"""The panel model: one attention-MIL classifier per marker on frozen tile embeddings, each with its own
calibration and its own accept threshold.

Training, per marker:
1. keep patients with a label (and whose own labels meet the marker's ``requires``, e.g. IDH-mutant for 1p/19q);
2. site-grouped cross-validation gives an out-of-fold logit for every labelled patient;
3. a temperature is fitted on those logits, and each call gets its own accept threshold: the lowest calibrated
   confidence at which both the wrong-call rate and the miss rate of that call stay within the marker's limits,
   as upper 95% bounds (``gate.py``); a call with no such threshold always orders the test;
4. the final classifier is trained on every labelled patient.

Markers per patient are separate models (not one shared head) so that each one can use every patient that has
that label; TCGA labels are patchy across markers. The tile pipeline, ABMIL and calibration are the shared ones
in ``entip_train.wsi``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..wsi.abmil import ABMIL
from ..wsi.calibrate import ece, fit_temperature
from .dataset import PatientBag
from .gate import NEVER, marker_thresholds
from .labels import Labels, apply_requirements
from .markers import PanelConfig, load_config

__version__ = "0.2.0"


@dataclass
class MarkerFit:
    finding: str
    trained: bool
    reason: str = ""
    temperature: float = 1.0
    thresholds: List[float] = field(default_factory=lambda: [NEVER, NEVER])   # per call (class 0, class 1)
    metrics: dict = field(default_factory=dict)
    clf: Optional[ABMIL] = None


def _groups_to_folds(groups: Sequence[str], k: int, seed: int = 0) -> np.ndarray:
    """Assign whole groups to k folds, balancing patient counts (largest groups first)."""
    groups = np.asarray([g or f"_p{i}" for i, g in enumerate(groups)])
    uniq, n = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = sorted(range(len(uniq)), key=lambda i: (-n[i], rng.random()))
    load = np.zeros(k)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[uniq[i]] = f
        load[f] += n[i]
    return np.array([fold_of[g] for g in groups])


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _auroc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


class PanelModel:
    def __init__(self, config: Optional[PanelConfig] = None, model_id: str = "glioma-panel",
                 version: str = __version__, licence: str = "head: this repository; encoder: H-optimus-0 Apache-2.0"):
        self.config = config or load_config()
        self.model_id, self.version, self.licence = model_id, version, licence
        self.fits: Dict[str, MarkerFit] = {}
        self.encoder = ""
        self.tile_um = 0.0
        self.ood: dict = {}
        self.trained_on: dict = {}

    # ------------------------------------------------------------------ out-of-distribution check
    def _bag_means(self, bags: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([b.mean(0) for b in bags]).astype(np.float64)

    def _fit_ood(self, bags: Sequence[np.ndarray]) -> None:
        m = self._bag_means(bags)
        mu, sd = m.mean(0), m.std(0) + 1e-6
        d = np.sqrt((((m - mu) / sd) ** 2).mean(1))
        self.ood = {"mu": mu, "sd": sd, "limit": float(np.quantile(d, self.config.ood_quantile))}

    def ood_distance(self, bag: np.ndarray) -> float:
        z = (bag.mean(0) - self.ood["mu"]) / self.ood["sd"]
        return float(np.sqrt((z ** 2).mean()))

    # ------------------------------------------------------------------ training
    def fit(self, patients: Sequence[PatientBag], labels: Labels, source: str = "", log=print) -> "PanelModel":
        if not patients:
            raise ValueError("no patients")
        encs = {p.encoder for p in patients}
        if len(encs) != 1:
            raise ValueError(f"patients embedded with different encoders: {sorted(encs)}")
        self.encoder, self.tile_um = encs.pop(), float(patients[0].tile_um)
        labels = apply_requirements(labels)
        self._fit_ood([p.features for p in patients])
        self.trained_on = {"source": source, "n_patients": len(patients), "encoder": self.encoder,
                           "tile_um": self.tile_um, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for m in self.config.active:
            idx = [i for i, p in enumerate(patients) if m.finding in labels.get(p.study_id, {})]
            y = np.array([m.calls.index(labels[patients[i].study_id][m.finding]) for i in idx], int)
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            if min(n0, n1) < m.min_per_class:
                self.fits[m.finding] = MarkerFit(m.finding, False,
                                                 f"too few labelled patients ({m.calls[0]} {n0}, {m.calls[1]} {n1}; "
                                                 f"need {m.min_per_class} of each)",
                                                 metrics={"n": len(y), "n_per_class": [n0, n1]})
                log(f"[panel] {m.finding}: not trained, {self.fits[m.finding].reason}")
                continue
            self.fits[m.finding] = self._fit_marker(m, [patients[i] for i in idx], y, log)
        return self

    def _new_clf(self) -> ABMIL:
        return ABMIL(2, **self.config.abmil)

    def _fit_marker(self, m, pats: List[PatientBag], y: np.ndarray, log) -> MarkerFit:
        bags = [p.features for p in pats]
        k = min(self.config.cv_folds, len({p.group or p.study_id for p in pats}))
        folds = _groups_to_folds([p.group for p in pats], max(2, k))
        oof = np.zeros((len(y), 2))
        scored = np.zeros(len(y), bool)       # a fold whose training side lacks a class cannot be scored
        for f in range(folds.max() + 1):
            te, tr = folds == f, folds != f
            if len(np.unique(y[tr])) < 2 or not te.any():
                continue
            clf = self._new_clf().fit([bags[i] for i in np.flatnonzero(tr)], y[tr])
            oof[te] = clf.logits([bags[i] for i in np.flatnonzero(te)])
            scored[te] = True
        ys, zs = y[scored], oof[scored]
        metrics = {"n": int(len(y)), "n_per_class": [int((y == 0).sum()), int((y == 1).sum())],
                   "n_scored_oof": int(scored.sum()), "folds": int(folds.max() + 1),
                   "cv_groups": "tissue source site" if all(p.group for p in pats) else "patient"}
        if len(np.unique(ys)) < 2:
            reason = "cross-validation could not score both classes (too few sites or patients)"
            log(f"[panel] {m.finding}: not trained, {reason}")
            return MarkerFit(m.finding, False, reason, metrics=metrics)
        t = fit_temperature(zs, ys)
        p = _softmax(zs / t)
        conf, pred = p.max(1), p.argmax(1)
        acc = marker_thresholds(conf, pred, ys, m.max_error, m.max_miss)
        metrics.update({
            "auroc_oof": _auroc(ys, p[:, 1]), "balanced_accuracy_oof": _balanced_acc(ys, pred),
            "ece_oof_raw": ece(_softmax(zs), ys), "ece_oof_calibrated": ece(p, ys), "temperature": t,
            "accept": {m.calls[c]: acc[c] for c in (0, 1)},
            "coverage_oof": float(sum(acc[c]["n_accepted"] for c in (0, 1)) / len(ys)),
        })
        clf = self._new_clf().fit(bags, y)
        log(f"[panel] {m.finding}: n={len(y)} AUROC(oof)={metrics['auroc_oof']:.3f}; accepts "
            + ", ".join(f"{m.calls[c]} {acc[c]['n_accepted']}/{int((pred == c).sum())}" for c in (0, 1)))
        return MarkerFit(m.finding, True, temperature=t, thresholds=[float(acc[0]["threshold"]),
                                                                    float(acc[1]["threshold"])],
                         metrics=metrics, clf=clf)

    # ------------------------------------------------------------------ inference
    def predict_proba(self, finding: str, bag: np.ndarray) -> np.ndarray:
        f = self.fits[finding]
        return f.clf.predict_proba([bag], temperature=f.temperature)[0]

    def attention(self, finding: str, bag: np.ndarray) -> np.ndarray:
        return self.fits[finding].clf.attention(bag)

    # ------------------------------------------------------------------ persistence (no pickle)
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"model_id": self.model_id, "version": self.version, "licence": self.licence,
                    "encoder": self.encoder, "tile_um": self.tile_um, "trained_on": self.trained_on,
                    "config": self.config.to_dict(), "ood_limit": self.ood.get("limit"), "markers": {}}
        for name, f in self.fits.items():
            manifest["markers"][name] = {"trained": f.trained, "reason": f.reason, "temperature": f.temperature,
                                         "thresholds": f.thresholds, "metrics": f.metrics}
            if f.trained:
                f.clf.save(out / f"{name}.npz")
        np.savez(out / "ood.npz", mu=self.ood["mu"], sd=self.ood["sd"])
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> "PanelModel":
        out = Path(out_dir)
        man = json.loads((out / "manifest.json").read_text())
        c = man["config"]
        cfg = load_config(None, markers=c["markers"], disabled=c["disabled"], overrides=c["overrides"],
                          min_tiles=c["min_tiles"], ood_quantile=c["ood_quantile"], top_evidence=c["top_evidence"],
                          cv_folds=c["cv_folds"], abmil=c["abmil"])
        m = cls(cfg, man["model_id"], man["version"], man["licence"])
        m.encoder, m.tile_um, m.trained_on = man["encoder"], man["tile_um"], man["trained_on"]
        with np.load(out / "ood.npz", allow_pickle=False) as z:
            m.ood = {"mu": z["mu"], "sd": z["sd"], "limit": man["ood_limit"]}
        for name, f in man["markers"].items():
            m.fits[name] = MarkerFit(name, f["trained"], f["reason"], f["temperature"], list(f["thresholds"]), f["metrics"],
                                     ABMIL.load(out / f"{name}.npz") if f["trained"] else None)
        return m
