"""The step 1 triage model: train with cross-fitting, calibrate, gate, and emit contract records.

Training (``TriageModel.fit``):

1. Classes with fewer than ``min_patients`` patients are left out and listed in the report (public data has
   almost no metastasis or lymphoma and no reactive brain; those classes wait for Edinburgh slides).
2. ``folds``-fold cross-validation, patients never split across folds, gives an out-of-fold probability for
   every slide. Each fold trains an ABMIL with early stopping on a patient-held-out validation share.
3. Temperature scaling is fitted on those out-of-fold predictions. The accept threshold is fitted on one row
   per patient (the patient's most abnormal slide, the same rule used at prediction), because slides of one
   patient are not independent.
4. A novelty check (``wsi/novelty.py``) learns how far training slides sit from their class centroids; its
   cut-off also comes from out-of-fold slides.
5. The final model is trained on everything and carries that temperature, threshold and novelty check.
   The cross-validated numbers describe the fold models; the final model is assumed to behave alike.

A patient gets a confident call only if the chosen slide is above the threshold, is not novel, and no other
slide of the patient confidently says something else.

Checks against shortcuts and blind spots, all in the report:

* ``source_check``: within one class, how well the source can be told apart from mean-pooled features
  (high = strong source signal in the features). Public sources are mostly single-class (all GTEx cortex is
  normal and PAXgene-fixed, all TCGA is glioma), so a model can learn the source rather than the tissue;
* ``leave_one_source_out``: for classes with several sources, accuracy on a source the model never saw;
* ``unseen_classes``: slides of the left-out classes are scored by the final model. They can only be wrong,
  so the report gives how many would have been accepted as a confident call, and how many the novelty check
  catches.

Probabilities reflect the class mix of the training data after balanced weighting, not Edinburgh's case mix.
Recalibrate on local slides before any use.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import __version__
from ..wsi.abmil import ABMIL, _softmax
from ..wsi.bags import Bag
from ..wsi.calibrate import ece, fit_temperature, selective_threshold
from ..wsi.novelty import Novelty, pooled
from .labels import CLASSES, GLIAL, IDX, TUMOUR, derived

CONTRACT = "entip.glioma_step/1"
FEATURE = "cns_tissue_class"


@dataclass
class TriageConfig:
    hidden: int = 128
    attn: int = 64
    epochs: int = 30
    lr: float = 1e-3
    max_tiles: int = 512
    patience: int = 8
    folds: int = 5
    val_share: float = 0.15
    min_patients: int = 5
    max_error: float = 0.05            # accepted calls may be wrong at most this often (upper 95% bound)
    min_accepted: int = 20
    # Calls that always go to a pathologist even when confident: anything not heading into steps 2-4.
    review_calls: tuple = ("normal_brain", "reactive", "other_primary_cns", "metastasis", "lymphoma")
    # Adequacy is tissue area only (lesional content is not judged). 1 mm² is a placeholder for Al to set.
    min_area_mm2: float = 1.0
    top_k: int = 20
    novelty_quantile: float = 0.99     # training slides beyond this out-of-fold distance count as novel
    seed: int = 0


def _abmil(cfg: TriageConfig, seed: int) -> ABMIL:
    return ABMIL(len(CLASSES), hidden=cfg.hidden, attn=cfg.attn, lr=cfg.lr, epochs=cfg.epochs,
                 max_tiles=cfg.max_tiles, patience=cfg.patience, seed=seed)


def _patient_split(patients: np.ndarray, share: float, rng: np.random.Generator):
    uniq = np.unique(patients)
    rng.shuffle(uniq)
    n_val = max(1, int(round(share * len(uniq))))
    val = np.isin(patients, uniq[:n_val])
    return np.flatnonzero(~val), np.flatnonzero(val)


def _fit_one(cfg, X, y, patients, seed) -> ABMIL:
    rng = np.random.default_rng(seed)
    tr, va = _patient_split(patients, cfg.val_share, rng)
    if len(np.unique(y[tr])) < len(np.unique(y)):          # tiny classes: keep them all in training
        tr, va = np.arange(len(y)), np.array([], int)
    m = _abmil(cfg, seed)
    val = ([X[i] for i in va], y[va]) if len(va) else None
    return m.fit([X[i] for i in tr], y[tr], val=val)


def _folds(y, patients, k, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    k = min(k, len(np.unique(patients)))
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y,
                                                                                         patients))


def p_tumour(probs: np.ndarray) -> np.ndarray:
    return np.atleast_2d(probs)[:, [IDX[c] for c in TUMOUR]].sum(1)


def patient_rows(probs: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """Index of each patient's most abnormal slide (highest tumour probability): one row per patient."""
    pt = p_tumour(probs)
    return np.array([np.flatnonzero(patients == p)[np.argmax(pt[patients == p])] for p in np.unique(patients)])


def metrics(probs: np.ndarray, y: np.ndarray, trained: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = probs.argmax(1)
    out = {"n": int(len(y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "recall": {CLASSES[c]: float((pred[y == c] == c).mean()) for c in np.unique(y)},
           "ece": ece(probs, y)}
    yt = np.isin(y, [IDX[c] for c in TUMOUR]).astype(int)
    yg = np.isin(y, [IDX[c] for c in GLIAL]).astype(int)
    d = [derived(p) for p in probs]
    if 0 < yt.sum() < len(yt):
        out["auroc_tumour"] = float(roc_auc_score(yt, [v["p_tumour"] for v in d]))
    if 0 < yg.sum() < len(yg):
        out["auroc_glial"] = float(roc_auc_score(yg, [v["p_glial"] for v in d]))
    return out


def source_check(X: Sequence[np.ndarray], labels: Sequence[str], groups: Sequence[str],
                 patients: Sequence[str], min_slides: int = 5, seed: int = 0) -> dict:
    """Within each class, can the source be predicted from mean-pooled tile features? Returns balanced
    accuracy of a grouped-CV logistic regression against chance (1 / number of sources)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    labels, groups, patients = np.asarray(labels), np.asarray(groups), np.asarray(patients)
    pooled = np.stack([np.asarray(b).mean(0) for b in X])
    out = {"per_class": {}}
    single = Counter()
    for g in np.unique(groups):
        single[len(np.unique(labels[groups == g])) == 1] += int((groups == g).sum())
    out["share_of_slides_in_single_class_sources"] = single[True] / max(1, len(groups))
    for c in np.unique(labels):
        m = labels == c
        cnt = Counter(groups[m])
        keep = m & np.isin(groups, [g for g, n in cnt.items() if n >= min_slides])
        srcs = np.unique(groups[keep])
        if len(srcs) < 2 or len(np.unique(patients[keep])) < 3:
            out["per_class"][c] = {"sources": [str(s) for s in srcs], "checked": False}
            continue
        Xc, yc, pc = pooled[keep], groups[keep], patients[keep]
        k = min(3, len(np.unique(pc)))
        pred = np.empty(len(yc), dtype=yc.dtype)
        for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(Xc, yc, pc):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            if len(np.unique(yc[tr])) < 2:
                pred[te] = yc[tr][0]
                continue
            pred[te] = clf.fit(Xc[tr], yc[tr]).predict(Xc[te])
        out["per_class"][c] = {"sources": [str(s) for s in srcs], "checked": True,
                               "source_balanced_accuracy": float(balanced_accuracy_score(yc, pred)),
                               "chance": 1.0 / len(srcs)}
    return out


class TriageModel:
    def __init__(self, cfg: Optional[TriageConfig] = None):
        self.cfg = cfg or TriageConfig()
        self.mil: Optional[ABMIL] = None
        self.temperature = 1.0
        self.threshold = 1.01
        self.trained = np.ones(len(CLASSES), bool)
        self.encoder = ""
        self.report: dict = {}
        self.model_id = ""
        self.version = ""
        self.novelty: Optional[Novelty] = None

    # ------------------------------------------------------------------ training
    def fit(self, bags: Sequence[Bag], labels: Sequence[str], groups: Sequence[str], patients: Sequence[str],
            loso: bool = True) -> dict:
        cfg = self.cfg
        labels, groups, patients = np.asarray(labels), np.asarray(groups), np.asarray(patients)
        encs = {b.encoder for b in bags}
        if len(encs) != 1:
            raise ValueError(f"bags come from different encoders: {sorted(encs)}")
        self.encoder = encs.pop()
        n_pat = {c: len(np.unique(patients[labels == c])) for c in CLASSES}
        use = [c for c in CLASSES if n_pat[c] >= cfg.min_patients]
        if len(use) < 2:
            raise ValueError(f"need at least two classes with >= {cfg.min_patients} patients; have {n_pat}")
        keep = np.isin(labels, use)
        self.trained = np.array([c in use for c in CLASSES])
        X = [bags[i].features for i in np.flatnonzero(keep)]
        y = np.array([IDX[c] for c in labels[keep]])
        P, G = patients[keep], groups[keep]

        Z = pooled(X)
        oof = np.zeros((len(y), len(CLASSES)))
        oof_dist = np.zeros(len(y))
        for f, (tr, te) in enumerate(_folds(y, P, cfg.folds, cfg.seed)):
            m = _fit_one(cfg, [X[i] for i in tr], y[tr], P[tr], cfg.seed + f)
            oof[te] = m.logits([X[i] for i in te])
            oof_dist[te] = Novelty(quantile=cfg.novelty_quantile).fit(Z[tr], y[tr]).distance(Z[te])
        oof[:, ~self.trained] = -1e9
        raw = _softmax(oof, axis=1)
        self.temperature = fit_temperature(oof, y)
        cal = _softmax(oof / self.temperature, axis=1)
        rows = patient_rows(cal, P)
        gate = selective_threshold(cal[rows].max(1), cal[rows].argmax(1) == y[rows], cfg.max_error,
                                   cfg.min_accepted)
        gate["unit"] = "patient (most abnormal slide)"
        self.threshold = gate["threshold"]
        self.novelty = Novelty(quantile=cfg.novelty_quantile).fit(Z, y)
        self.novelty.fit_threshold(oof_dist)

        rep = {"n_slides": int(len(y)), "n_patients": int(len(np.unique(P))),
               "slides_per_class": {c: int((labels[keep] == c).sum()) for c in use},
               "patients_per_class": n_pat, "classes_trained": use,
               "classes_excluded": [c for c in CLASSES if c not in use],
               "cv": {"folds": cfg.folds, "split": "patient", "uncalibrated": metrics(raw, y, self.trained),
                      "calibrated": metrics(cal, y, self.trained),
                      "per_patient": metrics(cal[rows], y[rows], self.trained),
                      "temperature": self.temperature, "gate": gate},
               "novelty": {"threshold": self.novelty.threshold, "quantile": cfg.novelty_quantile,
                           "share_of_training_slides_flagged": float((oof_dist > self.novelty.threshold).mean())},
               "per_source_accuracy": {str(g): float((cal.argmax(1)[G == g] == y[G == g]).mean())
                                       for g in np.unique(G)},
               "source_check": source_check(X, labels[keep], G, P, seed=cfg.seed)}
        if loso:
            rep["leave_one_source_out"] = self._loso(X, y, G, P)
        self.mil = _fit_one(cfg, X, y, P, cfg.seed + 100)
        h = hashlib.sha1(b"".join(self.mil.p[k].tobytes() for k in sorted(self.mil.p))).hexdigest()[:8]
        self.model_id, self.version = f"cns-triage-{h}", f"{__version__}+{h}"
        rep["unseen_classes"] = self._unseen([bags[i] for i in np.flatnonzero(~keep)], labels[~keep])
        rep.update({"model_id": self.model_id, "version": self.version, "encoder": self.encoder,
                    "config": asdict(self.cfg)})
        self.report = rep
        return rep

    def _unseen(self, bags, labels) -> dict:
        """Slides of classes the model was not trained on: every call is wrong, so count the accepted ones."""
        out = {}
        for c in np.unique(labels):
            ps = [self.predict(b) for b, l in zip(bags, labels) if l == c and len(b)]
            if not ps:
                continue
            acc = [p for p in ps if p["probs"].max() >= self.threshold]
            out[str(c)] = {"n": len(ps), "called": dict(Counter(CLASSES[int(p["probs"].argmax())] for p in ps)),
                           "above_threshold": len(acc), "flagged_novel": int(sum(p["novel"] for p in ps)),
                           "accepted_wrongly": int(sum(not p["novel"] for p in acc))}
        return out

    def _loso(self, X, y, G, P) -> dict:
        """For each source of a class that has another source too: train without it, test on it."""
        out = {}
        for g in np.unique(G):
            c = np.unique(y[G == g])
            if len(c) != 1 or len(np.unique(G[(y == c[0]) & (G != g)])) == 0:
                continue
            tr, te = np.flatnonzero(G != g), np.flatnonzero(G == g)
            if len(np.unique(y[tr])) < 2:
                continue
            m = _fit_one(self.cfg, [X[i] for i in tr], y[tr], P[tr], self.cfg.seed + 200)
            lg = m.logits([X[i] for i in te])
            lg[:, ~self.trained] = -1e9
            out[str(g)] = {"class": CLASSES[int(c[0])], "n": int(len(te)),
                           "accuracy": float((lg.argmax(1) == y[te]).mean())}
        return out

    # ------------------------------------------------------------------ inference
    def predict(self, bag: Bag) -> dict:
        """One slide: calibrated class probabilities, attention per tile, novelty distance."""
        if self.mil is None:
            raise RuntimeError("model is not trained or loaded")
        if bag.encoder != self.encoder:
            raise ValueError(f"bag encoder {bag.encoder!r} differs from the model's {self.encoder!r}")
        if len(bag) == 0:
            return {"scan_id": bag.scan_id, "n_tiles": 0, "probs": None, "attention": np.zeros(0),
                    "distance": None, "novel": False}
        lg = self.mil.logits([bag.features])[0]
        lg[~self.trained] = -1e9
        p = _softmax(lg / self.temperature)
        dist = float(self.novelty.distance(pooled([bag.features]))[0]) if self.novelty is not None else None
        return {"scan_id": bag.scan_id, "n_tiles": len(bag), "probs": p,
                "attention": self.mil.attention(bag.features), "distance": dist,
                "novel": bool(dist is not None and dist > self.novelty.threshold)}

    def records(self, bags: Sequence[Bag], study_id: str) -> List[dict]:
        """Contract records (entip.glioma_step/1) for one patient: ``adequacy`` and ``tissue_class``.

        With several H&E scans the call comes from the most abnormal one (highest tumour probability):
        averaging would let three bland slides outvote the one that holds the tumour."""
        cfg = self.cfg
        model = {"model_id": self.model_id, "version": self.version, "licence": "MIT (code); see model card"}
        prov = {"kind": "model", "actor": self.model_id, "version": self.version}
        scan_ids = [b.scan_id for b in bags]
        area = float(sum(len(b) * b.tile_um ** 2 for b in bags) / 1e6)
        lo, hi = 0.5 * cfg.min_area_mm2, 2.0 * cfg.min_area_mm2
        adequate = area >= cfg.min_area_mm2
        clear = area >= hi or area < lo
        val = {"tissue_area_mm2": round(area, 4), "min_area_mm2": cfg.min_area_mm2}
        sharp = [b.sharpness for b in bags if b.sharpness is not None and len(b.sharpness)]
        if sharp:
            sh = np.concatenate(sharp)
            val.update({"sharpness_median": round(float(np.median(sh)), 1),
                        "sharpness_p10": round(float(np.percentile(sh, 10)), 1)})
        out = [{"contract": CONTRACT, "step": "triage", "finding": "adequacy", "study_id": study_id,
                "scan_ids": scan_ids, "call": "adequate" if adequate else "inadequate", "source": "ai",
                "confidence": 1.0 if clear else 0.5, "threshold": 0.9,
                "status": "confident" if clear else "uncertain",
                "probabilities": {}, "value": val,
                "evidence": [{"kind": "scan", "id": s} for s in scan_ids],
                "order_test": None if (clear and adequate) else "additional_sampling",
                "reasons": [f"{area:.2f} mm² of tissue tiles (rule: at least {cfg.min_area_mm2} mm²); "
                            "tissue area only; lesional content and focus are reported, not judged"],
                "model": {"model_id": "triage-adequacy-rule", "version": __version__, "licence": "MIT"},
                "provenance": {"kind": "program", "actor": "entip-train triage", "version": __version__}}]
        scored = [(self.predict(b), b) for b in bags]
        scored = [(g, b) for g, b in scored if g["probs"] is not None]
        if not scored:
            return out
        g, b = max(scored, key=lambda gb: float(p_tumour(gb[0]["probs"])[0]))
        p = g["probs"]
        call = CLASSES[int(p.argmax())]
        conf = float(p.max())
        never = self.threshold > 1.0
        reasons = [f"calibrated confidence {conf:.3f} vs threshold {min(self.threshold, 1.0):.3f}"]
        problems = []
        if never:
            problems.append("no threshold met the error target in validation: this model never accepts")
        elif conf < self.threshold:
            problems.append("below threshold: pathologist to classify")
        if g["novel"]:
            problems.append(f"slide is unlike the training slides (distance {g['distance']:.2f} vs "
                            f"{self.novelty.threshold:.2f}): possible unseen tumour type, stain or scanner")
        others = sorted({CLASSES[int(o["probs"].argmax())] for o, _ in scored
                         if o is not g and o["probs"].max() >= self.threshold} - {call})
        if others:
            problems.append(f"another slide of this patient is confidently called {', '.join(others)}")
        confident = not problems
        order = None
        if problems:
            order = "pathologist_review"
            reasons += problems
        elif call in cfg.review_calls:
            order = "pathologist_review"
            reasons.append(f"{call}: outside the glioma pathway, pathologist to confirm and work up")
        if len(scored) > 1:
            reasons.append(f"call taken from the most abnormal of {len(scored)} slides ({b.scan_id})")
        evidence = []
        for r, i in enumerate(np.argsort(-g["attention"])[:cfg.top_k]):
            x, y = (float(v) for v in b.coords_um[i])
            evidence.append({"kind": "tile", "id": tile_region_id(b.scan_id, self.model_id, r),
                             "x_um": x, "y_um": y, "w_um": b.tile_um, "h_um": b.tile_um,
                             "weight": float(g["attention"][i])})
        value = {k: round(v, 4) for k, v in derived(p).items()}
        value.update({"n_tiles": int(g["n_tiles"]), "novel": g["novel"],
                      "classes_trained": [c for c, t in zip(CLASSES, self.trained) if t],
                      "per_scan": [{"scan_id": o["scan_id"], "call": CLASSES[int(o["probs"].argmax())],
                                    "confidence": round(float(o["probs"].max()), 4),
                                    "p_tumour": round(float(p_tumour(o["probs"])[0]), 4), "novel": o["novel"]}
                                   for o, _ in scored]})
        out.append({"contract": CONTRACT, "step": "triage", "finding": "tissue_class", "study_id": study_id,
                    "scan_ids": [o["scan_id"] for o, _ in scored], "call": call, "source": "ai",
                    "confidence": round(conf, 4), "threshold": round(min(self.threshold, 1.0), 4),
                    "status": "confident" if confident else "uncertain",
                    "probabilities": {c: round(float(v), 4) for c, v, t in zip(CLASSES, p, self.trained) if t},
                    "value": value, "evidence": evidence, "order_test": order, "reasons": reasons,
                    "model": model, "provenance": prov})
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.mil.save(d / "abmil.npz")
        if self.novelty is not None:
            np.savez(d / "novelty.npz", **self.novelty.state())
        meta = {"model_id": self.model_id, "version": self.version, "encoder": self.encoder,
                "temperature": self.temperature, "threshold": self.threshold,
                "trained": [c for c, t in zip(CLASSES, self.trained) if t], "config": asdict(self.cfg),
                "report": self.report}
        (d / "triage.json").write_text(json.dumps(meta, indent=2, default=float))
        return d

    @classmethod
    def load(cls, model_dir: str | Path) -> "TriageModel":
        d = Path(model_dir)
        meta = json.loads((d / "triage.json").read_text())
        cfg = meta["config"]
        cfg["review_calls"] = tuple(cfg["review_calls"])
        m = cls(TriageConfig(**cfg))
        m.mil = ABMIL.load(d / "abmil.npz")
        if (d / "novelty.npz").exists():
            with np.load(d / "novelty.npz", allow_pickle=False) as z:
                m.novelty = Novelty.from_state(z)
        m.model_id, m.version, m.encoder = meta["model_id"], meta["version"], meta["encoder"]
        m.temperature, m.threshold = float(meta["temperature"]), float(meta["threshold"])
        m.trained = np.array([c in meta["trained"] for c in CLASSES])
        m.report = meta.get("report", {})
        return m


def tile_region_id(scan_id: str, model_id: str, rank: int) -> str:
    return f"{scan_id}:{model_id}:t{rank:03d}"


def slide_region_id(scan_id: str, model_id: str) -> str:
    return f"{scan_id}:{model_id}:slide"
