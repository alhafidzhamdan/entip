"""Train the step-4 model: site-grouped cross-validation, temperature scaling and selective thresholds.

1. Every case gets a fold so that all cases from one TCGA tissue source site share a fold (``tcga.group_folds``).
2. For each fold: train ABMIL on the other folds, early-stop and fit the temperature on one of them, and
   predict the held-out fold. This gives an out-of-fold (OOF) prediction for every case from a site the model
   never saw.
3. From the OOF predictions: accuracy, balanced accuracy and ECE at class and family level, per-family recall
   and share of confident calls, and the selective thresholds (``wsi.calibrate.selective_threshold``) for
   class and family at ``max_error``. The novelty cut-off is set the same way (``novelty.oof_threshold``).
4. The deployed model is the ensemble of the fold members, so no part of it is untested.

Labels with a classifier score carry it as a sample weight, so borderline calls count less.

Limits to keep in mind when reading the numbers: the thresholds were measured on single fold members while
the ensemble is deployed, and everything is TCGA. An external set (Edinburgh) is the real test. With the
exact bound in ``selective_threshold``, no call is accepted at a 5% error limit before 59 error-free cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..wsi.abmil import ABMIL
from ..wsi.bags import Bag
from ..wsi.calibrate import ece, fit_temperature, selective_threshold
from .classes import Taxonomy
from .model import MethylationModel, pool_bags
from .novelty import Novelty, descriptor, oof_threshold
from .tcga import group_folds, site_of


@dataclass
class Case:
    case_id: str
    bags: List[Bag]
    cls: str
    weight: float = 1.0
    group: Optional[str] = None      # CV group; default: TCGA tissue source site from the case id


def _balanced_acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([(pred[y == k] == k).mean() for k in np.unique(y)]))


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(cases: Sequence[Case], taxonomy: Taxonomy, model_id: str, version: str, k: int = 5,
          max_error: float = 0.05, min_accepted: int = 20, min_tiles: int = 50, label_source: str = "",
          min_per_class: int = 10, novelty_quantile: float = 0.99, mil_kw: Optional[dict] = None, seed: int = 0,
          log: Callable[[str], None] = lambda s: None) -> MethylationModel:
    """Classes with fewer than ``min_per_class`` cases are dropped (logged and listed in the metrics): the
    model cannot learn them, and slides of those classes fall back to the array."""
    mil_kw = dict(mil_kw or {})
    n_by = {}
    for c in cases:
        n_by[c.cls] = n_by.get(c.cls, 0) + 1
    dropped = {c: n for c, n in sorted(n_by.items()) if n < min_per_class}
    if dropped:
        log(f"dropping classes with < {min_per_class} cases: {dropped}")
        cases = [c for c in cases if c.cls not in dropped]
    present = sorted({c.cls for c in cases})
    if len(present) < 2:
        raise ValueError(f"need at least 2 classes with >= {min_per_class} cases; have {n_by}")
    unknown = [c for c in present if c not in taxonomy.family_of]
    if unknown:
        raise ValueError(f"classes not in the taxonomy: {unknown}; give a taxonomy CSV")
    tax = taxonomy.restrict(present)
    classes = tax.classes
    y = np.array([classes.index(c.cls) for c in cases])
    w = np.array([c.weight for c in cases], np.float64)
    feats = [pool_bags(c.bags)[0] for c in cases]
    encoders = {b.encoder for c in cases for b in c.bags}
    if len(encoders) != 1:
        raise ValueError(f"all bags must come from one encoder, got {sorted(encoders)}")
    groups = [c.group if c.group is not None else site_of(c.case_id) for c in cases]
    fold = np.array(group_folds(groups, k=k, seed=seed))
    k = int(fold.max()) + 1
    if k < 3:
        raise ValueError("need at least 3 site groups for train / validation / test folds")

    oof = np.zeros((len(cases), len(classes)))
    temps, members = [], []
    for f in range(k):
        v = (f + 1) % k
        tr, va, te = np.where((fold != f) & (fold != v))[0], np.where(fold == v)[0], np.where(fold == f)[0]
        m = ABMIL(len(classes), seed=seed + f, **mil_kw)
        m.fit([feats[i] for i in tr], y[tr], sample_weight=w[tr], val=([feats[i] for i in va], y[va]))
        t = fit_temperature(m.logits([feats[i] for i in va]), y[va])
        temps.append(t)
        members.append(m)
        oof[te] = _softmax(m.logits([feats[i] for i in te]) / t)
        log(f"fold {f}: train {len(tr)}, val {len(va)}, test {len(te)}, T={t:.2f}, "
            f"test acc {(oof[te].argmax(1) == y[te]).mean():.3f}")

    M = MethylationModel(model_id, version, classes, tax.family_of, encoders.pop(), min_tiles=min_tiles,
                         label_source=label_source, temperatures=temps, members=members)
    F = M.family_matrix()
    yf = np.array([M.families.index(tax.family_of[classes[i]]) for i in y])
    pf = oof @ F
    th_c = selective_threshold(oof.max(1), oof.argmax(1) == y, max_error, min_accepted)
    th_f = selective_threshold(pf.max(1), pf.argmax(1) == yf, max_error, min_accepted)
    M.class_threshold, M.family_threshold = th_c["threshold"], th_f["threshold"]
    descs = np.stack([descriptor(f) for f in feats])
    nov_thr, nov_scores = oof_threshold(descs, fold, quantile=novelty_quantile)
    M.novelty = Novelty.fit(descs)
    M.novelty.threshold, M.novelty.quantile = nov_thr, novelty_quantile
    M.metrics = {
        "n_cases": len(cases), "folds": k, "cv": "grouped by tissue source site", "max_error": max_error,
        "class": {"accuracy": float((oof.argmax(1) == y).mean()), "balanced_accuracy": _balanced_acc(oof.argmax(1), y),
                  "ece": ece(oof, y), "selective": th_c},
        "family": {"accuracy": float((pf.argmax(1) == yf).mean()), "balanced_accuracy": _balanced_acc(pf.argmax(1), yf),
                   "ece": ece(pf, yf), "selective": th_f},
        "per_family": _per_family(pf, yf, M.families, th_f["threshold"]),
        "novelty": {"threshold": nov_thr, "quantile": novelty_quantile,
                    "share_flagged_oof": float((nov_scores > nov_thr).mean())},
        "fold_temperatures": temps,
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)},
        "dropped_classes": dropped,
    }
    log(f"OOF family accuracy {M.metrics['family']['accuracy']:.3f}; confident on "
        f"{th_f['coverage']:.0%} at threshold {th_f['threshold']:.2f}")

    return M


def _per_family(pf: np.ndarray, yf: np.ndarray, families: List[str], thr: float) -> Dict[str, dict]:
    """Recall and share of confident calls per true family, on the OOF predictions."""
    pred, conf = pf.argmax(1), pf.max(1)
    out = {}
    for i, f in enumerate(families):
        m = yf == i
        if m.any():
            ok = m & (conf >= thr)
            out[f] = {"n": int(m.sum()), "recall": float((pred[m] == i).mean()),
                      "confident_share": float(ok.sum() / m.sum()),
                      "confident_errors": int((pred[ok] != i).sum())}
    return out
