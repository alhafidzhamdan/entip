"""Train the step-4 model: site-grouped cross-validation, temperature scaling and selective thresholds.

1. Every case gets a fold so that all cases from one TCGA tissue source site share a fold (``tcga.site_folds``).
2. For each outer fold: train ABMIL on the other folds, early-stop and fit the temperature on one of them, and
   predict the held-out fold. This gives an out-of-fold (OOF) prediction for every case from a site the model
   never saw.
3. From the OOF predictions: accuracy, balanced accuracy and ECE at class and family level, and the selective
   thresholds (``wsi.calibrate.selective_threshold``) for class and family at ``max_error``.
4. The final model trains on all folds but one (used for early stopping and its temperature) and carries the
   OOF thresholds and metrics.

Labels with a classifier score carry it as a sample weight, so borderline calls count less.
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
from .tcga import site_folds


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
          mil_kw: Optional[dict] = None, seed: int = 0, log: Callable[[str], None] = lambda s: None
          ) -> MethylationModel:
    mil_kw = dict(mil_kw or {})
    present = sorted({c.cls for c in cases})
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
    groups = [c.group for c in cases]
    if all(g is None for g in groups):
        fold = np.array(site_folds([c.case_id for c in cases], k=k, seed=seed))
    else:
        fold = np.array(_group_folds(groups, k, seed))
    k = int(fold.max()) + 1
    if k < 3:
        raise ValueError("need at least 3 site groups for train / validation / test folds")

    oof = np.zeros((len(cases), len(classes)))
    temps = []
    for f in range(k):
        v = (f + 1) % k
        tr, va, te = np.where((fold != f) & (fold != v))[0], np.where(fold == v)[0], np.where(fold == f)[0]
        m = ABMIL(len(classes), seed=seed + f, **mil_kw)
        m.fit([feats[i] for i in tr], y[tr], sample_weight=w[tr], val=([feats[i] for i in va], y[va]))
        t = fit_temperature(m.logits([feats[i] for i in va]), y[va])
        temps.append(t)
        oof[te] = _softmax(m.logits([feats[i] for i in te]) / t)
        log(f"fold {f}: train {len(tr)}, val {len(va)}, test {len(te)}, T={t:.2f}, "
            f"test acc {(oof[te].argmax(1) == y[te]).mean():.3f}")

    M = MethylationModel(model_id, version, classes, tax.family_of, encoders.pop(), min_tiles=min_tiles,
                         label_source=label_source)
    F = M.family_matrix()
    yf = np.array([M.families.index(tax.family_of[classes[i]]) for i in y])
    pf = oof @ F
    th_c = selective_threshold(oof.max(1), oof.argmax(1) == y, max_error, min_accepted)
    th_f = selective_threshold(pf.max(1), pf.argmax(1) == yf, max_error, min_accepted)
    M.class_threshold, M.family_threshold = th_c["threshold"], th_f["threshold"]
    M.metrics = {
        "n_cases": len(cases), "folds": k, "cv": "grouped by tissue source site", "max_error": max_error,
        "class": {"accuracy": float((oof.argmax(1) == y).mean()), "balanced_accuracy": _balanced_acc(oof.argmax(1), y),
                  "ece": ece(oof, y), "selective": th_c},
        "family": {"accuracy": float((pf.argmax(1) == yf).mean()), "balanced_accuracy": _balanced_acc(pf.argmax(1), yf),
                   "ece": ece(pf, yf), "selective": th_f},
        "fold_temperatures": temps,
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)},
    }
    log(f"OOF family accuracy {M.metrics['family']['accuracy']:.3f}; confident on "
        f"{th_f['coverage']:.0%} at threshold {th_f['threshold']:.2f}")

    # final model: all folds; the largest fold early-stops it and sets its temperature
    v = int(np.bincount(fold).argmax())
    tr, va = np.where(fold != v)[0], np.where(fold == v)[0]
    M.mil = ABMIL(len(classes), seed=seed, **mil_kw)
    M.mil.fit([feats[i] for i in tr], y[tr], sample_weight=w[tr], val=([feats[i] for i in va], y[va]))
    M.temperature = fit_temperature(M.mil.logits([feats[i] for i in va]), y[va])
    return M


def _group_folds(groups: Sequence[Optional[str]], k: int, seed: int) -> List[int]:
    fake = [f"TCGA-{g or 'NA'}-{i:04d}" for i, g in enumerate(groups)]   # reuse site_folds' balancing
    return site_folds(fake, k=k, seed=seed)
