"""Accept thresholds for a two-call marker, one per call.

A single "error among accepted calls <= 5%" rule is not safe for rare alterations: if 3% of tumours carry an
alteration, a model that confidently says "wildtype" for everyone is 97% right and would pass, while missing
every altered case. So a call of class c is accepted only above a threshold where BOTH hold, each as an upper
95% bound (exact Clopper-Pearson, one-sided, the same bound as ``wsi.calibrate``) on out-of-fold predictions:

* wrong-call rate: among accepted calls of c, the share that are wrong, is within ``max_error``;
* miss rate: among patients whose true class is NOT c, the share accepted as c, is within ``max_miss``.

The second rule means a negative call can never be accepted until enough altered patients have been seen:
with zero misses, the bound only drops to 5% from 59 altered patients upward (1 - 0.05 ** (1 / n) <= 0.05).

Thresholds are searched from the most confident call downward, starting where enough calls exist for the bound
to be reachable at all, and the search stops at the first cut that fails, so the bound is not picked as the best
of many tries (fixed-sequence testing).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..wsi.calibrate import exact_upper as _upper

NEVER = 1.01          # a threshold no confidence reaches: always order the test


def class_threshold(conf: np.ndarray, pred: np.ndarray, y: np.ndarray, c: int, max_error: float = 0.05,
                    max_miss: float = 0.05, min_accepted: int = 20) -> dict:
    """Threshold for accepting calls of class ``c``. conf: calibrated confidence of the predicted class;
    pred, y: predicted and true class per patient (out-of-fold)."""
    conf, pred, y = np.asarray(conf, float), np.asarray(pred, int), np.asarray(y, int)
    n_other = int((y != c).sum())                       # patients a call of c could be wrong about
    idx = np.flatnonzero(pred == c)
    order = idx[np.argsort(-conf[idx], kind="stable")]
    best = {"threshold": NEVER, "n_accepted": 0, "coverage": 0.0, "error": None, "error_upper": None,
            "miss": None, "miss_upper": None, "n_other_class": n_other, "max_error": max_error,
            "max_miss": max_miss}
    errs = 0
    for k, i in enumerate(order, start=1):
        errs += int(y[i] != c)
        if k < len(order) and conf[order[k]] == conf[i]:
            continue                                     # only cut between distinct confidence values
        if k < min_accepted or _upper(0, k) > max_error:
            continue                                     # too few calls to pass even with no errors
        e_ub, m_ub = _upper(errs, k), _upper(errs, n_other)
        if e_ub > max_error or m_ub > max_miss:
            break                                        # first failure ends the search
        best.update(threshold=float(conf[i]), n_accepted=k, coverage=k / max(1, len(y)), error=errs / k,
                    error_upper=e_ub, miss=errs / max(1, n_other), miss_upper=m_ub)
    return best


def marker_thresholds(conf, pred, y, max_error: float = 0.05, max_miss: float = 0.05,
                      min_accepted: int = 20) -> Dict[int, dict]:
    return {c: class_threshold(conf, pred, y, c, max_error, max_miss, min_accepted) for c in (0, 1)}
