"""Calibration and the accept / order-the-test gate shared by the glioma workflow models.

* ``fit_temperature``: one scalar T that minimises validation log loss (temperature scaling,
  Guo et al. 2017, https://arxiv.org/abs/1706.04599). It changes confidence, never the predicted class.
* ``ece``: expected calibration error with equal-width bins.
* ``selective_threshold``: the lowest confidence above which the model's error rate on held-out slides stays
  under ``max_error``, judged by a one-sided Wilson upper bound so a handful of lucky slides cannot set it.
  Below the threshold the model orders the real test.
"""
from __future__ import annotations

import math

import numpy as np


def _softmax(v, axis=-1):
    v = v - v.max(axis=axis, keepdims=True)
    e = np.exp(v)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    logits, y = np.asarray(logits, float), np.asarray(y, int)

    def nll(t):
        p = _softmax(logits / t, axis=1)
        return -np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
    t = grid[int(np.argmin([nll(g) for g in grid]))]
    lo, hi = t / 1.1, t * 1.1                         # refine by golden-section search
    for _ in range(40):
        a, b = hi - (hi - lo) / 1.618, lo + (hi - lo) / 1.618
        lo, hi = (lo, b) if nll(a) < nll(b) else (a, hi)
    return float((lo + hi) / 2)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def wilson_upper(errors: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% upper bound on an error rate (z = 1.645)."""
    if n == 0:
        return 1.0
    p = errors / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + rad) / den)


def selective_threshold(conf: np.ndarray, correct: np.ndarray, max_error: float = 0.05,
                        min_accepted: int = 20) -> dict:
    """Pick the confidence threshold for "accept". Returns threshold, coverage (share accepted), the observed
    error and its upper bound among accepted slides. If no threshold meets ``max_error``, threshold is 1.01:
    the model never accepts and always orders the test."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    order = np.argsort(-conf)
    best = {"threshold": 1.01, "coverage": 0.0, "error": None, "error_upper": None, "n_accepted": 0,
            "max_error": max_error}
    errs = 0
    for k, i in enumerate(order, start=1):
        errs += int(not correct[i])
        if k < min_accepted or (k < len(order) and conf[order[k]] == conf[i]):
            continue                                   # only cut between distinct confidence values
        ub = wilson_upper(errs, k)
        if ub <= max_error:
            best = {"threshold": float(conf[i]), "coverage": k / len(conf), "error": errs / k,
                    "error_upper": ub, "n_accepted": k, "max_error": max_error}
    return best
