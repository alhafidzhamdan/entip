"""Tier 2: gated champion/challenger retraining (after MitoEd's src/mitoed/train/retrain.py).

One round:
  1. Gather every firm, non-locked mark for the feature across scans (plus any external training set).
  2. Train a challenger. Live heads are drafts; only a gated round makes a champion.
  3. Score champion and challenger on the LOCKED set: marks flagged ``locked`` (or on locked scans),
     which are never trained on.
  4. Promote only if the challenger passes the gate: overall balanced accuracy not lower than the
     champion's minus ``tol_overall``; no scan group worse by more than ``tol_group``; calibration (ECE)
     not worse by more than ``tol_ece``. With no champion yet, the challenger needs ``min_locked_ba``.
  5. Record the round either way (Model row + action log); the old champion becomes ``retired``.

The gate only sees metric dicts, so models that are not live heads (e.g. MitoEd's classifier, or a
detector retrained on GPU) plug in by implementing ``Trainable``: ``fit`` on the training set and
``predict`` on the locked set. ``HeadTrainable`` is the built-in one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


@dataclass
class GateConfig:
    min_new_labels: int = 20       # do not retrain on fewer new marks than this (since the champion)
    min_locked_per_class: int = 5  # locked set must have this many of each class to judge anything
    min_locked_ba: float = 0.7     # floor for the first champion
    tol_overall: float = 0.0
    tol_group: float = 0.05
    tol_ece: float = 0.02


class Trainable(Protocol):
    kind: str

    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> "Trainable": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def balanced_accuracy(p: np.ndarray, y: np.ndarray) -> float:
    pred = p >= 0.5
    parts = []
    if (y == 1).any():
        parts.append(pred[y == 1].mean())
    if (y == 0).any():
        parts.append((~pred[y == 0]).mean())
    return float(np.mean(parts)) if parts else float("nan")


def score(p: np.ndarray, y: np.ndarray, groups: Sequence[str]) -> dict:
    g = np.asarray(groups)
    per = {}
    for k in sorted(set(g)):
        m = g == k
        if (y[m] == 1).any() and (y[m] == 0).any():
            per[str(k)] = balanced_accuracy(p[m], y[m])
    return {"n": int(len(y)), "balanced_accuracy": balanced_accuracy(p, y),
            "ece": expected_calibration_error(p, y), "per_group": per}


def gate(champ: Optional[dict], chall: dict, cfg: GateConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if champ is None:
        ok = chall["balanced_accuracy"] >= cfg.min_locked_ba
        reasons.append(f"first champion: locked balanced accuracy {chall['balanced_accuracy']:.3f} "
                       f"{'>=' if ok else '<'} floor {cfg.min_locked_ba}")
        return ok, reasons
    ok = True
    if chall["balanced_accuracy"] < champ["balanced_accuracy"] - cfg.tol_overall:
        ok = False
        reasons.append(f"balanced accuracy {chall['balanced_accuracy']:.3f} < champion "
                       f"{champ['balanced_accuracy']:.3f} - {cfg.tol_overall}")
    for k, v in champ.get("per_group", {}).items():
        c = chall.get("per_group", {}).get(k)
        if c is not None and c < v - cfg.tol_group:
            ok = False
            reasons.append(f"{k}: {c:.3f} < champion {v:.3f} - {cfg.tol_group}")
    if chall["ece"] > champ["ece"] + cfg.tol_ece:
        ok = False
        reasons.append(f"ECE {chall['ece']:.3f} > champion {champ['ece']:.3f} + {cfg.tol_ece}")
    if ok:
        reasons.append("passed: no regression on the locked set")
    return ok, reasons


@dataclass
class RoundResult:
    promoted: bool
    reasons: List[str]
    champion: Optional[dict]
    challenger: Optional[dict]
    n_train: int
    n_locked: int
    new_labels: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_round(challenger: Trainable, champion: Optional[Trainable], train: Tuple[np.ndarray, np.ndarray, np.ndarray],
              locked: Tuple[np.ndarray, np.ndarray, List[str]], new_labels: int, cfg: GateConfig,
              force: bool = False) -> RoundResult:
    """``train`` = (X, y, w); ``locked`` = (X, y, groups). Fits ``challenger`` in place."""
    X, y, w = train
    Xl, yl, gl = locked
    n_l = int(len(yl))
    base = dict(champion=None, challenger=None, n_train=int(len(y)), n_locked=n_l, new_labels=int(new_labels))
    if new_labels < cfg.min_new_labels and not force:
        return RoundResult(False, [f"only {new_labels} new marks (< {cfg.min_new_labels}); not retrained"], **base)
    if min(int((yl == 1).sum()), int((yl == 0).sum())) < cfg.min_locked_per_class:
        return RoundResult(False, [f"locked set too small: need {cfg.min_locked_per_class} of each class, have "
                                   f"{int((yl == 1).sum())} positive / {int((yl == 0).sum())} negative"], **base)
    if len(np.unique(y)) < 2:
        return RoundResult(False, ["training marks have only one class"], **base)
    challenger.fit(X, y, w)
    s_chall = score(challenger.predict(Xl), yl, gl)
    s_champ = score(champion.predict(Xl), yl, gl) if champion is not None else None
    ok, reasons = gate(s_champ, s_chall, cfg)
    base.update(champion=s_champ, challenger=s_chall)
    return RoundResult(ok, reasons, **base)
