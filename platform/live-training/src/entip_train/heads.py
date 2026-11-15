"""Tier 1: live heads on frozen embeddings.

A head is a small classifier (scikit-learn, BSD-3-Clause; XGBoost, Apache-2.0, optional) trained on the
cached embeddings of the objects a person has marked. It retrains in milliseconds after every click,
warm-starting from the previous version, and ranks the unlabelled objects for review.

Marks become labels like this:
* a point marks the nearest cell within ``snap_um`` (cell level) or the tile that contains it;
* a polygon marks every cell inside it, or every tile whose centre is inside it;
* if one object is marked twice, the latest mark wins.

For ``positives_only`` features, or while a head has no negative marks yet, a random sample of unmarked
objects is added as down-weighted background negatives, away from the positives.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ontology import POSITIVE, Annotation, Cell, point_in_polygon
from .slides import centres


@dataclass
class HeadConfig:
    kind: str = "logreg"            # logreg | hgb | xgboost
    snap_um: float = 8.0            # a click snaps to a cell centre within this distance
    bg_ratio: float = 2.0           # background negatives per positive when negatives are missing
    bg_min: int = 20
    bg_weight: float = 0.3          # sample weight of a background negative (a marked object has 1)
    bg_exclusion_um: float = 20.0   # no background negative this close to a positive
    random_share: float = 0.1       # share of the review queue drawn at random from confident objects
    C: float = 1.0
    seed: int = 0


# ------------------------------------------------------------------ marks -> labels

def labels_from_annotations(items: Sequence, annotations: Sequence[Annotation], level: str,
                            snap_um: float = 8.0) -> Dict[int, Tuple[int, str]]:
    """{item index: (label 1/0, annotation_id)}; the latest annotation wins."""
    if not items:
        return {}
    xy = np.asarray(centres(items), float)
    out: Dict[int, Tuple[int, str]] = {}
    is_cell = level == "cell"
    for a in annotations:
        y = 1 if a.label == POSITIVE else 0
        g = a.geometry
        if g.get("type") == "object":                     # explicit object id (cell_id / region_id)
            idx = [i for i, it in enumerate(items) if _oid(it) == g["object_id"]]
        elif g.get("type") == "point":
            px, py = float(g["x_um"]), float(g["y_um"])
            if is_cell:
                d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
                j = int(d.argmin())
                idx = [j] if d[j] <= snap_um else []
            else:
                idx = [i for i, it in enumerate(items)
                       if it.x_um <= px < it.x_um + it.w_um and it.y_um <= py < it.y_um + it.h_um]
        elif g.get("type") == "polygon":
            pts = g["points_um"]
            idx = [i for i, (cx, cy) in enumerate(xy) if point_in_polygon(cx, cy, pts)]
        else:
            raise ValueError(f"unknown geometry {g.get('type')}")
        for i in idx:
            out[i] = (y, a.annotation_id)
    return out


def _oid(it) -> str:
    return it.cell_id if isinstance(it, Cell) else it.region_id


def background_negatives(items: Sequence, labelled: Dict[int, Tuple[int, str]], cfg: HeadConfig,
                         rng: np.random.Generator) -> List[int]:
    pos = [i for i, (y, _) in labelled.items() if y == 1]
    n_neg = sum(1 for y, _ in labelled.values() if y == 0)
    if not pos:
        return []
    want = max(cfg.bg_min, int(cfg.bg_ratio * len(pos))) - n_neg
    xy = np.asarray(centres(items), float)
    pxy = xy[pos]
    d = np.sqrt(((xy[:, None, :] - pxy[None, :, :]) ** 2).sum(-1)).min(1)
    pool = [i for i in range(len(items)) if i not in labelled and d[i] > cfg.bg_exclusion_um]
    if not pool or want <= 0:
        return []
    return sorted(rng.choice(pool, size=min(want, len(pool)), replace=False).tolist())


# ------------------------------------------------------------------ the head

class LiveHead:
    """Standardise -> classifier. ``fit`` warm-starts from ``parent`` when the kind allows it."""

    def __init__(self, cfg: HeadConfig, parent: Optional["LiveHead"] = None):
        self.cfg = cfg
        self.dim: Optional[int] = None
        if parent is not None and parent.cfg.kind == cfg.kind:
            self.scaler, self.clf = copy.deepcopy(parent.scaler), copy.deepcopy(parent.clf)
            self.dim, self.warm = parent.dim, True
        else:
            self.scaler, self.clf, self.warm = None, None, False

    @property
    def kind(self) -> str:
        return self.cfg.kind

    def _new(self):
        from sklearn.preprocessing import StandardScaler
        k = self.cfg.kind
        if k == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=self.cfg.C, class_weight="balanced", max_iter=500, warm_start=True)
        elif k == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, class_weight="balanced",
                                                 random_state=self.cfg.seed)
        elif k == "xgboost":
            from xgboost import XGBClassifier  # optional, Apache-2.0
            clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=self.cfg.seed)
        else:
            raise ValueError(f"unknown head kind {k}")
        return StandardScaler(), clf

    def fit(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> "LiveHead":
        if len(np.unique(y)) < 2:
            raise ValueError("need at least one positive and one negative (or background) example")
        if self.clf is None or self.dim not in (None, X.shape[1]):
            self.scaler, self.clf = self._new()
            self.warm = False
        self.dim = X.shape[1]
        self.scaler.fit(X)
        Z = self.scaler.transform(X)
        if self.cfg.kind == "xgboost" and self.warm:
            self.clf.fit(Z, y, sample_weight=w, xgb_model=self.clf.get_booster())
        else:
            self.clf.fit(Z, y, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def cv_balanced_accuracy(X: np.ndarray, y: np.ndarray, cfg: HeadConfig, w: Optional[np.ndarray] = None,
                         marked: Optional[np.ndarray] = None) -> Optional[float]:
    """Stratified k-fold balanced accuracy over the *marked* examples (background negatives train but are
    not scored). None if there are fewer than 3 marks per class."""
    from sklearn.model_selection import StratifiedKFold
    marked = np.ones(len(y), bool) if marked is None else marked
    ym = y[marked]
    k = min(5, int((ym == 1).sum()), int((ym == 0).sum()))
    if k < 3:
        return None
    mi = np.flatnonzero(marked)
    bg = np.flatnonzero(~marked)
    accs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=cfg.seed).split(mi, ym):
        tri = np.concatenate([mi[tr], bg])
        h = LiveHead(cfg).fit(X[tri], y[tri], None if w is None else w[tri])
        pred = h.predict(X[mi[te]]) >= 0.5
        yt = y[mi[te]]
        accs.append(np.mean([pred[yt == 1].mean(), (~pred[yt == 0]).mean()]))
    return float(np.mean(accs))


# ------------------------------------------------------------------ review queue

def review_queue(ids: Sequence[str], p: np.ndarray, exclude: set, n: int = 20, random_share: float = 0.1,
                 seed: int = 0) -> List[dict]:
    """Most uncertain first (smallest |p - 0.5|), plus a random share of the rest to catch confident errors.
    Marked objects are left out. Each row: {object_id, p, uncertainty, reason}."""
    rng = np.random.default_rng(seed)
    cand = [i for i, k in enumerate(ids) if k not in exclude]
    if not cand:
        return []
    cand = np.asarray(cand)
    u = np.abs(p[cand] - 0.5)
    order = cand[np.argsort(u, kind="stable")]
    n = min(n, len(order))
    n_rand = int(round(n * random_share)) if n >= 5 else 0
    top = list(order[:n - n_rand])
    rest = order[n - n_rand:]
    rand = list(rng.choice(rest, size=min(n_rand, len(rest)), replace=False)) if n_rand and len(rest) else []
    rows = [{"object_id": ids[i], "p": float(p[i]), "uncertainty": float(1 - 2 * abs(p[i] - 0.5)),
             "reason": "uncertain"} for i in top]
    rows += [{"object_id": ids[i], "p": float(p[i]), "uncertainty": float(1 - 2 * abs(p[i] - 0.5)),
              "reason": "random"} for i in rand]
    return rows
