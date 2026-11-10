"""Region detectors: necrosis and microvascular proliferation.

Each is a small classifier on frozen tile embeddings (H-optimus-0 for real slides), then:

1. tile probabilities, calibrated with a sigmoid fit (Platt) on held-out folds when there are enough labels;
2. tiles with p >= ``tile_threshold`` are joined into 8-connected components (one per focus);
3. a component's score is the mean of its top ``top_k`` tile probabilities; components smaller than
   ``min_tiles`` are dropped;
4. ``p_present`` for the material is the best component score, or the highest tile p when no tile passes
   ``tile_threshold``;
5. ``contract.decide`` turns p_present into present / absent / indeterminate. Indeterminate, or any guard
   tripping, marks the record ``uncertain`` and orders a pathologist review with the top components as evidence.

Guards that force ``uncertain`` (never a silent absent):
* the bag's encoder version differs from the one the head was trained on (refused outright);
* too little tissue (``min_tissue_tiles``);
* too many tiles unlike the training tiles (``max_ood_fraction``; distance in standardised embedding space
  above the 99th centile seen in training);
* ``validated`` is False: the head and its threshold have not been checked on held-out pathologist-labelled
  slides. This is the default, so every call stays ``uncertain`` until someone validates the model.

Thresholds here are starting values for synthetic data; they must be set on a validation set with
pathologist labels before any result is used.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..ontology import point_in_polygon
from .bags import TileBag
from .contract import FeatureResult, decide

ENCODER_LICENCE = {"h-optimus-0": "Apache-2.0", "handcrafted": "MIT"}


@dataclass
class RegionConfig:
    feature: str
    tile_threshold: float = 0.5
    top_k: int = 3
    min_tiles: int = 1
    threshold: float = 0.8
    min_tissue_tiles: int = 20
    max_ood_fraction: float = 0.25
    C: float = 0.5
    max_evidence: int = 5
    validated: bool = False
    training_data: str = ""        # name and licence of the labels the head was trained on
    seed: int = 0


DEFAULTS = {
    "necrosis": RegionConfig("necrosis"),
    "microvascular_proliferation": RegionConfig("microvascular_proliferation"),
}


def tile_labels(bag: TileBag, positives: Sequence[Sequence], negatives: Sequence[Sequence] = ()) -> np.ndarray:
    """Labels from polygons (lists of [x, y] µm): 1 for a positive, 0 for a negative, -1 unlabelled.

    A tile takes a polygon's label when its centre is inside the polygon, or when the polygon's centroid is
    inside the tile (so a vessel tuft smaller than a tile still labels the tile it sits in). Positive wins."""
    y = np.full(len(bag), -1, int)
    c = bag.centres_um()
    x0, y0 = bag.coords_um[:, 0], bag.coords_um[:, 1]
    for polys, val in ((negatives, 0), (positives, 1)):
        for pts in polys:
            gx, gy = np.asarray(pts, float).mean(0)
            hit = (x0 <= gx) & (gx < x0 + bag.tile_um) & (y0 <= gy) & (gy < y0 + bag.tile_um)
            for i, (x, yy) in enumerate(c):
                if hit[i] or point_in_polygon(x, yy, pts):
                    y[i] = val
    return y


def _components(rc: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """8-connected components among tiles where ``mask`` is True; returns lists of tile indices."""
    where = {tuple(rc[i]): i for i in np.flatnonzero(mask)}
    seen, out = set(), []
    for start in where:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            comp.append(where[(r, c)])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in where and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        out.append(np.asarray(sorted(comp)))
    return out


class RegionDetector:
    """One feature, one head. ``fit`` on labelled tiles from one or more bags, ``assess`` on new bags."""

    kind = "highgrade:tile-logreg"

    def __init__(self, cfg: RegionConfig):
        self.cfg = cfg
        self.clf = None
        self.encoder_version: Optional[str] = None
        self.tile_um: Optional[float] = None
        self.mu = self.sd = None
        self.ood_threshold: Optional[float] = None
        self.metrics: Dict[str, float] = {}
        self.n_labels = 0

    # ------------------------------------------------------------------ training
    def fit(self, bags: Sequence[TileBag], labels: Sequence[np.ndarray]) -> "RegionDetector":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        vers = {b.encoder_version for b in bags}
        sizes = {b.tile_um for b in bags}
        if len(vers) != 1 or len(sizes) != 1:
            raise ValueError(f"bags mix encoders {vers} or tile sizes {sizes}")
        X = np.concatenate([b.features[np.asarray(y) >= 0] for b, y in zip(bags, labels)])
        y = np.concatenate([np.asarray(y)[np.asarray(y) >= 0] for y in labels])
        if len(np.unique(y)) < 2:
            raise ValueError("need positive and negative tiles")
        base = make_pipeline(StandardScaler(), LogisticRegression(C=self.cfg.C, max_iter=2000))
        n_min = int(min((y == 1).sum(), (y == 0).sum()))
        self.clf = (CalibratedClassifierCV(base, method="sigmoid", cv=3) if n_min >= 6 else base).fit(X, y)
        self.encoder_version, self.tile_um = vers.pop(), sizes.pop()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.ood_threshold = float(np.quantile(self._dist(X), 0.99))
        self.n_labels = int(len(y))
        self.metrics = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
                        "calibrated": bool(n_min >= 6)}
        return self

    def _dist(self, X: np.ndarray) -> np.ndarray:
        return np.sqrt((((X - self.mu) / self.sd) ** 2).mean(1))

    @property
    def model_id(self) -> str:
        h = hashlib.sha1(repr((asdict(self.cfg), self.encoder_version, self.n_labels,
                               None if self.mu is None else float(self.mu.sum()))).encode()).hexdigest()[:10]
        return f"hg-{self.cfg.feature}-{h}"

    def tile_p(self, bag: TileBag) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("detector is not trained")
        if bag.encoder_version != self.encoder_version or bag.tile_um != self.tile_um:
            raise ValueError(f"bag {bag.scan_id} uses {bag.encoder_version} at {bag.tile_um} µm; this head was "
                             f"trained on {self.encoder_version} at {self.tile_um} µm")
        return self.clf.predict_proba(bag.features)[:, 1] if len(bag) else np.zeros(0)

    # ------------------------------------------------------------------ inference
    def assess(self, bags: Sequence[TileBag]) -> tuple[FeatureResult, Dict[str, np.ndarray]]:
        """One result for all ``bags`` (the scans of one specimen). Also returns tile p per scan."""
        cfg = self.cfg
        per_scan, comps, reasons = {}, [], []
        n_tiles, n_ood = 0, 0
        for bag in bags:
            p = self.tile_p(bag)
            per_scan[bag.scan_id] = p
            n_tiles += len(bag)
            n_ood += int((self._dist(bag.features) > self.ood_threshold).sum()) if len(bag) else 0
            ids, cen, rc = bag.tile_ids(), bag.centres_um(), bag.grid()
            for k, idx in enumerate(_components(rc, p >= cfg.tile_threshold)):
                if len(idx) < cfg.min_tiles:
                    continue
                score = float(np.sort(p[idx])[::-1][:cfg.top_k].mean())
                x0, y0 = bag.coords_um[idx].min(0)
                x1, y1 = bag.coords_um[idx].max(0) + bag.tile_um
                comps.append({"kind": "region", "id": f"{bag.scan_id}:{cfg.feature}:{k:03d}",
                              "scan_id": bag.scan_id, "weight": score, "n_tiles": int(len(idx)),
                              "area_mm2": float(len(idx) * bag.tile_area_mm2),
                              "x_um": float(x0), "y_um": float(y0), "w_um": float(x1 - x0), "h_um": float(y1 - y0),
                              "centre_um": [float(cen[idx, 0].mean()), float(cen[idx, 1].mean())],
                              "tile_ids": [ids[i] for i in idx]})
        comps.sort(key=lambda d: -d["weight"])
        all_p = np.concatenate(list(per_scan.values())) if per_scan else np.zeros(0)
        p_present = comps[0]["weight"] if comps else float(all_p.max()) if len(all_p) else 0.0
        if n_tiles < cfg.min_tissue_tiles:
            reasons.append(f"only {n_tiles} tissue tiles (< {cfg.min_tissue_tiles})")
        ood = n_ood / n_tiles if n_tiles else 1.0
        if ood > cfg.max_ood_fraction:
            reasons.append(f"{ood:.0%} of tiles are unlike the training tiles (> {cfg.max_ood_fraction:.0%})")
        if not cfg.validated:
            reasons.append("model not yet validated on held-out pathologist-labelled slides")
        call, conf, status, order_test, reasons = decide(p_present, cfg.threshold, reasons)
        value = {"n_foci": len(comps), "area_mm2": float(sum(c["area_mm2"] for c in comps)),
                 "largest_focus_mm2": float(max((c["area_mm2"] for c in comps), default=0.0)),
                 "tissue_mm2": float(sum(b.tissue_area_mm2 for b in bags)), "n_tiles": n_tiles,
                 "ood_fraction": float(ood), "tile_threshold": cfg.tile_threshold}
        enc = (self.encoder_version or "").split("@")[0]
        res = FeatureResult(cfg.feature, [b.scan_id for b in bags], call, float(p_present), conf, cfg.threshold,
                            status, self.model_id, self.encoder_version or "", ENCODER_LICENCE.get(enc, "unknown"),
                            value, comps[:cfg.max_evidence], order_test, reasons)
        return res, per_scan

    # ------------------------------------------------------------------ persistence
    def model_record(self, status: str = "draft") -> dict:
        enc = (self.encoder_version or "").split("@")[0]
        return {"model_id": self.model_id, "feature": self.cfg.feature, "kind": self.kind, "encoder": enc,
                "status": status, "version": self.encoder_version, "n_labels": self.n_labels,
                "metrics": self.metrics, "licence": ENCODER_LICENCE.get(enc, "unknown"),
                "meta": {"config": asdict(self.cfg), "tile_um": self.tile_um}}

    def save(self, path: str | Path) -> Path:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({k: getattr(self, k) for k in ("cfg", "clf", "encoder_version", "tile_um", "mu", "sd",
                                                   "ood_threshold", "metrics", "n_labels")}, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RegionDetector":
        """Joblib runs code on load: only load heads this pipeline saved itself."""
        import joblib
        d = joblib.load(path)
        det = cls(d.pop("cfg"))
        for k, v in d.items():
            setattr(det, k, v)
        return det
