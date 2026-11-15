"""Is this slide like the slides the model was trained on?

A closed-set classifier must put every slide into one of its classes, so a tumour type it never saw (or a new
scanner, stain or fixative) can come out as a confident wrong call. This check is a safety net for that:
slides far from every training class are sent to a person whatever the classifier says.

Method: mean-pool the tile features of a slide, standardise, project on whitened principal components, and
take the distance to the nearest class centroid. The cut-off is a quantile of out-of-fold distances of the
training slides. It is a coarse, slide-level check: a small unusual focus inside ordinary tissue will not
move the mean, so it complements the confidence gate and does not replace a pathologist.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def pooled(bags: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(b, np.float32).mean(0) for b in bags]).astype(np.float64)


class Novelty:
    def __init__(self, n_components: int = 32, quantile: float = 0.99):
        self.n_components, self.quantile = n_components, quantile
        self.mu = self.sd = self.comp = self.scale = self.centroids = None
        self.threshold = np.inf

    def fit(self, Z: np.ndarray, y: Sequence[int]) -> "Novelty":
        """Z: pooled features (n, D); y: class per slide."""
        y = np.asarray(y)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        X = (Z - self.mu) / self.sd
        k = max(1, min(self.n_components, X.shape[1], len(X) - 1))
        _, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        self.comp = vt[:k]
        self.scale = np.maximum(s[:k] / np.sqrt(max(1, len(X) - 1)), 1e-6)
        T = self._project(Z)
        self.centroids = np.stack([T[y == c].mean(0) for c in np.unique(y)])
        return self

    def _project(self, Z: np.ndarray) -> np.ndarray:
        return (((Z - self.mu) / self.sd) @ self.comp.T) / self.scale

    def distance(self, Z: np.ndarray) -> np.ndarray:
        T = self._project(np.atleast_2d(Z))
        return np.sqrt(((T[:, None, :] - self.centroids[None]) ** 2).sum(-1)).min(1)

    def fit_threshold(self, oof_distances: np.ndarray) -> float:
        self.threshold = float(np.quantile(oof_distances, self.quantile))
        return self.threshold

    def is_novel(self, Z: np.ndarray) -> np.ndarray:
        return self.distance(Z) > self.threshold

    def state(self) -> dict:
        return {"nov_mu": self.mu, "nov_sd": self.sd, "nov_comp": self.comp, "nov_scale": self.scale,
                "nov_centroids": self.centroids, "nov_threshold": np.float64(self.threshold)}

    @classmethod
    def from_state(cls, z) -> Optional["Novelty"]:
        if "nov_mu" not in getattr(z, "files", z):
            return None
        n = cls()
        n.mu, n.sd, n.comp, n.scale = z["nov_mu"], z["nov_sd"], z["nov_comp"], z["nov_scale"]
        n.centroids, n.threshold = z["nov_centroids"], float(z["nov_threshold"])
        return n
