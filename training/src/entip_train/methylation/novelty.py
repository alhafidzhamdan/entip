"""A "does this slide look like anything we trained on?" check, behind the ``no_match`` call.

A softmax classifier always picks one of its classes, however strange the slide. Methylation classifiers
answer "no match" instead, and so should this step. Each case is summarised by the mean of its tile
embeddings; the novelty score is the mean distance to its ``k`` nearest training cases (features standardised
on the training cases). The cut-off is a high quantile of the scores of held-out-site cases in
cross-validation, so ordinary site-to-site variation is not flagged.

``wsi/novelty.py`` (step 1) is the class-centroid version of the same idea; this one uses nearest
training cases because methylation classes are many and some are small.

This is a coarse check. It catches slides far from the training data (another tissue, another stain, a
scanner artefact, a very different tumour); it will not catch a tumour type that merely resembles a
trained class. Those rely on the confidence threshold and, in the workflow, on triage (step 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def descriptor(feats: np.ndarray) -> np.ndarray:
    return np.asarray(feats, np.float64).mean(axis=0)


@dataclass
class Novelty:
    ref: np.ndarray                # (n, D) standardised training descriptors
    mu: np.ndarray
    sd: np.ndarray
    k: int = 5
    threshold: float = float("inf")
    quantile: float = 0.99

    @classmethod
    def fit(cls, descs: np.ndarray, k: int = 5) -> "Novelty":
        descs = np.asarray(descs, np.float64)
        mu, sd = descs.mean(0), descs.std(0) + 1e-6
        return cls((descs - mu) / sd, mu, sd, k=min(k, len(descs)))

    def score(self, descs: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(np.asarray(descs, np.float64)) - self.mu) / self.sd
        d2 = (x ** 2).sum(1)[:, None] - 2 * x @ self.ref.T + (self.ref ** 2).sum(1)[None, :]
        d = np.sqrt(np.maximum(d2, 0) / x.shape[1])
        return np.sort(d, axis=1)[:, : self.k].mean(axis=1)

    def is_novel(self, feats: np.ndarray) -> tuple[bool, float]:
        s = float(self.score(descriptor(feats))[0])
        return s > self.threshold, s

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, ref=self.ref.astype(np.float32), mu=self.mu, sd=self.sd,
                            cfg=np.array([self.k, self.threshold, self.quantile], np.float64))

    @classmethod
    def load(cls, path: str | Path) -> "Novelty":
        with np.load(path, allow_pickle=False) as z:
            k, thr, q = z["cfg"]
            return cls(z["ref"].astype(np.float64), z["mu"], z["sd"], int(k), float(thr), float(q))


def oof_threshold(descs: np.ndarray, fold: Sequence[int], k: int = 5, quantile: float = 0.99) -> tuple[float, np.ndarray]:
    """Score every case against the cases of the other folds; the cut-off is ``quantile`` of those scores."""
    descs, fold = np.asarray(descs, np.float64), np.asarray(fold)
    s = np.zeros(len(descs))
    for f in np.unique(fold):
        te = fold == f
        s[te] = Novelty.fit(descs[~te], k).score(descs[te])
    return float(np.quantile(s, quantile)), s
