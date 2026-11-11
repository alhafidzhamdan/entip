"""Gated attention-based multiple-instance learning (Ilse, Tomczak and Welling 2018,
https://arxiv.org/abs/1802.04712), written here in NumPy so the deployable path carries no CLAM / GPL code.

A slide is a bag of tile embeddings ``x`` (N, D). The model is::

    h   = relu(x Wp + bp)                       (N, H)   projection
    a   = (tanh(h Wv) * sigmoid(h Wu)) w        (N,)     gated attention score
    att = softmax(a)                                      one weight per tile, sums to 1
    z   = att @ h                               (H,)     slide embedding
    p   = softmax(z Wc + bc)                    (C,)     class probabilities

Trained by Adam on weighted cross-entropy, one bag per step, with random tile dropout (``max_tiles``).
Features are standardised with statistics from a sample of the training tiles. Bags are never copied or
concatenated, so they can be float16 arrays (or any sequence whose items convert with ``np.asarray``). CPU is enough once bags are cached:
the expensive part, the encoder, has already run.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def _softmax(v: np.ndarray, axis: int = -1) -> np.ndarray:
    v = v - v.max(axis=axis, keepdims=True)
    e = np.exp(v)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * v))


PARAMS = ("Wp", "bp", "Wv", "Wu", "w", "Wc", "bc")


class ABMIL:
    def __init__(self, n_classes: int, hidden: int = 128, attn: int = 64, lr: float = 1e-3, epochs: int = 30,
                 weight_decay: float = 1e-4, max_tiles: Optional[int] = 512, class_weight: str | None = "balanced",
                 patience: int = 8, seed: int = 0):
        self.n_classes, self.hidden, self.attn = int(n_classes), int(hidden), int(attn)
        self.lr, self.epochs, self.weight_decay = lr, epochs, weight_decay
        self.max_tiles, self.class_weight, self.patience, self.seed = max_tiles, class_weight, patience, seed
        self.p: dict = {}
        self.mu = self.sd = None
        self.history: List[dict] = []

    # ------------------------------------------------------------------ parameters
    def _init(self, d: int, rng: np.random.Generator) -> None:
        H, A, C = self.hidden, self.attn, self.n_classes
        g = lambda i, o: rng.normal(0, np.sqrt(2.0 / (i + o)), (i, o))
        self.p = {"Wp": g(d, H), "bp": np.zeros(H), "Wv": g(H, A), "Wu": g(H, A), "w": rng.normal(0, 0.1, A),
                  "Wc": g(H, C), "bc": np.zeros(C)}

    def _fit_scaler(self, bags, rng, per_bag: int = 256) -> None:
        """Feature mean and standard deviation from up to ``per_bag`` tiles of each bag (streamed)."""
        n, s1, s2 = 0, 0.0, 0.0
        for b in bags:
            x = np.asarray(b)
            if len(x) > per_bag:
                x = x[rng.choice(len(x), per_bag, replace=False)]
            x = x.astype(np.float64)
            n += len(x)
            s1 = s1 + x.sum(0)
            s2 = s2 + (x ** 2).sum(0)
        self.mu = s1 / n
        self.sd = np.sqrt(np.maximum(s2 / n - self.mu ** 2, 0)) + 1e-6

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, np.float64) - self.mu) / self.sd

    # ------------------------------------------------------------------ forward / backward
    def _forward(self, x: np.ndarray):
        p = self.p
        s0 = x @ p["Wp"] + p["bp"]
        h = np.maximum(s0, 0)
        v, u = np.tanh(h @ p["Wv"]), _sigmoid(h @ p["Wu"])
        a = (v * u) @ p["w"]
        att = _softmax(a)
        z = att @ h
        logits = z @ p["Wc"] + p["bc"]
        return logits, (x, s0, h, v, u, att, z)

    def _backward(self, dlogits: np.ndarray, cache) -> dict:
        p = self.p
        x, s0, h, v, u, att, z = cache
        g = {"Wc": np.outer(z, dlogits), "bc": dlogits}
        dz = p["Wc"] @ dlogits
        dh = np.outer(att, dz)
        datt = h @ dz
        da = att * (datt - (att * datt).sum())
        gate = v * u
        g["w"] = gate.T @ da
        dg = np.outer(da, p["w"])
        ds1 = dg * u * (1 - v ** 2)
        ds2 = dg * v * u * (1 - u)
        g["Wv"], g["Wu"] = h.T @ ds1, h.T @ ds2
        dh += ds1 @ p["Wv"].T + ds2 @ p["Wu"].T
        ds0 = dh * (s0 > 0)
        g["Wp"], g["bp"] = x.T @ ds0, ds0.sum(0)
        return g

    def _loss_grad(self, x: np.ndarray, y: int, wt: float):
        logits, cache = self._forward(x)
        pr = _softmax(logits)
        loss = -wt * np.log(max(pr[y], 1e-12))
        d = pr.copy()
        d[y] -= 1.0
        return loss, self._backward(wt * d, cache)

    # ------------------------------------------------------------------ public API
    def fit(self, bags: Sequence[np.ndarray], y: Sequence[int], sample_weight: Optional[Sequence[float]] = None,
            val: Optional[tuple] = None) -> "ABMIL":
        """bags: list of (N_i, D) arrays; y: int class per bag (0..C-1); sample_weight: per bag (0 drops a bag,
        so it doubles as a label mask); val: optional (bags, y) for early stopping on validation loss."""
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, int)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.class_weight == "balanced":
            counts = np.bincount(y[sw > 0], minlength=self.n_classes).astype(float)
            cw = np.where(counts > 0, counts[counts > 0].sum() / (np.count_nonzero(counts) * np.maximum(counts, 1)), 0)
            sw = sw * cw[y]
        self._fit_scaler(bags, rng)
        X = bags                       # normalised one bag at a time: thousands of slides must fit in memory
        self._init(np.asarray(X[0]).shape[1], rng)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        s = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, t = 0.9, 0.999, 0
        best, best_p, bad = np.inf, None, 0
        idx = np.flatnonzero(sw > 0)
        for ep in range(self.epochs):
            rng.shuffle(idx)
            tot = 0.0
            for i in idx:
                x = np.asarray(X[i])
                if self.max_tiles and len(x) > self.max_tiles:
                    x = x[rng.choice(len(x), self.max_tiles, replace=False)]
                loss, g = self._loss_grad(self._norm(x), y[i], sw[i])
                tot += loss
                t += 1
                for k in self.p:
                    gk = g[k] + (self.weight_decay * self.p[k] if k.startswith("W") else 0)
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    s[k] = b2 * s[k] + (1 - b2) * gk ** 2
                    self.p[k] -= self.lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(s[k] / (1 - b2 ** t)) + 1e-8)
            rec = {"epoch": ep, "train_loss": tot / max(1, sw[idx].sum())}
            if val is not None:
                vb, vy = val
                pv = self.predict_proba(vb)
                rec["val_loss"] = float(-np.mean(np.log(np.maximum(pv[np.arange(len(vy)), vy], 1e-12))))
                if rec["val_loss"] < best - 1e-4:
                    best, best_p, bad = rec["val_loss"], {k: v.copy() for k, v in self.p.items()}, 0
                else:
                    bad += 1
            self.history.append(rec)
            if val is not None and bad >= self.patience:
                break
        if best_p is not None:
            self.p = best_p
        return self

    def logits(self, bags: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self._forward(self._norm(b))[0] for b in bags])

    def predict_proba(self, bags: Sequence[np.ndarray], temperature: float = 1.0) -> np.ndarray:
        return _softmax(self.logits(bags) / temperature, axis=1)

    def attention(self, bag: np.ndarray) -> np.ndarray:
        """Per-tile attention weights for one slide (sum to 1): the heatmap shown in the GUI."""
        return self._forward(self._norm(bag))[1][5]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = np.array([self.n_classes, self.hidden, self.attn], np.int64)
        np.savez(path, cfg=cfg, mu=self.mu, sd=self.sd, **{k: self.p[k] for k in PARAMS})
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ABMIL":
        with np.load(path, allow_pickle=False) as z:
            c, h, a = (int(v) for v in z["cfg"])
            m = cls(c, h, a)
            m.mu, m.sd = z["mu"], z["sd"]
            m.p = {k: z[k] for k in PARAMS}
        return m
