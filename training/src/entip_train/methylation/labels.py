"""Where the training labels come from: methylation results for the same tumour as the slide.

Three sources, best first:

1. ``from_report_csv``: classes already called by a methylation classifier (e.g. your lab's reports), one
   row per case with ``case_id,class,score``. Calls below ``min_score`` are dropped, as Capper et al. 2018
   and Hetairos did with low classifier scores.
2. ``ReferenceClassifier``: our own classifier trained on a public reference set of beta values with known
   classes (e.g. GEO GSE90496, the Capper 2018 reference set), then applied to TCGA 450K betas. This avoids
   depending on the Heidelberg classifier software, whose licence terms we have not checked.
3. ``cluster_labels``: unsupervised clusters of the TCGA betas. The clusters have no names until a
   neuropathologist maps them to classes; use only for exploration.

Betas are a samples x CpG matrix; the CpG ids must match between reference and target (450K and EPIC share
most 450K probes). Missing CpGs in the target are filled with the reference median and reported.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Label:
    case_id: str
    cls: str
    score: float
    source: str                    # "report", "reference-classifier@<version>", "cluster@k"

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "class": self.cls, "score": self.score, "source": self.source}


def from_report_csv(path: str | Path, min_score: float = 0.9, source: str = "report",
                    keep_unscored: bool = False) -> List[Label]:
    """Rows without a score are dropped unless ``keep_unscored`` (they then carry score NaN and weight 1)."""
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("score") or "").strip()
            s = float(raw) if raw else float("nan")
            if (np.isnan(s) and keep_unscored) or (not np.isnan(s) and s >= min_score):
                out.append(Label(r["case_id"].strip(), r["class"].strip(), s, source))
    return out


def probe_id(name: str) -> str:
    """450K / EPIC v1 ids are kept; EPIC v2 ids carry a suffix (``cg00000029_TC21``) that is dropped so all
    three arrays share one id space."""
    return name.split("_", 1)[0]


def read_beta_file(path: str | Path) -> Dict[str, float]:
    """One sample: two columns, CpG id and beta (tab-separated; a header row and 'NA' values are allowed).
    Replicate EPIC v2 probes for one CpG are averaged."""
    acc: Dict[str, List[float]] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[0].startswith(("cg", "ch", "rs")):
                continue
            try:
                v = float(parts[1])
            except ValueError:
                continue
            if not np.isnan(v):
                acc.setdefault(probe_id(parts[0]), []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def stack_betas(samples: Dict[str, Dict[str, float]], cpgs: Optional[Sequence[str]] = None,
                max_missing: float = 0.1) -> Tuple[List[str], List[str], np.ndarray]:
    """{sample_id: {cpg: beta}} -> (sample_ids, cpgs, matrix with NaN where missing).

    Without ``cpgs``, keeps every CpG measured in at least ``1 - max_missing`` of samples (a CpG that failed
    detection in a few samples is kept, not dropped for everyone)."""
    ids = list(samples)
    if cpgs is None:
        counts: Dict[str, int] = {}
        for v in samples.values():
            for c in v:
                counts[c] = counts.get(c, 0) + 1
        need = (1 - max_missing) * len(ids)
        cpgs = sorted(c for c, n in counts.items() if n >= need)
    cpgs = list(cpgs)
    X = np.full((len(ids), len(cpgs)), np.nan, np.float32)
    col = {c: j for j, c in enumerate(cpgs)}
    for i, s in enumerate(ids):
        for c, b in samples[s].items():
            j = col.get(c)
            if j is not None:
                X[i, j] = b
    return ids, cpgs, X


@dataclass
class ReferenceClassifier:
    """Random forest on the most variable CpGs, the approach of Capper et al. 2018, reimplemented with
    scikit-learn (BSD-3-Clause). Not a copy of their classifier and not validated against it."""
    n_cpgs: int = 10000
    n_trees: int = 500
    seed: int = 0
    version: str = "refrf@1"
    cpgs: List[str] = field(default_factory=list)
    median: Optional[np.ndarray] = None
    model: object = None

    def fit(self, X: np.ndarray, cpgs: Sequence[str], y: Sequence[str]) -> "ReferenceClassifier":
        from sklearn.ensemble import RandomForestClassifier
        sd = np.nanstd(X, axis=0)
        keep = np.argsort(-np.nan_to_num(sd))[: min(self.n_cpgs, X.shape[1])]
        self.cpgs = [cpgs[j] for j in keep]
        Xk = X[:, keep]
        self.median = np.nanmedian(Xk, axis=0)
        Xk = np.where(np.isnan(Xk), self.median, Xk)
        self.model = RandomForestClassifier(self.n_trees, random_state=self.seed, n_jobs=-1,
                                            class_weight="balanced_subsample", oob_score=True).fit(Xk, list(y))
        return self

    @property
    def classes(self) -> List[str]:
        return list(self.model.classes_)

    def _align(self, X: np.ndarray, cpgs: Sequence[str]) -> Tuple[np.ndarray, float]:
        col = {c: j for j, c in enumerate(cpgs)}
        out = np.tile(self.median, (X.shape[0], 1)).astype(np.float32)
        found = 0
        for k, c in enumerate(self.cpgs):
            j = col.get(c)
            if j is not None:
                v = X[:, j]
                out[:, k] = np.where(np.isnan(v), self.median[k], v)
                found += 1
        return out, found / max(1, len(self.cpgs))

    def predict_proba(self, X: np.ndarray, cpgs: Sequence[str]) -> Tuple[np.ndarray, float]:
        Xa, coverage = self._align(X, cpgs)
        return self.model.predict_proba(Xa), coverage

    def label(self, ids: Sequence[str], X: np.ndarray, cpgs: Sequence[str], min_score: float = 0.9,
              min_coverage: float = 0.9) -> List[Label]:
        P, coverage = self.predict_proba(X, cpgs)
        if coverage < min_coverage:
            raise ValueError(f"only {coverage:.0%} of the classifier's CpGs are in the target data "
                             f"(< {min_coverage:.0%}); labels would not be trustworthy")
        out = []
        for i, sid in enumerate(ids):
            k = int(P[i].argmax())
            if P[i, k] >= min_score:
                out.append(Label(sid, self.classes[k], float(P[i, k]), f"reference-classifier@{self.version}"))
        return out


def cluster_labels(ids: Sequence[str], X: np.ndarray, k: int = 6, n_cpgs: int = 5000, seed: int = 0) -> List[Label]:
    """K-means on the most variable CpGs. Classes are ``cluster_<n>``: unnamed until a pathologist maps them."""
    from sklearn.cluster import KMeans
    sd = np.nanstd(X, axis=0)
    keep = np.argsort(-np.nan_to_num(sd))[: min(n_cpgs, X.shape[1])]
    Xk = X[:, keep]
    Xk = np.where(np.isnan(Xk), np.nanmedian(Xk, axis=0), Xk)
    km = KMeans(k, random_state=seed, n_init=10).fit(Xk)
    d = km.transform(Xk)
    # a soft score: how much closer the sample is to its own centre than to the next one
    srt = np.sort(d, axis=1)
    score = 1 - srt[:, 0] / np.maximum(srt[:, 1], 1e-9)
    return [Label(s, f"cluster_{int(c)}", float(sc), f"cluster@{k}") for s, c, sc in zip(ids, km.labels_, score)]
