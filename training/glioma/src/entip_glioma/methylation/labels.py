"""Where the training labels come from: methylation results for the same tumour as the slide.

Three sources, best first:

1. ``from_report_csv``: classes already called by a methylation classifier (e.g. your lab's reports), one
   row per case with ``case_id,class,score``. Calls below ``min_score`` are dropped, as Capper et al. 2018
   and Hetairos did with low classifier scores.
2. ``ReferenceClassifier``: our own classifier (random forest plus score calibration) trained on a public
   reference set of beta values with known classes (e.g. GEO GSE90496, the Capper 2018 reference set), then
   applied to TCGA 450K betas. This avoids
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


def usable_probes(cpgs: Sequence[str], exclude: Optional[Sequence[str]] = None) -> np.ndarray:
    """Mask of probes to use: CpG probes only (``cg``; SNP ``rs`` and non-CpG ``ch`` probes are dropped) and
    none of ``exclude``. Capper et al. 2018 also removed sex-chromosome and SNP-affected probes; the array
    manifest says which those are, so pass them in ``exclude`` (``label --exclude-probes``)."""
    ex = set(exclude or ())
    return np.array([c.startswith("cg") and c not in ex for c in cpgs], bool)


@dataclass
class ReferenceClassifier:
    """Random forest on the most variable CpGs with calibrated scores, the approach of Capper et al. 2018,
    reimplemented with scikit-learn (BSD-3-Clause). Not a copy of their classifier and not validated against it.

    Raw forest votes are spread over many classes and are rarely near 1, so a 0.9 cut-off on them would
    drop most samples. As in the paper, a multinomial logistic regression turns the vote vector into a
    calibrated score; it is fitted on out-of-bag votes, so no sample is scored by trees that saw it."""
    n_cpgs: int = 10000
    n_trees: int = 500
    seed: int = 0
    version: str = "refrf@2"
    exclude: Optional[Sequence[str]] = None
    cpgs: List[str] = field(default_factory=list)
    median: Optional[np.ndarray] = None
    model: object = None
    calibrator: object = None
    oob: dict = field(default_factory=dict)

    def fit(self, X: np.ndarray, cpgs: Sequence[str], y: Sequence[str]) -> "ReferenceClassifier":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        sd = np.nan_to_num(np.nanstd(X, axis=0))
        sd[~usable_probes(cpgs, self.exclude)] = -1
        keep = np.argsort(-sd)[: min(self.n_cpgs, int((sd >= 0).sum()))]
        self.cpgs = [cpgs[j] for j in keep]
        Xk = X[:, keep]
        self.median = np.nanmedian(Xk, axis=0)
        Xk = np.where(np.isnan(Xk), self.median, Xk)
        y = np.asarray(list(y))
        self.model = RandomForestClassifier(self.n_trees, random_state=self.seed, n_jobs=-1, bootstrap=True,
                                            class_weight="balanced_subsample", oob_score=True).fit(Xk, y)
        votes = self.model.oob_decision_function_
        ok = ~np.isnan(votes).any(axis=1)                 # a sample every tree saw has no out-of-bag votes
        self.calibrator = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced").fit(votes[ok], y[ok])
        P = self.calibrator.predict_proba(votes[ok])
        pred = self.calibrator.classes_[P.argmax(1)]
        self.oob = {"n": int(ok.sum()), "accuracy": float((pred == y[ok]).mean()),
                    "share_score_ge_0.9": float((P.max(1) >= 0.9).mean()),
                    "accuracy_score_ge_0.9": float((pred == y[ok])[P.max(1) >= 0.9].mean())
                    if (P.max(1) >= 0.9).any() else None}
        return self

    @property
    def classes(self) -> List[str]:
        return list(self.calibrator.classes_)

    def _align(self, X: np.ndarray, cpgs: Sequence[str]) -> Tuple[np.ndarray, float, np.ndarray]:
        """Target matrix in the classifier's CpG order; share of its CpGs present; per-sample share missing."""
        col = {c: j for j, c in enumerate(cpgs)}
        out = np.tile(self.median, (X.shape[0], 1)).astype(np.float32)
        miss = np.ones((X.shape[0], len(self.cpgs)), bool)
        found = 0
        for k, c in enumerate(self.cpgs):
            j = col.get(c)
            if j is not None:
                v = X[:, j]
                miss[:, k] = np.isnan(v)
                out[:, k] = np.where(miss[:, k], self.median[k], v)
                found += 1
        return out, found / max(1, len(self.cpgs)), miss.mean(axis=1)

    def predict_proba(self, X: np.ndarray, cpgs: Sequence[str]) -> Tuple[np.ndarray, float]:
        Xa, coverage, _ = self._align(X, cpgs)
        return self.calibrator.predict_proba(self.model.predict_proba(Xa)), coverage

    def label(self, ids: Sequence[str], X: np.ndarray, cpgs: Sequence[str], min_score: float = 0.9,
              min_coverage: float = 0.9, max_missing: float = 0.1) -> List[Label]:
        """Labels for samples scoring at least ``min_score``. Samples missing more than ``max_missing`` of
        the classifier's CpGs are skipped (their values would mostly be the reference median)."""
        Xa, coverage, missing = self._align(X, cpgs)
        if coverage < min_coverage:
            raise ValueError(f"only {coverage:.0%} of the classifier's CpGs are in the target data "
                             f"(< {min_coverage:.0%}); labels would not be trustworthy")
        P = self.calibrator.predict_proba(self.model.predict_proba(Xa))
        out = []
        for i, sid in enumerate(ids):
            k = int(P[i].argmax())
            if P[i, k] >= min_score and missing[i] <= max_missing:
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
