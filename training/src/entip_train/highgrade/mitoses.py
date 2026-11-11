"""Mitotic activity from MitoEd's output.

MitoEd (github.com/alhafidzhamdan/mitoed, MIT) finds and classifies mitotic-figure candidates and writes one row
per candidate (``mitoed predict``: ``x_um, y_um, p_mitosis, call, ...``). This detector reads that table, or
any list of ``{id?, x_um, y_um, p}``, and decides whether mitotic activity is elevated for a diffuse glioma.

Criterion. cIMPACT-NOW update 5 (Brat et al., Acta Neuropathol 2020; doi:10.1007/s00401-020-02127-9) keeps
"significant" mitotic activity as what separates IDH-mutant astrocytoma grade 3 from grade 2, says no validated
cut-off exists, and notes that most neuropathologists use >= 2 mitoses in the whole specimen, or 1 in a very
small biopsy, while large specimens may need more. So the count here is over the whole specimen (all its scans)
and the cut-off ``min_count`` is a setting (default 2); ``small_biopsy_mm2`` (default off) lowers it to 1 below
that tissue area. Neither number is validated; both are for Al to set.

Confidence. Candidates are treated as independent Bernoulli trials with their calibrated p; the number of true
mitoses then follows a Poisson-binomial distribution and ``p_present = P(N >= min_count)``. This uses every
borderline candidate rather than a hard 0.5 cut, so two candidates at p = 0.6 give an uncertain call, not a
confident one.

Which candidates count. MitoEd writes every candidate its first stage finds, most of them mimics with a small p.
Thousands of those would add up to a spurious count, so a candidate counts only if MitoEd's own call is
``mitosis`` or ``uncertain`` (its calibrated band), or, for tables without a call, if p >= ``min_p``.

Limits. Candidates are assumed independent and distinct: the same mitosis seen on two serial sections is
counted twice. Pool scans of different blocks, not levels of one block.

Guards (force ``uncertain``): the mitosis model is not validated on CNS tumours (the default: MitoEd is
trained on MIDOG++, whose released data has no CNS tumours; the MIDOG 2022 test set with astrocytoma and
meningioma was never released, see MitoEd's LICENCES.md), or the tissue area is unknown.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .contract import FeatureResult, decide


@dataclass
class MitosisConfig:
    min_count: int = 2
    min_p: float = 0.2             # floor for candidates that carry no MitoEd call
    small_biopsy_mm2: Optional[float] = None
    threshold: float = 0.8
    max_evidence: int = 10
    validated: bool = False


def p_at_least(p: Sequence[float], k: int) -> float:
    """P(sum of independent Bernoulli(p_i) >= k), exact dynamic programme over counts 0..k."""
    if k <= 0:
        return 1.0
    dist = np.zeros(k + 1)              # dist[j] = P(count == j) for j < k, dist[k] = P(count >= k)
    dist[0] = 1.0
    for q in np.clip(np.asarray(p, float), 0, 1):
        moved = dist[:-1] * q
        dist[:-1] *= 1 - q
        dist[1:] += moved
    return float(min(1.0, max(0.0, dist[k])))


def read_mitoed_cells(path: str | Path, scan_id: str) -> List[dict]:
    """Rows of a ``mitoed predict`` CSV as candidates ``{id, scan_id, x_um, y_um, p}``."""
    out = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            out.append({"id": r.get("cell_id") or f"{scan_id}:m{i:06d}", "scan_id": scan_id,
                        "x_um": float(r["x_um"]), "y_um": float(r["y_um"]), "p": float(r["p_mitosis"]),
                        "call": r.get("call") or None})
    return out


class MitosisCounter:
    kind = "highgrade:mitoed-count"

    def __init__(self, cfg: Optional[MitosisConfig] = None, model_id: str = "mitoed", version: str = "",
                 licence: str = "MIT (code); see MitoEd LICENCES.md for weights and data"):
        self.cfg = cfg or MitosisConfig()
        self.model_id, self.version, self.licence = model_id, version, licence

    def assess(self, candidates: Sequence[dict], scan_ids: Sequence[str], tissue_mm2: Optional[float]) -> FeatureResult:
        cfg = self.cfg
        n_all = len(candidates)
        candidates = self.counted(candidates)
        p = np.asarray([c["p"] for c in candidates], float)
        guards: List[str] = []
        if not cfg.validated:
            guards.append("mitosis model not yet validated on CNS tumours")
        k = cfg.min_count
        if tissue_mm2 is None or tissue_mm2 <= 0:
            guards.append("tissue area unknown: mitoses per mm2 not computed")
        elif cfg.small_biopsy_mm2 is not None and tissue_mm2 < cfg.small_biopsy_mm2:
            k = 1
        p_present = p_at_least(p, k)
        call, conf, status, order_test, reasons = decide(p_present, cfg.threshold, guards)
        expected = float(p.sum())
        order = np.argsort(-p)[:cfg.max_evidence]
        evidence = [{"kind": "cell", "id": candidates[i]["id"], "x_um": float(candidates[i]["x_um"]),
                     "y_um": float(candidates[i]["y_um"]), "weight": float(p[i]),
                     "scan_id": candidates[i].get("scan_id")} for i in order]
        value = {"count_expected": round(expected, 2), "count_p_ge_0_5": int((p >= 0.5).sum()),
                 "n_candidates": int(n_all), "n_counted": int(len(p)), "min_count": k, "tissue_mm2": tissue_mm2,
                 "mitoses_per_mm2": (round(expected / tissue_mm2, 3) if tissue_mm2 else None),
                 "criterion": "cIMPACT-NOW 5: >= 2 in whole specimen (1 in very small biopsy); not validated"}
        return FeatureResult("mitotic_activity", list(scan_ids), call, p_present, conf, cfg.threshold, status,
                             self.model_id, self.version, self.licence, value, evidence, order_test, reasons)

    def counted(self, candidates: Sequence[dict]) -> List[dict]:
        """Candidates that enter the count (and the database): not MitoEd mimics, not below the floor."""
        return [c for c in candidates
                if (c["call"] != "mimic" if c.get("call") else c["p"] >= self.cfg.min_p)]

    def model_record(self) -> dict:
        return {"model_id": self.model_id, "feature": "mitotic_figure", "kind": self.kind, "encoder": "mitoed",
                "status": "draft", "version": self.version, "licence": self.licence,
                "meta": {"config": asdict(self.cfg)}}
