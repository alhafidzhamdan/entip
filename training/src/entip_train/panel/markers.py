"""The glioma panel: which markers the model predicts, their calls, and the real test each falls back to.

Names, calls and test names follow the step 5 contract (``entip.glioma_step/1``, ``dx/contract/README.md``).
Which markers are switched on is configuration (``configs/glioma_panel.json``), because the Edinburgh panel's
exact gene list is not yet confirmed. A marker that is off is still reported, as ``not_assessed``, so the rules
engine never mistakes "not predicted" for "normal".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

STEP = "glioma_panel"
CONTRACT = "entip.glioma_step/1"


@dataclass(frozen=True)
class Marker:
    finding: str                  # contract name, e.g. "IDH"
    calls: Tuple[str, str]        # (negative / reference call, altered call); class index 0 and 1
    order_test: str               # contract test name used when the model is not confident
    description: str
    # The model is only asked about this marker when another marker has a given call, e.g. 1p/19q is only
    # meaningful in IDH-mutant glioma (WHO CNS5: oligodendroglioma is IDH-mutant and 1p/19q-codeleted;
    # Louis et al. 2021, https://doi.org/10.1093/neuonc/noab106). Applied to labels in training and to
    # confident upstream calls at prediction time.
    requires: Optional[Tuple[str, str]] = None
    # The label available for training is not the clinical test itself (e.g. TCGA ATRX mutation for ATRX IHC).
    label_proxy: str = ""
    max_error: float = 0.05       # wrong-call rate allowed among accepted calls of each class, upper 95% bound
    max_miss: float = 0.05        # share of patients of the other class accepted as this call, upper 95% bound
    min_per_class: int = 15       # fewer labelled patients than this in either class: don't train, order test

    @property
    def altered(self) -> str:
        return self.calls[1]


DEFAULT_MARKERS: Dict[str, Marker] = {m.finding: m for m in [
    Marker("IDH", ("wildtype", "mutant"), "IDH1_R132H_IHC", "IDH1 or IDH2 mutation"),
    Marker("ATRX", ("retained", "lost"), "ATRX_IHC", "ATRX nuclear expression",
           label_proxy="TCGA labels are ATRX mutation status, used as a proxy for loss of nuclear expression"),
    Marker("TP53", ("wildtype", "mutant"), "p53_IHC", "TP53 mutation (p53 IHC in the panel)",
           label_proxy="TCGA labels are TP53 mutation status, used as a proxy for an aberrant p53 IHC pattern"),
    Marker("1p19q", ("not_codeleted", "codeleted"), "1p19q_testing", "1p/19q whole-arm codeletion",
           requires=("IDH", "mutant")),
    Marker("CDKN2A", ("retained", "homozygous_deletion"), "CDKN2A_B_testing", "CDKN2A/B homozygous deletion"),
    Marker("EGFR", ("not_amplified", "amplified"), "EGFR_amplification_testing", "EGFR amplification"),
    Marker("chr7_chr10", ("absent", "gain7_loss10"), "chr7_chr10_copy_number",
           "combined whole chromosome 7 gain and chromosome 10 loss"),
    Marker("TERT", ("wildtype", "mutant"), "TERT_promoter_sequencing", "TERT promoter mutation"),
    Marker("H3K27", ("wildtype", "altered"), "H3K27M_IHC", "H3 K27 alteration (H3 K27M)"),
    Marker("H3G34", ("wildtype", "mutant"), "H3G34_testing", "H3 G34R/V mutation"),
    Marker("MGMT", ("unmethylated", "methylated"), "MGMT_promoter_methylation", "MGMT promoter methylation"),
]}


@dataclass
class PanelConfig:
    markers: List[Marker]
    disabled: List[str] = field(default_factory=list)      # reported as not_assessed
    min_tiles: int = 16             # a bag with fewer tissue tiles than this is not assessed confidently
    ood_quantile: float = 0.99      # bag looks unlike training (distance above this training quantile): uncertain
    top_evidence: int = 5           # highest-attention tiles reported as evidence
    cv_folds: int = 5
    abmil: dict = field(default_factory=lambda: {"hidden": 128, "attn": 64, "epochs": 30, "lr": 1e-3,
                                                 "max_tiles": 512, "seed": 0})

    def marker(self, finding: str) -> Marker:
        for m in self.markers:
            if m.finding == finding:
                return m
        raise KeyError(finding)

    @property
    def active(self) -> List[Marker]:
        return [m for m in self.markers if m.finding not in self.disabled]

    def to_dict(self) -> dict:
        return {"markers": [m.finding for m in self.markers], "disabled": list(self.disabled),
                "min_tiles": self.min_tiles, "ood_quantile": self.ood_quantile, "top_evidence": self.top_evidence,
                "cv_folds": self.cv_folds, "abmil": dict(self.abmil),
                "overrides": {m.finding: {"max_error": m.max_error, "max_miss": m.max_miss, "min_per_class": m.min_per_class}
                              for m in self.markers}}


def load_config(path: Optional[str | Path] = None, **kw) -> PanelConfig:
    """Read ``configs/glioma_panel.json``-style JSON. Keys: ``markers`` (list of contract names; default all),
    ``disabled``, ``overrides`` ({finding: {max_error, max_miss, min_per_class}}), ``min_tiles``, ``ood_quantile``,
    ``top_evidence``, ``cv_folds``, ``abmil``. Unknown marker names are refused rather than ignored."""
    raw: dict = json.loads(Path(path).read_text()) if path else {}
    raw.update(kw)
    known_keys = {"_comment", "markers", "disabled", "overrides", "min_tiles", "ood_quantile", "top_evidence",
                  "cv_folds", "abmil"}
    if set(raw) - known_keys:
        raise ValueError(f"unknown config keys {sorted(set(raw) - known_keys)}")
    bad = {k for ov in raw.get("overrides", {}).values() for k in ov} - {"max_error", "max_miss", "min_per_class"}
    if bad:
        raise ValueError(f"unknown override keys {sorted(bad)}")
    names = raw.get("markers") or list(DEFAULT_MARKERS)
    unknown = [n for n in names + list(raw.get("disabled", [])) + list(raw.get("overrides", {}))
               if n not in DEFAULT_MARKERS]
    if unknown:
        raise ValueError(f"unknown panel markers {unknown}; known: {list(DEFAULT_MARKERS)}")
    markers = []
    for n in names:
        m = DEFAULT_MARKERS[n]
        ov = raw.get("overrides", {}).get(n, {})
        markers.append(replace(m, **{k: v for k, v in ov.items() if k in ("max_error", "max_miss", "min_per_class")}))
    cfg = PanelConfig(markers=markers, disabled=list(raw.get("disabled", [])))
    for k in ("min_tiles", "ood_quantile", "top_evidence", "cv_folds"):
        if k in raw:
            setattr(cfg, k, type(getattr(cfg, k))(raw[k]))
    if "abmil" in raw:
        cfg.abmil.update(raw["abmil"])
    return cfg
