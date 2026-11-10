"""Methylation classes and the families they roll up to.

The default list is the diffuse glioma part of the class names used in the Capper et al. 2018 reference
set (GEO GSE90496), plus control tissue. Families are the step-5 contract's vocabulary (``dx/contract/README.md``); check the
family of each class with a neuropathologist (e.g. GBM MYCN is filed under paediatric-type HGG here). Class names change between classifier versions, so load your own
list with ``Taxonomy.from_csv`` (columns ``class,family``) to match the labels you train on.

When the model cannot choose between classes in one family (say GBM RTK I vs RTK II) but is sure of the
family, the agent reports the family and still orders the array.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DEFAULT_CLASSES: Dict[str, str] = {
    "GBM, RTK I": "glioblastoma_idh_wildtype",
    "GBM, RTK II": "glioblastoma_idh_wildtype",
    "GBM, MES": "glioblastoma_idh_wildtype",
    "GBM, MID": "glioblastoma_idh_wildtype",
    "GBM, MYCN": "pediatric_hgg_h3wt_idhwt",
    "GBM, G34": "diffuse_hemispheric_glioma_h3g34",
    "DMG, K27": "diffuse_midline_glioma_h3k27",
    "A IDH": "astrocytoma_idh_mutant",
    "A IDH, HG": "astrocytoma_idh_mutant",
    "O IDH": "oligodendroglioma_idh_mutant_codeleted",
    "CONTR, HEMI": "control_tissue",
    "CONTR, REACT": "control_tissue",
    "CONTR, WM": "control_tissue",
}


@dataclass
class Taxonomy:
    family_of: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASSES))

    @property
    def classes(self) -> List[str]:
        return list(self.family_of)

    @property
    def families(self) -> List[str]:
        return sorted(set(self.family_of.values()))

    def family(self, cls: str) -> str:
        return self.family_of.get(cls, "other")

    def restrict(self, present: List[str]) -> "Taxonomy":
        """Keep only classes that occur in the training labels (order kept)."""
        s = set(present)
        return Taxonomy({c: f for c, f in self.family_of.items() if c in s})

    @classmethod
    def from_csv(cls, path: str | Path) -> "Taxonomy":
        with open(path, newline="") as f:
            return cls({r["class"].strip(): r["family"].strip() for r in csv.DictReader(f)})
