"""Triage classes and the mapping from public slide metadata to them.

Classes follow the step 5 contract (entip.glioma_step/1, finding ``tissue_class``)::

    normal_brain       non-tumour, normal CNS parenchyma
    reactive           non-tumour, gliosis / inflammation / other non-neoplastic change
    glioma             glial and glioneuronal tumours (the input to steps 2-4)
    other_primary_cns  primary CNS tumours that are not glial (meningioma, embryonal, germ cell, ...)
    metastasis         carcinoma, melanoma etc. in the CNS
    lymphoma           lymphoma in the CNS

The keyword lists below are a first pass for IDC ``admittingDiagnosis_CodeMeaning`` strings. They are a
neuropathologist's call to finalise: edit ``GLIAL_TERMS`` / ``OTHER_PRIMARY_TERMS`` rather than the code.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

CLASSES = ("normal_brain", "reactive", "glioma", "other_primary_cns", "metastasis", "lymphoma")
TUMOUR = ("glioma", "other_primary_cns", "metastasis", "lymphoma")
GLIAL = ("glioma",)
IDX = {c: i for i, c in enumerate(CLASSES)}

CNS_SITES = ("brain", "brain stem", "cerebrum", "cerebellum", "cerebral cortex", "cerebral ventricle",
             "spinal cord", "central nervous system", "cerebral meninges", "spinal meninges", "meninges",
             "optic nerve")

# Glial and glioneuronal tumours (WHO CNS5 groups these together as gliomas, glioneuronal and neuronal
# tumours; neuronal-only ones such as central neurocytoma are left in other_primary_cns here).
GLIAL_TERMS = ("glioma", "glioblastoma", "astrocytoma", "oligodendroglioma", "oligoastrocytoma", "ependymoma",
               "subependymoma", "astroblastoma", "xanthoastrocytoma", "gliosarcoma", "ganglioglioma",
               "dysembryoplastic neuroepithelial", "glioneuronal")
OTHER_PRIMARY_TERMS = ("medulloblastoma", "atypical teratoid", "embryonal", "pineoblastoma", "pineocytoma",
                       "meningioma", "craniopharyngioma", "choroid plexus", "germinoma", "germ cell",
                       "teratoma", "schwannoma", "hemangioblastoma", "chordoma", "neurocytoma",
                       "solitary fibrous", "melanocytoma", "pituitary adenoma", "neuroblastoma")
METASTASIS_TERMS = ("metastatic", "carcinoma", "adenocarcinoma", "melanoma")
LYMPHOMA_TERMS = ("lymphoma",)
# Too vague to label: excluded rather than guessed.
VAGUE_TERMS = ("neoplasm,", "tumor cells", "uncertain whether", "malignant tumor,", "nos with borderline")

# Collections whose diagnosis column is empty in IDC but whose class is known from the collection itself.
COLLECTION_CLASS = {"tcga_gbm": "glioma", "tcga_lgg": "glioma", "cptac_gbm": "glioma"}
# Never use: canine (ICDC-Glioma is dogs).
EXCLUDE_COLLECTIONS = ("icdc_glioma",)


def _text(v) -> str:
    """Lower-case text; None, NaN and other non-strings (empty cells in pandas output) become ''."""
    return v.strip().lower() if isinstance(v, str) else ""


def map_idc(collection_id: str, site: Optional[str], dx: Optional[str]) -> Optional[str]:
    """IDC collection + primaryAnatomicStructure + admittingDiagnosis -> triage class, or None to exclude."""
    col, s, d = (_text(v) for v in (collection_id, site, dx))
    if col in EXCLUDE_COLLECTIONS:
        return None
    if col == "gtex":
        return "normal_brain" if s == "cerebral cortex" else None
    if s and s not in CNS_SITES:
        return None
    if col in COLLECTION_CLASS and not d:
        return COLLECTION_CLASS[col]
    if any(t in d for t in LYMPHOMA_TERMS):
        return "lymphoma"
    if "metastatic" in d and "uncertain whether primary" not in d:
        return "metastasis"
    if not d or any(t in d for t in VAGUE_TERMS):
        return None
    if any(t in d for t in GLIAL_TERMS):
        return "glioma"
    if any(t in d for t in OTHER_PRIMARY_TERMS):
        return "other_primary_cns"
    if any(t in d for t in METASTASIS_TERMS):
        return "metastasis"     # carcinoma / melanoma at a CNS site
    return None


def derived(probs: np.ndarray) -> dict:
    """Tumour and glial probabilities from the six class probabilities (one slide)."""
    p = np.asarray(probs, float)
    p_t = float(sum(p[IDX[c]] for c in TUMOUR))
    p_g = float(sum(p[IDX[c]] for c in GLIAL))
    return {"p_tumour": p_t, "p_glial": p_g, "p_glial_given_tumour": p_g / p_t if p_t > 0 else 0.0}
