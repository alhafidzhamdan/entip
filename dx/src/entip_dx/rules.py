"""WHO CNS5 rules for adult-type and paediatric-type diffuse gliomas, with the sources each rule comes from.

The rules are written out by hand so every call can be traced to a criterion. Where a rule is a summary that
was not checked against the primary text, its citation says so. Research use only.
"""
from __future__ import annotations

RULES_VERSION = "who-cns5-2021/entip-0.1"

# Each citation is a short label plus a DOI link. "checked" means the rule was read in that source's full text.
CITATIONS = {
    "who5": ("Louis DN et al. The 2021 WHO Classification of Tumors of the CNS: a summary. Neuro-Oncology "
             "2021;23:1231-51 (tumour types, Table 2 markers, Table 3 grades, Table 4 layered report)",
             "https://doi.org/10.1093/neuonc/noab106"),
    "cimpact3": ("Brat DJ et al. cIMPACT-NOW update 3: diffuse astrocytic glioma, IDH-wildtype, with molecular "
                 "features of glioblastoma. Acta Neuropathol 2018;136:805-10",
                 "https://doi.org/10.1007/s00401-018-1913-0"),
    "cimpact5": ("Brat DJ et al. cIMPACT-NOW update 5: grading criteria and terminologies for IDH-mutant "
                 "astrocytomas. Acta Neuropathol 2020;139:603-8",
                 "https://doi.org/10.1007/s00401-020-02127-9"),
    "cimpact1": ("Louis DN et al. cIMPACT-NOW update 1: Not Otherwise Specified (NOS) and Not Elsewhere "
                 "Classified (NEC). Acta Neuropathol 2018;135:481-4 (definitions summarised, text not read)",
                 "https://doi.org/10.1007/s00401-018-1808-0"),
    "cimpact6": ("Louis DN et al. cIMPACT-NOW update 6: new entity and diagnostic principle recommendations. "
                 "Brain Pathol 2020;30:844-56", "https://doi.org/10.1111/bpa.12832"),
}


def cite(key: str) -> str:
    label, url = CITATIONS[key]
    return f"{label} {url}"


# WHO CNS5 type names (Louis 2021, Table 1) and their grades (Table 3).
TYPES = {
    "gbm": "Glioblastoma, IDH-wildtype",
    "astro": "Astrocytoma, IDH-mutant",
    "oligo": "Oligodendroglioma, IDH-mutant, and 1p/19q-codeleted",
    "dmg": "Diffuse midline glioma, H3 K27-altered",
    "dhg": "Diffuse hemispheric glioma, H3 G34-mutant",
    "phgg": "Diffuse paediatric-type high-grade glioma, H3-wildtype and IDH-wildtype",
}

# Methylation family (contract vocabulary) to the WHO type it supports.
FAMILY_TO_TYPE = {
    "glioblastoma_idh_wildtype": "gbm",
    "astrocytoma_idh_mutant": "astro",
    "oligodendroglioma_idh_mutant_codeleted": "oligo",
    "diffuse_midline_glioma_h3k27": "dmg",
    "diffuse_hemispheric_glioma_h3g34": "dhg",
    "pediatric_hgg_h3wt_idhwt": "phgg",
}

# The written criteria, used for the "based on" list. Each entry: (id, text, citation key).
CRITERIA = {
    "gbm_def": ("Diffuse astrocytic glioma, IDH-wildtype and H3-wildtype", "who5"),
    "gbm_histo": ("Microvascular proliferation or necrosis", "who5"),
    "gbm_mol": ("TERT promoter mutation, EGFR amplification or +7/-10 (any one) in an IDH-wildtype diffuse "
                "astrocytic glioma", "cimpact3"),
    "astro_def": ("IDH mutation without 1p/19q codeletion", "who5"),
    "astro_g4": ("Microvascular proliferation, necrosis or CDKN2A/B homozygous deletion (any one) gives "
                 "CNS WHO grade 4", "cimpact5"),
    "astro_g3": ("Significant mitotic activity with anaplasia, without grade 4 features, gives CNS WHO grade 3; "
                 "most neuropathologists use 2 or more mitoses in the whole specimen (1 in a very small biopsy)",
                 "cimpact5"),
    "oligo_def": ("IDH mutation with 1p/19q codeletion", "who5"),
    "oligo_grade": ("CNS WHO grade 2 or 3; grade is assigned by the pathologist from histology", "who5"),
    "dmg_def": ("Diffuse glioma with H3 K27 alteration (K27M or K27me3 loss); WHO requires a midline location, "
                "confirmed clinically", "who5"),
    "dhg_def": ("Diffuse glioma with H3 G34 mutation; WHO requires a cerebral hemispheric location, "
                "confirmed clinically", "who5"),
    "phgg_def": ("Diffuse high-grade glioma, H3-wildtype and IDH-wildtype, matching the paediatric-type "
                 "methylation class", "who5"),
    "nec": ("NEC: tests done but the result does not fit a WHO type; NOS: a needed test was not done or failed",
            "cimpact1"),
}

# Findings each type needs before it can be signed out (blocking), and the ones reported but not
# type-changing (non-blocking).
REPORT_ONLY = ("MGMT", "ATRX", "TP53")
