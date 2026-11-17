"""WHO CNS5 terms for ENTIP's classes and diagnoses (WHO Classification of Tumours of the Central Nervous
System, 5th edition, 2021).

Conventions, as in WHO CNS5:

* grades in Arabic numerals as "CNS WHO grade N", given within a tumour type;
* **NOS** (not otherwise specified) when the molecular test a type needs was not done or is not available to us
  (e.g. "Oligodendroglioma, NOS" when 1p/19q is unknown); **NEC** (not elsewhere classified) only when testing
  was done and the result fits no type - we rarely have that, so we rarely use it;
* a source term that WHO CNS5 dropped or renamed is translated, and the translation is explained in ``basis``.

Every diagnosis ENTIP records has two parts: ``who_cns5`` (the term) and ``who_cns5_basis`` (what it rests on,
e.g. "IDH and 1p/19q from TCGA pan-glioma; grade from histology; CDKN2A/B not assessed"), so a reader can see how
far each diagnosis is integrated. This is a judgement table: a neuropathologist should review it, and edit the
tables below rather than the code.
"""
from __future__ import annotations

import re
from typing import Optional

# ------------------------------------------------------------------ triage classes in WHO CNS5 terms
CLASS_CNS5 = {
    "normal_brain": ("Non-neoplastic CNS tissue (normal)",
                     "not a WHO CNS5 tumour category; normal brain parenchyma"),
    "reactive": ("Non-neoplastic CNS tissue (reactive)",
                 "not a WHO CNS5 tumour category; gliosis, inflammation, infarct and other non-neoplastic change"),
    "glioma": ("Gliomas and glioneuronal tumours",
               "WHO CNS5 chapter 1 (gliomas, glioneuronal tumours and neuronal tumours) without the neuronal "
               "tumours: adult-type diffuse gliomas, paediatric-type diffuse low- and high-grade gliomas, "
               "circumscribed astrocytic gliomas, glioneuronal tumours and ependymal tumours"),
    "other_primary_cns": ("Other primary CNS tumours",
                          "the remaining WHO CNS5 chapters: neuronal tumours, choroid plexus, embryonal, pineal, "
                          "cranial and paraspinal nerve, meningiomas, mesenchymal non-meningothelial, melanocytic, "
                          "haematolymphoid (primary CNS lymphomas), germ cell and sellar region tumours"),
    "metastasis": ("Metastases to the CNS",
                   "WHO CNS5 chapter 'Metastases to the CNS'; in public training data stood in for by primary "
                   "tumours of the commonest origins (lung, breast, melanoma, kidney, colorectum)"),
    "lymphoma": ("Secondary (systemic) lymphoma involving the CNS",
                 "systemic lymphoma types (e.g. follicular lymphoma); primary CNS lymphomas belong to the WHO CNS5 "
                 "haematolymphoid chapter and are counted as other primary CNS"),
}


def class_name(c: str) -> str:
    return CLASS_CNS5.get(c, (c, ""))[0]


def grade(n) -> str:
    return f"CNS WHO grade {n}"


# ------------------------------------------------------------------ adult-type diffuse glioma with molecular data
TCGA_GRADE = {"G2": 2, "G3": 3, "G4": 4}


def diffuse_glioma(idh: Optional[str], codel: Optional[str] = None, grade_src: Optional[str] = None,
                   tert: Optional[str] = None, chr7_10: Optional[str] = None, gbm_study: bool = False,
                   source: str = "") -> tuple[str, str]:
    """Adult-type diffuse glioma from IDH, 1p/19q, histological grade and the molecular glioblastoma features.

    ``idh``: Mutant / WT; ``codel``: TCGA IDH_CODEL_SUBTYPE (IDHwt / IDHmut-codel / IDHmut-non-codel);
    ``grade_src``: G2 / G3 / G4; ``tert``: TERT promoter Mutant / WT; ``chr7_10``: TCGA's 'Gain chr 7 & loss chr
    10' or not; ``gbm_study``: the tumour was collected as glioblastoma (TCGA-GBM, CPTAC-GBM)."""
    g = TCGA_GRADE.get((grade_src or "").strip().upper()) or (4 if gbm_study else None)
    src = f"{source}; " if source else ""
    idh = (idh or "").strip()
    codel = (codel or "").strip()
    if codel == "IDHmut-codel":
        if g == 4:      # an IDH-mutant, codeleted tumour graded 4 histologically: CNS5 grades oligodendroglioma 2-3
            return ("Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, " + grade(3),
                    f"{src}IDH and 1p/19q; source grade 4, capped at 3 (WHO CNS5 grades oligodendroglioma 2-3)")
        return ("Oligodendroglioma, IDH-mutant and 1p/19q-codeleted" + (f", {grade(g)}" if g else ""),
                f"{src}IDH and 1p/19q; grade from histology")
    if codel == "IDHmut-non-codel" or (idh == "Mutant" and not codel):
        basis = f"{src}IDH" + (" and 1p/19q (not codeleted)" if codel else ", 1p/19q not assessed") + \
                "; grade from histology; CDKN2A/B not assessed (a homozygous deletion would make it grade 4)"
        if not codel and g in (2, 3):
            return f"Diffuse glioma, IDH-mutant, NOS (histological grade {g})", basis
        return "Astrocytoma, IDH-mutant" + (f", {grade(g)}" if g else ""), basis
    if codel == "IDHwt" or idh == "WT":
        if gbm_study or g == 4:
            return "Glioblastoma, IDH-wildtype, " + grade(4), f"{src}IDH-wildtype; glioblastoma histology"
        molecular = [w for w, ok in (("TERT promoter mutation", (tert or "").strip() == "Mutant"),
                                     ("+7/-10", "gain chr 7" in (chr7_10 or "").lower())) if ok]
        if molecular:
            return ("Glioblastoma, IDH-wildtype, " + grade(4),
                    f"{src}IDH-wildtype, lower-grade histology, molecular glioblastoma features: "
                    + ", ".join(molecular))
        return ("Diffuse glioma, IDH-wildtype, NOS" + (f" (histological grade {g})" if g else ""),
                f"{src}IDH-wildtype without TERT promoter mutation or +7/-10 in the data; EGFR, H3 and "
                "paediatric-type alterations not assessed")
    return "Diffuse glioma, NOS", f"{src}IDH status unknown"


# ------------------------------------------------------------------ source terms (ICD-O, as in IDC) -> CNS5
# First match wins: put specific phrases before general ones. (pattern, WHO CNS5 term, note)
ICDO_RULES = [
    (r"pilomyxoid", "Pilocytic astrocytoma (pilomyxoid subtype)", "pilomyxoid is a subtype in WHO CNS5"),
    (r"pilocytic", "Pilocytic astrocytoma, " + grade(1), ""),
    (r"pleomorphic xanthoastrocytoma, anaplastic|anaplastic pleomorphic xanthoastrocytoma",
     "Pleomorphic xanthoastrocytoma, " + grade(3), ""),
    (r"pleomorphic xanthoastrocytoma", "Pleomorphic xanthoastrocytoma", "grade 2 or 3 not stated"),
    (r"subependymal giant cell", "Subependymal giant cell astrocytoma, " + grade(1), ""),
    (r"subependymoma", "Subependymoma, " + grade(1), ""),
    (r"myxopapillary", "Myxopapillary ependymoma, " + grade(2), ""),
    (r"ependymoma, anaplastic|anaplastic ependymoma", "Ependymoma, NOS, " + grade(3),
     "site and molecular group not available"),
    (r"ependymoma", "Ependymoma, NOS", "site, molecular group and grade not available"),
    (r"diffuse midline glioma", "Diffuse midline glioma, H3 K27-altered, " + grade(4), ""),
    (r"desmoplastic infantile", "Desmoplastic infantile ganglioglioma / desmoplastic infantile astrocytoma, "
     + grade(1), ""),
    (r"ganglioglioma, anaplastic|anaplastic ganglioglioma", "Glioneuronal tumour, NOS",
     "anaplastic ganglioglioma is not a WHO CNS5 type"),
    (r"ganglioglioma", "Ganglioglioma, " + grade(1), ""),
    (r"dysembryoplastic", "Dysembryoplastic neuroepithelial tumour, " + grade(1), ""),
    (r"glioneuronal", "Glioneuronal tumour, NOS", ""),
    (r"astroblastoma", "Astroblastoma, NOS", "MN1 alteration not confirmed"),
    (r"gliosarcoma", "Glioblastoma, IDH-wildtype, " + grade(4) + " (gliosarcoma pattern)",
     "IDH status assumed wildtype for gliosarcoma; not tested here"),
    (r"glioblastoma", "High-grade glioma, NOS",
     "WHO CNS5 needs IDH (and in children H3 and other alterations) to name the type"),
    (r"oligodendroglioma", "Oligodendroglioma, NOS", "IDH and 1p/19q not available"),
    (r"oligoastrocytoma|mixed glioma", "Diffuse glioma, NOS",
     "oligoastrocytoma is not a WHO CNS5 type; IDH and 1p/19q decide"),
    (r"astrocytoma, anaplastic|anaplastic astrocytoma", "Diffuse astrocytic glioma, NOS (histological grade 3)",
     "IDH not available"),
    (r"astrocytoma", "Astrocytoma, NOS", "IDH not available"),
    (r"chordoid glioma", "Chordoid glioma, " + grade(2), ""),
    (r"angiocentric glioma", "Angiocentric glioma, " + grade(1), ""),
    (r"glioma", "Glioma, NOS", ""),
    (r"medulloblastoma", "Medulloblastoma, NOS, " + grade(4), "molecular group not available"),
    (r"atypical teratoid|rhabdoid tumou?r","Atypical teratoid/rhabdoid tumour, " + grade(4), ""),
    (r"multilayered rosettes", "Embryonal tumour with multilayered rosettes, " + grade(4), ""),
    (r"pineoblastoma", "Pineoblastoma, " + grade(4), ""),
    (r"pineocytoma", "Pineocytoma, " + grade(1), ""),
    (r"pineal parenchymal", "Pineal parenchymal tumour of intermediate differentiation", "grade 2 or 3 not stated"),
    (r"embryonal tumou?r|primitive neuroectodermal|pnet", "CNS embryonal tumour, NOS",
     "PNET is not a WHO CNS5 type; molecular group not available"),
    (r"neuroblastoma", "Neuroblastoma, NOS", "CNS neuroblastoma, FOXR2-activated needs molecular testing"),
    (r"choroid plexus carcinoma", "Choroid plexus carcinoma, " + grade(3), ""),
    (r"atypical choroid plexus papilloma|choroid plexus papilloma, atypical",
     "Atypical choroid plexus papilloma, " + grade(2), ""),
    (r"choroid plexus papilloma", "Choroid plexus papilloma, " + grade(1), ""),
    (r"central neurocytoma", "Central neurocytoma, " + grade(2), ""),
    (r"neurocytoma", "Extraventricular neurocytoma, " + grade(2), ""),
    (r"gangliocytoma", "Gangliocytoma, " + grade(1), ""),
    (r"meningioma, anaplastic|anaplastic meningioma|malignant meningioma", "Meningioma, " + grade(3), ""),
    (r"atypical meningioma|meningioma, atypical", "Meningioma, " + grade(2), ""),
    (r"meningioma", "Meningioma, NOS", "grade not stated"),
    (r"melanotic schwannoma", "Malignant melanotic nerve sheath tumour", "renamed in WHO CNS5"),
    (r"schwannoma", "Schwannoma, " + grade(1), ""),
    (r"malignant peripheral nerve sheath", "Malignant peripheral nerve sheath tumour", ""),
    (r"neurofibroma", "Neurofibroma, " + grade(1), ""),
    (r"perineurioma", "Perineurioma, " + grade(1), ""),
    (r"paraganglioma", "Paraganglioma", ""),
    (r"melanocytoma", "Meningeal melanocytoma", ""),
    (r"craniopharyngioma, adamantinomatous|adamantinomatous", "Adamantinomatous craniopharyngioma, " + grade(1), ""),
    (r"craniopharyngioma, papillary|papillary craniopharyngioma", "Papillary craniopharyngioma, " + grade(1), ""),
    (r"craniopharyngioma", "Craniopharyngioma, NOS", "adamantinomatous or papillary not stated"),
    (r"pituitary adenoma|pituitary neuroendocrine", "Pituitary adenoma / PitNET", ""),
    (r"pituicytoma", "Pituicytoma", ""),
    (r"germinoma|dysgerminoma|seminoma", "Germinoma", ""),
    (r"yolk sac", "Yolk sac tumour", ""),
    (r"embryonal carcinoma", "Embryonal carcinoma", ""),
    (r"choriocarcinoma", "Choriocarcinoma", ""),
    (r"teratoma, immature|immature teratoma", "Immature teratoma", ""),
    (r"teratoma.*malignant|malignant.*teratoma", "Teratoma with somatic-type malignancy", "renamed in WHO CNS5"),
    (r"teratoma", "Teratoma, NOS", ""),
    (r"germ cell", "Germ cell tumour, NOS", ""),
    (r"h(a)?emangioblastoma", "Haemangioblastoma, " + grade(1), ""),
    (r"h(a)?emangiopericytoma|solitary fibrous", "Solitary fibrous tumour", "haemangiopericytoma renamed in WHO CNS5"),
    (r"chordoma", "Chordoma", ""),
    (r"lymphoma", "Lymphoma, NOS", "primary CNS or systemic not stated"),
    (r"melanoma", "Melanoma, NOS", "primary meningeal or metastatic not stated"),
    (r"metastatic|carcinoma|adenocarcinoma", "Metastatic tumour, NOS", ""),
]
_RULES = [(re.compile(p, re.I), t, n) for p, t, n in ICDO_RULES]

# Collections whose tissue, not their diagnosis column, gives the term.
COLLECTION_TERM = {
    "gtex": ("Non-neoplastic cerebral cortex", "GTEx post-mortem donor tissue"),
    "tcga_luad": ("Primary lung adenocarcinoma", "TCGA-LUAD; stands in for metastasis to the CNS"),
    "tcga_lusc": ("Primary lung squamous cell carcinoma", "TCGA-LUSC; stands in for metastasis to the CNS"),
    "tcga_brca": ("Primary breast carcinoma", "TCGA-BRCA; stands in for metastasis to the CNS"),
    "tcga_skcm": ("Cutaneous melanoma", "TCGA-SKCM (primary or metastatic, mostly not CNS); stands in for "
                                        "metastasis to the CNS"),
    "tcga_kirc": ("Primary clear cell renal cell carcinoma", "TCGA-KIRC; stands in for metastasis to the CNS"),
    "tcga_coad": ("Primary colon adenocarcinoma", "TCGA-COAD; stands in for metastasis to the CNS"),
    "tcga_read": ("Primary rectal adenocarcinoma", "TCGA-READ; stands in for metastasis to the CNS"),
    "cptac_luad": ("Primary lung adenocarcinoma", "CPTAC-LUAD; stands in for metastasis to the CNS"),
    "cptac_lscc": ("Primary lung squamous cell carcinoma", "CPTAC-LSCC; stands in for metastasis to the CNS"),
    "cptac_brca": ("Primary breast carcinoma", "CPTAC-BRCA; stands in for metastasis to the CNS"),
    "cptac_cm": ("Cutaneous melanoma", "CPTAC-CM; stands in for metastasis to the CNS"),
    "cptac_ccrcc": ("Primary clear cell renal cell carcinoma", "CPTAC-CCRCC; stands in for metastasis to the CNS"),
    "cptac_coad": ("Primary colon adenocarcinoma", "CPTAC-COAD; stands in for metastasis to the CNS"),
}
GBM_COLLECTIONS = ("tcga_gbm", "cptac_gbm")
DIFFUSE_GLIOMA_COLLECTIONS = ("tcga_gbm", "tcga_lgg", "cptac_gbm")


def from_term(dx: Optional[str]) -> Optional[tuple[str, str]]:
    """A source diagnosis (ICD-O text as in IDC) -> (WHO CNS5 term, basis), or None when no rule matches."""
    d = (dx or "").strip()
    if not d:
        return None
    for rx, term, note in _RULES:
        if rx.search(d):
            return term, f"from the source diagnosis '{d}'" + (f"; {note}" if note else "")
    return None


def for_row(row: dict, molecular: Optional[dict] = None, corrections: Optional[dict] = None) -> tuple[str, str]:
    """WHO CNS5 term and basis for one manifest row (IDC training data).

    Order: a confirmed correction; the collection's tissue (GTEx, the non-CNS primaries); IDH/1p19q data for
    TCGA and CPTAC gliomas (``molecular``: {patient: {IDH_STATUS, IDH_CODEL_SUBTYPE, GRADE, ...}}); the source
    diagnosis; else the triage class name, marked for review."""
    pid, col = row.get("PatientID", ""), (row.get("collection_id") or "").strip().lower()
    c = (corrections or {}).get(pid)
    if c and (c.get("diagnosis") or "").strip():
        return c["diagnosis"].strip(), f"label correction: {c.get('source', '').strip()}"
    if col in COLLECTION_TERM:
        return COLLECTION_TERM[col]
    if col in DIFFUSE_GLIOMA_COLLECTIONS:
        m = (molecular or {}).get(pid)
        if m:
            return diffuse_glioma(m.get("IDH_STATUS"), m.get("IDH_CODEL_SUBTYPE"), m.get("GRADE"),
                                  m.get("TERT_PROMOTER_STATUS"), m.get("CHR_7_GAIN_CHR_10_LOSS"),
                                  gbm_study=col in GBM_COLLECTIONS, source=m.get("SOURCE", ""))
        if col in GBM_COLLECTIONS:
            return "High-grade glioma, NOS", f"{col}: collected as glioblastoma; IDH status not available"
        return "Diffuse glioma, NOS", f"{col}: IDH status not available"
    t = from_term(row.get("dx"))
    if t:
        return t
    label = row.get("label", "")
    return f"{class_name(label)}, NOS", "no WHO CNS5 rule for the source diagnosis " \
                                        f"'{(row.get('dx') or '').strip() or 'none'}': review"


# ------------------------------------------------------------------ manifests
def read_molecular(path) -> dict:
    """``fetch_idh.py``'s table -> {patient: {IDH_STATUS, IDH_CODEL_SUBTYPE, GRADE, ..., SOURCE}}. A patient
    with several samples keeps the primary tumour (-01) when there is one, else the first sample."""
    import csv
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"):
            pid = (r.get("PATIENT_ID") or "").strip()
            if pid and (pid not in out or (r.get("SAMPLE_ID") or "").endswith("-01")):
                out[pid] = r
    return out


def annotate(rows: list, molecular: Optional[dict] = None, corrections: Optional[dict] = None) -> list:
    """Add ``diagnosis`` (the WHO CNS5 term) and ``who_cns5_basis`` to manifest rows; the source diagnosis stays
    in ``dx``."""
    out = []
    for r in rows:
        term, basis = for_row(r, molecular, corrections)
        out.append({**r, "diagnosis": term, "who_cns5_basis": basis})
    return out
