"""Panel labels per patient, from TCGA (training) or a local results table (Edinburgh, later).

Two readers, both giving ``{patient_id: {finding: call}}`` in the contract's call words:

* ``from_cbioportal_pan_glioma``: the sample clinical table of cBioPortal study ``lgggbm_tcga_pub``
  (TCGA pan-glioma, Ceccarelli et al. 2016, https://doi.org/10.1016/j.cell.2015.12.028). Column ids checked
  against https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-attributes on 2026-11-10. It gives
  IDH, 1p/19q (through IDH_CODEL_SUBTYPE), ATRX, TERT promoter, MGMT promoter and +7/-10.
* ``from_long_table``: any CSV with columns ``patient_id, finding, call``. Use it for the markers that come
  from mutation and copy-number calls (TP53, CDKN2A/B, EGFR, H3) and for local panel results.

Values that are not recognised are left unlabelled and counted, never guessed. The patient id of a TCGA
barcode is its first three fields (``TCGA-XX-XXXX``); the second field is the tissue source site, used to
keep each site's patients in one cross-validation fold (https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/).
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

from .markers import DEFAULT_MARKERS

Labels = Dict[str, Dict[str, str]]

# column -> (finding, {source value: call}). Values seen in the cBioPortal API on 2026-11-10:
# IDH_STATUS Mutant/WT; ATRX_STATUS, TERT_PROMOTER_STATUS Mutant/WT; MGMT_PROMOTER_STATUS
# Methylated/Unmethylated; CHR_7_GAIN_CHR_10_LOSS "Gain chr 7 & loss chr 10"/"No combined CNA";
# IDH_CODEL_SUBTYPE IDHmut-non-codel/IDHwt. "IDHmut-codel" is assumed by symmetry; check the download.
CBIOPORTAL_PAN_GLIOMA = {
    "IDH_STATUS": ("IDH", {"Mutant": "mutant", "WT": "wildtype"}),
    "ATRX_STATUS": ("ATRX", {"Mutant": "lost", "WT": "retained"}),
    "TERT_PROMOTER_STATUS": ("TERT", {"Mutant": "mutant", "WT": "wildtype"}),
    "MGMT_PROMOTER_STATUS": ("MGMT", {"Methylated": "methylated", "Unmethylated": "unmethylated"}),
    "CHR_7_GAIN_CHR_10_LOSS": ("chr7_chr10", {"Gain chr 7 & loss chr 10": "gain7_loss10",
                                              "No combined CNA": "absent"}),
    # IDH-wildtype tumours get no 1p/19q label: the marker is only modelled in IDH-mutant glioma.
    "IDH_CODEL_SUBTYPE": ("1p19q", {"IDHmut-codel": "codeleted", "IDHmut-non-codel": "not_codeleted"}),
}


def tcga_patient(barcode: str) -> str:
    """``TCGA-02-0001-01Z-00-DX1.xxxx.svs`` -> ``TCGA-02-0001``."""
    parts = Path(barcode).name.split("-")
    if len(parts) < 3 or parts[0] != "TCGA":
        raise ValueError(f"not a TCGA barcode: {barcode!r}")
    return "-".join(parts[:3])[:12]


def tcga_site(barcode: str) -> str:
    """Tissue source site code, the second barcode field."""
    return tcga_patient(barcode).split("-")[1]


def _read_table(path: str | Path):
    text = Path(path).read_text().splitlines()
    rows = [r for r in text if r and not r.startswith("#")]    # cBioPortal downloads start with # comment lines
    delim = "\t" if rows and "\t" in rows[0] else ","
    return list(csv.DictReader(rows, delimiter=delim))


def from_cbioportal_pan_glioma(path: str | Path, report: Optional[Counter] = None) -> Labels:
    """Read the cBioPortal sample table (TSV or CSV; needs ``PATIENT_ID`` or ``SAMPLE_ID``)."""
    report = report if report is not None else Counter()
    out: Labels = {}
    for row in _read_table(path):
        pid = row.get("PATIENT_ID") or row.get("patientId") or tcga_patient(row.get("SAMPLE_ID") or row["sampleId"])
        lab = out.setdefault(pid, {})
        for col, (finding, mapping) in CBIOPORTAL_PAN_GLIOMA.items():
            v = (row.get(col) or "").strip()
            if not v or v.upper() in ("NA", "[NOT AVAILABLE]", "NAN"):
                continue
            if v in mapping:
                prev = lab.get(finding)
                if prev and prev != mapping[v]:
                    report[f"{finding}:conflict"] += 1        # two samples of one patient disagree
                    lab[finding] = "__conflict__"
                elif prev != "__conflict__":
                    lab[finding] = mapping[v]
            elif not (finding == "1p19q" and v == "IDHwt"):
                report[f"{col}:unrecognised:{v}"] += 1
    return {p: {f: c for f, c in lab.items() if c != "__conflict__"} for p, lab in out.items()}


def from_long_table(path: str | Path, report: Optional[Counter] = None) -> Labels:
    """CSV/TSV with ``patient_id, finding, call``; calls must be the contract's words."""
    report = report if report is not None else Counter()
    out: Labels = {}
    for row in _read_table(path):
        f, c = row["finding"].strip(), row["call"].strip()
        m = DEFAULT_MARKERS.get(f)
        if m is None or c not in m.calls:
            report[f"unrecognised:{f}:{c}"] += 1
            continue
        out.setdefault(row["patient_id"].strip(), {})[f] = c
    return out


def merge(*tables: Labels) -> Labels:
    """Later tables win per (patient, finding)."""
    out: Labels = {}
    for t in tables:
        for p, lab in t.items():
            out.setdefault(p, {}).update(lab)
    return out


def apply_requirements(labels: Labels) -> Labels:
    """Drop labels for markers whose ``requires`` condition is not met by the patient's own labels
    (e.g. a 1p/19q label on an IDH-wildtype or IDH-unknown tumour)."""
    out: Labels = {}
    for p, lab in labels.items():
        keep = {}
        for f, c in lab.items():
            req = DEFAULT_MARKERS[f].requires if f in DEFAULT_MARKERS else None
            if req is None or lab.get(req[0]) == req[1]:
                keep[f] = c
        out[p] = keep
    return out


def counts(labels: Labels) -> Dict[str, Tuple[int, int]]:
    """{finding: (n reference call, n altered call)}."""
    out: Dict[str, Tuple[int, int]] = {}
    for f, m in DEFAULT_MARKERS.items():
        vals = [lab[f] for lab in labels.values() if f in lab]
        out[f] = (vals.count(m.calls[0]), vals.count(m.calls[1]))
    return out
