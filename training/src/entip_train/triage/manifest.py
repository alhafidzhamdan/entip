"""Training manifest for the triage model from the Imaging Data Commons (IDC) slide index.

Run on a machine with internet access (Al's Mac or Eddie), not in the cloud sessions:

    pip install idc-index           # MIT; pulls idc-index-data (MIT)
    entip-train triage manifest --out triage_manifest.csv

The query joins ``index`` (collection, patient, licence, scanner) with ``sm_index`` (stain, fixative,
site, admitting diagnosis); column names are from idc-index-data 25.0.5's schema files. Only H&E slides with
an open licence (CC BY 4.0 / CC BY 3.0) are kept. Each row gets a triage class from ``labels.map_idc``; rows
that map to None are dropped and counted.

Counts on IDC v25 (idc-index-data 25.0.5, run 2026-11-10; slides / patients):

=================  ===============  ================================================================
class              all preparations  FFPE only
=================  ===============  ================================================================
glioma             5,433 / 2,562    2,499 / 1,646 (TCGA-GBM 860, TCGA-LGG 844, CCDI-MCI 786, HCMI 9)
other_primary_cns  856 / 829        550 / 534 (CCDI-MCI)
normal_brain       423 / 422        0 (GTEx cerebral cortex is PAXgene-fixed, paraffin-embedded)
metastasis         18 / 12          3 / 3
lymphoma           2 / 1            0
reactive           0                0
=================  ===============  ================================================================

So public data can train glioma vs other primary CNS vs normal brain only, and normal brain comes from one
source with a different fixative. Suggested first run: ``--prep ffpe paxgene_paraffin``. Metastasis,
lymphoma and reactive need Edinburgh slides.

``group`` is the source used for grouped splits: collection plus scanner, because whole collections are
single-class (all GTEx cortex is normal, all TCGA-GBM is glioma) and a model can learn the collection
instead of the tissue. See ``model.source_check``.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

from .labels import map_idc

OPEN_LICENCES = ("CC BY 4.0", "CC BY 3.0")

IDC_SQL = """
SELECT i.collection_id, i.PatientID, i.SeriesInstanceUID, i.license_short_name, i.Manufacturer,
       i.series_size_MB, s.primaryAnatomicStructure_CodeMeaning AS site,
       s.admittingDiagnosis_CodeMeaning AS dx, s.min_PixelSpacing_2sf AS pixel_spacing_mm,
       s.ContainerIdentifier AS container_id,
       array_to_string(s.tissueFixative_CodeMeaning, '; ') AS fixative,
       array_to_string(s.embeddingMedium_CodeMeaning, '; ') AS embedding
FROM index i JOIN sm_index s ON i.SeriesInstanceUID = s.SeriesInstanceUID
WHERE list_contains(s.staining_usingSubstance_CodeMeaning, 'hematoxylin stain')
  AND list_contains(s.staining_usingSubstance_CodeMeaning, 'water soluble eosin stain')
"""

FIELDS = ("collection_id", "PatientID", "SeriesInstanceUID", "license_short_name", "Manufacturer",
          "series_size_MB", "site", "dx", "pixel_spacing_mm", "container_id", "fixative", "embedding", "prep",
          "label", "group")


def prep_of(fixative, embedding) -> str:
    """ffpe | paxgene_paraffin | frozen | unknown, from the DICOM specimen preparation codes."""
    f = fixative if isinstance(fixative, str) else ""
    e = embedding if isinstance(embedding, str) else ""
    if "freezing" in e.lower() or "freezing" in f.lower():
        return "frozen"
    if "paraffin" in e.lower() and "formalin" in f.lower():
        return "ffpe"
    if "paraffin" in e.lower() and "paxgene" in f.lower():
        return "paxgene_paraffin"
    return "unknown"


def label_rows(rows: Iterable[dict], preps: Optional[tuple] = None) -> tuple[List[dict], Counter]:
    """Keep open-licence rows that map to a triage class; ``preps`` limits preparation (e.g. ("ffpe",))."""
    kept, dropped = [], Counter()
    for r in rows:
        if r.get("license_short_name") not in OPEN_LICENCES:
            dropped["licence"] += 1
            continue
        prep = prep_of(r.get("fixative"), r.get("embedding"))
        if preps and prep not in preps:
            dropped["prep"] += 1
            continue
        lab = map_idc(r.get("collection_id"), r.get("site"), r.get("dx"))
        if lab is None:
            dropped["unmapped"] += 1
            continue
        out = {k: r.get(k) for k in FIELDS}
        out["label"], out["prep"] = lab, prep
        out["group"] = f"{r.get('collection_id')}|{r.get('Manufacturer') or 'unknown'}|{prep}"
        kept.append(out)
    return kept, dropped


def from_idc() -> List[dict]:   # pragma: no cover - needs network and idc-index
    from idc_index import IDCClient
    c = IDCClient()
    c.fetch_index("sm_index")
    return c.sql_query(IDC_SQL).to_dict("records")


def write_manifest(rows: List[dict], out: str | Path) -> Path:
    out = Path(out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


def read_manifest(path: str | Path) -> List[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
