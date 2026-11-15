"""Download TCGA glioma IDH status from cBioPortal (needs internet) as a table pick_slides.py reads.

    python fetch_idh.py idh.tsv

Source: cBioPortal study ``lgggbm_tcga_pub``, TCGA pan-glioma (Ceccarelli et al. 2016, Cell,
https://doi.org/10.1016/j.cell.2015.12.028), sample attribute ``IDH_STATUS`` (Mutant / WT), through the
public API. Writes one row per sample: ``PATIENT_ID``, ``SAMPLE_ID``, ``IDH_STATUS``.
"""
from __future__ import annotations

import json
import sys
import urllib.request

URL = ("https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-data"
       "?clinicalDataType=SAMPLE&attributeId=IDH_STATUS&projection=SUMMARY&pageSize=100000")


def main(out: str) -> int:
    req = urllib.request.Request(URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.load(r)
    with open(out, "w") as f:
        f.write(f"# {URL}\nPATIENT_ID\tSAMPLE_ID\tIDH_STATUS\n")
        for x in rows:
            f.write(f"{x['patientId']}\t{x['sampleId']}\t{x['value']}\n")
    print(f"{len(rows)} samples with IDH status -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
