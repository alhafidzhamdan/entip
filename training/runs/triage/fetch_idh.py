"""Download glioma IDH status from cBioPortal (needs internet) as a table pick_slides.py reads.

    python fetch_idh.py idh.tsv

Two sources, one row per sample: ``PATIENT_ID``, ``SAMPLE_ID``, ``IDH_STATUS`` (Mutant / WT / Uncertain), ``SOURCE``.

* TCGA-GBM and TCGA-LGG: study ``lgggbm_tcga_pub``, TCGA pan-glioma (Ceccarelli et al. 2016, Cell,
  https://doi.org/10.1016/j.cell.2015.12.028), sample attribute ``IDH_STATUS`` as curated by the authors.
* CPTAC-GBM: study ``gbm_cptac_2021`` (Wang et al. 2021, Cancer Cell, https://doi.org/10.1016/j.ccell.2021.01.006),
  derived from its whole-exome mutation calls: ``Mutant`` with a canonical IDH1 R132 or IDH2 R172 mutation,
  ``WT`` when the sample was sequenced and has neither, ``Uncertain`` with any other IDH1/IDH2 variant
  (pick_slides.py leaves those out).

CCDI-MCI (paediatric) has no public IDH status here, so its gliomas cannot pass an IDH filter.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request

API = "https://www.cbioportal.org/api"
TCGA_URL = (f"{API}/studies/lgggbm_tcga_pub/clinical-data"
            "?clinicalDataType=SAMPLE&attributeId=IDH_STATUS&projection=SUMMARY&pageSize=100000")
CANONICAL = {"IDH1": re.compile(r"^R132"), "IDH2": re.compile(r"^R172")}


def _get(url: str, data: dict | None = None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"Accept": "application/json", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def tcga_rows() -> list:
    return [(x["patientId"], x["sampleId"], x["value"], "cBioPortal lgggbm_tcga_pub IDH_STATUS")
            for x in _get(TCGA_URL)]


def cptac_gbm_rows() -> list:
    study, seq = "gbm_cptac_2021", "gbm_cptac_2021_sequenced"
    samples = _get(f"{API}/sample-lists/{seq}/sample-ids")
    owner = {x["sampleId"]: x["patientId"] for x in _get(f"{API}/studies/{study}/samples?pageSize=100000")}
    muts = _get(f"{API}/molecular-profiles/{study}_mutations/mutations/fetch?projection=DETAILED",
                {"entrezGeneIds": [3417, 3418], "sampleListId": seq})       # IDH1, IDH2
    status, patient = {}, {}
    for m in muts:
        gene, change = m["gene"]["hugoGeneSymbol"], m.get("proteinChange") or ""
        canonical = bool(CANONICAL.get(gene, re.compile("$^")).match(change))
        prev = status.get(m["sampleId"])
        status[m["sampleId"]] = "Mutant" if canonical or prev == "Mutant" else "Uncertain"
        patient[m["sampleId"]] = m["patientId"]
    src = "cBioPortal gbm_cptac_2021 mutations (IDH1 R132 / IDH2 R172)"
    rows = []
    for s in samples:
        pid = owner.get(s) or patient.get(s)
        if not pid:
            continue                                   # no patient for this sample: leave it out, never guess
        rows.append((pid, s, status.get(s, "WT"), src))
    return rows


def main(out: str) -> int:
    rows = tcga_rows() + cptac_gbm_rows()
    with open(out, "w") as f:
        f.write("# glioma IDH status from cBioPortal; see training/runs/triage/fetch_idh.py\n")
        f.write("PATIENT_ID\tSAMPLE_ID\tIDH_STATUS\tSOURCE\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    from collections import Counter
    print(f"{len(rows)} samples with IDH status -> {out}:",
          dict(Counter((r[3].split()[1], r[2]) for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
