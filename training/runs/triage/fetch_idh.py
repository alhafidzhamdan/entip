"""Download glioma IDH status from cBioPortal (needs internet) as a table pick_slides.py reads.

    python fetch_idh.py idh.tsv

Two sources, one row per sample: ``PATIENT_ID``, ``SAMPLE_ID``, ``IDH_STATUS`` (Mutant / WT / Uncertain), then
what a WHO CNS5 integrated diagnosis needs where the source has it - ``IDH_CODEL_SUBTYPE`` (IDHwt /
IDHmut-codel / IDHmut-non-codel), ``GRADE`` (G2-G4, histological), ``TERT_PROMOTER_STATUS``,
``CHR_7_GAIN_CHR_10_LOSS`` - and ``SOURCE``. ``who_cns5.diffuse_glioma`` turns these into WHO CNS5 terms.

* TCGA-GBM and TCGA-LGG: study ``lgggbm_tcga_pub``, TCGA pan-glioma (Ceccarelli et al. 2016, Cell,
  https://doi.org/10.1016/j.cell.2015.12.028), sample attributes as curated by the authors.
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
TCGA_ATTRS = ("IDH_STATUS", "IDH_CODEL_SUBTYPE", "GRADE", "TERT_PROMOTER_STATUS", "CHR_7_GAIN_CHR_10_LOSS")
EXTRA = TCGA_ATTRS[1:]
CANONICAL = {"IDH1": re.compile(r"^R132"), "IDH2": re.compile(r"^R172")}


def _get(url: str, data: dict | None = None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"Accept": "application/json", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def tcga_rows() -> list:
    vals, owner = {}, {}
    for attr in TCGA_ATTRS:
        for x in _get(f"{API}/studies/lgggbm_tcga_pub/clinical-data?clinicalDataType=SAMPLE&attributeId={attr}"
                      "&projection=SUMMARY&pageSize=100000"):
            vals.setdefault(x["sampleId"], {})[attr] = x["value"]
            owner[x["sampleId"]] = x["patientId"]
    rows = []
    for s, v in vals.items():
        idh = v.get("IDH_STATUS") or {"IDHwt": "WT", "IDHmut-codel": "Mutant",
                                      "IDHmut-non-codel": "Mutant"}.get(v.get("IDH_CODEL_SUBTYPE", ""))
        if idh:
            rows.append((owner[s], s, idh, *(v.get(k, "") for k in EXTRA), "cBioPortal lgggbm_tcga_pub"))
    return rows


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
        rows.append((pid, s, status.get(s, "WT"), "", "", "", "", src))
    return rows


def main(out: str) -> int:
    rows = tcga_rows() + cptac_gbm_rows()
    with open(out, "w") as f:
        f.write("# glioma IDH status from cBioPortal; see training/runs/triage/fetch_idh.py\n")
        f.write("\t".join(("PATIENT_ID", "SAMPLE_ID", "IDH_STATUS", *EXTRA, "SOURCE")) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    from collections import Counter
    print(f"{len(rows)} samples with IDH status -> {out}:",
          dict(Counter((r[-1].split()[1], r[2]) for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
