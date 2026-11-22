"""Step 3a: size a glioma-only download of unused patients for validating a frozen MVP head.

    python validation_size.py --manifest manifest.csv --idh idh.tsv --bags bags/ --used review-key.csv --used pilot.csv

Read-only. It downloads nothing, selects no patient, loads no model and writes nothing, and prints counts per tumour
category only. It covers gliomas with a confirmed IDH status from TCGA and CPTAC in the given IDC manifest, at least
``--min-mb`` per slide with no upper limit, because the largest resections may be the likeliest to show MVP. The size
assumes one slide per patient, using each patient's median slide size.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import CATEGORIES, category, read_patients  # noqa: E402

COLLECTIONS = ("tcga_gbm", "tcga_lgg", "cptac_gbm")
CALIBRATION, TEST = 40, 59


def candidates(rows: list, molecular: dict, corrections: dict, min_mb: float) -> tuple[dict, Counter]:
    """patient -> their usable slides with a step 3 category, and why other glioma slides were left out."""
    from entip_glioma.triage import who_cns5
    out, left = defaultdict(list), Counter()
    for r in rows:
        if r.get("label") != "glioma" or r.get("collection_id") not in COLLECTIONS:
            continue
        if float(r.get("series_size_MB") or 0) < min_mb:
            left["slide_too_small"] += 1
        elif (molecular.get(r["PatientID"]) or {}).get("IDH_STATUS") not in ("Mutant", "WT"):
            left["idh_not_confirmed"] += 1
        else:
            out[r["PatientID"]].append(who_cns5.annotate([dict(r)], molecular, corrections)[0])
    return out, left


def summarise(people: dict, used: set, bags: Path) -> dict:
    rows = defaultdict(Counter)
    mb = Counter()
    for pid, slides in people.items():
        cats = {category(s) for s in slides}
        cat = cats.pop() if len(cats) == 1 else "mixed"
        c = rows[cat]
        c["patients"] += 1
        if pid in used:
            c["already_used"] += 1
        elif any((bags / f"{s['SeriesInstanceUID']}.npz").exists() for s in slides):
            c["eligible_with_bag"] += 1
        else:
            c["eligible_to_download"] += 1
            mb[cat] += statistics.median(float(s["series_size_MB"]) for s in slides)
    keys = ("patients", "already_used", "eligible_with_bag", "eligible_to_download")
    order = [c for c in (*CATEGORIES, "mixed") if c in rows]
    table = {c: {**{k: rows[c][k] for k in keys}, "download_gb": round(mb[c] / 1000, 1)} for c in order}
    table["total"] = {**{k: sum(rows[c][k] for c in order) for k in keys}, "download_gb": round(sum(mb.values()) / 1000, 1)}
    return table


def main() -> int:
    from entip_glioma.triage import who_cns5
    from entip_glioma.triage.manifest import read_corrections
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="IDC manifest.csv from a triage run")
    ap.add_argument("--idh", required=True, help="idh.tsv from fetch_idh.py")
    ap.add_argument("--bags", required=True)
    ap.add_argument("--used", action="append", required=True, help="CSV with a patient or PatientID column (repeat)")
    ap.add_argument("--min-mb", type=float, default=100)
    a = ap.parse_args()
    gone = [p for p in [a.manifest, a.idh, *a.used] if not Path(p).exists()]
    if gone:
        raise SystemExit(f"missing files, so the count would be wrong: {gone}")
    rows = list(csv.DictReader(open(a.manifest, newline="")))
    people, left = candidates(rows, who_cns5.read_molecular(a.idh), read_corrections(), a.min_mb)
    used = read_patients(a.used)
    table = summarise(people, used, Path(a.bags))
    total = table["total"]["eligible_with_bag"] + table["total"]["eligible_to_download"]
    print(json.dumps({"read_only": True, "used_patients_listed": len(used),
                      "collections_in_manifest": dict(Counter(r.get("collection_id") for r in rows
                                                              if r.get("collection_id") in COLLECTIONS)),
                      "glioma_slides_left_out": dict(left), "by_category": table,
                      "targets": {"calibration": CALIBRATION, "test": TEST, "eligible": total,
                                  "enough_patients": total >= CALIBRATION + TEST}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
