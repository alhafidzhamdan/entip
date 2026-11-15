"""Pick the trial slides from a triage manifest: the smallest slides of each class, one slide per patient.

    python pick_slides.py manifest.csv trial.csv --per-label glioma=5 normal_brain=5
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--per-label", nargs="+", default=["glioma=5", "normal_brain=5"], metavar="LABEL=N")
    a = ap.parse_args()
    want = {k: int(v) for k, v in (x.split("=") for x in a.per_label)}
    with open(a.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    rows.sort(key=lambda r: float(r.get("series_size_MB") or 1e9))
    got, patients, pick = Counter(), set(), []
    for r in rows:                                   # smallest first, one slide per patient
        lab = r["label"]
        if got[lab] < want.get(lab, 0) and r["PatientID"] not in patients:
            got[lab] += 1
            patients.add(r["PatientID"])
            pick.append(r)
    short = {k: n - got[k] for k, n in want.items() if got[k] < n}
    if short:
        raise SystemExit(f"manifest too small, missing slides: {short}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pick)
    mb = sum(float(r["series_size_MB"]) for r in pick)
    print(f"{len(pick)} slides, {mb:.0f} MB to download: " +
          ", ".join(f"{k} {v}" for k, v in sorted(got.items())))
    for r in pick:
        print(f"  {r['label']:<14} {r['collection_id']:<10} {float(r['series_size_MB']):>7.1f} MB  {r['SeriesInstanceUID']}")


if __name__ == "__main__":
    main()
