"""Step 3a: set aside fresh patients for the final slide-level MVP test, before the next retraining.

    python reserve_test.py --manifest trial.csv --bags bags/ --exclude review-key.csv --exclude new1-key.csv \
        --out mvp-final-test

Patients are drawn at random, one slide each, with no model involved, from those in no ``--exclude`` file. Writes
``patients.csv`` (``patient``, ``slide``, ``category``, ``diagnosis``): pass it to training as ``--holdout`` so these
patients are never trained on. It is never overwritten. A claim of under 5% error among confident calls needs at least
59 independent patients with no errors, so the script warns below that.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import pick_slides, quotas, read_manifest, read_patients  # noqa: E402

MIN_PATIENTS = 59


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--exclude", action="append", required=True, help="CSV of patients already used (repeat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", nargs="*", default=["gbm=25", "astro4=15", "oligo3=15", "astro3=15", "grade2=10"])
    ap.add_argument("--seed", type=int, default=20261122)
    a = ap.parse_args()
    path = Path(a.out) / "patients.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists: the reserved patients are never redrawn")
    used = read_patients(a.exclude)
    if not used:
        raise SystemExit("--exclude names no patients")
    want = quotas(a.per_category)
    picked = pick_slides(read_manifest(a.manifest), want, a.bags, a.seed, used)
    if not picked:
        raise SystemExit("no unused patients with bags to reserve")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "slide", "category", "diagnosis"])
        for r in sorted(picked, key=lambda r: r["PatientID"]):
            w.writerow([r["PatientID"], r["SeriesInstanceUID"], r["category"], r.get("diagnosis", "")])
    got = Counter(r["category"] for r in picked)
    info = {"reserved_patients": len(picked), "by_category": dict(got), "wanted": want,
            "enough_for_5_percent_claim": len(picked) >= MIN_PATIENTS, "written": str(path)}
    print(json.dumps(info, indent=2))
    if len(picked) < MIN_PATIENTS:
        print(f"only {len(picked)} patients reserved; {MIN_PATIENTS} are needed for the 5% claim", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
