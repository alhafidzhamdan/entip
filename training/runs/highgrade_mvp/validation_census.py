"""Step 3a: count the patients still available to calibrate and test a frozen MVP head.

    python validation_census.py --internal diverse trial.csv bags/ --used review-key.csv --used pilot.csv \
        --external ipd manifest.csv bags/

Read-only. It selects no patient, loads no model and writes nothing, and prints counts per tumour category only. A patient
is eligible when they appear in no ``--used`` file and have a bag. ``--held-out`` patients are counted apart, because
they were set aside for another purpose. A missing file is reported, and a missing ``--used`` file stops the census,
because it would make used patients look fresh.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import CATEGORIES, read_manifest, read_patients  # noqa: E402

CALIBRATION, TEST = 40, 59


def patients(manifests: list) -> dict:
    """patient -> {"categories": set, "bag": whether any of their slides has one}."""
    out = defaultdict(lambda: {"categories": set(), "bag": False})
    for manifest, bags in manifests:
        for r in read_manifest(manifest):
            if r.get("label", "glioma") != "glioma":
                continue
            p = out[r["PatientID"]]
            p["categories"].add(r["category"])
            p["bag"] |= (Path(bags) / f"{r['SeriesInstanceUID']}.npz").exists()
    return out


def count(people: dict, used: set, held: set) -> dict:
    rows = defaultdict(Counter)
    for pid, p in people.items():
        cat = next(iter(p["categories"])) if len(p["categories"]) == 1 else "mixed"
        c = rows[cat]
        c["patients"] += 1
        if pid in used:
            c["already_used"] += 1
        elif pid in held:
            c["held_out_elsewhere"] += 1
        elif p["bag"]:
            c["eligible_with_bag"] += 1
        else:
            c["eligible_needs_embedding"] += 1
    keys = ("patients", "already_used", "held_out_elsewhere", "eligible_with_bag", "eligible_needs_embedding")
    order = [c for c in (*CATEGORIES, "mixed") if c in rows]
    table = {c: {k: rows[c][k] for k in keys} for c in order}
    table["total"] = {k: sum(rows[c][k] for c in order) for k in keys}
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--internal", nargs=3, action="append", default=[], metavar=("NAME", "MANIFEST", "BAGS"))
    ap.add_argument("--external", nargs=3, action="append", default=[], metavar=("NAME", "MANIFEST", "BAGS"))
    ap.add_argument("--used", action="append", required=True, help="CSV with a patient or PatientID column (repeat)")
    ap.add_argument("--held-out", action="append", default=[])
    a = ap.parse_args()
    gone = [p for p in a.used if not Path(p).exists()]
    if gone:
        raise SystemExit(f"missing --used files, so used patients could look fresh: {gone}")
    used, held = read_patients(a.used), read_patients([p for p in a.held_out if Path(p).exists()])
    found = [(n, m, b) for n, m, b in a.internal if Path(m).exists()]
    if not found:
        raise SystemExit("none of the --internal manifests exist")
    internal = count(patients([(m, b) for _, m, b in found]), used, held)
    out = {"read_only": True, "used_patients_listed": len(used), "internal_cohorts": [n for n, _, _ in found],
           "internal_missing": [n for n, m, _ in a.internal if not Path(m).exists()], "internal": internal,
           "targets": {"calibration": CALIBRATION, "test": TEST,
                       "eligible_now": internal["total"]["eligible_with_bag"],
                       "enough_with_bags": internal["total"]["eligible_with_bag"] >= CALIBRATION + TEST,
                       "enough_after_embedding": internal["total"]["eligible_with_bag"] +
                       internal["total"]["eligible_needs_embedding"] >= CALIBRATION + TEST},
           "external": {}, "external_missing": []}
    for n, m, b in a.external:
        if Path(m).exists():
            out["external"][n] = count(patients([(m, b)]), used, held)
        else:
            out["external_missing"].append(n)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
