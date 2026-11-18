"""Add WHO CNS5 diagnoses to a run's slide list (trial.csv) that was picked before they existed.

    python add_who_cns5.py trial.csv --idh idh.tsv

Writes ``diagnosis`` (the WHO CNS5 term) and ``who_cns5_basis`` (what it rests on) into the file, keeping a copy
as ``trial.before-who-cns5.csv``, and prints the diagnoses by class (the label training uses, after label_corrections.csv) and every row no rule could name (to review).
New runs get the columns from pick_slides.py. ``--idh``: fetch_idh.py's table, rerun after 2026-11-17 so it
holds 1p/19q, grade, TERT and +7/-10.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

from entip_glioma.triage import who_cns5
from entip_glioma.triage.manifest import apply_corrections, read_corrections


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--idh", help="fetch_idh.py table (IDH, 1p/19q, grade) for TCGA and CPTAC gliomas")
    a = ap.parse_args()
    p = Path(a.trial)
    rows = list(csv.DictReader(p.open(newline="")))
    mol = who_cns5.read_molecular(a.idh) if a.idh else None
    if mol is not None and not any(m.get("IDH_CODEL_SUBTYPE") for m in mol.values()):
        print("note: the --idh table has no 1p/19q column; rerun fetch_idh.py for integrated glioma diagnoses",
              file=sys.stderr)
    corrections = read_corrections()
    out = who_cns5.annotate(rows, mol, corrections)
    backup = p.with_name(p.stem + ".before-who-cns5.csv")
    if not backup.exists():
        shutil.copy(p, backup)
    fields = list(rows[0]) + [k for k in ("diagnosis", "who_cns5_basis") if k not in rows[0]]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    # Grouped by the label training uses: trial.csv keeps the source's label, corrections apply at training.
    trained, changed = apply_corrections(out, corrections)
    pts = {}
    for r in trained:
        pts.setdefault(r["PatientID"], r)
    for label in sorted({r["label"] for r in pts.values()}):
        print(f"{label} ({who_cns5.class_name(label)}):")
        for d, n in Counter(r["diagnosis"] for r in pts.values() if r["label"] == label).most_common():
            print(f"  {n:>5}  {d}")
    moved = {c["patient"]: c for c in changed}
    if moved:
        print("label corrections applied at training (label_corrections.csv):")
        for c in moved.values():
            print(f"  {c['patient']}: {c['from']} -> {c['to'] or 'dropped'} ({c['diagnosis']}; {c['source']})")
    review = Counter(r.get("dx") or "" for r in pts.values() if r["who_cns5_basis"].endswith(": review"))
    if review:
        print("no WHO CNS5 rule (add one to who_cns5.ICDO_RULES):", dict(review))
    print(f"wrote {p} ({len(out)} slides); copy of the original: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
