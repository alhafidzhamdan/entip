"""The triage model's out-of-fold mistakes, most confident first, and why the gate accepts what it does.

    python mistakes.py report-h-optimus-0.json [--all] [--manifest trial.csv]

Reads ``oof_patients`` from a triage training report: one row per patient (their most abnormal slide), with the
call made by a model that never saw that patient. The gate tests cuts from the most confident patient down and
needs the error bound among accepted patients to stay under the limit; at 5% that needs at least 59 patients with
no error, so a mistake ranked within the top 59 blocks the gate on its own.

Each patient's WHO CNS5 diagnosis is shown under its row when the run's ``trial.csv`` has one (``diagnosis``,
written by pick_slides.py or add_who_cns5.py): from ``--manifest``, else the trial.csv next to the report.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from collections import Counter


def diagnoses(path) -> dict:
    """{patient: WHO CNS5 diagnosis} from a manifest with a ``diagnosis`` column; {} when there is none."""
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {r["PatientID"]: r["diagnosis"] for r in csv.DictReader(f) if r.get("diagnosis")}


def main(path: str, show_all: bool = False, manifest: str | None = None) -> int:
    dx = diagnoses(manifest or Path(path).with_name("trial.csv"))
    rep = json.load(open(path))
    rep = rep.get("report", rep)                   # a saved model's model.json, or the train command's output
    rows = rep.get("oof_patients")
    if not rows:
        raise SystemExit("this report has no oof_patients: retrain with the current code (eddie/retrain.sh)")
    gate = rep["cv"]["gate"]
    wrong = [r for r in rows if not r["correct"]]
    print(f"{len(wrong)} of {len(rows)} patients wrong out of fold; gate accepted {gate['n_accepted']} "
          f"(first wrong call at confidence rank {gate.get('first_error_rank')})")
    print("confusions (truth -> call):", dict(Counter(f"{r['label']} -> {r['call']}" for r in wrong).most_common()))
    print()
    print(f"{'rank':>4}  {'patient':<16} {'truth':<18} {'call':<18} {'conf':>6}  {'novel':<5}  source / slide")
    for r in rows if show_all else wrong:
        mark = "" if r["correct"] else "  <- wrong"
        print(f"{r['rank']:>4}  {r['patient']:<16} {r['label']:<18} {r['call']:<18} {r['confidence']:>6.3f}  "
              f"{str(r['novel']):<5}  {r['source']} / {r['slide']}{mark if show_all else ''}")
        if r["patient"] in dx:
            print(f"{'':>6}WHO CNS5: {dx[r['patient']]}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    m = args[args.index("--manifest") + 1] if "--manifest" in args else None
    sys.exit(main(args[0], "--all" in args, m))
