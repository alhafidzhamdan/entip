"""The triage model's out-of-fold mistakes, most confident first, and why the gate accepts what it does.

    python mistakes.py report-h-optimus-0.json [--all]

Reads ``oof_patients`` from a triage training report: one row per patient (their most abnormal slide), with the
call made by a model that never saw that patient. The gate tests cuts from the most confident patient down and
needs the error bound among accepted patients to stay under the limit; at 5% that needs at least 59 patients with
no error, so a mistake ranked within the top 59 blocks the gate on its own.
"""
from __future__ import annotations

import json
import sys

from collections import Counter


def main(path: str, show_all: bool = False) -> int:
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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--all" in sys.argv[2:]))
