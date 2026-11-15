"""Teacher vs student: print the triage report trained on each one's bags side by side, plus embedding agreement.

    python compare.py report-h-optimus-0.json report-student.json evaluation.json
"""
from __future__ import annotations

import json
import sys


def _row(r: dict) -> dict:
    cv = r["cv"]["calibrated"]
    gate = r["cv"]["gate"]
    loso = r.get("leave_one_source_out") or {}
    loso_acc = [v.get("accuracy") for v in loso.values() if isinstance(v, dict) and v.get("accuracy") is not None] \
        if isinstance(loso, dict) else []
    return {"classes trained": ", ".join(r["classes_trained"]),
            "CV balanced accuracy": cv.get("balanced_accuracy"), "CV calibration error (ECE)": cv.get("ece"),
            "AUROC tumour": cv.get("auroc_tumour"), "AUROC glial": cv.get("auroc_glial"),
            "gate threshold": gate.get("threshold"), "gate coverage": gate.get("coverage"),
            "gate patients accepted": gate.get("n_accepted"),
            "leave-one-source-out accuracy (mean)": sum(loso_acc) / len(loso_acc) if loso_acc else None}


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else ("-" if v is None else str(v))


def main(teacher: str, student: str, evaluation: str) -> int:
    t, s = _row(json.load(open(teacher))), _row(json.load(open(student)))
    ev = json.load(open(evaluation))
    print(f"student: {ev['student']}")
    print(f"embedding agreement on {ev['heldout_slides']} held-out slides ({ev['heldout_tiles']} tiles): "
          f"tile cosine {_fmt(ev['tile_cosine_mean'])} (5th percentile {_fmt(ev['tile_cosine_p05'])}), "
          f"slide-mean cosine {_fmt(ev['slide_mean_cosine_mean'])}")
    print("by class: " + ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(ev["tile_cosine_by_label"].items())))
    print()
    w = max(len(k) for k in t)
    print(f"{'triage, trained on':<{w}}  {'H-optimus-0':>14}  {'student':>14}")
    for k in t:
        print(f"{k:<{w}}  {_fmt(t[k]):>14}  {_fmt(s[k]):>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
