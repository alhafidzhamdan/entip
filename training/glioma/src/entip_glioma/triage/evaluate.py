"""Score a trained triage model on an external set it never saw, patient by patient, exactly as it runs in use.

Each patient's call comes from ``TriageModel.records`` - the same code that writes the clinical record - so the
external numbers describe what a pathologist would see: the call, its confidence, and whether it was confident
(accepted) or sent for review. Nothing is refitted: the temperature, threshold and novelty check are the ones
set on the training sources.

The report separates:

* **known classes** (the model was trained on them): accuracy, balanced accuracy, recall per class, and the
  tumour / glial questions that triage exists to answer;
* **the gate**: how many patients were accepted (confident), how many of those were wrong, and the exact upper
  bound on that error rate - the external check of the guarantee set in cross-validation;
* **unseen classes** (in the external set but not trained, e.g. lymphoma): how often the model was confidently
  wrong about them;
* **per diagnosis**, when the manifest has a ``diagnosis`` column.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..wsi.bags import Bag
from ..wsi.calibrate import exact_upper
from .labels import CLASSES, GLIAL, TUMOUR


def evaluate(model, rows: List[dict], bags_dir: str | Path, id_col: str = "SeriesInstanceUID",
             max_error: Optional[float] = None) -> dict:
    trained = [c for c, t in zip(CLASSES, model.trained) if t]
    max_error = model.cfg.max_error if max_error is None else max_error
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["PatientID"]].append(r)
    patients, missing, empty = [], 0, 0
    for pid, rs in sorted(by_patient.items()):
        bags = []
        for r in rs:
            f = Path(bags_dir) / f"{r[id_col]}.npz"
            if not f.exists():
                missing += 1
                continue
            b = Bag.load(f)
            if len(b) == 0:
                empty += 1
                continue
            bags.append(b)
        if not bags:
            continue
        recs = [x for x in model.records(bags, pid) if x["finding"] == "tissue_class"]
        if not recs:
            continue
        x = recs[0]
        truth = rs[0]["label"]
        patients.append({"patient": pid, "label": truth, "diagnosis": rs[0].get("diagnosis") or None,
                         "call": x["call"], "confidence": x["confidence"], "confident": x["status"] == "confident",
                         "novel": bool(x["value"].get("novel")), "slides": len(bags),
                         "correct": x["call"] == truth, "probs": x["probabilities"]})

    known = [p for p in patients if p["label"] in trained]
    unseen = [p for p in patients if p["label"] not in trained]
    rep = {"model_id": model.model_id, "encoder": model.encoder, "classes_trained": trained,
           "threshold": round(min(model.threshold, 1.0), 4), "model_never_accepts": model.threshold > 1.0,
           "patients_scored": len(patients), "slides_missing_bags": missing, "slides_empty_bags": empty,
           "patients_per_class": dict(Counter(p["label"] for p in patients)),
           "known": _known(known, trained), "gate": _gate(known, max_error), "unseen": _unseen(unseen),
           "per_diagnosis": _per_diagnosis(patients),
           "confident_mistakes": sorted(({k: p[k] for k in ("patient", "label", "diagnosis", "call", "confidence",
                                                            "novel")} for p in known
                                         if p["confident"] and not p["correct"]),
                                        key=lambda p: -p["confidence"]),
           "patients": [{k: v for k, v in p.items() if k != "probs"} for p in patients]}
    return rep


def _known(ps: list, trained: list) -> dict:
    if not ps:
        return {"n": 0}
    rec = {c: _share(ps, lambda p, c=c: p["label"] == c, lambda p: p["correct"]) for c in trained}
    rec = {c: v for c, v in rec.items() if v is not None}
    in_tumour = lambda c: c in TUMOUR                                   # noqa: E731
    in_glial = lambda c: c in GLIAL                                     # noqa: E731
    return {"n": len(ps), "accuracy": round(float(np.mean([p["correct"] for p in ps])), 4),
            "balanced_accuracy": round(float(np.mean(list(rec.values()))), 4) if rec else None,
            "recall": {c: round(v, 4) for c, v in rec.items()},
            "tumour_vs_not_correct": round(float(np.mean([in_tumour(p["call"]) == in_tumour(p["label"])
                                                          for p in ps])), 4),
            "glial_vs_not_correct": round(float(np.mean([in_glial(p["call"]) == in_glial(p["label"]) for p in ps])), 4),
            "confusions": dict(Counter(f"{p['label']} -> {p['call']}" for p in ps if not p["correct"]).most_common()),
            "novel_share": round(float(np.mean([p["novel"] for p in ps])), 4)}


def _gate(ps: list, max_error: float) -> dict:
    acc = [p for p in ps if p["confident"]]
    errs = sum(not p["correct"] for p in acc)
    ub = exact_upper(errs, len(acc)) if acc else None
    return {"n": len(ps), "accepted": len(acc), "coverage": round(len(acc) / len(ps), 4) if ps else None,
            "errors_among_accepted": errs, "error": round(errs / len(acc), 4) if acc else None,
            "error_upper_95": round(ub, 4) if ub is not None else None, "max_error": max_error,
            "guarantee_held": (ub <= max_error) if ub is not None else None}


def _unseen(ps: list) -> dict:
    out = {}
    for c in sorted({p["label"] for p in ps}):
        cs = [p for p in ps if p["label"] == c]
        out[c] = {"n": len(cs), "confidently_called": sum(p["confident"] for p in cs),
                  "flagged_novel": sum(p["novel"] for p in cs), "calls": dict(Counter(p["call"] for p in cs))}
    return out


def _per_diagnosis(ps: list) -> dict:
    out = {}
    for d in sorted({p["diagnosis"] for p in ps if p["diagnosis"]}):
        ds = [p for p in ps if p["diagnosis"] == d]
        out[d] = {"label": ds[0]["label"], "n": len(ds), "correct": round(float(np.mean([p["correct"] for p in ds])), 4),
                  "confident": sum(p["confident"] for p in ds),
                  "confident_wrong": sum(p["confident"] and not p["correct"] for p in ds),
                  "calls": dict(Counter(p["call"] for p in ds))}
    return out


def _share(ps, sel, ok) -> Optional[float]:
    s = [p for p in ps if sel(p)]
    return float(np.mean([ok(p) for p in s])) if s else None
