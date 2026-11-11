"""Per-step model confidence for one patient: what each step of the AI glioma workflow predicted and how sure it is.

Nothing here is a new score. Steps 1 to 4 already post finding records (``entip.glioma_step/1``, verb
``glioma_step``) that carry a confidence, the accept threshold set on held-out data, and a status. Step 5, the
rules engine, posts ``entip.integrated_dx/1`` (verb ``integrated_dx``). This module reads the latest of each
from the action log and lays them out step by step, saying plainly where a score is not a calibrated
probability or where there is no score at all.

``summarise`` is a pure function (no database), so other tools and tests can use it.
"""
from __future__ import annotations

from typing import Any

CONTRACT = "entip.step_confidence/1"

# (step key, number, title). Keys match the step contract; step 5 has its own output.
STEPS = (
    ("triage", 1, "Triage"),
    ("high_grade_features", 2, "High-grade features"),
    ("glioma_panel", 3, "Glioma panel"),
    ("methylation_class", 4, "Methylation class"),
    ("integrated_diagnosis", 5, "Integrated diagnosis"),
)

# What each finding predicts, in plain words. Unknown findings fall back to their own name.
PREDICTS = {
    "adequacy": "Enough tissue to assess",
    "tissue_class": "Tissue class (normal, reactive, glioma, other primary, metastasis, lymphoma)",
    "glioma_pattern": "Glioma growth pattern on H&E",
    "necrosis": "Necrosis",
    "microvascular_proliferation": "Microvascular proliferation",
    "mitotic_activity": "Elevated mitotic activity",
    "IDH": "IDH mutation",
    "ATRX": "ATRX loss",
    "TP53": "TP53 mutation",
    "1p19q": "1p/19q codeletion",
    "CDKN2A": "CDKN2A/B homozygous deletion",
    "EGFR": "EGFR amplification",
    "chr7_chr10": "Chromosome 7 gain and 10 loss",
    "TERT": "TERT promoter mutation",
    "H3K27": "H3 K27 alteration",
    "H3G34": "H3 G34 mutation",
    "MGMT": "MGMT promoter methylation",
    "methylation_class": "DNA methylation class (family) from H&E",
}

# What kind of number ``confidence`` is.
SCORE_NOTES = {
    "calibrated_probability": "Calibrated probability that the call is right.",
    "reported_probability": "The step reports this as a calibrated probability; the record does not say how "
                            "it was calibrated.",
    "uncalibrated": "Model score that has not been calibrated on held-out slides. Not a probability.",
    "rule": "Fixed rule, not a model: the number is a placeholder, not a probability.",
    "none": "No score: the step did not make a call.",
    "not_a_model": "Not a model call (a lab result or a person's call).",
    "rules_engine": "Rules engine: it applies the WHO CNS5 criteria to the findings above and has no "
                    "probability of its own. It is only as sure as the findings it used.",
}

_UNCALIBRATED_WORDS = ("not calibrated", "uncalibrated", "not been calibrated")


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _score_kind(rec: dict) -> str:
    """Decide what the record's confidence is. An explicit ``calibration`` block in the record wins."""
    if rec.get("source", "ai") != "ai":
        return "not_a_model"
    if rec.get("call") == "indeterminate" or rec.get("status") == "not_assessed":
        return "none"
    cal = rec.get("calibration")
    if isinstance(cal, dict):
        method = str(cal.get("method") or "").lower()
        if method == "rule":
            return "rule"
        if cal.get("calibrated") is False or method in ("none", "uncalibrated"):
            return "uncalibrated"
        return "calibrated_probability"
    model_id = str((rec.get("model") or {}).get("model_id") or "")
    if model_id.endswith("-rule"):
        return "rule"
    if any(w in str(r).lower() for r in rec.get("reasons") or [] for w in _UNCALIBRATED_WORDS):
        return "uncalibrated"
    return "reported_probability"


def _status(rec: dict) -> str:
    """The record's status, or the contract default when it is left out."""
    if rec.get("status"):
        return rec["status"]
    if rec.get("source", "ai") != "ai":
        return "confident"
    conf, thr = _num(rec.get("confidence")), _num(rec.get("threshold"))
    return "confident" if conf is not None and thr is not None and conf >= thr else "uncertain"


def _fmt(p: float) -> str:
    """Two decimals, without rounding a near-certain score up to 1.00."""
    return ">0.99" if 0.995 <= p < 1.0 else f"{p:.2f}"


def _explain(f: dict) -> str:
    """One plain sentence for the pathologist."""
    kind, conf, thr = f["score"]["kind"], f["confidence"], f["threshold"]
    conf_s = _fmt(conf) if conf is not None else ""
    first = f["reasons"][0] if f["reasons"] else None
    test = f" Advised: {f['order_test'].replace('_', ' ')}." if f["order_test"] else ""
    if kind == "not_a_model":
        return "Lab result or pathologist call, not a model prediction."
    if kind == "none":
        return f"No call made{': ' + first if first else ''}.{test}"
    if kind == "rule":
        return f"Set by a fixed rule, not a model{': ' + first if first else ''}."
    if thr is None:
        return f"Confidence {conf_s}; the step gave no accept threshold, so the call is not accepted.{test}"
    if thr >= 1.0:
        return ("No accept threshold could be set for this model at the agreed error rate, so none of its "
                f"calls are accepted (confidence {conf_s}).{test}")
    if f["passed"]:
        return f"Confidence {conf_s}, above the accept threshold {_fmt(thr)}."
    if conf >= thr:
        why = f": {'; '.join(f['reasons'])}" if f["reasons"] else ""
        return f"Confidence {conf_s} is above the accept threshold {_fmt(thr)} but the step held it back{why}.{test}"
    return f"Confidence {conf_s}, below the accept threshold {_fmt(thr)}.{test}"


def _finding(rec: dict, action: dict) -> dict:
    kind = _score_kind(rec)
    has_score = kind in ("calibrated_probability", "reported_probability", "uncalibrated")
    conf = _num(rec.get("confidence")) if has_score else None
    thr = _num(rec.get("threshold")) if has_score else None
    status = _status(rec)
    model = rec.get("model") or {}
    f = {
        "finding": rec.get("finding"),
        "predicts": PREDICTS.get(rec.get("finding"), rec.get("finding")),
        "scan_ids": list(rec.get("scan_ids") or []),
        "call": rec.get("call"),
        "source": rec.get("source", "ai"),
        "confidence": conf,
        "threshold": thr,
        "margin": round(conf - thr, 4) if conf is not None and thr is not None else None,
        "above_threshold": (conf >= thr) if conf is not None and thr is not None else None,
        "status": status,
        "passed": status == "confident",
        "score": {"kind": kind, "note": SCORE_NOTES[kind]},
        "calibration": rec.get("calibration") if isinstance(rec.get("calibration"), dict) else None,
        "order_test": rec.get("order_test"),
        "reasons": [str(r) for r in rec.get("reasons") or []],
        "model": {"model_id": model.get("model_id"), "version": model.get("version")} if model else None,
        "at": action.get("at"),
        "action_id": action.get("action_id"),
    }
    f["explain"] = _explain(f)
    return f


def _step_block(key: str, n: int, title: str, findings: list[dict]) -> dict:
    scored = [f for f in findings if f["confidence"] is not None]
    lowest = min(scored, key=lambda f: f["confidence"]) if scored else None
    if not findings:
        status = "not_run"
    elif all(f["passed"] for f in findings):
        status = "passed"
    else:
        status = "needs_review"
    return {
        "step": key, "n": n, "title": title, "status": status,
        "n_findings": len(findings), "n_passed": sum(f["passed"] for f in findings),
        "lowest_confidence": ({"finding": lowest["finding"], "confidence": lowest["confidence"]}
                              if lowest else None),
        "findings": findings,
    }


def _dx_block(n: int, title: str, action: dict | None) -> dict:
    out = {"step": "integrated_diagnosis", "n": n, "title": title, "status": "not_run",
           "score": {"kind": "rules_engine", "note": SCORE_NOTES["rules_engine"]},
           "diagnosis": None, "inputs": [], "weakest_ai_input": None, "ai_decisive": [], "n_tests_ordered": 0,
           "at": None, "action_id": None}
    if not action:
        return out
    dx = action.get("detail") or {}
    d = dx.get("integrated_diagnosis") or {}
    inputs = []
    for layer, items in (dx.get("layers") or {}).items():
        for it in items or []:
            inputs.append({"layer": layer, "finding": it.get("finding"), "call": it.get("call"),
                           "source": it.get("source"), "status": it.get("status"),
                           "confidence": _num(it.get("confidence")) if it.get("source") == "ai" else None})
    ai = [i for i in inputs if i["source"] == "ai" and i["confidence"] is not None and i["call"]]
    out.update({
        "status": d.get("status") or "not_run",
        "diagnosis": {"text": d.get("text"), "status": d.get("status"), "cns_who_grade": d.get("cns_who_grade"),
                      "favoured": d.get("favoured")},
        "inputs": inputs,
        "weakest_ai_input": min(ai, key=lambda i: i["confidence"]) if ai else None,
        "ai_decisive": list(dx.get("ai_decisive") or []),
        "n_tests_ordered": len(dx.get("orders") or []),
        "at": action.get("at"), "action_id": action.get("action_id"),
    })
    return out


def summarise(study_id: str, actions: list[dict]) -> dict:
    """Build the per-step confidence summary from a patient's ``glioma_step`` and ``integrated_dx`` actions.

    ``actions`` are rows of the action log (``action_id``, ``verb``, ``at``, ``detail``), oldest first. For each
    finding the latest run wins: all of that run's records for the finding are kept (one per scan group)."""
    latest: dict[tuple[str, str], tuple[dict, list[dict]]] = {}
    dx_action = None
    for a in actions:
        detail = a.get("detail") or {}
        if detail.get("study_id") != study_id:
            continue
        if a.get("verb") == "integrated_dx":
            dx_action = a
        elif a.get("verb") == "glioma_step":
            run: dict[tuple[str, str], list[dict]] = {}
            for r in detail.get("records") or []:
                if isinstance(r, dict) and r.get("step") and r.get("finding") \
                        and r.get("study_id", study_id) == study_id:
                    run.setdefault((r["step"], r["finding"]), []).append(r)
            for k, recs in run.items():
                latest[k] = (a, recs)

    order = {name: i for i, name in enumerate(PREDICTS)}
    steps = []
    for key, n, title in STEPS:
        if key == "integrated_diagnosis":
            steps.append(_dx_block(n, title, dx_action))
            continue
        findings = [_finding(r, a) for (s, _), (a, recs) in latest.items() if s == key for r in recs]
        findings.sort(key=lambda f: (order.get(f["finding"], 99), f["scan_ids"]))
        steps.append(_step_block(key, n, title, findings))

    model_steps = steps[:-1]
    all_f = [f for s in model_steps for f in s["findings"]]
    return {
        "contract": CONTRACT,
        "study_id": study_id,
        "steps": steps,
        "overall": {
            "steps_run": sum(s["status"] != "not_run" for s in steps),
            "n_findings": len(all_f),
            "n_passed": sum(f["passed"] for f in all_f),
            "needs_review": [f["finding"] for f in all_f if not f["passed"]],
            "note": "Per-finding scores are not combined into one number: the findings are not independent "
                    "and the steps are calibrated separately.",
        },
    }


def patient_confidence(conn, study_id: str) -> dict:
    """Read the patient's step and diagnosis events from the log and summarise them."""
    rows = conn.execute(
        "SELECT action_id, verb, at, detail FROM entip.action "
        "WHERE verb IN ('glioma_step', 'integrated_dx') AND detail->>'study_id' = %s ORDER BY at, action_id",
        (study_id,)).fetchall()
    for r in rows:
        if hasattr(r["at"], "isoformat"):
            r["at"] = r["at"].isoformat()
    return summarise(study_id, rows)
