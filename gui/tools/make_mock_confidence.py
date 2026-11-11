"""Write gui/web/mock/confidence.json: per-step model confidence for the mock patients.

The finding records are invented, in the shape steps 1 to 4 post (entip.glioma_step/1), and are summarised by
the same code the database endpoint uses (db/src/entip_db/confidence.py), so the mock and
GET /patients/{study_id}/confidence have one shape. Synthetic data only; every score here is made up.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "db" / "src"))
from entip_db.confidence import summarise  # noqa: E402  (pure Python, no database needed)

MOCK = ROOT / "gui" / "web" / "mock"

# finding -> (mock molecular field, {mock value -> call}, call when the field is absent, accept threshold)
PANEL = {
    "IDH": ("IDH", {"wildtype": "wildtype"}, "mutant", 0.83),
    "ATRX": ("ATRX", {"loss": "lost", "retained": "retained"}, "retained", 0.72),
    "TP53": ("TP53", {"mutant": "mutant"}, "wildtype", 0.83),
    "1p19q": ("1p/19q", {"codeleted": "codeleted"}, "not_codeleted", 0.94),
    "CDKN2A": ("CDKN2A/B", {"homozygous deletion": "homozygous_deletion"}, "retained", 0.90),
    "EGFR": ("EGFR", {"amplified": "amplified"}, "not_amplified", 0.81),
    "chr7_chr10": ("+7/-10", {"present": "gain7_loss10"}, "absent", 0.85),
    "TERT": ("TERT promoter", {"C228T": "mutant", "C250T": "mutant"}, "wildtype", 0.83),
    "H3K27": (None, {}, "wildtype", 0.60),
}
FAMILY = {"Glioblastoma": "glioblastoma_idh_wildtype", "Astrocytoma": "astrocytoma_idh_mutant",
          "Oligodendroglioma": "oligodendroglioma_idh_mutant_codeleted"}
TEST = {"IDH": "IDH1_R132H_IHC", "ATRX": "ATRX_IHC", "TP53": "p53_IHC", "1p19q": "1p19q_testing",
        "CDKN2A": "CDKN2A_B_testing", "EGFR": "EGFR_amplification_testing", "chr7_chr10": "chr7_chr10_copy_number",
        "TERT": "TERT_promoter_sequencing", "H3K27": "H3K27M_IHC", "MGMT": "MGMT_promoter_methylation"}


def records(p: dict, scan_id: str, rnd: random.Random) -> list[dict]:
    sid = p["study_id"]

    def ai(step, finding, call, conf, thr, model, status=None, **kw):
        status = status or ("confident" if conf >= thr else "uncertain")
        return {"contract": "entip.glioma_step/1", "step": step, "finding": finding, "study_id": sid,
                "scan_ids": [scan_id], "call": call, "source": "ai", "confidence": round(conf, 4),
                "threshold": thr, "status": status, "reasons": [],
                "model": {"model_id": model, "version": "mock"}, **kw}

    high = p.get("who_grade") == 4
    out = [
        ai("triage", "adequacy", "adequate", 1.0, 0.9, "triage-adequacy-rule",
           reasons=["tissue area above the 1.0 mm² rule; lesional content not judged"]),
        ai("triage", "tissue_class", "glioma", rnd.uniform(0.93, 0.999), 0.87, "mock-triage"),
        ai("high_grade_features", "necrosis", "present" if high else "absent", rnd.uniform(0.9, 0.98), 0.9,
           "mock-necrosis"),
        ai("high_grade_features", "microvascular_proliferation", "present" if high else "absent",
           rnd.uniform(0.74, 0.97), 0.9, "mock-mvp", order_test=None),
        ai("high_grade_features", "mitotic_activity", "present" if (p.get("who_grade") or 0) >= 3 else "absent",
           rnd.uniform(0.85, 0.97), 0.9, "mock-mitoses", status="uncertain", order_test="pathologist_review",
           reasons=["mitosis model not yet validated on CNS tumours"]),
    ]
    mol = p.get("molecular") or {}
    for finding, (field, calls, default, thr) in PANEL.items():
        call = calls.get(mol.get(field), default)
        r = ai("glioma_panel", finding, call, rnd.uniform(thr - 0.04, 0.999), thr, f"mock-panel:{finding}")
        if r["status"] == "uncertain":
            r["order_test"] = TEST[finding]
        out.append(r)
    out.append(ai("glioma_panel", "MGMT", mol.get("MGMT", "unmethylated") if mol.get("MGMT") in
                  ("methylated", "unmethylated") else "unmethylated", rnd.uniform(0.55, 0.7), 1.0,
                  "mock-panel:MGMT", order_test=TEST["MGMT"],
                  reasons=["no confidence level met the 5% error target in cross-validation"]))
    out.append({**ai("glioma_panel", "H3G34", "indeterminate", 0.0, 1.0, "mock-panel:H3G34", status="uncertain",
                     order_test="H3G34_testing"),
                "reasons": ["no model for this marker: too few labelled patients"]})
    family = next((v for k, v in FAMILY.items() if k in (p.get("diagnosis") or "")), "other")
    out.append(ai("methylation_class", "methylation_class", family, rnd.uniform(0.6, 0.99), 0.68, "mock-methylation",
                  calibration={"method": "temperature", "max_error": 0.05, "validated": False}))
    for r in out:
        if r["status"] == "uncertain" and not r.get("order_test"):
            r["order_test"] = "methylation_array" if r["step"] == "methylation_class" else "pathologist_review"
    return out


def draft(p: dict, recs: list[dict]) -> dict:
    """A stand-in for the rules engine's output (entip.integrated_dx/1), enough for the step 5 row."""
    layer = {"triage": "histology", "high_grade_features": "grade_features", "glioma_panel": "molecular",
             "methylation_class": "methylation"}
    layers: dict[str, list] = {}
    for r in recs:
        if r["finding"] == "adequacy":
            continue
        ok = r["status"] == "confident"
        layers.setdefault(layer[r["step"]], []).append(
            {"finding": r["finding"], "call": r["call"] if ok else None, "source": "ai" if ok else "missing",
             "status": r["status"] if ok else "pending", "confidence": r["confidence"] if ok else None})
    orders = sorted({r["order_test"] for r in recs if r.get("order_test")})
    return {"contract": "entip.integrated_dx/1", "study_id": p["study_id"],
            "integrated_diagnosis": {"text": f"{p['diagnosis']}, CNS WHO grade {p['who_grade']} (mock draft)",
                                     "status": "provisional" if orders else "complete",
                                     "cns_who_grade": p["who_grade"], "favoured": p["diagnosis"]},
            "layers": layers, "orders": [{"test": t, "reason": "mock", "blocking": True} for t in orders],
            "ai_decisive": [r["finding"] for r in recs if r["step"] == "glioma_panel" and r["status"] == "confident"
                            and r["finding"] in ("IDH", "1p19q")]}


def first_scan(o) -> str | None:
    if isinstance(o, dict):
        if o.get("scan_id") and o.get("tile_source"):
            return o["scan_id"]
        o = list(o.values())
    if isinstance(o, list):
        for v in o:
            hit = first_scan(v)
            if hit:
                return hit
    return None


def main() -> None:
    data = json.loads((MOCK / "patients.json").read_text())
    out = {}
    for i, p in enumerate(data["patients"]):
        sid = p["study_id"]
        recs = records(p, first_scan(p) or f"{sid}-scan", random.Random(20261111 + i))
        at = "2026-11-11T09:00:00+00:00"
        out[sid] = summarise(sid, [
            {"action_id": f"mock-steps-{sid}", "verb": "glioma_step", "at": at,
             "detail": {"study_id": sid, "records": recs}},
            {"action_id": f"mock-dx-{sid}", "verb": "integrated_dx", "at": at, "detail": draft(p, recs)}])
    (MOCK / "confidence.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote confidence for {len(out)} patients to {MOCK / 'confidence.json'}")


if __name__ == "__main__":
    main()
