"""Per-step confidence: the pure summary, then the endpoint. Records are shaped like the steps' real outputs."""
import json
from pathlib import Path

import pytest

from entip_db.confidence import CONTRACT, summarise

SID = "SYN-0003"


def rec(step, finding, call, conf, thr, status=None, **kw):
    r = {"contract": "entip.glioma_step/1", "step": step, "finding": finding, "study_id": SID,
         "scan_ids": ["SCN-0003-A1-01"], "call": call, "source": "ai", "confidence": conf,
         "model": {"model_id": kw.pop("model_id", f"m-{finding}"), "version": "0.1.0"}, "reasons": [], **kw}
    if thr is not None:
        r["threshold"] = thr
    if status:
        r["status"] = status
    return r


def act(i, verb, detail):
    return {"action_id": f"a{i}", "verb": verb, "at": f"2026-11-11T10:00:{i:02d}+00:00",
            "detail": {"study_id": SID, **detail}}


STEP_RECORDS = [
    rec("triage", "adequacy", "adequate", 1.0, 0.9, "confident", model_id="triage-adequacy-rule",
        reasons=["2.01 mm² of tissue tiles (rule: at least 1.0 mm²)"]),
    rec("triage", "tissue_class", "glioma", 0.9999, 0.8748, "confident"),
    rec("high_grade_features", "necrosis", "present", 0.8879, 0.8, "uncertain", order_test="pathologist_review",
        reasons=["model not signed off on pathologist-labelled slides"]),
    rec("high_grade_features", "microvascular_proliferation", "present", 0.71, 0.8, "uncertain",
        order_test="pathologist_review", reasons=["slide score not calibrated on held-out slides"]),
    rec("glioma_panel", "IDH", "mutant", 0.9907, 0.8252, "confident",
        calibration={"method": "temperature", "max_error": 0.05, "n_heldout": 180}),
    rec("glioma_panel", "MGMT", "methylated", 0.6379, 1.0, "uncertain", order_test="MGMT_promoter_methylation",
        reasons=["no confidence level met the 5% error target in cross-validation"]),
    rec("glioma_panel", "H3G34", "indeterminate", 0.0, None, "uncertain", order_test="H3G34_testing",
        reasons=["no model for this marker: too few labelled patients"]),
    rec("methylation_class", "methylation_class", "indeterminate", 0.0, 0.6765, "not_assessed",
        order_test="methylation_array", reasons=["only 5 tissue tiles (< 20)"]),
]

DX = {"contract": "entip.integrated_dx/1",
      "integrated_diagnosis": {"text": "Pending: diffuse glioma, IDH-mutant, 1p/19q status awaited",
                               "status": "pending", "cns_who_grade": None, "favoured": "Astrocytoma, IDH-mutant"},
      "layers": {"histology": [{"finding": "tissue_class", "call": "glioma", "source": "ai",
                                "status": "confident", "confidence": 0.9999}],
                 "molecular": [{"finding": "IDH", "call": "mutant", "source": "ai", "status": "confident",
                                "confidence": 0.9907},
                               {"finding": "ATRX", "call": "lost", "source": "test", "status": "confident",
                                "confidence": 1.0},
                               {"finding": "1p19q", "call": None, "source": "missing", "status": "pending",
                                "confidence": None}]},
      "orders": [{"test": "1p19q_testing", "reason": "needed", "blocking": True}], "ai_decisive": ["IDH"]}


def by_finding(out):
    return {f["finding"]: f for s in out["steps"] for f in s.get("findings", [])}


def test_summary_has_every_step_in_order():
    out = summarise(SID, [act(1, "glioma_step", {"records": STEP_RECORDS}), act(2, "integrated_dx", DX)])
    assert out["contract"] == CONTRACT
    assert [(s["n"], s["step"]) for s in out["steps"]] == [
        (1, "triage"), (2, "high_grade_features"), (3, "glioma_panel"), (4, "methylation_class"),
        (5, "integrated_diagnosis")]
    assert [s["status"] for s in out["steps"]] == ["passed", "needs_review", "needs_review", "needs_review",
                                                   "pending"]
    assert out["overall"]["n_findings"] == 8 and out["overall"]["n_passed"] == 3
    assert "necrosis" in out["overall"]["needs_review"]


def test_confidence_threshold_and_pass():
    f = by_finding(summarise(SID, [act(1, "glioma_step", {"records": STEP_RECORDS})]))
    idh = f["IDH"]
    assert idh["predicts"] == "IDH mutation" and idh["call"] == "mutant"
    assert idh["confidence"] == 0.9907 and idh["threshold"] == 0.8252 and idh["margin"] == 0.1655
    assert idh["passed"] and idh["above_threshold"]
    assert idh["score"]["kind"] == "calibrated_probability" and idh["calibration"]["max_error"] == 0.05
    assert "above the accept threshold 0.83" in idh["explain"]
    # no calibration details in the record: say so rather than claim it
    assert f["tissue_class"]["score"]["kind"] == "reported_probability"
    mvp = f["microvascular_proliferation"]
    assert not mvp["passed"] and mvp["above_threshold"] is False and mvp["margin"] == -0.09
    assert mvp["score"]["kind"] == "uncalibrated"


def test_held_back_above_threshold_keeps_the_step_status():
    nec = by_finding(summarise(SID, [act(1, "glioma_step", {"records": STEP_RECORDS})]))["necrosis"]
    assert nec["above_threshold"] and not nec["passed"]
    assert "held it back" in nec["explain"] and "not signed off" in nec["explain"]


def test_no_score_is_said_plainly():
    f = by_finding(summarise(SID, [act(1, "glioma_step", {"records": STEP_RECORDS})]))
    for name in ("H3G34", "methylation_class"):
        assert f[name]["confidence"] is None and f[name]["threshold"] is None and not f[name]["passed"]
        assert f[name]["score"]["kind"] == "none" and f[name]["explain"].startswith("No call made")
    assert f["adequacy"]["score"]["kind"] == "rule" and f["adequacy"]["confidence"] is None
    assert f["adequacy"]["passed"]
    assert "No accept threshold could be set" in f["MGMT"]["explain"]


def test_status_default_follows_the_contract():
    recs = [rec("glioma_panel", "ATRX", "lost", 0.9, 0.8), rec("glioma_panel", "TP53", "mutant", 0.7, 0.8),
            rec("glioma_panel", "EGFR", "amplified", 0.99, None)]
    f = by_finding(summarise(SID, [act(1, "glioma_step", {"records": recs})]))
    assert f["ATRX"]["passed"] and not f["TP53"]["passed"] and not f["EGFR"]["passed"]


def test_latest_run_wins_per_finding():
    old = [rec("glioma_panel", "IDH", "wildtype", 0.6, 0.83, "uncertain"),
           rec("glioma_panel", "ATRX", "retained", 0.95, 0.72, "confident")]
    new = [rec("glioma_panel", "IDH", "mutant", 0.99, 0.83, "confident")]
    out = summarise(SID, [act(1, "glioma_step", {"records": old}), act(2, "glioma_step", {"records": new})])
    f = by_finding(out)
    assert f["IDH"]["call"] == "mutant" and f["IDH"]["action_id"] == "a2"
    assert f["ATRX"]["action_id"] == "a1"  # not re-run, so the earlier call stands
    assert out["steps"][2]["n_findings"] == 2


def test_other_patients_and_lab_results():
    other = {"action_id": "x", "verb": "glioma_step", "at": "2026-11-11T10:00:00+00:00",
             "detail": {"study_id": "SYN-0001", "records": STEP_RECORDS}}
    assert summarise(SID, [other])["overall"]["n_findings"] == 0
    lab = {**rec("glioma_panel", "IDH", "mutant", 1.0, None, "confident"), "source": "test"}
    f = by_finding(summarise(SID, [act(1, "glioma_step", {"records": [lab]})]))["IDH"]
    assert f["score"]["kind"] == "not_a_model" and f["confidence"] is None and f["passed"]


def test_step5_has_no_score_of_its_own():
    out = summarise(SID, [act(1, "glioma_step", {"records": STEP_RECORDS}), act(2, "integrated_dx", DX)])
    s5 = out["steps"][4]
    assert s5["score"]["kind"] == "rules_engine" and "confidence" not in s5
    assert s5["diagnosis"]["favoured"] == "Astrocytoma, IDH-mutant" and s5["n_tests_ordered"] == 1
    assert s5["weakest_ai_input"]["finding"] == "IDH" and s5["ai_decisive"] == ["IDH"]
    assert [i["confidence"] for i in s5["inputs"] if i["source"] != "ai"] == [None, None]
    assert summarise(SID, [])["steps"][4]["status"] == "not_run"


def test_real_step_outputs_when_present():
    """The outputs of steps 1 to 4 kept with the rules engine (dx/), once that is merged."""
    path = Path(__file__).resolve().parents[2] / "dx" / "tests" / "fixtures" / "step_outputs.jsonl"
    if not path.exists():
        pytest.skip("dx/tests/fixtures/step_outputs.jsonl not in this checkout")
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for sid in {r["study_id"] for r in recs}:
        mine = [r for r in recs if r["study_id"] == sid]
        out = summarise(sid, [{"action_id": "a", "verb": "glioma_step", "at": None,
                               "detail": {"study_id": sid, "records": mine}}])
        assert out["overall"]["n_findings"] == len(mine)
        for f in by_finding(out).values():
            assert f["explain"] and (f["confidence"] is None or 0 <= f["confidence"] <= 1)


def test_endpoint(client):
    assert client.get("/patients/NOPE/confidence").status_code == 404
    empty = client.get(f"/patients/{SID}/confidence").json()
    assert [s["status"] for s in empty["steps"]] == ["not_run"] * 5
    for verb, detail, actor in (("glioma_step", {"study_id": SID, "records": STEP_RECORDS}, "glioma-panel"),
                                ("integrated_dx", {"study_id": SID, **DX}, "entip-dx")):
        r = client.post("/actions", json={"verb": verb, "provenance": {"kind": "model", "actor": actor},
                                          "detail": detail})
        assert r.status_code == 201, r.text
    out = client.get(f"/patients/{SID}/confidence").json()
    assert out["study_id"] == SID and out["overall"]["n_passed"] == 3
    idh = by_finding(out)["IDH"]
    assert idh["confidence"] == 0.9907 and idh["passed"] and idh["at"] and idh["action_id"]
    assert out["steps"][4]["diagnosis"]["status"] == "pending"
    # another patient is untouched
    assert client.get("/patients/SYN-0002/confidence").json()["overall"]["n_findings"] == 0
