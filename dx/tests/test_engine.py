import json
from pathlib import Path

import jsonschema
import pytest

from entip_dx import contract
from entip_dx.cli import export_contract
from entip_dx.dbio import records_from_actions, records_from_findings
from entip_dx.engine import diagnose

SID = "SYN-0001"
M = {"model_id": "test-head", "version": "0.0.1", "licence": "MIT"}


def ai(finding, call, conf=0.95, thr=0.9, **kw):
    return {"contract": contract.STEP_CONTRACT, "step": contract.step_of(finding), "finding": finding,
            "study_id": SID, "call": call, "confidence": conf, "threshold": thr, "model": M, **kw}


def lab(finding, call, method="NGS panel"):
    return {"step": contract.step_of(finding), "finding": finding, "study_id": SID, "call": call,
            "source": "test", "value": {"method": method}, "provenance": {"kind": "program", "actor": "lab"}}


GLIOMA = [ai("adequacy", "adequate"), ai("tissue_class", "glioma")]
H3WT = [lab("H3K27", "wildtype", "IHC"), lab("H3G34", "wildtype")]


def orders(out):
    return {o["test"] for o in out["orders"]}


def test_classic_glioblastoma_from_histology():
    out = diagnose(GLIOMA + H3WT + [lab("IDH", "wildtype"), ai("necrosis", "present"),
                                    ai("microvascular_proliferation", "absent")])
    d = out["integrated_diagnosis"]
    assert d["text"] == "Glioblastoma, IDH-wildtype, CNS WHO grade 4" and d["status"] == "complete"
    assert d["cns_who_grade"] == 4
    assert out["ai_decisive"] == ["necrosis"]
    mgmt = [o for o in out["orders"] if o["test"] == "MGMT_promoter_methylation"]
    assert mgmt and mgmt[0]["blocking"] is False
    assert "Glioblastoma, IDH-wildtype" in out["report_text"]
    assert "https://doi.org/10.1093/neuonc/noab106" in out["report_text"]


def test_molecular_glioblastoma_from_tert_alone():
    out = diagnose(GLIOMA + H3WT + [lab("IDH", "wildtype"), lab("TERT", "mutant"),
                                    ai("necrosis", "absent"), ai("microvascular_proliferation", "absent")])
    d = out["integrated_diagnosis"]
    assert d["who_type"] == "gbm" and d["status"] == "complete"
    crit = [b for b in out["based_on"] if b["met"] and "TERT" in b["findings"]]
    assert crit and "s00401-018-1913-0" in crit[0]["citation"]


def test_idh_wildtype_without_criteria_is_nec_and_orders_methylation():
    out = diagnose(GLIOMA + H3WT + [lab("IDH", "wildtype"), lab("TERT", "wildtype"),
                                    lab("EGFR", "not_amplified"), lab("chr7_chr10", "absent"),
                                    ai("necrosis", "absent"), ai("microvascular_proliferation", "absent")])
    d = out["integrated_diagnosis"]
    assert d["suffix"] == "NEC" and d["who_type"] is None
    assert "methylation_array" in orders(out)


def test_idh_wildtype_missing_molecular_criteria_orders_only_what_is_unknown():
    out = diagnose(GLIOMA + H3WT + [lab("IDH", "wildtype"), ai("necrosis", "absent"),
                                    ai("microvascular_proliferation", "absent"),
                                    ai("TERT", "mutant", conf=0.6), lab("EGFR", "not_amplified")])
    assert out["integrated_diagnosis"]["status"] == "pending"
    assert out["integrated_diagnosis"]["favoured"] == "Glioblastoma, IDH-wildtype"
    o = orders(out)
    assert {"TERT_promoter_sequencing", "chr7_chr10_copy_number"} <= o
    assert "EGFR_amplification_testing" not in o


def test_astrocytoma_grade_4_by_cdkn2a():
    out = diagnose(GLIOMA + [lab("IDH", "mutant", "IHC"), lab("1p19q", "not_codeleted"),
                             lab("CDKN2A", "homozygous_deletion"), ai("necrosis", "absent"),
                             ai("microvascular_proliferation", "absent")])
    d = out["integrated_diagnosis"]
    assert d["text"] == "Astrocytoma, IDH-mutant, CNS WHO grade 4" and d["status"] == "complete"
    assert out["ai_decisive"] == []  # decided on a test


def test_astrocytoma_grade_waits_for_cdkn2a():
    out = diagnose(GLIOMA + [lab("IDH", "mutant", "IHC"), lab("1p19q", "not_codeleted"),
                             ai("necrosis", "absent"), ai("microvascular_proliferation", "absent"),
                             ai("mitotic_activity", "absent")])
    d = out["integrated_diagnosis"]
    assert d["status"] == "provisional" and d["cns_who_grade"] is None
    assert "CDKN2A_B_testing" in orders(out)


def test_astrocytoma_grade_3_and_2():
    base = GLIOMA + [lab("IDH", "mutant", "IHC"), lab("1p19q", "not_codeleted"),
                     lab("CDKN2A", "retained"), ai("necrosis", "absent"),
                     ai("microvascular_proliferation", "absent")]
    assert diagnose(base + [ai("mitotic_activity", "present")])["integrated_diagnosis"]["cns_who_grade"] == 3
    assert diagnose(base + [ai("mitotic_activity", "absent")])["integrated_diagnosis"]["cns_who_grade"] == 2


def test_oligodendroglioma_grade_left_to_pathologist():
    out = diagnose(GLIOMA + [lab("IDH", "mutant", "IHC"), lab("1p19q", "codeleted"),
                             ai("microvascular_proliferation", "present")])
    d = out["integrated_diagnosis"]
    assert d["who_type"] == "oligo" and d["cns_who_grade"] == "2 or 3" and "favour 3" in d["text"]


def test_uncertain_idh_orders_test_and_stays_pending():
    out = diagnose(GLIOMA + [ai("IDH", "wildtype", conf=0.7), ai("necrosis", "present")])
    d = out["integrated_diagnosis"]
    assert d["status"] == "pending" and d["favoured"] == "Glioblastoma, IDH-wildtype"
    assert "IDH1_R132H_IHC" in orders(out)
    assert "IDH: pending (AI suggests wildtype, 0.70, below threshold)" in out["report_text"]


def test_test_result_beats_ai():
    out = diagnose(GLIOMA + H3WT + [ai("IDH", "mutant", conf=0.99), lab("IDH", "wildtype"),
                                    ai("necrosis", "present")])
    assert out["integrated_diagnosis"]["who_type"] == "gbm"


def test_h3_k27_altered():
    out = diagnose(GLIOMA + [lab("H3K27", "altered", "IHC")])
    d = out["integrated_diagnosis"]
    assert d["who_type"] == "dmg" and d["cns_who_grade"] == 4
    assert any("midline" in n for n in out["notes"])


def test_not_a_glioma():
    out = diagnose([ai("adequacy", "adequate"), ai("tissue_class", "metastasis")])
    assert out["integrated_diagnosis"]["status"] == "not_diffuse_glioma"
    assert orders(out) == {"pathologist_review"}


def test_inadequate():
    out = diagnose([ai("adequacy", "inadequate"), ai("tissue_class", "glioma")])
    assert out["integrated_diagnosis"]["status"] == "inadequate"
    assert "additional_sampling" in orders(out)


def test_ai_methylation_conflict_orders_array():
    out = diagnose(GLIOMA + H3WT + [lab("IDH", "wildtype"), ai("necrosis", "present"),
                                    ai("methylation_class", "astrocytoma_idh_mutant",
                                       value={"class": "A_IDH_HG"})])
    assert out["integrated_diagnosis"]["status"] == "provisional"
    assert {"methylation_array", "pathologist_review"} <= orders(out)
    assert out["conflicts"]


def test_priors_never_reach_the_output():
    priors = {"age_years": 41, "site": "left thalamus", "radiology": "ring enhancing lesion"}
    out = diagnose(GLIOMA + [ai("IDH", "wildtype", conf=0.5)], priors=priors)
    text = json.dumps(out)
    for word in ("thalamus", "ring enhancing", "41", "age_years", "radiology"):
        assert word not in text


def test_contract_refuses_priors_and_bad_calls():
    with pytest.raises(contract.ContractError):
        contract.validate_finding(ai("tissue_class", "glioma", value={"age": 60}))
    with pytest.raises(contract.ContractError):
        contract.validate_finding(ai("IDH", "maybe"))
    with pytest.raises(contract.ContractError):
        contract.validate_finding(ai("IDH", "indeterminate", status="confident"))
    r = contract.validate_finding(ai("EGFR", "amplified", conf=0.5))
    assert r["status"] == "uncertain" and r["order_test"] == "EGFR_amplification_testing"


def test_schemas_match_code_and_validate(tmp_path):
    export_contract(tmp_path)
    here = Path(__file__).resolve().parents[1] / "contract"
    for name in ("glioma_step.schema.json", "integrated_dx.schema.json"):
        assert (tmp_path / name).read_text() == (here / name).read_text(), f"run entip-dx export-contract ({name})"
    step = json.loads((here / "glioma_step.schema.json").read_text())
    jsonschema.validate(contract.validate_finding(ai("necrosis", "present")), step)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**ai("necrosis", "mutant"), "contract": contract.STEP_CONTRACT}, step)
    dxs = json.loads((here / "integrated_dx.schema.json").read_text())
    jsonschema.validate(diagnose(GLIOMA + [lab("IDH", "mutant", "IHC")]), dxs)


def test_db_rows_become_test_records():
    rows = [{"finding_id": "GF-1", "marker": "IDH", "result": "mutant", "detail": "IDH1 R132H", "method": "IHC"},
            {"finding_id": "GF-2", "marker": "1p19q", "result": "not codeleted", "method": "NGS panel"},
            {"finding_id": "GF-3", "marker": "ATRX", "result": "loss", "method": "IHC"},
            {"finding_id": "GF-4", "marker": "methylation_class_ai", "result": "x"},
            {"finding_id": "GF-5", "marker": "BRAF", "result": "wildtype"}]
    recs = records_from_findings(SID, rows)
    assert [(r["finding"], r["call"]) for r in recs] == [("IDH", "mutant"), ("1p19q", "not_codeleted"),
                                                         ("ATRX", "lost")]
    acts = [{"verb": "glioma_step", "detail": {"study_id": SID, "records": [ai("necrosis", "present")]}},
            {"verb": "create", "detail": {}}]
    assert len(records_from_actions(SID, acts)) == 1
    out = diagnose(GLIOMA + recs + records_from_actions(SID, acts))
    assert out["integrated_diagnosis"]["who_type"] == "astro"
    assert out["integrated_diagnosis"]["cns_who_grade"] == 4


def test_high_grade_feature_on_any_scan_counts():
    recs = GLIOMA + H3WT + [lab("IDH", "wildtype"),
                            ai("necrosis", "absent", scan_ids=["S1"]), ai("necrosis", "present", scan_ids=["S2"])]
    out = diagnose(recs)
    assert out["integrated_diagnosis"]["who_type"] == "gbm"


def test_absent_needs_every_scan_confident():
    recs = GLIOMA + H3WT + [lab("IDH", "wildtype"), lab("TERT", "wildtype"), lab("EGFR", "not_amplified"),
                            lab("chr7_chr10", "absent"), ai("microvascular_proliferation", "absent"),
                            ai("necrosis", "absent", scan_ids=["S1"]),
                            ai("necrosis", "absent", conf=0.6, scan_ids=["S2"])]
    out = diagnose(recs)
    assert out["integrated_diagnosis"]["status"] == "pending"
    assert "pathologist_review" in orders(out)


def test_newer_ai_run_replaces_older_one():
    old = ai("IDH", "mutant", conf=0.99, scan_ids=["S1"])
    new = ai("IDH", "mutant", conf=0.6, scan_ids=["S1"])
    out = diagnose(GLIOMA + [old, new])
    assert out["integrated_diagnosis"]["status"] == "pending"
    assert "IDH1_R132H_IHC" in orders(out)


def test_tumour_call_beats_normal_brain_on_another_scan():
    recs = [ai("adequacy", "adequate"), ai("tissue_class", "normal_brain", scan_ids=["S1"]),
            ai("tissue_class", "glioma", scan_ids=["S2"]), ai("IDH", "wildtype", conf=0.5)]
    assert diagnose(recs)["integrated_diagnosis"]["text"] == "Pending: diffuse glioma, IDH status awaited"


def test_panel_calls_that_differ_between_scans_are_uncertain():
    recs = GLIOMA + [ai("IDH", "mutant", scan_ids=["S1"]), ai("IDH", "wildtype", scan_ids=["S2"])]
    out = diagnose(recs)
    assert out["integrated_diagnosis"]["status"] == "pending"
    assert any("differ between scans" in x for x in out["conflicts"])


def test_negative_r132h_ihc_needs_sequencing():
    recs = GLIOMA + H3WT + [lab("IDH", "wildtype", "IHC"), ai("necrosis", "present")]
    out = diagnose(recs)
    assert out["integrated_diagnosis"]["status"] == "provisional"
    assert [o for o in out["orders"] if o["test"] == "IDH_sequencing" and o["blocking"]]
    # sequencing beats IHC whatever the order the results arrive in
    out2 = diagnose([lab("IDH", "wildtype", "NGS panel")] + recs)
    assert out2["integrated_diagnosis"]["status"] == "complete"


def test_nec_waits_for_h3():
    recs = GLIOMA + [lab("IDH", "wildtype"), lab("TERT", "wildtype"), lab("EGFR", "not_amplified"),
                     lab("chr7_chr10", "absent"), ai("necrosis", "absent"), ai("microvascular_proliferation", "absent")]
    out = diagnose(recs)
    assert out["integrated_diagnosis"]["status"] == "pending" and out["integrated_diagnosis"]["suffix"] is None
    assert "H3K27M_IHC" in orders(out)


def test_ai_disagreement_with_test_is_logged():
    out = diagnose(GLIOMA + H3WT + [ai("IDH", "mutant"), lab("IDH", "wildtype"), ai("necrosis", "present")])
    assert out["ai_disagreements"] == ["IDH: AI said mutant; test says wildtype"]


def test_report_wording_for_indeterminate_and_methylation():
    out = diagnose(GLIOMA + [ai("IDH", "indeterminate", conf=0.0, status="uncertain"),
                             ai("methylation_class", "glioblastoma_idh_wildtype")])
    assert "IDH: pending (AI could not call it)" in out["report_text"]
    assert "matches Glioblastoma, IDH-wildtype" in out["report_text"]


def test_real_step_outputs_run_through_the_engine():
    """Records produced by the step 1-4 code on their PR branches (synthetic data), as one patient."""
    path = Path(__file__).parent / "fixtures" / "step_outputs.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert {r["step"] for r in recs} == set(contract.STEPS)
    for r in recs:
        r["study_id"] = SID
    out = diagnose(recs)
    jsonschema.validate(out, json.loads((Path(__file__).resolve().parents[1] / "contract" /
                                         "integrated_dx.schema.json").read_text()))
    assert out["integrated_diagnosis"]["status"] in contract.DX_STATUSES
