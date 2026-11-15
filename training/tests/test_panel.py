"""Glioma panel model (step 3) on synthetic patients only."""
import json

import numpy as np
import pytest

from entip_train.panel import labels as L
from entip_train.panel.db import model_actions, step_action, validate
from entip_train.panel.gate import NEVER, class_threshold
from entip_train.panel.markers import DEFAULT_MARKERS, load_config
from entip_train.panel.model import PanelModel
from entip_train.panel.predict import predict_patient, summarise
from entip_train.panel.synthetic import synthetic_cohort

SMALL = {"hidden": 32, "attn": 16, "epochs": 12, "max_tiles": 128}
CONTRACT_CALLS = {  # from dx/contract/README.md (step 5)
    "IDH": {"mutant", "wildtype"}, "ATRX": {"retained", "lost"}, "TP53": {"mutant", "wildtype"},
    "1p19q": {"codeleted", "not_codeleted"}, "CDKN2A": {"homozygous_deletion", "retained"},
    "EGFR": {"amplified", "not_amplified"}, "chr7_chr10": {"gain7_loss10", "absent"},
    "TERT": {"mutant", "wildtype"}, "H3K27": {"altered", "wildtype"}, "H3G34": {"mutant", "wildtype"},
    "MGMT": {"methylated", "unmethylated"},
}
TESTS = {"IDH1_R132H_IHC", "ATRX_IHC", "p53_IHC", "1p19q_testing", "CDKN2A_B_testing", "EGFR_amplification_testing",
         "chr7_chr10_copy_number", "TERT_promoter_sequencing", "H3K27M_IHC", "H3G34_testing",
         "MGMT_promoter_methylation"}


@pytest.fixture(scope="module")
def trained():
    pats, labs, _ = synthetic_cohort(320, seed=0)
    m = PanelModel(load_config(None, abmil=SMALL)).fit(pats, labs, source="synthetic", log=lambda *_: None)
    return m


@pytest.fixture(scope="module")
def held_out():
    return synthetic_cohort(60, seed=1, prefix="TST")


def test_markers_match_step5_contract():
    for f, m in DEFAULT_MARKERS.items():
        assert set(m.calls) == CONTRACT_CALLS[f]
        assert m.order_test in TESTS


def test_config_subset_and_unknown(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"markers": ["IDH", "1p19q", "MGMT"], "disabled": ["MGMT"],
                             "overrides": {"IDH": {"max_error": 0.02}}}))
    cfg = load_config(p)
    assert [m.finding for m in cfg.markers] == ["IDH", "1p19q", "MGMT"]
    assert [m.finding for m in cfg.active] == ["IDH", "1p19q"]
    assert cfg.marker("IDH").max_error == 0.02
    with pytest.raises(ValueError):
        load_config(None, markers=["IDH", "BRAF"])
    with pytest.raises(ValueError):
        load_config(None, min_tile=3)                       # misspelt key is refused, not ignored
    with pytest.raises(ValueError):
        load_config(None, overrides={"IDH": {"max_err": 0.1}})


def test_gate_refuses_always_negative_on_rare_marker():
    # 3% altered; a model that says "wildtype" for everyone with high confidence is 97% right
    y = np.array([1] * 30 + [0] * 970)
    conf, pred = np.full(1000, 0.99), np.zeros(1000, int)
    g = class_threshold(conf, pred, y, 0)
    assert g["threshold"] == NEVER and g["n_accepted"] == 0


def test_gate_needs_enough_patients_of_the_other_class():
    rng = np.random.default_rng(0)
    for n_alt, ok in ((30, False), (80, True)):           # zero misses: bound <= 5% needs 59 or more
        y = np.array([1] * n_alt + [0] * 300)
        pred = y.copy()                                     # a perfect classifier
        conf = rng.uniform(0.9, 1.0, len(y))
        g = class_threshold(conf, pred, y, 0)
        assert (g["threshold"] < 1.0) == ok
        if ok:
            assert g["n_accepted"] == 300 and g["miss_upper"] <= 0.05 and g["error_upper"] <= 0.05


def test_gate_stops_at_first_failure():
    # 100 right calls, then a run of wrong ones, then many right ones at lower confidence
    y = np.array([0] * 100 + [1] * 40 + [0] * 2000 + [1] * 200)
    pred = np.zeros(len(y), int)
    conf = np.concatenate([np.linspace(0.999, 0.95, 100), np.linspace(0.94, 0.90, 40),
                           np.linspace(0.89, 0.60, 2000), np.full(200, 0.55)])
    g = class_threshold(conf, pred, y, 0)
    # a "best of all cuts" search would accept 2,140 here (40 errors); stopping at the first failure does not
    assert 100 <= g["n_accepted"] < 110 and g["threshold"] > 0.93


def test_confident_calls_are_mostly_right(trained, held_out):
    test, tlabs, _ = held_out
    right = called = 0
    for p in test:
        for r in predict_patient(trained, p):
            if r["status"] == "confident":
                called += 1
                right += r["call"] == tlabs[p.study_id].get(r["finding"])
    assert called > 200
    assert right / called >= 0.95


def test_no_signal_and_untrainable_markers_order_the_test(trained, held_out):
    test, _, _ = held_out
    assert not trained.fits["H3G34"].trained                    # no mutant patients in the synthetic world
    for p in test[:10]:
        recs = {r["finding"]: r for r in predict_patient(trained, p)}
        assert recs["H3G34"]["status"] == "uncertain" and recs["H3G34"]["call"] == "indeterminate"
        assert recs["H3G34"]["order_test"] == "H3G34_testing"
        assert recs["MGMT"]["status"] != "confident"            # MGMT has no image signal here
        assert recs["MGMT"]["order_test"] == "MGMT_promoter_methylation"
        # H3 K27 is separable here but rare (about 10%): too few altered patients to accept either call
        assert recs["H3K27"]["status"] == "uncertain" and recs["H3K27"]["order_test"] == "H3K27M_IHC"
        assert "never met" in " ".join(recs["H3K27"]["reasons"])


def test_1p19q_follows_idh(trained, held_out):
    test, tlabs, _ = held_out
    for p in test:
        recs = {r["finding"]: r for r in predict_patient(trained, p)}
        if recs["IDH"]["status"] == "confident" and recs["IDH"]["call"] == "wildtype":
            assert recs["1p19q"]["status"] == "not_assessed"
        if recs["IDH"]["status"] != "confident":
            assert recs["1p19q"]["status"] != "confident"
        if recs["1p19q"]["status"] == "confident":
            assert recs["1p19q"]["value"]["prerequisite"] == {"finding": "IDH", "call": "mutant", "source": "ai"}
    p = test[0]
    r = {x["finding"]: x for x in predict_patient(trained, p, known={"IDH": "wildtype"})}
    assert r["1p19q"]["status"] == "not_assessed" and r["1p19q"]["order_test"] is None


def test_gates_encoder_tiles_and_ood(trained, held_out):
    test, _, _ = held_out
    p = test[0]
    other = type(p)(p.study_id, p.features, p.scan_index, p.coords_um, p.tile_um, p.scan_ids, "h-optimus-0@x")
    assert all(r["status"] != "confident" for r in predict_patient(trained, other))
    few = type(p)(p.study_id, p.features[:5], p.scan_index[:5], p.coords_um[:5], p.tile_um, p.scan_ids, p.encoder)
    assert all(r["status"] != "confident" for r in predict_patient(trained, few))
    far = type(p)(p.study_id, p.features + 25.0, p.scan_index, p.coords_um, p.tile_um, p.scan_ids, p.encoder)
    recs = predict_patient(trained, far)
    assert all(r["status"] != "confident" for r in recs)
    assert any("unlike the training" in " ".join(r["reasons"]) for r in recs)


def test_records_shape_and_evidence(trained, held_out):
    test, _, _ = held_out
    recs = predict_patient(trained, test[0])
    assert [r["finding"] for r in recs] == list(DEFAULT_MARKERS)
    for r in recs:
        assert r["contract"] == "entip.glioma_step/1" and r["step"] == "glioma_panel"
        assert r["call"] in CONTRACT_CALLS[r["finding"]] | {"indeterminate"}
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["model"]["model_id"].startswith("glioma-panel:")
        if r["status"] == "uncertain":
            assert r["order_test"] in TESTS and r["reasons"]
        if r["status"] == "confident":
            assert r["confidence"] >= r["threshold"]
            cal = r["calibration"]
            assert cal["method"] == "temperature" and cal["validated"] and cal["error_upper"] <= cal["max_error"]
            assert 0 < cal["coverage"] <= 1 and cal["n_heldout"] > 0
            ev = r["evidence"]
            assert ev and all(e["kind"] == "tile" and e["id"].split(":tile:")[0] in r["scan_ids"] for e in ev)
            assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
        for banned in ("age", "site", "location", "sex", "radiology", "imaging"):
            assert banned not in json.dumps(list(r.keys()))
    if validate(recs) is None:
        pytest.skip("entip_dx not installed; contract checked by hand above")


def test_save_load_roundtrip(trained, held_out, tmp_path):
    test, _, _ = held_out
    trained.save(tmp_path / "m")
    m2 = PanelModel.load(tmp_path / "m")
    a = predict_patient(trained, test[3])
    b = predict_patient(m2, test[3])
    assert [(r["call"], r["status"]) for r in a] == [(r["call"], r["status"]) for r in b]
    assert np.allclose([r["confidence"] for r in a], [r["confidence"] for r in b])


def test_db_actions(trained, held_out):
    test, _, _ = held_out
    recs = predict_patient(trained, test[0])
    a = step_action(trained, test[0].study_id, recs)
    assert a["verb"] == "glioma_step" and a["detail"]["study_id"] == test[0].study_id and "records" not in a
    mod = model_actions(trained, uri="file:///models/panel")
    assert mod["verb"] == "upsert" and mod["object_type"] == "model"
    ids = {r["model_id"] for r in mod["records"]}
    assert "glioma-panel:IDH" in ids and "glioma-panel:H3G34" not in ids
    json.dumps(mod)
    s = summarise(recs)
    assert set(s) == {"confident", "order", "not_assessed"}


def test_cbioportal_labels(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("#Patient Identifier\tx\n"
                 "PATIENT_ID\tSAMPLE_ID\tIDH_STATUS\tIDH_CODEL_SUBTYPE\tATRX_STATUS\tMGMT_PROMOTER_STATUS\t"
                 "CHR_7_GAIN_CHR_10_LOSS\tTERT_PROMOTER_STATUS\n"
                 "TCGA-AA-0001\tTCGA-AA-0001-01\tMutant\tIDHmut-codel\tWT\tMethylated\tNo combined CNA\tMutant\n"
                 "TCGA-AA-0002\tTCGA-AA-0002-01\tWT\tIDHwt\tWT\tUnmethylated\tGain chr 7 & loss chr 10\tNA\n"
                 "TCGA-BB-0003\tTCGA-BB-0003-01\tMutant\tIDHmut-non-codel\tMutant\tweird\tNo combined CNA\tWT\n")
    rep = L.Counter()
    lab = L.from_cbioportal_pan_glioma(p, rep)
    assert lab["TCGA-AA-0001"] == {"IDH": "mutant", "1p19q": "codeleted", "ATRX": "retained", "MGMT": "methylated",
                                   "chr7_chr10": "absent", "TERT": "mutant"}
    assert "1p19q" not in lab["TCGA-AA-0002"] and "TERT" not in lab["TCGA-AA-0002"]
    assert lab["TCGA-BB-0003"]["ATRX"] == "lost" and "MGMT" not in lab["TCGA-BB-0003"]
    assert rep["MGMT_PROMOTER_STATUS:unrecognised:weird"] == 1


def test_long_labels_and_requirements(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("patient_id,finding,call\nP1,TP53,mutant\nP1,1p19q,codeleted\nP2,EGFR,amplified\nP2,EGFR2,x\n")
    rep = L.Counter()
    lab = L.from_long_table(p, rep)
    assert lab == {"P1": {"TP53": "mutant", "1p19q": "codeleted"}, "P2": {"EGFR": "amplified"}}
    assert L.apply_requirements(lab)["P1"] == {"TP53": "mutant"}          # 1p/19q needs IDH-mutant
    assert L.tcga_patient("TCGA-02-0001-01Z-00-DX1.83fce43e.svs") == "TCGA-02-0001"
    assert L.tcga_site("TCGA-02-0001-01Z-00-DX1.svs") == "02"
