import numpy as np

from entip_train.retrain import GateConfig, expected_calibration_error, gate


def mark(svc, slide, hid, n_pos, n_neg, seed, locked=None):
    rng = np.random.default_rng(seed)
    pos = [c for c in slide.cells if slide.cell_truth[c.cell_id]]
    neg = [c for c in slide.cells if not slide.cell_truth[c.cell_id]]
    for c in rng.choice(pos, n_pos, replace=False):
        svc.add_label(hid, slide.scan.scan_id, 1, "t", cell_id=c.cell_id, locked=locked, train=False)
    for c in rng.choice(neg, n_neg, replace=False):
        svc.add_label(hid, slide.scan.scan_id, 0, "t", cell_id=c.cell_id, locked=locked, train=False)


def test_first_round_promotes_then_gate_compares(svc, slide, slide_b):
    hid = svc.create_head("atypical", "cell", locked_scans=["syn-b"])
    mark(svc, slide, hid, 15, 15, 0)
    mark(svc, slide_b, hid, 10, 20, 1)                      # syn-b is locked: test only
    assert svc.head_info(hid)["n_locked"] == 30
    r1 = svc.retrain(hid, "t")
    assert r1["status"] == "champion" and r1["n_locked"] == 30 and r1["n_train"] == 30
    r2 = svc.retrain(hid, "t")
    assert "not retrained" in r2["reasons"][0]              # no new marks
    mark(svc, slide, hid, 10, 10, 5)
    r3 = svc.retrain(hid, "t", kind="hgb", force=True)  # some seed-5 marks repeat earlier ones
    assert r3["status"] in ("champion", "rejected") and r3["champion"] is not None
    statuses = [m.status for m in svc.store.models(hid) if m.kind.startswith("retrain")]
    assert statuses.count("champion") == 1                  # exactly one live champion


def test_locked_set_too_small_skips(svc, slide):
    hid = svc.create_head("z", "cell")
    mark(svc, slide, hid, 15, 15, 0)
    r = svc.retrain(hid, "t")
    assert r["challenger"] is None and "locked set too small" in r["reasons"][0]


def test_gate_rules():
    cfg = GateConfig(tol_overall=0.0, tol_group=0.05, tol_ece=0.02)
    champ = {"balanced_accuracy": 0.9, "ece": 0.05, "per_group": {"a": 0.9, "b": 0.9}}
    ok, _ = gate(champ, {"balanced_accuracy": 0.92, "ece": 0.06, "per_group": {"a": 0.95, "b": 0.86}}, cfg)
    assert ok
    ok, why = gate(champ, {"balanced_accuracy": 0.95, "ece": 0.05, "per_group": {"a": 1.0, "b": 0.8}}, cfg)
    assert not ok and why[0].startswith("b:")
    ok, why = gate(champ, {"balanced_accuracy": 0.95, "ece": 0.2, "per_group": {}}, cfg)
    assert not ok and "ECE" in why[0]
    ok, _ = gate(None, {"balanced_accuracy": 0.6, "ece": 0.1, "per_group": {}}, GateConfig(min_locked_ba=0.7))
    assert not ok


def test_ece():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(np.full(100, 0.5), y) < 1e-9
    assert abs(expected_calibration_error(np.where(y == 1, 0.0, 1.0), y) - 1.0) < 1e-9
