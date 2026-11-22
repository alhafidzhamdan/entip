"""Step 3 (high-grade features) on synthetic slides only."""
import json
from pathlib import Path

import numpy as np
import pytest

from entip_train.encoders import HandcraftedEncoder
from entip_glioma.highgrade.actions import build_actions
from entip_glioma.highgrade.assess import assess_specimen
from entip_glioma.highgrade.bags import tile_ids
from entip_glioma.highgrade.contract import CALLS, CONTRACT
from entip_glioma.highgrade.regions import RegionConfig, RegionDetector, tile_labels
from entip_glioma.highgrade.synthetic import synthetic_gbm
from entip_glioma.wsi.bags import Bag, extract_bag

ENC = HandcraftedEncoder()
FEATS = {"necrosis": "necrosis_polys", "microvascular_proliferation": "mvp_polys"}


def _bag(**kw):
    s = synthetic_gbm(tiles=10, **kw)
    return s, extract_bag(s.reader, s.scan.scan_id, s.scan.width_um, s.scan.height_um, ENC)


def _like(b, scan_id="X", features=None, n=None, encoder=None):
    f = b.features if features is None else features
    return Bag(scan_id, f[:n], b.coords_um[:n], b.tile_um, encoder or b.encoder, b.tissue_frac[:n])


@pytest.fixture(scope="module")
def train_set():
    out = [_bag(scan_id=f"SCN-SYN-TR-{i}", seed=i, necrosis=1, mvp=3) for i in range(3)]
    out.append(_bag(scan_id="SCN-SYN-TR-3", seed=3, necrosis=0, mvp=0))
    return out


@pytest.fixture(scope="module")
def held_out():
    return [_bag(scan_id=f"SCN-SYN-CAL-{i}", seed=50 + i, necrosis=i % 2, mvp=2 * (i % 2)) for i in range(8)]


@pytest.fixture(scope="module")
def detectors(train_set, held_out):
    dets = {}
    for f, attr in FEATS.items():
        labels = [tile_labels(b, getattr(s, attr), s.tumour_polys) for s, b in train_set]
        det = RegionDetector(RegionConfig(f, validated=True)).fit([b for _, b in train_set], labels)
        det.calibrate_slides([[b] for _, b in held_out], [i % 2 for i in range(len(held_out))])
        dets[f] = det
    return dets


# ---------------------------------------------------------------- necrosis and MVP

@pytest.mark.parametrize("feature", list(FEATS))
def test_region_detector_finds_feature(detectors, feature):
    det = detectors[feature]
    s_pos, b_pos = _bag(scan_id="SCN-SYN-T1", seed=101, necrosis=1, mvp=3)
    s_neg, b_neg = _bag(scan_id="SCN-SYN-T2", seed=102, necrosis=0, mvp=0)
    pos, _ = det.assess([b_pos])
    neg, _ = det.assess([b_neg])
    assert pos.call == "present" and pos.status == "confident" and pos.confidence >= 0.8
    assert neg.call == "absent" and neg.status == "confident"
    assert pos.evidence and pos.evidence[0]["kind"] == "region"
    # the top focus overlaps the truth
    e = pos.evidence[0]
    truth = getattr(s_pos, FEATS[feature])
    cents = [np.asarray(t).mean(0) for t in truth]
    assert any(e["x_um"] - 112 <= cx <= e["x_um"] + e["w_um"] + 112 and
               e["y_um"] - 112 <= cy <= e["y_um"] + e["h_um"] + 112 for cx, cy in cents)


def test_slide_calibration(train_set, held_out):
    three = train_set[:3]                                        # every slide has both classes
    labels = [tile_labels(b, s.necrosis_polys, s.tumour_polys) for s, b in three]
    det = RegionDetector(RegionConfig("necrosis", validated=True)).fit([b for _, b in three], labels)
    assert "grouped by slide" in det.metrics["tile_calibration"]
    before = det.model_id
    r, _ = det.assess([held_out[1][1]])                          # not calibrated yet: never confident
    assert r.status == "uncertain" and any("not calibrated" in x for x in r.reasons)
    with pytest.raises(ValueError):
        det.calibrate_slides([[held_out[0][1]]], [0])
    info = det.calibrate_slides([[b] for _, b in held_out], [i % 2 for i in range(len(held_out))])
    assert info["threshold_from"] == "config" and det.model_id != before   # too few slides to set a threshold
    r, _ = det.assess([held_out[1][1]])
    assert r.status == "confident" and r.call == "present" and r.model_id == det.model_id
    cal = r.to_record("SYN-0001")["calibration"]                 # fitted, but too few slides to meet the target
    assert cal == {"method": "platt", "calibrated": True, "validated": False, "max_error": 0.05, "n_heldout": 8}


def test_threshold_set_from_held_out_specimens(train_set, held_out):
    """With 40+ held-out specimens the threshold comes from their error bound and the block reports it."""
    labels = [tile_labels(b, s.necrosis_polys, s.tumour_polys) for s, b in train_set]
    det = RegionDetector(RegionConfig("necrosis", min_accepted=20, max_error=0.15)).fit(
        [b for _, b in train_set], labels)
    many = [[b] for _, b in held_out] * 6                         # 48 specimens (repeats: mechanics only)
    info = det.calibrate_slides(many, [i % 2 for i in range(len(held_out))] * 6)
    cal = det.calibration_block()
    assert info["threshold_from"] == "held-out slides" and info["met_target"] and cal["validated"]
    assert cal["n_heldout"] == 48 and cal["error_upper"] <= 0.15 and 0 < cal["coverage"] <= 1
    assert det.threshold == info["threshold"] <= 1.0


def test_features_are_specific(detectors):
    _, b = _bag(scan_id="SCN-SYN-T3", seed=103, necrosis=1, mvp=0)
    nec, _ = detectors["necrosis"].assess([b])
    mvp, _ = detectors["microvascular_proliferation"].assess([b])
    assert nec.call == "present" and mvp.call == "absent"


def test_guards(detectors, train_set):
    det = detectors["necrosis"]
    _, b = train_set[0]
    with pytest.raises(ValueError):
        det.assess([_like(b, encoder="h-optimus-0@other")])
    tiny = _like(b, n=5)
    r, _ = det.assess([tiny])
    assert r.status == "uncertain" and any("tissue tiles" in x for x in r.reasons)
    rng = np.random.default_rng(0)
    odd = _like(b, features=b.features + rng.normal(0, 50, b.features.shape).astype(np.float32))
    r, _ = det.assess([odd])
    assert r.status == "uncertain" and any("unlike the training" in x for x in r.reasons)
    unvalidated = RegionDetector(RegionConfig("necrosis"))
    unvalidated.__dict__.update({k: v for k, v in det.__dict__.items() if k != "cfg"})
    r, _ = unvalidated.assess([b])
    assert r.status == "uncertain" and r.order_test == "pathologist_review"


def test_save_load_roundtrip(detectors, train_set, tmp_path):
    det = detectors["microvascular_proliferation"]
    _, b = train_set[1]
    det2 = RegionDetector.load(det.save(tmp_path / "mvp.joblib"))
    assert np.allclose(det.tile_p(b), det2.tile_p(b)) and det2.model_id == det.model_id
    b2 = Bag.load(b.save(tmp_path / "b.npz"))                    # the step-1 file format (float16 on disk)
    assert tile_ids(b2) == tile_ids(b) and np.allclose(det.tile_p(b2), det.tile_p(b), atol=0.02)
    assert det2.threshold == det.threshold and det2.slide_cal == det.slide_cal


# ---------------------------------------------------------------- contract and database actions

def _keys(o):
    if isinstance(o, dict):
        return set(o) | set().union(*(_keys(v) for v in o.values()))
    if isinstance(o, list):
        return set().union(*(_keys(v) for v in o)) if o else set()
    return set()


DB_SCHEMA = Path(__file__).resolve().parents[3] / "platform" / "db" / "schema" / "schema.json"


def test_specimen_records_and_actions(detectors):
    s1, b1 = _bag(scan_id="SCN-SYN-0001-A1-01", seed=201, necrosis=1, mvp=2)
    s2, b2 = _bag(scan_id="SCN-SYN-0001-A2-01", seed=202, necrosis=0, mvp=0)
    a = assess_specimen("SYN-0001", [b1, b2], list(detectors.values()))
    recs = {r["finding"]: r for r in a.records()}
    assert set(recs) == {"necrosis", "microvascular_proliferation"}   # no mitosis record until a new counter exists
    for r in recs.values():
        assert r["contract"] == CONTRACT and r["step"] == "high_grade_features" and r["source"] == "ai"
        assert r["call"] in CALLS and 0 <= r["confidence"] <= 1 and r["model"]["model_id"]
        assert r["scan_ids"] == ["SCN-SYN-0001-A1-01", "SCN-SYN-0001-A2-01"]
        assert all(set(e) >= {"kind", "id"} for e in r["evidence"])
        assert not _keys(r) & {"age", "site", "location", "sex", "chi", "nhs_number", "dob", "imaging"}
    assert recs["necrosis"]["call"] == "present"

    verbs = [(x["verb"], x["object_type"]) for x in a.actions]
    assert verbs[0] == ("upsert", "feature") and verbs[-1] == ("glioma_step", None)
    assert a.actions[-1]["detail"]["study_id"] == "SYN-0001" and not a.actions[-1]["records"]
    if DB_SCHEMA.exists():                                      # every field must be one the db accepts
        tables = json.loads(DB_SCHEMA.read_text())["tables"]
        extra = {"region": {"x_um", "y_um", "w_um", "h_um", "geometry"}, "cell": {"outline_um"}}
        for act in a.actions:
            if act["object_type"]:
                cols = set(tables[act["object_type"]]["columns"]) | extra.get(act["object_type"], set())
                for rec in act["records"]:
                    assert set(rec) <= cols, (act["object_type"], set(rec) - cols)
    ids = [r["region_id"] for x in a.actions if x["object_type"] == "region" for r in x["records"]]
    assert len(ids) == len(set(ids))


def test_records_pass_step6_validator(detectors):
    contract = pytest.importorskip("entip_dx.contract")
    _, b = _bag(scan_id="SCN-SYN-0002-A1-01", seed=301)
    a = assess_specimen("SYN-0002", [b], list(detectors.values()))
    for r in a.records():
        contract.validate_finding(r)


def test_post_to_live_db(detectors):
    """Runs only with ENTIP_TEST_DB_API set (a db API with the seeded synthetic patients)."""
    import os
    import urllib.request
    url = os.environ.get("ENTIP_TEST_DB_API")
    if not url:
        pytest.skip("set ENTIP_TEST_DB_API=http://127.0.0.1:8705 to run")
    from entip_glioma.highgrade.actions import post_actions
    s, b = _bag(scan_id="SCN-0001-A1-01", seed=401, necrosis=1, mvp=2)
    a = assess_specimen("SYN-0001", [b], list(detectors.values()))
    res = post_actions(url, a.actions)
    assert len(res) == len(a.actions)
    with urllib.request.urlopen(f"{url}/actions?verb=glioma_step&limit=5000") as r:
        events = json.loads(r.read())
    assert any(e["action_id"] == res[-1]["action_id"] for e in events)


# ---------------------------------------------------------------- sub-step training additions

def test_context_features_are_neighbourhood_means(train_set):
    from entip_glioma.highgrade.bags import context_features, grid
    _, b = train_set[0]
    X = context_features(b, (1,))
    assert X.shape == (len(b), 2 * b.features.shape[1])
    rc = grid(b)
    at = {tuple(v): i for i, v in enumerate(rc)}
    i = len(b) // 2
    nb = [at[(rc[i][0] + dr, rc[i][1] + dc)] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
          if (rc[i][0] + dr, rc[i][1] + dc) in at]
    assert np.allclose(X[i, b.features.shape[1]:], b.features[nb].mean(0), atol=1e-5)
    assert context_features(b, ()) is b.features


def test_context_head_trains_and_round_trips(train_set, tmp_path):
    labels = [tile_labels(b, s.necrosis_polys, s.tumour_polys) for s, b in train_set]
    det = RegionDetector(RegionConfig("necrosis", context=(1, 2))).fit([b for _, b in train_set], labels)
    s, b = train_set[0]
    p = det.tile_p(b)
    y = tile_labels(b, s.necrosis_polys, s.tumour_polys)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.3
    det2 = RegionDetector.load(det.save(tmp_path / "n.joblib"))
    assert np.allclose(det2.tile_p(b), p)


def test_train_cli_takes_negative_bags(train_set, tmp_path):
    import argparse
    from entip_glioma.highgrade.cli import highgrade_train
    (s, b), (_, neg) = train_set[0], train_set[3]
    b.save(tmp_path / "a.npz")
    neg.save(tmp_path / "neg.npz")
    (tmp_path / "a.json").write_text(json.dumps({"positives": s.mvp_polys, "negatives": []}))
    args = argparse.Namespace(bag=[str(tmp_path / "a.npz")], labels=[str(tmp_path / "a.json")],
                              negative_bag=[str(tmp_path / "neg.npz")], negatives_per_bag=30, context=[1],
                              seed=0, feature="microvascular_proliferation", threshold=0.8, training_data="t",
                              out=str(tmp_path / "m.joblib"))
    assert highgrade_train(args) == 0
    det = RegionDetector.load(tmp_path / "m.joblib")
    assert det.metrics["negative_bags"] == 1 and det.metrics["n_neg"] == 30
    assert det.cfg.context == (1,)
