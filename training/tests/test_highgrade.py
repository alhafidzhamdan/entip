"""Step 2 (high-grade features) on synthetic slides only."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from entip_train.encoders import HandcraftedEncoder
from entip_train.highgrade.actions import build_actions
from entip_train.highgrade.assess import assess_specimen
from entip_train.highgrade.bags import TileBag, embed_bag, load_bag, save_bag
from entip_train.highgrade.contract import CALLS, CONTRACT
from entip_train.highgrade.mitoses import MitosisConfig, MitosisCounter, p_at_least
from entip_train.highgrade.regions import RegionConfig, RegionDetector, tile_labels
from entip_train.highgrade.synthetic import synthetic_gbm

ENC = HandcraftedEncoder()
FEATS = {"necrosis": "necrosis_polys", "microvascular_proliferation": "mvp_polys"}


def _bag(**kw):
    s = synthetic_gbm(tiles=12, **kw)
    return s, embed_bag(s.scan, s.reader, ENC)


@pytest.fixture(scope="module")
def train_set():
    out = [_bag(scan_id=f"SCN-SYN-TR-{i}", seed=i, necrosis=1, mvp=3) for i in range(3)]
    out.append(_bag(scan_id="SCN-SYN-TR-3", seed=3, necrosis=0, mvp=0))
    return out


@pytest.fixture(scope="module")
def detectors(train_set):
    dets = {}
    for f, attr in FEATS.items():
        labels = [tile_labels(b, getattr(s, attr), s.tumour_polys) for s, b in train_set]
        dets[f] = RegionDetector(RegionConfig(f, validated=True)).fit([b for _, b in train_set], labels)
    return dets


# ---------------------------------------------------------------- mitoses

def test_p_at_least_matches_brute_force():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 7)
    for k in range(0, 5):
        brute = sum(np.prod([q if b else 1 - q for q, b in zip(p, bits)])
                    for bits in itertools.product([0, 1], repeat=len(p)) if sum(bits) >= k)
        assert p_at_least(p, k) == pytest.approx(brute, abs=1e-12)


@pytest.mark.parametrize("ps, call", [([0.95, 0.97, 0.9] + [0.01] * 30, "present"),
                                      ([0.02] * 40, "absent"),
                                      ([0.6, 0.6] + [0.01] * 10, "indeterminate")])
def test_mitotic_activity_calls(ps, call):
    cands = [{"id": f"c{i}", "scan_id": "S", "x_um": float(i), "y_um": 0.0, "p": p} for i, p in enumerate(ps)]
    r = MitosisCounter(MitosisConfig(validated=True)).assess(cands, ["S"], tissue_mm2=10.0)
    assert r.call == call
    assert (r.status == "confident") == (call != "indeterminate")
    assert (r.order_test == "pathologist_review") == (call == "indeterminate")
    assert r.evidence[0]["weight"] == max(ps)


def test_small_biopsy_lowers_cutoff_and_unvalidated_is_uncertain():
    cands = [{"id": "c0", "scan_id": "S", "x_um": 0.0, "y_um": 0.0, "p": 0.97}]
    big = MitosisCounter(MitosisConfig(validated=True, small_biopsy_mm2=2.0)).assess(cands, ["S"], 20.0)
    small = MitosisCounter(MitosisConfig(validated=True, small_biopsy_mm2=2.0)).assess(cands, ["S"], 1.0)
    assert big.call == "absent" and small.call == "present" and small.value["min_count"] == 1
    default = MitosisCounter().assess(cands, ["S"], 1.0)
    assert default.status == "uncertain" and default.order_test == "pathologist_review"


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


def test_features_are_specific(detectors):
    _, b = _bag(scan_id="SCN-SYN-T3", seed=103, necrosis=1, mvp=0)
    nec, _ = detectors["necrosis"].assess([b])
    mvp, _ = detectors["microvascular_proliferation"].assess([b])
    assert nec.call == "present" and mvp.call == "absent"


def test_guards(detectors, train_set):
    det = detectors["necrosis"]
    _, b = train_set[0]
    with pytest.raises(ValueError):
        det.assess([TileBag("X", b.features, b.coords_um, b.tile_um, b.mpp, "h-optimus-0@other")])
    tiny = TileBag("X", b.features[:5], b.coords_um[:5], b.tile_um, b.mpp, b.encoder_version)
    r, _ = det.assess([tiny])
    assert r.status == "uncertain" and any("tissue tiles" in x for x in r.reasons)
    rng = np.random.default_rng(0)
    odd = TileBag("X", b.features + rng.normal(0, 50, b.features.shape), b.coords_um, b.tile_um, b.mpp,
                  b.encoder_version)
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
    b2 = load_bag(save_bag(b, tmp_path / "b.npz"))
    assert np.allclose(b2.features, b.features) and b2.tile_ids() == b.tile_ids()


# ---------------------------------------------------------------- contract and database actions

def _keys(o):
    if isinstance(o, dict):
        return set(o) | set().union(*(_keys(v) for v in o.values()))
    if isinstance(o, list):
        return set().union(*(_keys(v) for v in o)) if o else set()
    return set()


DB_SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema" / "schema.json"


def test_specimen_records_and_actions(detectors):
    s1, b1 = _bag(scan_id="SCN-SYN-0001-A1-01", seed=201, necrosis=1, mvp=2)
    s2, b2 = _bag(scan_id="SCN-SYN-0001-A2-01", seed=202, necrosis=0, mvp=0)
    cands = s1.candidates + s2.candidates
    a = assess_specimen("SYN-0001", [b1, b2], list(detectors.values()),
                        MitosisCounter(MitosisConfig(validated=True)), cands)
    recs = {r["finding"]: r for r in a.records()}
    assert set(recs) == {"necrosis", "microvascular_proliferation", "mitotic_activity"}
    for r in recs.values():
        assert r["contract"] == CONTRACT and r["step"] == "high_grade_features" and r["source"] == "ai"
        assert r["call"] in CALLS and 0 <= r["confidence"] <= 1 and r["model"]["model_id"]
        assert r["scan_ids"] == ["SCN-SYN-0001-A1-01", "SCN-SYN-0001-A2-01"]
        assert all(set(e) >= {"kind", "id"} for e in r["evidence"])
        assert not _keys(r) & {"age", "site", "location", "sex", "chi", "nhs_number", "dob", "imaging"}
    assert recs["necrosis"]["call"] == "present" and recs["mitotic_activity"]["call"] == "present"

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


def test_records_pass_step5_validator(detectors):
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
    from entip_train.highgrade.actions import post_actions
    s, b = _bag(scan_id="SCN-0001-A1-01", seed=401, necrosis=1, mvp=2)
    a = assess_specimen("SYN-0001", [b], list(detectors.values()), MitosisCounter(), s.candidates)
    res = post_actions(url, a.actions)
    assert len(res) == len(a.actions)
    with urllib.request.urlopen(f"{url}/actions?verb=glioma_step&limit=5000") as r:
        events = json.loads(r.read())
    assert any(e["action_id"] == res[-1]["action_id"] for e in events)
