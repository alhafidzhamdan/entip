"""Step 1 triage and the shared wsi/ pieces, on synthetic data only."""
import json

import numpy as np
import pytest

from entip_train.encoders import HandcraftedEncoder
from entip_train.slides import ArrayReader
from entip_train.triage.db import build_actions
from entip_train.triage.labels import CLASSES, derived, map_idc
from entip_train.triage.manifest import label_rows, prep_of
from entip_train.triage.model import TriageConfig, TriageModel, source_check
from entip_train.triage.synthetic import synthetic_cohort
from entip_train.wsi.abmil import ABMIL
from entip_train.wsi.bags import Bag, extract_bag
from entip_train.wsi.calibrate import ece, exact_upper, fit_temperature, selective_threshold
from entip_train.wsi.novelty import Novelty, pooled


# ------------------------------------------------------------------ wsi/abmil
def test_abmil_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    m = ABMIL(3, hidden=6, attn=4)
    m.mu, m.sd = np.zeros(5), np.ones(5)
    m._init(5, rng)
    x = rng.normal(size=(7, 5))
    _, g = m._loss_grad(x, 2, 1.0)
    for k, P in m.p.items():
        i = tuple(rng.integers(0, s) for s in P.shape)
        o = P[i]
        P[i] = o + 1e-6
        l1, _ = m._loss_grad(x, 2, 1.0)
        P[i] = o - 1e-6
        l2, _ = m._loss_grad(x, 2, 1.0)
        P[i] = o
        assert abs((l1 - l2) / 2e-6 - g[k][i]) < 1e-6, k


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    bags = []
    for c in y:
        x = rng.normal(size=(int(rng.integers(20, 40)), 8))
        x[:3, 0] += 4 * (1 if c else -1)
        bags.append(x)
    return bags, y


def test_abmil_learns_saves_and_masks(tmp_path):
    bags, y = _toy()
    m = ABMIL(2, hidden=16, attn=8, epochs=10).fit(bags[:45], y[:45])
    acc = (m.predict_proba(bags[45:]).argmax(1) == y[45:]).mean()
    assert acc > 0.8
    att = m.attention(bags[50])
    assert att.shape == (len(bags[50]),) and abs(att.sum() - 1) < 1e-9
    m2 = ABMIL.load(m.save(tmp_path / "m.npz"))
    assert np.allclose(m.predict_proba(bags[45:]), m2.predict_proba(bags[45:]))
    # weight 0 on every class-1 bag: the model never learns class 1
    w = (y[:45] == 0).astype(float)
    m3 = ABMIL(2, hidden=16, attn=8, epochs=5).fit(bags[:45], y[:45], sample_weight=w)
    assert (m3.predict_proba(bags[45:]).argmax(1) == 0).all()


# ------------------------------------------------------------------ wsi/calibrate
def test_temperature_and_ece():
    rng = np.random.default_rng(1)
    true = rng.normal(0, 2, (2000, 3))
    p = np.exp(true) / np.exp(true).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=q) for q in p])
    t = fit_temperature(true * 3.0, y)          # over-confident logits: T should come back near 3
    assert 2.4 < t < 3.6
    assert ece(p, y) < 0.05


def test_selective_threshold():
    conf = np.linspace(0.5, 1.0, 400)
    correct = conf > 0.7
    g = selective_threshold(conf, correct, max_error=0.05, min_accepted=20)
    assert g["threshold"] > 0.65 and g["error_upper"] <= 0.05 and 0 < g["coverage"] < 1
    never = selective_threshold(conf, np.zeros(400, bool), max_error=0.05)
    assert never["threshold"] > 1.0 and never["coverage"] == 0
    assert exact_upper(0, 58) > 0.05 > exact_upper(0, 59)      # 59 error-free cases are needed for 5%
    assert exact_upper(0, 0) == 1.0
    # an early error stops the search: a later, luckier stretch cannot rescue it
    conf = np.linspace(1.0, 0.5, 400)
    correct = np.ones(400, bool)
    correct[60:70] = False
    g = selective_threshold(conf, correct, max_error=0.05)
    assert g["n_accepted"] <= 60


def test_selective_threshold_keeps_its_promise():
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(200):
        conf = rng.uniform(0.5, 1, 400)
        correct = rng.random(400) > 1.2 * (1 - conf) ** 1.5
        g = selective_threshold(conf, correct, 0.05)
        if g["n_accepted"]:
            c = np.linspace(g["threshold"], 1, 2000)
            bad += np.mean(1.2 * (1 - c) ** 1.5) > 0.05         # true error among accepted cases
    assert bad / 200 <= 0.05


def test_novelty_flags_far_slides():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 60)
    Z = rng.normal(0, 1, (120, 20)) + y[:, None] * 3
    n = Novelty(n_components=8, quantile=0.99).fit(Z[::2], y[::2])
    n.fit_threshold(n.distance(Z[1::2]))
    assert n.is_novel(Z[1::2]).mean() <= 0.05
    assert n.is_novel(rng.normal(0, 1, (30, 20)) - 8).all()
    m = Novelty.from_state(n.state())
    assert np.allclose(m.distance(Z[:5]), n.distance(Z[:5]))


def test_abmil_takes_float16_bags_without_copying_them():
    bags, y = _toy()
    b16 = [b.astype(np.float16) for b in bags]
    m = ABMIL(2, hidden=16, attn=8, epochs=8).fit(b16[:45], y[:45])
    assert (m.predict_proba(b16[45:]).argmax(1) == y[45:]).mean() > 0.8
    assert all(b.dtype == np.float16 for b in b16)


# ------------------------------------------------------------------ wsi/bags
def test_extract_bag_skips_background_and_roundtrips(tmp_path, slide):
    rgb = np.full((1024, 2048, 3), 255, np.uint8)          # right half blank glass
    rgb[:, :1024] = slide.reader.rgb[:1024, :1024]
    r = ArrayReader(rgb, 0.5)
    enc = HandcraftedEncoder()
    bag = extract_bag(r, "syn-a", 1024.0, 512.0, enc, tile_um=64.0)
    assert len(bag) > 0 and (bag.coords_um[:, 0] < 512).all()  # no tile from the blank half
    assert bag.features.shape == (len(bag), enc.dim)
    assert bag.sharpness.shape == (len(bag),)
    small = Bag.load(bag.save(tmp_path / "b.npz"), max_tiles=5, dtype=np.float16)
    assert len(small) == 5 and small.features.dtype == np.float16 and small.sharpness.shape == (5,)
    b2 = Bag.load(tmp_path / "b.npz")
    assert b2.scan_id == "syn-a" and b2.encoder == enc.version and np.allclose(b2.features, bag.features, atol=1e-2)


def test_pyramid_reader_pads_and_picks_levels():
    from entip_train.slides import _LevelReader

    class Two(_LevelReader):
        mpp, width_px, height_px, levels = 0.5, 400, 400, [(0, 1.0), (1, 4.0)]
        calls = []

        def _read(self, key, x, y, w, h):
            self.calls.append(key)
            return np.full((h, w, 3), 100, np.uint8)

    r = Two()
    t = r.read_um(56, 56, 112, 224)                       # native field: finest level, no padding
    assert t.shape == (224, 224, 3) and (t == 100).all() and r.calls[-1] == 0
    o = r.read_um(100, 100, 400, 100)                     # 8x shrink: coarse level; half the field is off-slide
    assert r.calls[-1] == 1 and o.shape == (100, 100, 3)
    assert (o[:, :20] == 255).all() and (o[80:] == 255).all() and (o[30:70, 30:70] == 100).all()
    assert (r.width_um, r.height_um) == (200.0, 200.0)


# ------------------------------------------------------------------ labels / manifest
@pytest.mark.parametrize("col,site,dx,want", [
    ("gtex", "Cerebral cortex", None, "normal_brain"),
    ("gtex", "Pituitary", None, None),
    ("tcga_gbm", "Brain", float("nan"), "glioma"),
    ("icdc_glioma", "Brain", None, None),
    ("ccdi_mci", "Brain", "Pilocytic astrocytoma", "glioma"),
    ("ccdi_mci", "Brain", "Medulloblastoma, NOS", "other_primary_cns"),
    ("ccdi_mci", "Brain", "Neoplasm, malignant", None),
    ("ccdi_mci", "Brain", "Neoplasm, metastatic", "metastasis"),
    ("ccdi_mci", "Brain", "Neoplasm, malignant, uncertain whether primary or metastatic", None),
    ("hcmi_cmdc", "Brain", "Adenocarcinoma, NOS", "metastasis"),
    ("cgci_htmcp_dlbcl", "Brain", "Diffuse large B-cell lymphoma, NOS", "lymphoma"),
    ("cgci_blgsp", "Lymph node", "Burkitt lymphoma, NOS", None),
])
def test_map_idc(col, site, dx, want):
    assert map_idc(col, site, dx) == want


def test_label_rows_filters():
    rows = [{"collection_id": "tcga_gbm", "PatientID": "p1", "license_short_name": "CC BY 3.0", "site": "Brain",
             "dx": None, "fixative": "Formalin", "embedding": "Paraffin wax", "Manufacturer": "Aperio"},
            {"collection_id": "tcga_gbm", "PatientID": "p1", "license_short_name": "CC BY 3.0", "site": "Brain",
             "dx": None, "fixative": "", "embedding": "Tissue freezing medium", "Manufacturer": "Aperio"},
            {"collection_id": "x", "PatientID": "p2", "license_short_name": "CC BY-NC 4.0", "site": "Brain",
             "dx": "Glioblastoma", "fixative": "", "embedding": "", "Manufacturer": ""}]
    kept, dropped = label_rows(rows, preps=("ffpe",))
    assert [r["prep"] for r in kept] == ["ffpe"] and kept[0]["group"] == "tcga_gbm|Aperio|ffpe"
    assert dropped == {"prep": 1, "licence": 1}
    assert prep_of("PAXgene Tissue System", "Paraffin wax") == "paxgene_paraffin"


def test_derived_probabilities():
    p = np.zeros(len(CLASSES))
    p[CLASSES.index("glioma")], p[CLASSES.index("metastasis")], p[CLASSES.index("normal_brain")] = .6, .2, .2
    d = derived(p)
    assert d["p_tumour"] == pytest.approx(0.8) and d["p_glial_given_tumour"] == pytest.approx(0.75)


# ------------------------------------------------------------------ model
@pytest.fixture(scope="module")
def trained():
    bags, labels, groups, patients = synthetic_cohort(seed=0)
    # one lymphoma slide only: below min_patients, so the class is left out and reported
    lb, ll, lg, lp = synthetic_cohort({"lymphoma": 2}, seed=5)
    m = TriageModel(TriageConfig(epochs=10, folds=3, max_error=0.1, min_accepted=10))
    rep = m.fit(bags + lb, labels + ll, groups + lg, patients + lp)
    return m, rep, bags, labels


def test_fit_report(trained):
    m, rep, _, _ = trained
    assert "lymphoma" in rep["classes_excluded"] and "reactive" in rep["classes_excluded"]
    cv = rep["cv"]["calibrated"]
    assert cv["balanced_accuracy"] > 0.8 and cv["auroc_tumour"] > 0.9
    assert rep["source_check"]["share_of_slides_in_single_class_sources"] == 1.0
    assert len(rep["leave_one_source_out"]) == 8
    assert m.model_id.startswith("cns-triage-")


def test_records_follow_step5_contract(trained, tmp_path):
    m, _, bags, labels = trained
    i = labels.index("glioma")
    recs = m.records([bags[i], bags[i + 1]], "SYN-0001")
    assert [r["finding"] for r in recs] == ["adequacy", "tissue_class"]
    tc = recs[1]
    assert set(tc["probabilities"]) == {"normal_brain", "glioma", "other_primary_cns", "metastasis"}
    cal = tc["calibration"]
    assert cal["method"] == "temperature" and cal["calibrated"] and cal["n_heldout"] == m.report["n_patients"]
    assert cal["validated"] == (m.threshold <= 1.0) and cal["max_error"] == m.cfg.max_error
    assert recs[0]["calibration"] == {"method": "rule", "calibrated": False, "validated": False}
    assert tc["status"] in ("confident", "uncertain") and len(tc["evidence"]) == m.cfg.top_k
    assert '"site"' not in json.dumps(recs) and '"age"' not in json.dumps(recs)
    if tc["call"] != "glioma" or tc["status"] == "uncertain":
        assert tc["order_test"] == "pathologist_review"
    contract = pytest.importorskip("entip_dx.contract")
    for r in recs:
        contract.validate_finding(r)
    m2 = TriageModel.load(m.save(tmp_path / "model"))
    assert m2.records([bags[i]], "SYN-0001") == m.records([bags[i]], "SYN-0001")


def test_most_abnormal_slide_decides_and_disagreement_is_flagged(trained):
    m, _, bags, labels = trained
    normal = [bags[i] for i, l in enumerate(labels) if l == "normal_brain"][:3]
    tumour = bags[labels.index("glioma")]
    rec = m.records(normal + [tumour], "SYN-0004")[1]
    assert rec["call"] == "glioma" and len(rec["value"]["per_scan"]) == 4
    assert rec["evidence"][0]["id"].startswith(tumour.scan_id)
    other = bags[labels.index("metastasis")]
    rec = m.records([tumour, other], "SYN-0005")[1]
    if all(s["confidence"] >= m.threshold for s in rec["value"]["per_scan"]):
        assert rec["status"] == "uncertain" and rec["order_test"] == "pathologist_review"


def test_empty_scan_does_not_misalign_evidence(trained):
    m, _, bags, labels = trained
    b = bags[labels.index("glioma")]
    empty = Bag("SCN-EMPTY", np.zeros((0, b.features.shape[1]), np.float32), np.zeros((0, 2)), 112.0, b.encoder,
                np.zeros(0, np.float32))
    rec = m.records([empty, b], "SYN-0006")[1]
    assert rec["scan_ids"] == [b.scan_id] and all(e["id"].startswith(b.scan_id) for e in rec["evidence"])


def test_novel_slide_and_never_accepting_model_go_to_pathologist(trained):
    m, _, bags, labels = trained
    b = bags[labels.index("glioma")]
    far = Bag("SCN-FAR", b.features + 25.0, b.coords_um, b.tile_um, b.encoder, b.tissue_frac)
    rec = m.records([far], "SYN-0007")[1]
    assert rec["value"]["novel"] and rec["status"] == "uncertain" and rec["order_test"] == "pathologist_review"
    old = m.threshold
    m.threshold = 1.01
    try:
        rec = m.records([b], "SYN-0008")[1]
        assert rec["status"] == "uncertain" and "never accepts" in " ".join(rec["reasons"])
    finally:
        m.threshold = old


def test_unseen_classes_are_reported(trained):
    _, rep, _, _ = trained
    u = rep["unseen_classes"]["lymphoma"]
    assert u["n"] == 2 and sum(u["called"].values()) == 2 and "lymphoma" not in u["called"]
    assert rep["cv"]["gate"]["unit"].startswith("patient") and rep["cv"]["gate"]["n"] == rep["n_patients"]


def test_non_glial_confident_call_goes_to_pathologist(trained):
    m, _, bags, labels = trained
    rec = m.records([bags[labels.index("metastasis")]], "SYN-0002")[1]
    assert rec["order_test"] == "pathologist_review"


def test_source_check_sees_strong_source_offsets():
    strong = synthetic_cohort({"glioma": 30, "normal_brain": 12}, source_shift=3.0, seed=2)
    weak = synthetic_cohort({"glioma": 30, "normal_brain": 12}, source_shift=0.0, seed=2)
    s = source_check([b.features for b in strong[0]], *strong[1:])["per_class"]["glioma"]
    w = source_check([b.features for b in weak[0]], *weak[1:])["per_class"]["glioma"]
    assert s["source_balanced_accuracy"] > 0.9 and w["source_balanced_accuracy"] < 0.75


def test_db_actions(trained):
    from entip_train.triage.db import register_actions
    m, _, bags, labels = trained
    acts = build_actions(m, [bags[0]], "SYN-0003")
    assert [(a["verb"], a.get("object_type")) for a in acts] == [
        ("upsert", "region"), ("upsert", "prediction"), ("glioma_step", None)]   # never the model row
    regions = {r["region_id"]: r for r in acts[0]["records"]}
    assert {r["kind"] for r in regions.values()} == {"roi", "tile"}
    assert all(p["target_type"] == "region" and p["target_id"] in regions for p in acts[1]["records"])
    ev = [e["id"] for r in acts[2]["detail"]["records"] for e in r["evidence"] if e["kind"] == "tile"]
    assert ev and all(e in regions for e in ev)
    reg = register_actions(m, uri="file:///models/triage")
    assert [a["object_type"] for a in reg] == ["feature", "model"] and reg[1]["records"][0]["status"] == "draft"


def test_post_patient_registers_a_model_once(trained, monkeypatch):
    from entip_train.triage import db
    m, _, bags, _ = trained
    sent = []
    monkeypatch.setattr(db, "post_actions", lambda url, acts: sent.append([a.get("object_type") for a in acts]))
    monkeypatch.setattr(db, "model_registered", lambda url, mid: False)
    db.post_patient("http://db", m, [bags[0]], "SYN-0009")
    monkeypatch.setattr(db, "model_registered", lambda url, mid: True)
    db.post_patient("http://db", m, [bags[0]], "SYN-0009")
    assert sent[0][:2] == ["feature", "model"] and "model" not in sent[1]


def test_build_bags_dry_run_counts_what_is_left(tmp_path, capsys):
    from entip_train.cli import main
    from entip_train.triage.manifest import FIELDS, write_manifest
    rows = [dict.fromkeys(FIELDS, "") | {"SeriesInstanceUID": f"1.2.{i}", "label": "glioma", "group": "g",
                                         "PatientID": f"p{i}", "series_size_MB": "500"} for i in range(4)]
    write_manifest(rows, tmp_path / "m.csv")
    (tmp_path / "bags").mkdir()
    (tmp_path / "bags" / "1.2.0.npz").write_bytes(b"")
    assert main(["triage", "build-bags", "--manifest", str(tmp_path / "m.csv"), "--out", str(tmp_path / "bags"),
                 "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"slides": 4, "already_done": 1, "to_do": 3, "download_GB": 1.5}


def test_cli_demo(capsys):
    from entip_train.cli import main
    assert main(["triage", "demo", "--epochs", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["actions"][-1] == "glioma_step:event"
