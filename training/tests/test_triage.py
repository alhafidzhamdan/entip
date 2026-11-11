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
from entip_train.wsi.calibrate import ece, fit_temperature, selective_threshold, wilson_upper


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
    conf = np.linspace(0.5, 1.0, 200)
    correct = conf > 0.7
    g = selective_threshold(conf, correct, max_error=0.05, min_accepted=20)
    assert g["threshold"] > 0.6 and g["error_upper"] <= 0.05 and 0 < g["coverage"] < 1
    never = selective_threshold(conf, np.zeros(200, bool), max_error=0.05)
    assert never["threshold"] > 1.0 and never["coverage"] == 0
    assert wilson_upper(0, 100) < 0.03 and wilson_upper(0, 0) == 1.0


# ------------------------------------------------------------------ wsi/bags
def test_extract_bag_skips_background_and_roundtrips(tmp_path, slide):
    rgb = np.full((1024, 2048, 3), 255, np.uint8)          # right half blank glass
    rgb[:, :1024] = slide.reader.rgb[:1024, :1024]
    r = ArrayReader(rgb, 0.5)
    enc = HandcraftedEncoder()
    bag = extract_bag(r, "syn-a", 1024.0, 512.0, enc, tile_um=64.0)
    assert len(bag) > 0 and (bag.coords_um[:, 0] < 512).all()  # no tile from the blank half
    assert bag.features.shape == (len(bag), enc.dim)
    b2 = Bag.load(bag.save(tmp_path / "b.npz"))
    assert b2.scan_id == "syn-a" and b2.encoder == enc.version and np.allclose(b2.features, bag.features, atol=1e-2)


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
    assert tc["status"] in ("confident", "uncertain") and len(tc["evidence"]) == 2 * m.cfg.top_k
    assert '"site"' not in json.dumps(recs) and '"age"' not in json.dumps(recs)
    if tc["call"] != "glioma" or tc["status"] == "uncertain":
        assert tc["order_test"] == "pathologist_review"
    contract = pytest.importorskip("entip_dx.contract")
    for r in recs:
        contract.validate_finding(r)
    m2 = TriageModel.load(m.save(tmp_path / "model"))
    assert m2.records([bags[i]], "SYN-0001") == m.records([bags[i]], "SYN-0001")


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
    m, _, bags, labels = trained
    acts = build_actions(m, [bags[0]], "SYN-0003")
    assert [(a["verb"], a.get("object_type")) for a in acts] == [
        ("upsert", "feature"), ("upsert", "model"), ("upsert", "region"), ("upsert", "prediction"),
        ("glioma_step", None)]
    regions = {r["region_id"]: r for r in acts[2]["records"]}
    assert {r["kind"] for r in regions.values()} == {"roi", "tile"}
    assert all(p["target_type"] == "region" and p["target_id"] in regions for p in acts[3]["records"])
    ev = [e["id"] for r in acts[4]["detail"]["records"] for e in r["evidence"] if e["kind"] == "tile"]
    assert ev and all(e in regions for e in ev)
    assert acts[1]["records"][0]["status"] == "draft"


def test_cli_demo(capsys):
    from entip_train.cli import main
    assert main(["triage", "demo", "--epochs", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["actions"][-1] == "glioma_step:event"
