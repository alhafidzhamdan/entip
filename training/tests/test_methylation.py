"""Step 4 (methylation class from H&E) on synthetic data only."""
import csv
import json

import numpy as np
import pytest

from entip_train.methylation.classes import DEFAULT_CLASSES, Taxonomy
from entip_train.methylation.db import MARKER, build_actions
from entip_train.methylation.labels import ReferenceClassifier, cluster_labels, from_report_csv
from entip_train.methylation.model import MethylationModel
from entip_train.methylation.synthetic import synthetic_betas, synthetic_cases
from entip_train.methylation.tcga import pair_slides, site_folds, site_of
from entip_train.methylation.train import Case, train
from entip_train.wsi.bags import Bag

FORBIDDEN = ("age", "site", "location", "radiology", "imaging", "sex")


@pytest.fixture(scope="module")
def trained():
    data = synthetic_cases(DEFAULT_CLASSES, n_per_class=12, seed=0)
    cases = [Case(cid, [bag], c) for cid, bag, c in data]
    m = train(cases, Taxonomy(), "mc-test", "0.0.1", k=5, min_tiles=20, mil_kw={"epochs": 15}, seed=0)
    return m, data


def _validate(rec):
    dx = pytest.importorskip("entip_dx.contract")
    return dx.validate_finding(rec)


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


def test_pairing_keeps_diagnostic_primary_slides():
    slides = ["TCGA-06-0125-01Z-00-DX1.abc.svs", "TCGA-06-0125-01A-01-TS1.def.svs",
              "TCGA-02-0001-02Z-00-DX1.ghi.svs", "TCGA-12-3456-01Z-00-DX2.jkl.svs"]
    meth = ["TCGA-06-0125-01A-11D-xxxx-05.level3betas.txt", "TCGA-02-0001-01A-01D.txt"]
    pairs = pair_slides(slides, meth)
    assert [(p.case_id, p.slide) for p in pairs] == [("TCGA-06-0125", slides[0])]
    assert pairs[0].site == "06"


def test_site_folds_never_split_a_site():
    ids = [f"TCGA-{s:02d}-{i:04d}" for i in range(200) for s in [i % 13]]
    fold = site_folds(ids, k=5)
    by_site = {}
    for c, f in zip(ids, fold):
        by_site.setdefault(site_of(c), set()).add(f)
    assert all(len(v) == 1 for v in by_site.values())
    assert len(set(fold)) == 5


def test_reference_classifier_labels_and_refuses_low_coverage():
    classes = ["GBM, RTK II", "A IDH", "O IDH"]
    ids, cpgs, X, y = synthetic_betas(classes, n_per_class=15, seed=0)
    clf = ReferenceClassifier(n_cpgs=200, n_trees=100).fit(X, cpgs, y)
    tid, _, TX, ty = synthetic_betas(classes, n_per_class=5, seed=1)
    labs = clf.label(tid, TX, cpgs, min_score=0.6)
    assert len(labs) >= 12
    assert all(l.cls == t for l in labs for i, t in zip(tid, ty) if l.case_id == i)
    with pytest.raises(ValueError):
        clf.label(tid, TX[:, :50], cpgs[:50])


def test_report_csv_drops_low_scores(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("case_id,class,score\nTCGA-06-0001,GBM, RTK II,0.95\n".replace("GBM, RTK II", '"GBM, RTK II"')
                 + "TCGA-06-0002,A IDH,0.4\n")
    labs = from_report_csv(p, 0.9)
    assert [(l.case_id, l.cls) for l in labs] == [("TCGA-06-0001", "GBM, RTK II")]


def test_clusters_are_unnamed():
    ids, _, X, _ = synthetic_betas(["a", "b"], n_per_class=10)
    labs = cluster_labels(ids, X, k=2)
    assert {l.cls for l in labs} == {"cluster_0", "cluster_1"}


def test_training_metrics_and_thresholds(trained):
    m, _ = trained
    fam = m.metrics["family"]
    assert fam["accuracy"] > 0.8
    assert m.metrics["cv"].startswith("grouped")
    # families are easier than subclasses on this data, as on real slides
    assert fam["accuracy"] >= m.metrics["class"]["accuracy"]
    assert 0 < m.family_threshold <= 1.01


def test_record_follows_step5_contract(trained):
    m, data = trained
    rec = m.predict([data[0][1]], "SYN-0001")
    assert rec["call"] in set(DEFAULT_CLASSES.values())
    assert rec["status"] in ("confident", "uncertain")
    assert abs(sum(rec["probabilities"].values()) - 1) < 1e-3
    assert (rec["order_test"] is None) == (rec["status"] == "confident")
    assert rec["evidence"] and all(e["kind"] == "tile" for e in rec["evidence"])
    assert not set(_keys(rec)) & set(FORBIDDEN)
    json.dumps(rec)
    _validate(rec)


def test_uncertain_orders_the_array(trained):
    m, data = trained
    m2 = MethylationModel(**{**m.__dict__, "family_threshold": 1.01})
    rec = m2.predict([data[0][1]], "SYN-0001")
    assert rec["status"] == "uncertain" and rec["order_test"] == "methylation_array"
    _validate(rec)


@pytest.mark.parametrize("why", ["few_tiles", "triage", "encoder"])
def test_not_assessed(trained, why):
    m, data = trained
    bag = data[0][1]
    if why == "few_tiles":
        bag = Bag(bag.scan_id, bag.features[:5], bag.coords_um[:5], bag.tile_um, bag.encoder, bag.tissue_frac[:5])
        rec = m.predict([bag], "SYN-0001")
    elif why == "triage":
        rec = m.predict([bag], "SYN-0001", triage_call="metastasis")
    else:
        bag = Bag(bag.scan_id, bag.features, bag.coords_um, bag.tile_um, "other@1", bag.tissue_frac)
        rec = m.predict([bag], "SYN-0001")
    assert rec["status"] == "not_assessed" and rec["call"] == "indeterminate"
    assert rec["order_test"] == "methylation_array"
    _validate(rec)


def test_two_slides_pool_into_one_call(trained):
    m, data = trained
    rec = m.predict([data[0][1], data[1][1]], "SYN-0001")
    assert len(rec["scan_ids"]) == 2


def test_save_load_roundtrip(trained, tmp_path):
    m, data = trained
    m.save(tmp_path / "m")
    m2 = MethylationModel.load(tmp_path / "m")
    f = [data[3][1].features]
    assert np.allclose(m.proba(f), m2.proba(f), atol=1e-6)
    assert not list((tmp_path / "m").glob("*.pkl"))


def test_db_actions(trained):
    m, data = trained
    rec = m.predict([data[0][1]], "SYN-0001")
    acts = build_actions(rec, m)
    assert [a["verb"] for a in acts] == ["upsert", "create", "glioma_step"]
    assert acts[0]["records"][0]["status"] == "draft"
    gf = [r for r in acts[1]["records"] if r["type"] == "GenomicFinding"][0]
    assert gf["marker"] == MARKER and gf["marker"].endswith("_ai") and gf["result"] == rec["call"]
    regions = {r["region_id"] for r in acts[1]["records"] if r["type"] == "Region"}
    step = acts[2]["detail"]["records"][0]
    assert {e["id"] for e in step["evidence"]} == regions
    assert acts[2]["detail"]["study_id"] == "SYN-0001"
    _validate(step)
    assert build_actions(rec, m)[1]["action_id"] == acts[1]["action_id"]     # retries are idempotent


def test_unknown_class_needs_a_taxonomy():
    data = synthetic_cases({"X": "other", "Y": "other", "Z": "other"}, n_per_class=4)
    cases = [Case(cid, [b], c) for cid, b, c in data]
    with pytest.raises(ValueError):
        train(cases, Taxonomy(), "x", "0", k=3)


# ---------------------------------------------------------------- review fixes (second pass)

def test_gdc_methylation_files_pair_through_the_sample_sheet(tmp_path):
    from entip_train.methylation.tcga import read_sample_sheet
    sheet = tmp_path / "gdc_sample_sheet.tsv"
    sheet.write_text("File ID\tFile Name\tData Category\tProject ID\tCase ID\tSample ID\tSample Type\n"
                     "u1\t3f2a.methylation_array.sesame.level3betas.txt\tDNA Methylation\tTCGA-GBM\t"
                     "TCGA-06-0125\tTCGA-06-0125-01A\tPrimary Tumor\n")
    names = read_sample_sheet(str(sheet))
    slides = ["TCGA-06-0125-01Z-00-DX1.abc.svs"]
    assert pair_slides(slides, list(names)) == []                    # UUID file names carry no barcode
    pairs = pair_slides(slides, list(names), sample_of=names)
    assert [p.case_id for p in pairs] == ["TCGA-06-0125"]


def test_epic_v2_ids_and_missing_values(tmp_path):
    from entip_train.methylation.labels import read_beta_file, stack_betas
    f = tmp_path / "s.txt"
    f.write_text("probe\tbeta\ncg00000029_TC21\t0.2\ncg00000029_TC22\t0.4\ncg00000108_BC11\tNA\n")
    b = read_beta_file(f)
    assert b == {"cg00000029": pytest.approx(0.3)}
    samples = {f"s{i}": {"cg1": 0.1, "cg2": 0.2} for i in range(19)}
    samples["s19"] = {"cg1": 0.1}                                    # cg2 failed in one sample of 20
    _, cpgs, X = stack_betas(samples)
    assert cpgs == ["cg1", "cg2"] and np.isnan(X[-1, 1])


def test_report_without_scores(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("case_id,class,score\nTCGA-06-0001,A IDH,\n")
    assert from_report_csv(p) == []
    assert [l.cls for l in from_report_csv(p, keep_unscored=True)] == ["A IDH"]


def test_rare_classes_are_dropped_not_learned():
    cls = {"GBM, RTK II": "glioblastoma_idh_wildtype", "A IDH": "astrocytoma_idh_mutant",
           "DMG, K27": "diffuse_midline_glioma_h3k27"}
    data = synthetic_cases(cls, n_per_class=12, seed=2)
    data = [d for d in data if d[2] != "DMG, K27"] + [d for d in data if d[2] == "DMG, K27"][:3]
    m = train([Case(cid, [b], c) for cid, b, c in data], Taxonomy(), "x", "0", k=4, min_tiles=20,
              mil_kw={"epochs": 5})
    assert "DMG, K27" not in m.classes and m.metrics["dropped_classes"] == {"DMG, K27": 3}


def test_non_tcga_ids_group_by_their_own_id():
    assert site_of("EDN-0001") == "EDN-0001" and site_of("TCGA-06-0125") == "06"


def test_cli_train_and_predict_from_files(tmp_path, capsys):
    from entip_train.cli import main
    data = synthetic_cases(DEFAULT_CLASSES, n_per_class=10, seed=3)
    (tmp_path / "bags").mkdir()
    with open(tmp_path / "pairs.csv", "w", newline="") as f, open(tmp_path / "labels.csv", "w", newline="") as g:
        pw, lw = csv.writer(f), csv.writer(g)
        pw.writerow(["case_id", "slide", "methylation", "sample_type", "group"])
        lw.writerow(["case_id", "class", "score", "source"])
        for cid, bag, c in data:
            bag.save(tmp_path / "bags" / f"{bag.scan_id}.npz")
            pw.writerow([cid, f"{bag.scan_id}.svs", "m.txt", "01", site_of(cid)])
            lw.writerow([cid, c, 0.95, "report"])
    assert main(["methyl", "train", "--labels", str(tmp_path / "labels.csv"), "--pairs", str(tmp_path / "pairs.csv"),
                 "--bags", str(tmp_path / "bags"), "--out", str(tmp_path / "model"), "--folds", "4"]) == 0
    capsys.readouterr()
    assert main(["methyl", "predict", "--model", str(tmp_path / "model"), "--study-id", "SYN-0002",
                 "--bags", str(tmp_path / "bags" / f"{data[0][1].scan_id}.npz")]) == 0
    rec = json.loads(capsys.readouterr().out)
    _validate(rec)
