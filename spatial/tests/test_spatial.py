import json

import numpy as np
import pandas as pd
import pytest

from entip_spatial import labels as L
from entip_spatial import records as R
from entip_spatial import register as G
from entip_spatial import synthetic
from entip_spatial.cli import main
from entip_spatial.loaders import load_visium, load_xenium
from entip_spatial.spots import spot_table


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    truth = synthetic.make(d, seed=1)
    return d, truth


def test_fit_recovers_affine_and_reports_errors():
    rng = np.random.default_rng(0)
    T = synthetic._rigid(10, 1.02, 30, -12)
    src = rng.uniform(0, 1000, (8, 2))
    dst = G.apply(T, src)
    for model in ("affine", "similarity"):
        M = G.fit(src, dst, model)
        assert np.allclose(M, T, atol=1e-8)
        assert G.loo_rmse_um(src, dst, model) < 1e-6
    noisy = dst + rng.normal(0, 3, dst.shape)
    tf = G.register("a", "s", src, noisy)
    assert tf.src_frame == "assay:a" and tf.dst_frame == "scan:s"
    assert 0 < tf.rmse_um < tf.loo_rmse_um  # held-out error is the honest (larger) one
    assert np.allclose(G.apply(G.invert(tf.matrix), G.apply(tf.matrix, src)), src)


def test_fit_rejects_bad_landmarks():
    with pytest.raises(ValueError):
        G.fit([[0, 0], [1, 1]], [[0, 0], [1, 1]], "affine")
    with pytest.raises(ValueError):
        G.fit([[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]], "affine")  # collinear


def test_similarity_has_no_reflection_by_default():
    src = np.array([[0, 0], [100, 0], [0, 50], [70, 80]], float)
    dst = src * [-1, 1]  # mirror image
    M = G.fit(src, dst, "similarity")
    assert np.linalg.det(M[:2, :2]) > 0


def test_loaders_read_10x_layouts(data):
    d, _ = data
    x = load_xenium(d / "xenium")
    assert x.uns["entip"]["object_kind"] == "nucleus"
    assert not any(g.startswith("NegControl") for g in x.var_names)  # control probes dropped
    assert x.obsm["spatial_um"].shape == (x.n_obs, 2)
    v = load_visium(d / "visium")
    assert v.uns["entip"]["um_per_unit"] == pytest.approx(0.25)
    assert v.uns["entip"]["radius_um"] == pytest.approx(27.5)


def test_icp_refinement_beats_landmarks_alone(data):
    d, truth = data
    x = load_xenium(d / "xenium")
    lm = pd.read_csv(d / "xenium_landmarks.csv")
    he = pd.read_csv(d / "he_cells.csv")
    src, dst = lm[["assay_x_um", "assay_y_um"]].to_numpy(), lm[["he_x_um", "he_y_um"]].to_numpy()
    T = np.array(truth["xenium_to_he"])
    pts = x.obsm["spatial_um"]

    def err(M):
        return np.sqrt(np.mean(np.sum((G.apply(M, pts) - G.apply(T, pts)) ** 2, 1)))

    plain = G.register("x", "s", src, dst)
    refined = G.register("x", "s", src, dst, refine_assay_pts=pts, refine_he_pts=he[["x_um", "y_um"]].to_numpy())
    assert err(np.array(refined.matrix)) < 1.0 < err(np.array(plain.matrix))
    assert refined.method == "landmarks+icp" and refined.meta["icp_matched"] > 0.5 * x.n_obs


def test_match_nuclei_mutual_and_gated():
    he = np.array([[0, 0], [20, 0], [40, 0]], float)
    assay = np.array([[1, 0], [21, 0], [80, 0]], float)
    i, j, dist = L.match_nuclei(he, assay, max_dist_um=5)
    assert list(i) == [0, 1] and list(j) == [0, 1] and np.allclose(dist, 1)
    # ambiguous: two assay cells equally close to one nucleus
    i, *_ = L.match_nuclei(np.array([[0, 0]], float), np.array([[3, 0], [-3, 0]], float), max_dist_um=5)
    assert len(i) == 0


def test_spot_table_counts():
    spots = np.array([[0, 0], [100, 0]], float)
    cells = pd.DataFrame({"x_um": [1, 2, 99, 300], "y_um": [0, 0, 0, 0], "call": ["mitosis", "other", "other", "mitosis"],
                          "p_mitosis": [0.9, 0.1, 0.2, 0.8]})
    t = spot_table(spots, ["a", "b"], cells, radius_um=27.5)
    assert t.loc["a", "n_cells"] == 2 and t.loc["a", "n_mitosis"] == 1 and t.loc["b", "n_other"] == 1
    assert t.loc["a", "p_mitosis_max"] == pytest.approx(0.9)
    assert t.loc["b", "mitosis_local"] == 0


def test_demo_end_to_end(tmp_path, capsys):
    main(["demo", str(tmp_path / "demo"), "--seed", "2"])
    s = json.loads((tmp_path / "demo" / "summary.json").read_text())
    assert s["xenium_true_error_um"] < 1.0
    assert s["xenium_label_accuracy"] > 0.95 and s["xenium_labelled"] > 0.8 * s["n_he"]
    assert s["visium_label_accuracy"] > 0.75
    lab = pd.read_csv(tmp_path / "demo" / "xenium" / "training_labels.csv")
    assert list(lab.columns) == L.COLUMNS
    assert lab["confidence"].between(0, 1).all()
    rec = tmp_path / "demo" / "xenium" / "records"
    actions = [json.loads(l) for l in (rec / "actions.jsonl").read_text().splitlines()]
    assert [a["object_type"] for a in actions] == ["assay", "transform", "spatial_object"]
    tf = R.read_transform(tmp_path / "demo" / "xenium" / "transform.json")
    assert tf.dst_frame == "scan:scan_demo"
    spot = pd.read_csv(tmp_path / "demo" / "visium" / "spot_table.csv", index_col=0)
    assert "n_mitosis" in spot and spot["n_cells"].sum() > 0


def test_visium_pixel_size_uses_the_65_um_scale_factor(data):
    """10x define spot_diameter_fullres as the pixels across a theoretical 65 um spot (not the 55 um spot)."""
    d, _ = data
    sf = json.loads((d / "visium" / "spatial" / "scalefactors_json.json").read_text())
    v = load_visium(d / "visium")
    assert v.uns["entip"]["um_per_unit"] == pytest.approx(65.0 / sf["spot_diameter_fullres"])
    # spots are 100 um apart centre to centre: the nearest-neighbour distance checks the scale end to end
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(v.obsm["spatial_um"]).query(v.obsm["spatial_um"], k=2)
    assert np.median(dist[:, 1]) == pytest.approx(100.0, abs=0.5)


def test_records_follow_the_database_contract():
    from entip_spatial.ontology import Assay, Provenance
    prov = Provenance.program("register")
    assert prov.kind == "program"
    a = Assay("assay_1", "sec_1", "visium", "assay:assay_1", 10, 5, "/data/NP-24-1234/outs", "ab" * 32, prov,
              meta={"kind": "visium"})
    rec = R.assay_record(a)
    assert "source_path" not in rec and "uri" not in rec and "NP-24-1234" not in json.dumps(rec)
    assert rec["sha256"] == "ab" * 32 and rec["provenance"]["kind"] == "program"
    assert R.assay_record(a, uri="s3://bucket/assay_1")["uri"] == "s3://bucket/assay_1"
    tf = G.register("assay_1", "scan_1", [[0, 0], [100, 0], [0, 100], [90, 80]],
                    [[1, 1], [101, 1], [1, 101], [91, 81]])
    t = R.transform_record(tf)
    assert t["type"] == "Transform" and t["src_frame"] == "assay:assay_1" and len(t["matrix"]) == 3


def test_db_sink_chunks_large_assays(monkeypatch):
    from entip_spatial.ontology import Assay, Provenance, SpatialObject
    prov = Provenance.program("register")
    a = Assay("assay_1", "sec_1", "xenium", "assay:assay_1", 5, 5, "x", None, prov)
    tf = G.register("assay_1", "scan_1", [[0, 0], [100, 0], [0, 100]], [[0, 0], [100, 0], [0, 100]])
    objs = [SpatialObject(f"assay_1:{i}", "assay_1", "nucleus", str(i), tf.dst_frame, float(i), 0.0, None,
                          tf.transform_id) for i in range(5)]
    sent = []
    sink = R.DbSink("http://db.invalid", token="", chunk=2)
    monkeypatch.setattr(sink, "post_action", lambda body: sent.append(body) or {"n": len(body["records"])})
    sink.register(a, tf, objs, prov)
    assert [[r["type"] for r in b["records"]] for b in sent] == [
        ["Assay", "Transform", "SpatialObject", "SpatialObject"], ["SpatialObject"] * 2, ["SpatialObject"]]
    assert len({b["action_id"] for b in sent}) == 3 and all(b["verb"] == "upsert" for b in sent)


def test_register_writes_to_the_real_database(tmp_path, data, live_db, monkeypatch, capsys):
    import urllib.request
    d, _ = data
    monkeypatch.setenv("ENTIP_DB_WRITE_TOKEN", live_db["token"])
    prov = {"kind": "program", "actor": "test"}
    chain = [{"type": "Patient", "study_id": "SYN-SPATIAL", "synthetic": True},
             {"type": "Specimen", "specimen_id": "SP-1", "study_id": "SYN-SPATIAL"},
             {"type": "Block", "block_id": "BL-1", "specimen_id": "SP-1"},
             {"type": "Section", "section_id": "section_demo", "block_id": "BL-1", "stain": "H&E"},
             {"type": "Scan", "scan_id": "scan_demo", "section_id": "section_demo", "mpp": 0.25,
              "width_px": 16000, "height_px": 12000}]
    R.DbSink(live_db["url"]).post_action({"verb": "create", "provenance": prov, "records": chain})

    def get(path):
        with urllib.request.urlopen(live_db["url"] + path) as r:
            return json.loads(r.read())

    def run():
        main(["register", "--kind", "visium", "--assay", str(d / "visium"), "--landmarks",
              str(d / "visium_landmarks.csv"), "--scan-id", "scan_demo", "--section-id", "section_demo",
              "--assay-id", "assay_visium_db", "--out", str(tmp_path / "v"), "--db", live_db["url"]])

    run()
    assay = get("/assays/assay_visium_db")
    assert assay["kind"] == "visium" and assay["study_id"] == "SYN-SPATIAL" and assay.get("uri") is None
    assert str(d) not in json.dumps(assay)                       # the local path is not sent
    objs = get("/assays/assay_visium_db/objects")
    assert len(objs) == assay["n_obs"] and {o["frame"] for o in objs} == {"scan:scan_demo"}
    tfs = get("/transforms?frame=scan:scan_demo")
    assert len(tfs) == 1 and tfs[0]["rmse_um"] > 0 and objs[0]["transform_id"] == tfs[0]["transform_id"]
    tree = get("/patients/SYN-SPATIAL")
    assert tree["assay_ids"] == ["assay_visium_db"]
    # a second run replaces the positions rather than adding a second set
    run()
    again = get("/assays/assay_visium_db/objects")
    assert len(again) == len(objs) and len(get("/transforms?frame=scan:scan_demo")) == 2
    assert {o["transform_id"] for o in again} != {o["transform_id"] for o in objs}
    # an unknown section is refused with a plain message, and the local records are still written
    with pytest.raises(SystemExit, match="missing parent"):
        main(["register", "--kind", "visium", "--assay", str(d / "visium"), "--landmarks",
              str(d / "visium_landmarks.csv"), "--scan-id", "scan_demo", "--section-id", "no_such_section",
              "--assay-id", "assay_orphan", "--out", str(tmp_path / "o"), "--db", live_db["url"]])
    assert (tmp_path / "o" / "records" / "assay.jsonl").exists()
