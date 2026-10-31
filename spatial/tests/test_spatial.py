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
