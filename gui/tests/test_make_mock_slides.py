"""Run: python -m pytest gui/tests"""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("mms", Path(__file__).parents[1] / "tools" / "make_mock_slides.py")
mms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mms)


def test_small_slide(tmp_path):
    p = dict(seed=1, w=1024, h=768, density=3000, mitosis=0.05, apoptotic=0.05, lymph=0.05,
             vessels=1, necrosis=True, oligo=0.0)
    s = mms.make("SCN-TEST", p, tmp_path)
    d = tmp_path / "slides" / "SCN-TEST"
    table = json.loads((d / "cells.json").read_text())
    assert table["frame"] == "scan:SCN-TEST" and table["mpp"] == mms.MPP
    assert len(table["cells"]) == s["n_cells"] > 0
    w_um, h_um = 1024 * mms.MPP, 768 * mms.MPP
    assert all(0 <= c["x_um"] <= w_um and 0 <= c["y_um"] <= h_um for c in table["cells"])
    assert all(len(c["emb"]) == mms.EMB_DIM for c in table["cells"])
    assert (d / "slide.dzi").exists() and (d / "slide_files" / "10" / "0_0.jpg").exists()
    truth = json.loads((d / "truth.json").read_text())
    assert set(truth) == {c["cell_id"] for c in table["cells"]}
