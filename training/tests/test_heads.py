"""Marks to labels on large slides, and which saved heads may be loaded."""
import joblib
import numpy as np
import pytest

from entip_train.heads import (HeadConfig, LiveHead, background_negatives, labels_from_annotations,
                               load_head_file)
from entip_train.ontology import Annotation, Cell


def _cells(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return [Cell(f"c{i}", "s", float(x), float(y), "t") for i, (x, y) in enumerate(rng.uniform(0, 1000, (n, 2)))]


def _ann(i, label, geom):
    return Annotation(f"a{i}", "s", "f", label, geom, "person:t")


def test_object_point_and_polygon_marks():
    cells = _cells()
    c7, c9 = cells[7], cells[9]
    anns = [_ann(0, "positive", {"type": "object", "object_id": "c7"}),
            _ann(1, "negative", {"type": "object", "object_id": "no-such-cell"}),
            _ann(2, "negative", {"type": "point", "x_um": c9.x_um + 1, "y_um": c9.y_um}),
            _ann(3, "positive", {"type": "polygon", "points_um": [[0, 0], [200, 0], [200, 200], [0, 200]]}),
            _ann(4, "negative", {"type": "object", "object_id": "c7"})]          # the latest mark wins
    lab = labels_from_annotations(cells, anns, "cell")
    inside = {i for i, c in enumerate(cells) if 0 < c.x_um < 200 and 0 < c.y_um < 200}
    assert lab[7] == (0, "a4")
    assert lab[9][0] == (1 if 9 in inside else 0)
    assert inside <= set(lab) and all(lab[i][0] == 1 for i in inside - {7, 9})
    assert set(lab) == inside | {7, 9}


def test_background_negatives_keep_away_from_positives():
    cells = _cells(2000)
    labelled = {i: (1, f"a{i}") for i in range(0, 40, 2)}
    cfg = HeadConfig()
    bg = background_negatives(cells, labelled, cfg, np.random.default_rng(0))
    assert len(bg) == max(cfg.bg_min, int(cfg.bg_ratio * len(labelled))) and not set(bg) & set(labelled)
    pos = np.array([(cells[i].x_um, cells[i].y_um) for i in labelled])
    for j in bg:
        d = np.hypot(pos[:, 0] - cells[j].x_um, pos[:, 1] - cells[j].y_um).min()
        assert d > cfg.bg_exclusion_um
    enough = {**labelled, **{i: (0, "n") for i in range(100, 200)}}
    assert background_negatives(cells, enough, cfg, np.random.default_rng(0)) == []


def test_large_slide_marks_are_fast():
    import time
    cells = _cells(50000)
    anns = [_ann(i, "positive", {"type": "object", "object_id": f"c{i * 97}"}) for i in range(300)]
    t = time.perf_counter()
    lab = labels_from_annotations(cells, anns, "cell")
    assert len(lab) == 300 and time.perf_counter() - t < 1.0      # was over a second per click before


def _fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    return LiveHead(HeadConfig()).fit(X, (X[:, 0] > 0).astype(int))


def test_only_heads_in_the_service_folder_are_loaded(tmp_path):
    root = tmp_path / "artifacts" / "heads"
    root.mkdir(parents=True)
    joblib.dump(_fitted(), root / "head_ok.joblib")
    assert isinstance(load_head_file(root / "head_ok.joblib", root), LiveHead)
    outside = tmp_path / "elsewhere.joblib"
    joblib.dump(_fitted(), outside)
    for bad in (outside, root / ".." / ".." / "elsewhere.joblib", root / "head_ok.txt"):
        with pytest.raises(ValueError, match="refusing"):
            load_head_file(bad, root)
    joblib.dump({"not": "a head"}, root / "other.joblib")
    with pytest.raises(ValueError, match="not a saved live head"):
        load_head_file(root / "other.joblib", root)
