import numpy as np
from sklearn.metrics import roc_auc_score

from entip_train.heads import HeadConfig, background_negatives, labels_from_annotations, review_queue
from entip_train.ontology import Annotation


def click_cells(svc, slide, hid, n_pos, n_neg, seed=0, **kw):
    rng = np.random.default_rng(seed)
    pos = [c for c in slide.cells if slide.cell_truth[c.cell_id]]
    neg = [c for c in slide.cells if not slide.cell_truth[c.cell_id]]
    out = None
    for c in list(rng.choice(pos, n_pos, replace=False)):
        out = svc.add_label(hid, slide.scan.scan_id, 1, "tester", x_um=c.x_um + 1, y_um=c.y_um - 1, **kw)
    for c in list(rng.choice(neg, n_neg, replace=False)) if n_neg else []:
        out = svc.add_label(hid, slide.scan.scan_id, 0, "tester", cell_id=c.cell_id, **kw)
    return out


def test_milestone_twenty_clicks_gives_ranked_queue(svc, slide):
    hid = svc.create_head("Atypical nucleus", "cell", actor="tester")
    assert hid == "atypical_nucleus"                      # database-style key
    out = click_cells(svc, slide, hid, 10, 10)
    assert out["n_pos"] == 10 and out["n_neg"] == 10
    assert out["model_version"].startswith("head_")
    q = out["queue"]
    assert len(q) == 20
    unc = [r["uncertainty"] for r in q if r["reason"] == "uncertain"]
    assert unc == sorted(unc, reverse=True)                  # most uncertain first
    assert sum(r["reason"] == "random" for r in q) == 2      # 10% random share
    marked = {a.geometry.get("object_id") for a in svc.store.annotations(hid)}
    assert not marked & {r["cell_id"] for r in q}            # marked cells never queued
    _, ids, _, p = svc.predict(hid, slide.scan.scan_id)
    assert roc_auc_score([slide.cell_truth[i] for i in ids], p) > 0.9


def test_every_click_makes_a_warm_started_draft(svc, slide):
    hid = svc.create_head("x", "cell")
    click_cells(svc, slide, hid, 3, 3)
    drafts = svc.store.models(hid, "draft")
    assert len(drafts) == 6                                   # a lone positive trains against background
    assert drafts[0].metrics["n_background"] > 0 and drafts[-1].metrics["n_background"] == 0
    assert drafts[0].parent is None and drafts[-1].parent == drafts[-2].model_id
    assert drafts[-1].metrics["warm_start"] is True


def test_positives_only_uses_background_negatives(svc, slide):
    hid = svc.create_head("mitosis-like", "cell", positives_only=True)
    out = click_cells(svc, slide, hid, 12, 0)
    assert out["n_neg"] == 0 and out["metrics"]["n_background"] >= 20
    _, ids, _, p = svc.predict(hid, slide.scan.scan_id)
    assert roc_auc_score([slide.cell_truth[i] for i in ids], p) > 0.8


def test_background_negatives_keep_away_from_positives(slide):
    items = slide.cells
    pos_i = [i for i, c in enumerate(items) if slide.cell_truth[c.cell_id]][:5]
    lab = {i: (1, "a") for i in pos_i}
    cfg = HeadConfig(bg_exclusion_um=20)
    bg = background_negatives(items, lab, cfg, np.random.default_rng(0))
    assert len(bg) == cfg.bg_min and not set(bg) & set(pos_i)
    for b in bg:
        d = min(np.hypot(items[b].x_um - items[i].x_um, items[b].y_um - items[i].y_um) for i in pos_i)
        assert d > 20


def test_click_snaps_to_nearest_cell_and_latest_wins(slide):
    c = slide.cells[0]
    a1 = Annotation("a1", "syn-a", "f", "positive", {"type": "point", "x_um": c.x_um + 2, "y_um": c.y_um}, "t")
    a2 = Annotation("a2", "syn-a", "f", "negative", {"type": "object", "object_id": c.cell_id}, "t")
    far = Annotation("a3", "syn-a", "f", "positive", {"type": "point", "x_um": -500, "y_um": -500}, "t")
    lab = labels_from_annotations(slide.cells, [a1, a2, far], "cell", snap_um=8)
    assert lab == {0: (0, "a2")}


def test_tile_head_from_polygons(svc, slide):
    hid = svc.create_head("pale patch", "tile")
    regs = slide.regions
    pos = [r for r in regs if slide.region_truth[r.region_id]][:6]
    neg = [r for r in regs if not slide.region_truth[r.region_id]][:6]
    for r in pos:
        svc.add_label(hid, "syn-a", 1, "t", x_um=r.centre[0], y_um=r.centre[1])
    box = [[0, 0], [130, 0], [130, 130], [0, 130]]          # covers 2x2 tiles, all non-pale for seed 0
    svc.add_label(hid, "syn-a", 0, "t", polygon_um=box)
    out = svc.add_label(hid, "syn-a", 0, "t", object_id=neg[-1].region_id)
    assert out["n_pos"] == 6
    preds = svc.predictions(hid, "syn-a")["predictions"]
    p = np.array([r["p"] for r in preds])
    y = np.array([slide.region_truth[r["object_id"]] for r in preds])
    assert roc_auc_score(y, p) > 0.9


def test_retraction_is_logged_not_deleted(svc, slide):
    hid = svc.create_head("y", "cell")
    click_cells(svc, slide, hid, 3, 3)
    a = svc.store.annotations(hid)[0]
    svc.retract_label(hid, a.annotation_id, "tester")
    assert a.annotation_id not in {x.annotation_id for x in svc.store.annotations(hid)}
    assert a.annotation_id in {x.annotation_id for x in svc.store.annotations(hid, include_retracted=True)}
    verbs = [x["verb"] for x in svc.store.actions(a.annotation_id)]
    assert verbs == ["annotate", "retract"]


def test_review_queue_random_share_and_exclusion():
    ids = [f"c{i}" for i in range(100)]
    p = np.linspace(0, 1, 100)
    q = review_queue(ids, p, {"c50"}, n=20, random_share=0.1)
    assert len(q) == 20 and "c50" not in {r["object_id"] for r in q}
    assert q[0]["object_id"] in ("c49", "c51")


def test_embedding_cache_persists(tmp_path, slide):
    from entip_train.cache import EmbeddingCache
    from entip_train.encoders import HandcraftedEncoder
    enc = HandcraftedEncoder()
    c1 = EmbeddingCache(tmp_path)
    e1 = c1.get("syn-a", "cell", slide.cells[:10], slide.reader, enc)
    c2 = EmbeddingCache(tmp_path)

    class NoRead:
        mpp = 0.5

        def read_um(self, *a):
            raise AssertionError("should come from disk")

    e2 = c2.get("syn-a", "cell", slide.cells[:10], NoRead(), enc)
    np.testing.assert_allclose(e1, e2, rtol=1e-2, atol=1e-3)   # float16 on disk
