"""End to end against the db/ pillar: a real PostgreSQL + PostGIS behind its FastAPI app.

Set ENTIP_TEST_ADMIN_DSN like db/tests (default postgresql://entip:entip@localhost:5432/postgres).
Skipped when entip-db, psycopg or the server is missing.
"""
import os
import uuid

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("entip_db")

from fastapi.testclient import TestClient  # noqa: E402

from entip_train.dbstore import DbStore  # noqa: E402
from entip_train.encoders import HandcraftedEncoder  # noqa: E402
from entip_train.ontology import Cell, Scan  # noqa: E402
from entip_train.service import TrainingService  # noqa: E402
from entip_train.slides import synthetic_slide  # noqa: E402

ADMIN = os.environ.get("ENTIP_TEST_ADMIN_DSN", "postgresql://entip:entip@localhost:5432/postgres")
SCAN = "SCN-0001-A1-T1"


@pytest.fixture(scope="module")
def db_http():
    try:
        admin = psycopg.connect(ADMIN, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"no PostgreSQL server ({e})")
    name = f"entip_train_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    url = psycopg.conninfo.make_conninfo(ADMIN, dbname=name)
    from entip_db.api import create_app
    from entip_db.db import connect, migrate
    from entip_db.seed import seed
    conn = connect(url)
    migrate(conn)
    seed(conn, n_cells=50)
    conn.close()
    with TestClient(create_app(url)) as c:
        yield c
    admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture(scope="module")
def setup(db_http, tmp_path_factory):
    s = synthetic_slide(SCAN, seed=4)
    store = DbStore(http=db_http)
    svc = TrainingService(store, HandcraftedEncoder(), tmp_path_factory.mktemp("art"), write_predictions=True)
    store.add_scan(Scan(SCAN, "synthetic:4", s.scan.mpp, s.scan.width_um, s.scan.height_um, "SEC-0001-A1-01"))
    ids = {c.cell_id: f"{SCAN}:c{i:06d}" for i, c in enumerate(s.cells)}     # database-style cell ids
    store.add_cells([Cell(ids[c.cell_id], SCAN, c.x_um, c.y_um, "synthetic") for c in s.cells])
    svc.readers[SCAN] = s.reader
    truth = {ids[k]: v for k, v in s.cell_truth.items()}
    return svc, store, truth, db_http


def test_reads_scan_cells_and_feature_keys(setup):
    svc, store, truth, http = setup
    assert store.scan(SCAN).mpp == 0.5
    assert len(store.cells(SCAN)) == 600 and store.cells(SCAN)[0].cell_id.startswith(SCAN + ":c")
    assert SCAN in store.scan_ids()
    assert svc.create_head("mitotic_figure", "cell") == "mitotic_figure"    # the database's key, as is
    assert store.feature("mitotic_figure").positives_only is True


def test_clicks_train_and_write_back(setup):
    svc, store, truth, http = setup
    hid = "mitotic_figure"
    rng = np.random.default_rng(0)
    pos = [k for k, v in truth.items() if v][:10]
    neg = [k for k, v in truth.items() if not v][:10]
    # the GUI writes its mark to the database first, then tells us its annotation_id
    c0 = next(c for c in store.cells(SCAN) if c.cell_id == pos[0])
    r = http.post("/actions", json={"verb": "create", "object_type": "annotation",
                                    "provenance": {"kind": "person", "actor": "person:al"},
                                    "records": [{"annotation_id": "ANN-GUI-001", "scan_id": SCAN, "feature": hid,
                                                 "label": "positive", "target_type": "cell", "target_id": pos[0],
                                                 "geometry": {"type": "point", "x_um": c0.x_um, "y_um": c0.y_um}}]})
    assert r.status_code == 201, r.text
    out = svc.add_label(hid, SCAN, 1, "al", annotation_id="ANN-GUI-001")
    assert out["label_id"] == "ANN-GUI-001"
    for k in pos[1:]:
        out = svc.add_label(hid, SCAN, 1, "al", cell_id=k)
    for k in rng.permutation(neg):
        out = svc.add_label(hid, SCAN, 0, "al", cell_id=str(k))
    mine = store.annotations(hid, scan_id=SCAN)
    assert sum(a.label == "positive" for a in mine) == 10 and len(mine) == 20
    assert out["n_pos"] == 10 + 12 and out["n_neg"] == 10 + 6         # plus the seeded marks on other scans
    assert out["metrics"]["skipped_scans"]                             # seeded scans have no pixels
    anns = http.get(f"/scans/{SCAN}/annotations", params={"feature": hid}).json()
    assert len(anns) == 20                                               # ANN-GUI-001 not written twice
    assert all(a["target_type"] == "cell" for a in anns)
    m = out["model_version"]
    assert http.get("/models", params={"feature": hid, "status": "draft"}).json()[-1]["model_id"] == m
    preds = http.get(f"/scans/{SCAN}/predictions", params={"model_id": m, "order": "uncertain"}).json()
    assert len(preds) == 200 and abs(preds[0]["score"] - 0.5) <= abs(preds[-1]["score"] - 0.5)
    q = svc.queue(hid, SCAN, 20)
    assert q[0]["cell_id"].startswith(SCAN)
    acts = http.get("/actions", params={"verb": "create", "object_type": "annotation"}).json()
    assert any(a["provenance"].get("tool") == "entip-train" for a in acts)


def test_retract_and_gated_retrain(setup):
    svc, store, truth, http = setup
    hid = "mitotic_figure"
    a = store.annotations(hid, scan_id=SCAN)[-1]
    svc.retract_label(hid, a.annotation_id, "al")
    assert a.annotation_id not in {x.annotation_id for x in store.annotations(hid, scan_id=SCAN)}
    pos = [k for k, v in truth.items() if v][10:20]
    neg = [k for k, v in truth.items() if not v][10:30]
    for k in pos:
        svc.add_label(hid, SCAN, 1, "al", cell_id=k, locked=True, train=False)
    for k in neg:
        svc.add_label(hid, SCAN, 0, "al", cell_id=k, locked=True, train=False)
    r1 = svc.retrain(hid, "al", force=True)
    assert r1["status"] == "champion", r1["reasons"]
    r2 = svc.retrain(hid, "al", force=True)
    champs = http.get("/models", params={"feature": hid, "status": "champion"}).json()
    assert len(champs) == 1
    if r2["status"] == "champion":
        retired = http.get("/models", params={"feature": hid, "status": "retired"}).json()
        assert retired[0]["model_id"] == r1["model_id"]
