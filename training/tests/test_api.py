import time

from fastapi.testclient import TestClient

from entip_train.api import create_app
from entip_train.encoders import HandcraftedEncoder
from entip_train.service import TrainingService
from entip_train.store import Store


def test_gui_contract(tmp_path):
    app = create_app(TrainingService(Store(tmp_path / "t.sqlite"), HandcraftedEncoder(), tmp_path / "a"))
    c = TestClient(app)
    assert c.get("/health").json()["ok"]
    sc = c.post("/scans/synthetic", json={"scan_id": "s1", "seed": 3}).json()
    assert sc["n_cells"] == 600
    hid = c.post("/heads", json={"feature": "atypical", "level": "cell", "scan_id": "s1"}).json()["head_id"]
    assert c.get(f"/heads/{hid}/queue", params={"scan_id": "s1"}).json() == []   # before any mark
    truth = app.state.truth["s1"]
    pos = [x for x in truth.cells if truth.cell_truth[x.cell_id]][:10]
    neg = [x for x in truth.cells if not truth.cell_truth[x.cell_id]][:10]
    for x, y in [(p, 1) for p in pos] + [(n, 0) for n in neg]:
        r = c.post(f"/heads/{hid}/labels", json={"scan_id": "s1", "cell_id": x.cell_id, "label": y, "actor": "al"})
        assert r.status_code == 200, r.text
    body = r.json()
    assert {"model_version", "n_pos", "n_neg", "metrics", "queue", "label_id"} <= set(body)
    q = c.get(f"/heads/{hid}/queue", params={"scan_id": "s1", "n": 20}).json()
    assert len(q) == 20 and {"cell_id", "p", "reason", "x_um", "y_um"} <= set(q[0])
    pr = c.get(f"/heads/{hid}/predictions", params={"scan_id": "s1"}).json()
    assert len(pr["predictions"]) == 600 and "cell_id" in pr["predictions"][0]
    assert c.delete(f"/heads/{hid}/labels/{body['label_id']}", params={"actor": "al"}).json()["retracted"]
    jid = c.post(f"/heads/{hid}/retrain", json={"actor": "al", "force": True}).json()["job_id"]
    for _ in range(100):
        j = c.get(f"/jobs/{jid}").json()
        if j["state"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert j["state"] == "done" and "locked set too small" in j["result"]["reasons"][0]
    assert c.get("/heads/nope").status_code == 404
    assert c.post(f"/heads/{hid}/labels", json={"scan_id": "s1", "label": 2}).status_code == 422
    acts = c.get("/actions").json()
    assert any(a["verb"] == "retract" for a in acts) and any(a["actor"] == "person:al" for a in acts)
