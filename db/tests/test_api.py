P = {"kind": "person", "actor": "person:trainee-01"}


def test_health_and_schema(client):
    assert client.get("/health").json()["schema_version"] == "0001"
    s = client.get("/schema").json()
    assert "patient" in s["tables"] and s["tables"]["scan"]["columns"]["section_id"]["references"] == \
        "section.section_id"
    assert "molecular" in s["patient_row"]


def test_patients_list_and_marker_filter(client):
    rows = client.get("/patients").json()
    assert [r["study_id"] for r in rows] == ["SYN-0001", "SYN-0002", "SYN-0003"]
    idh = client.get("/patients", params={"marker.IDH": "mutant"}).json()
    assert {r["study_id"] for r in idh} == {"SYN-0002", "SYN-0003"}


def test_patient_tree(client):
    t = client.get("/patients/SYN-0001").json()
    sec = t["specimens"][0]["blocks"][0]["sections"]
    assert [s["stain"] for s in sec] == ["H&E", "Ki67", "H&E"]
    scan = sec[0]["scans"][0]
    assert scan["scan_id"] == "SCN-0001-A1-01" and scan["width_um"] == 10000.0 and scan["n_cells"] >= 200
    assert sec[2]["assays"][0]["kind"] == "visium"
    assert t["specimens"][0]["assays"][0]["kind"] == "panel"
    assert t["transforms"][0]["rmse_um"] == 6.5
    assert client.get("/patients/NOPE").status_code == 404


def test_cells_bbox_and_scores(client):
    all_cells = client.get("/scans/SCN-0003-A1-01/cells").json()
    assert len(all_cells) == 200
    x, y = all_cells[0]["x_um"], all_cells[0]["y_um"]
    box = client.get("/scans/SCN-0003-A1-01/cells", params={"bbox": f"{x-1},{y-1},{x+1},{y+1}"}).json()
    assert all_cells[0]["cell_id"] in {c["cell_id"] for c in box} and len(box) < 200
    r = client.post("/actions", json={"verb": "create", "object_type": "prediction", "provenance":
                                      {"kind": "model", "actor": "model:mdl-cellpose-sam-synthetic"},
                                      "records": [{"model_id": "mdl-cellpose-sam-synthetic", "target_type": "cell",
                                                   "target_id": c["cell_id"], "scan_id": "SCN-0003-A1-01",
                                                   "score": i / 200} for i, c in enumerate(all_cells)]})
    assert r.status_code == 201, r.text
    scored = client.get("/scans/SCN-0003-A1-01/cells", params={"model_id": "mdl-cellpose-sam-synthetic",
                                                                "limit": 5}).json()
    assert scored[0]["score"] is not None
    q = client.get("/scans/SCN-0003-A1-01/predictions", params={"model_id": "mdl-cellpose-sam-synthetic",
                                                                 "limit": 3}).json()
    assert abs(q[0]["score"] - 0.5) <= abs(q[2]["score"] - 0.5)


def test_click_to_teach_then_retract(client):
    body = {"verb": "create", "object_type": "annotation", "provenance": P,
            "records": [{"annotation_id": "ANN-T-1", "scan_id": "SCN-0002-A1-01", "feature": "mitotic_figure",
                         "label": "positive", "geometry": {"type": "point", "x_um": 100.0, "y_um": 200.0}}]}
    r = client.post("/actions", json=body)
    assert r.status_code == 201, r.text
    assert client.post("/actions", json=body).status_code == 422  # duplicate id
    ids = {a["annotation_id"] for a in client.get("/scans/SCN-0002-A1-01/annotations").json()}
    assert "ANN-T-1" in ids
    r = client.post("/actions", json={"verb": "retract", "object_type": "annotation", "provenance": P,
                                      "records": [{"annotation_id": "ANN-T-1"}]})
    assert r.status_code == 201
    ids = {a["annotation_id"] for a in client.get("/scans/SCN-0002-A1-01/annotations").json()}
    assert "ANN-T-1" not in ids
    hist = client.get("/objects/annotation/ANN-T-1/history").json()
    assert [h["verb"] for h in hist] == ["create", "retract"]
    assert hist[1]["changes"][0]["before"]["retracted"] is False
    log = client.get("/patients/SYN-0002/actions").json()
    assert {"create", "retract"} <= {a["verb"] for a in log}
    assert all(c["study_id"] == "SYN-0002" for a in log for c in a["changes"])


def test_polygon_region_and_unknown_fields(client):
    r = client.post("/actions", json={"verb": "create", "object_type": "region", "provenance": P,
                                      "records": [{"region_id": "R-T-1", "scan_id": "SCN-0002-A1-01",
                                                   "source": "person", "geometry": {
                                                       "type": "polygon",
                                                       "points_um": [[0, 0], [100, 0], [100, 100], [0, 100]]}}]})
    assert r.status_code == 201, r.text
    reg = client.get("/scans/SCN-0002-A1-01/regions", params={"bbox": "50,50,60,60"}).json()
    assert "R-T-1" in {x["region_id"] for x in reg}
    bad = client.post("/actions", json={"verb": "create", "object_type": "scan", "provenance": P,
                                        "records": [{"scan_id": "S-x", "section_id": "SEC-0002-A1-01", "mpp": 0.25,
                                                     "width_px": 10, "height_px": 10, "stain": "H&E"}]})
    assert bad.status_code == 422 and "unknown field" in bad.json()["detail"]


def test_spatial_objects_and_transforms(client):
    objs = client.get("/assays/ASY-0001-VIS/objects").json()
    assert len(objs) == 50 and objs[0]["transform_id"].startswith("TF-")
    tf = client.get("/transforms", params={"frame": "scan:SCN-0001-A1-01"}).json()
    assert tf and tf[0]["src_frame"] == "assay:ASY-0001-VIS"


def test_gui_feedback_round(client):
    # region labels
    r = client.post("/actions", json={"verb": "create", "object_type": "region", "provenance": P,
                                      "records": [{"region_id": "R-NEC-1", "scan_id": "SCN-0003-A1-01",
                                                   "source": "person", "label": "necrosis",
                                                   "x_um": 10, "y_um": 10, "w_um": 50, "h_um": 50}]})
    assert r.status_code == 201, r.text
    assert [x["region_id"] for x in client.get("/scans/SCN-0003-A1-01/regions",
                                               params={"label": "necrosis"}).json()] == ["R-NEC-1"]
    # copy-number findings
    r = client.post("/actions", json={"verb": "create", "object_type": "genomic_finding", "provenance": P,
                                      "records": [{"finding_id": "GF-T-CN", "study_id": "SYN-0001",
                                                   "marker": "chr7_chr10", "kind": "copy_number",
                                                   "result": "+7/-10"}]})
    assert r.status_code == 201, r.text
    assert client.get("/patients/SYN-0001").json()["molecular"]["chr7_chr10"] == "+7/-10"
    # clear one model's predictions on one scan
    m = "mdl-cellpose-sam-synthetic"
    cells = client.get("/scans/SCN-0001-A1-01/cells", params={"limit": 3}).json()
    client.post("/actions", json={"verb": "upsert", "object_type": "prediction", "provenance": P, "records": [
        {"model_id": m, "target_type": "cell", "target_id": c["cell_id"], "scan_id": "SCN-0001-A1-01",
         "score": 0.4} for c in cells]})
    r = client.post("/actions", json={"verb": "delete", "object_type": "prediction", "provenance": P,
                                      "records": [{"model_id": m, "scan_id": "SCN-0001-A1-01"}]}).json()
    assert r["n"] == 3
    assert client.get("/scans/SCN-0001-A1-01/predictions", params={"model_id": m}).json() == []
    # the scan carries its patient's diagnosis
    s = client.get("/scans/SCN-0001-A1-01").json()
    assert s["stain"] == "H&E" and s["patient"]["diagnosis"].startswith("Glioblastoma")
