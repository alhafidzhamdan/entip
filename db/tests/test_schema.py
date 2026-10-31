import psycopg
import pytest

from entip_db.db import apply_action, migrate
from entip_db.db_errors import EntipError

P = {"kind": "person", "actor": "person:test"}


def test_migrate_is_idempotent(conn):
    assert migrate(conn) == []


def test_patient_row(conn):
    r = conn.execute("SELECT * FROM entip.patient_row WHERE study_id = 'SYN-0002'").fetchone()
    assert r["molecular"]["IDH"] == "mutant"
    assert r["molecular"]["1p19q"] == "codeleted"
    assert r["n_scans"] == 1 and r["n_assays"] == 1 and r["synthetic"] is True
    r1 = conn.execute("SELECT * FROM entip.patient_row WHERE study_id = 'SYN-0001'").fetchone()
    assert r1["n_scans"] == 2 and r1["n_assays"] == 2 and set(r1["assay_kinds"]) == {"panel", "visium"}


@pytest.mark.parametrize("stmt", [
    "INSERT INTO entip.patient (study_id, provenance) VALUES ('SYN-9999', '{\"kind\":\"person\",\"actor\":\"x\"}')",
    "UPDATE entip.patient SET diagnosis = 'x' WHERE study_id = 'SYN-0001'",
    "DELETE FROM entip.cell WHERE scan_id = 'SCN-0001-A1-01'",
    "INSERT INTO entip.cell (cell_id, scan_id, frame, x_um, y_um, source) "
    "VALUES ('c', 'SCN-0001-A1-01', 'scan:SCN-0001-A1-01', 1, 1, 't')",
])
def test_writes_outside_an_action_are_refused(conn, stmt):
    with pytest.raises(psycopg.Error, match="must belong to an action"):
        conn.execute(stmt)


@pytest.mark.parametrize("stmt", ["UPDATE entip.action SET verb = 'x'", "DELETE FROM entip.change",
                                  "TRUNCATE entip.action CASCADE"])
def test_log_is_append_only(conn, stmt):
    with pytest.raises(psycopg.Error, match="append-only"):
        conn.execute(stmt)


def test_every_change_is_logged(conn):
    r = apply_action(conn, verb="update", object_type="patient", provenance=P,
                     records=[{"study_id": "SYN-0003", "clinical": {"extent_of_resection": "subtotal"}}])
    ch = conn.execute("SELECT * FROM entip.change WHERE action_id = %s", (r["action_id"],)).fetchall()
    assert len(ch) == 1 and ch[0]["op"] == "update" and ch[0]["study_id"] == "SYN-0003"
    assert ch[0]["before"]["clinical"] == {} and ch[0]["after"]["clinical"]["extent_of_resection"] == "subtotal"


def test_bulk_cells_log_one_change_per_statement(conn):
    cells = [{"cell_id": f"SCN-0002-A1-01:t{i}", "scan_id": "SCN-0002-A1-01", "x_um": i, "y_um": i,
              "source": "test"} for i in range(500)]
    r = apply_action(conn, verb="create", object_type="cell", records=cells, provenance=P)
    assert r["n"] == 500 and r["changes"] == 1
    ch = conn.execute("SELECT * FROM entip.change WHERE action_id = %s", (r["action_id"],)).fetchone()
    assert ch["n_rows"] == 500 and ch["study_id"] == "SYN-0002"


@pytest.mark.parametrize("rec,msg", [
    ({"study_id": "SYN-0100", "chi": "0101011234"}, "identifier"),
    ({"study_id": "SYN-0100", "clinical": {"date_of_birth": "x"}}, "identifier"),
    ({"study_id": "0101011234"}, "CHI"),
    ({"study_id": "SYN-0100", "favourite_colour": "x"}, "unknown field"),
    ({"study_id": "SYN-0100", "who_grade": 7}, "check constraint"),
    ({"study_id": "SYN-0100", "sex": "M"}, "check constraint"),
])
def test_bad_patient_records_are_refused(conn, rec, msg):
    with pytest.raises(EntipError, match=msg):
        apply_action(conn, verb="create", object_type="patient", records=[rec], provenance=P)
    assert conn.execute("SELECT count(*) AS n FROM entip.patient WHERE study_id = 'SYN-0100'").fetchone()["n"] == 0


def test_failed_action_leaves_nothing(conn):
    before = conn.execute("SELECT count(*) AS n FROM entip.action").fetchone()["n"]
    with pytest.raises(EntipError):
        apply_action(conn, verb="create", object_type="specimen", provenance=P,
                     records=[{"specimen_id": "SPC-X", "study_id": "SYN-0001"},
                              {"specimen_id": "SPC-Y", "study_id": "NOPE-404"}])
    assert conn.execute("SELECT count(*) AS n FROM entip.action").fetchone()["n"] == before
    assert conn.execute("SELECT count(*) AS n FROM entip.specimen WHERE specimen_id = 'SPC-X'").fetchone()["n"] == 0


def test_cell_frame_must_match_scan(conn):
    with pytest.raises(EntipError, match="check constraint"):
        apply_action(conn, verb="create", object_type="cell", provenance=P,
                     records=[{"cell_id": "bad", "scan_id": "SCN-0001-A1-01", "frame": "scan:SCN-0002-A1-01",
                               "x_um": 1, "y_um": 1, "source": "t"}])


def test_transform_needs_error(conn):
    with pytest.raises(EntipError, match="rmse_um"):
        apply_action(conn, verb="create", object_type="transform", provenance=P,
                     records=[{"transform_id": "tf-x", "src_frame": "assay:ASY-0001-VIS",
                               "dst_frame": "scan:SCN-0001-A1-01", "model": "affine",
                               "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}])


def test_spatial_object_in_scan_frame_needs_transform(conn):
    with pytest.raises(EntipError, match="check constraint"):
        apply_action(conn, verb="create", object_type="spatial_object", provenance=P,
                     records=[{"object_id": "o1", "assay_id": "ASY-0001-VIS", "kind": "spot", "barcode": "B1x",
                               "frame": "scan:SCN-0001-A1-01", "x_um": 1, "y_um": 1}])


def test_one_champion_per_feature(conn):
    base = {"feature": "necrosis", "kind": "live-head:xgboost", "encoder": "h-optimus-0"}
    apply_action(conn, verb="create", object_type="model", provenance=P,
                 records=[{**base, "model_id": "m-a", "status": "champion"}])
    with pytest.raises(EntipError, match="already exists"):
        apply_action(conn, verb="create", object_type="model", provenance=P,
                     records=[{**base, "model_id": "m-b", "status": "champion"}])
    # demote then promote in one action
    apply_action(conn, verb="create", object_type="model", provenance=P,
                 records=[{**base, "model_id": "m-b", "status": "challenger", "parent_id": "m-a"}])
    apply_action(conn, verb="update", object_type="model", provenance=P,
                 records=[{"model_id": "m-a", "status": "retired"}, {"model_id": "m-b", "status": "champion"}])
    assert conn.execute("SELECT model_id FROM entip.model WHERE feature = 'necrosis' AND status = 'champion'"
                        ).fetchone()["model_id"] == "m-b"


def test_patients_are_never_deleted(conn):
    with pytest.raises(EntipError, match="never deleted"):
        apply_action(conn, verb="delete", object_type="patient", records=[{"study_id": "SYN-0001"}], provenance=P)


def test_event_verbs_record_without_records(conn):
    r = apply_action(conn, verb="fork", object_type="scan", provenance={"kind": "program", "actor": "entip-fork"},
                     detail={"inputs": {"slide": "synthetic"}, "outputs": {"scan_id": "SCN-0001-A1-01"}})
    assert r["changes"] == 0
    with pytest.raises(EntipError, match="records an event only"):
        apply_action(conn, verb="fork", object_type="scan", provenance=P, records=[{"scan_id": "x"}])


def test_mixed_batch_like_pillar1_registration(conn):
    prov = {"kind": "program", "actor": "person:fork-tester", "tool": "entip-fork", "tool_version": "0.1.0"}
    objs = [
        {"type": "Patient", "study_id": "SYN-0200", "synthetic": True},
        {"type": "Specimen", "specimen_id": "SPC-0200-1", "study_id": "SYN-0200"},
        {"type": "Block", "block_id": "BLK-0200-A1", "specimen_id": "SPC-0200-1", "label": "A1"},
        {"type": "Section", "section_id": "SEC-0200-A1-01", "block_id": "BLK-0200-A1", "label": "1", "stain": "H&E"},
        {"type": "Scan", "scan_id": "LNK-0200", "link_code": "LNK-0200", "section_id": "SEC-0200-A1-01",
         "mpp": 0.25, "width_px": 1000, "height_px": 800, "format": "tiff", "meta": {"levels": [[1000, 800]]}},
    ]
    r = apply_action(conn, verb="upsert", object_type=None, records=objs, provenance=prov,
                     detail={"inputs": {"link_code": "LNK-0200"}})
    assert r["n"] == 5 and r["changes"] == 5
    row = conn.execute("SELECT * FROM entip.patient_row WHERE study_id = 'SYN-0200'").fetchone()
    assert row["scan_ids"] == ["LNK-0200"]
    studies = {c["study_id"] for c in conn.execute("SELECT study_id FROM entip.change WHERE action_id = %s",
                                                    (r["action_id"],))}
    assert studies == {"SYN-0200"}
