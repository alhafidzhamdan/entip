"""Registration with the central database: outbox first, then POST /actions."""
import json
import urllib.request

import pytest

from conftest import fork_case
from entip_fork import cli, deid, registry, synth
from entip_fork.fork import Stores


class FakeDb:
    """Stands in for DbClient: remembers what was posted; can be switched off."""

    def __init__(self):
        self.have, self.posted, self.up = set(), [], True

    def exists(self, object_type, object_id):
        if not self.up:
            raise registry.DbError("cannot reach the database")
        return (object_type, object_id) in self.have

    def post_action(self, body):
        if not self.up:
            raise registry.DbError("cannot reach the database")
        self.posted.append(body)
        for r in body["records"]:
            self.have.add((r["type"].lower(), r[registry._ID_FIELD[r["type"]]]))
        return {"action_id": body["action_id"], "n": len(body["records"])}


@pytest.fixture
def fake(monkeypatch):
    db = FakeDb()
    monkeypatch.setattr(Stores, "db", lambda self: registry.DbRegistry(db) if self.db_url else None)
    return db


def test_objects_follow_the_database_contract(stores, case):
    r = fork_case(stores, case)
    rec = stores.registry().records()[0]
    scan = rec["objects"][-1]
    assert scan["scan_id"] == scan["link_code"] == r.link_code
    assert {"width_px", "height_px", "mpp", "format", "uri", "sha256", "meta"} <= set(scan)
    assert not {"width", "height", "levels", "source_format", "frame"} & set(scan)
    assert scan["mpp"] == pytest.approx(0.25) and scan["meta"]["source_format"] == "svs"
    assert all(o["provenance"]["kind"] == "program" for o in rec["objects"])
    body = registry.to_db_action(rec["objects"], rec["action"])
    assert body["verb"] == "upsert" and body["object_type"] is None
    assert body["action_id"] == rec["action"]["action_id"] and body["detail"]["event"] == "fork_slide"
    assert r.registered is False and r.register_error is None      # no database given: waits in the outbox


def test_fork_posts_to_the_database(tmp_path, case, fake):
    stores = Stores(tmp_path / "nhs", tmp_path / "research", db_url="http://db.invalid")
    r = fork_case(stores, case)
    assert r.registered and stores.registry().pending() == []
    assert [o["type"] for o in fake.posted[0]["records"]] == ["Patient", "Specimen", "Block", "Section", "Scan"]
    # a second section of the same block: only the new section and scan are sent
    other = synth.make_slide(tmp_path / "incoming" / "other.svs", seed=2)
    fork_case(stores, other, chi=case.chi, accession=case.accession, block=case.block, section="L2")
    assert [o["type"] for o in fake.posted[1]["records"]] == ["Section", "Scan"]


def test_database_down_keeps_the_fork_and_replay_posts_it(tmp_path, case, fake):
    stores = Stores(tmp_path / "nhs", tmp_path / "research", db_url="http://db.invalid")
    fake.up = False
    r = fork_case(stores, case)
    assert r.research_path.exists() and not r.registered and "cannot reach" in r.register_error
    assert len(stores.registry().pending()) == 1
    fake.up = True
    again = fork_case(stores, case)                    # the same slide again: nothing re-forked, queue flushed
    assert again.reused and again.registered
    assert len(fake.posted) == 1 and stores.registry().pending() == []
    assert stores.registry().replay(registry.DbRegistry(fake)) == 0


def test_crash_before_forked_status_is_repaired_on_the_next_run(stores, case, monkeypatch):
    from entip_fork.keystore import KeyStore

    def boom(self, *a, **k):
        raise RuntimeError("power cut")

    monkeypatch.setattr(KeyStore, "set_research_copy", boom)
    with pytest.raises(RuntimeError):
        fork_case(stores, case)
    monkeypatch.undo()
    r = fork_case(stores, case)
    assert not r.reused                                # forked again, and registered
    recs = stores.registry().records()
    assert recs and all(x["objects"][-1]["scan_id"] == r.link_code for x in recs)


def test_slide_without_pixel_size_needs_mpp(tmp_path, stores):
    import numpy as np
    import tifffile

    p = tmp_path / "in" / "nompp.tif"
    p.parent.mkdir()
    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), np.uint8)
    tifffile.imwrite(p, img, tile=(256, 256), photometric="rgb", metadata=None)
    with pytest.raises(ValueError, match="pixel size"):
        deid.write_research_copy(p, tmp_path / "out.tiff", research_meta={})
    assert not (tmp_path / "out.tiff").exists()
    info = deid.write_research_copy(p, tmp_path / "out.tiff", research_meta={}, mpp=0.5)
    assert info.mpp == 0.5
    with tifffile.TiffFile(tmp_path / "out.tiff") as t:
        assert json.loads(t.pages[0].description)["mpp"] == 0.5


def test_identifier_keys_are_refused_in_any_case():
    with pytest.raises(ValueError):
        registry.assert_no_identifiers({"objects": [{"CHI": "x"}]})


# -- against the real database API ---------------------------------------------

def _get(live_db, path):
    with urllib.request.urlopen(live_db["url"] + path) as r:
        return json.loads(r.read())


def test_fork_registers_in_the_real_database(tmp_path, case, live_db, monkeypatch, capsys):
    monkeypatch.setenv("ENTIP_DB_WRITE_TOKEN", live_db["token"])
    stores = Stores(tmp_path / "nhs", tmp_path / "research", db_url=live_db["url"])
    r = fork_case(stores, case)
    assert r.registered, r.register_error
    tree = _get(live_db, f"/patients/{r.study_id}")
    scan = tree["specimens"][0]["blocks"][0]["sections"][0]["scans"][0]
    assert scan["scan_id"] == r.link_code and scan["mpp"] == pytest.approx(0.25)
    assert scan["width_um"] == pytest.approx(scan["width_px"] * 0.25) and scan["stain"] == case.stain
    # the log shows the fork as one action, by the program, with no identifiers anywhere
    log = _get(live_db, f"/patients/{r.study_id}/actions")
    assert len(log) == 1 and log[0]["detail"]["event"] == "fork_slide"
    assert log[0]["provenance"]["tool"] == "entip-fork"
    blob = json.dumps([tree, log])
    for ident in (case.chi, case.accession, case.patient_name, "case.svs"):
        assert ident not in blob
    # replay after everything is posted does nothing; posting the same registration again changes nothing
    assert cli.main(["replay", "--research-root", str(stores.research_root), "--db", live_db["url"]]) == 0
    assert json.loads(capsys.readouterr().out) == {"posted": 0, "waiting": 0}
    rec = stores.registry().records()[0]
    out = registry.DbRegistry(registry.DbClient(live_db["url"])).register(rec["objects"], rec["action"])
    assert out["replayed"] is True
    assert len(_get(live_db, f"/patients/{r.study_id}/actions")) == 1


def test_wrong_token_leaves_the_registration_waiting(tmp_path, case, live_db, monkeypatch):
    monkeypatch.setenv("ENTIP_DB_WRITE_TOKEN", "not-the-token")
    stores = Stores(tmp_path / "nhs", tmp_path / "research", db_url=live_db["url"])
    r = fork_case(stores, case)
    assert not r.registered and "401" in r.register_error
    assert len(stores.registry().pending()) == 1
