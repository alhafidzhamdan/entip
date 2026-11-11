"""Safe retries, clean errors, identifier checks on values, and token access."""
import pytest
from fastapi.testclient import TestClient

from entip_db.api import create_app
from entip_db.db import apply_action, chi_check_ok
from entip_db.db_errors import EntipError

PROV = {"kind": "person", "actor": "person:tester"}


def _synthetic_chi() -> str:
    """CHI-shaped, valid check digit, day 00 (not a real date of birth, so nobody's CHI)."""
    for n in range(1000):
        first9 = f"000199{n:03d}"
        for d in range(10):
            if chi_check_ok(first9 + str(d)):
                return first9 + str(d)
    raise AssertionError("no synthetic CHI found")


def test_retry_with_same_action_id_is_applied_once(client):
    body = {"verb": "create", "object_type": "feature", "provenance": PROV, "action_id": "act_retry_feature",
            "records": [{"name": "retry_feature", "level": "cell"}]}
    first = client.post("/actions", json=body)
    assert first.status_code == 201 and first.json()["replayed"] is False
    again = client.post("/actions", json=body)
    assert again.status_code == 200
    assert again.json() == {**first.json(), "replayed": True}
    log = client.get("/objects/feature/retry_feature/history").json()
    assert len(log) == 1


def test_action_id_cannot_be_reused_for_another_action(client):
    client.post("/actions", json={"verb": "note", "provenance": PROV, "action_id": "act_reuse"})
    r = client.post("/actions", json={"verb": "other_note", "provenance": PROV, "action_id": "act_reuse"})
    assert r.status_code == 422 and "already used" in r.json()["detail"]
    r = client.post("/actions", json={"verb": "note", "provenance": PROV, "action_id": "bad id!"})
    assert r.status_code == 422


@pytest.mark.parametrize("path,params", [
    ("/actions", {"since": "garbage"}),
    ("/patients", {"limit": -1}),
    ("/patients", {"offset": -1}),
    ("/scans/SCN-0001-A1-01/cells", {"limit": 0}),
    ("/scans/SCN-0001-A1-01/cells", {"bbox": "nan,0,1,1"}),
])
def test_bad_query_values_are_422_not_500(client, path, params):
    assert client.get(path, params=params).status_code == 422


def test_since_filters_the_log(client):
    assert client.get("/actions", params={"since": "2999-01-01T00:00:00Z"}).json() == []
    assert client.get("/actions", params={"since": "2000-01-01T00:00:00Z", "limit": 1}).json()


@pytest.mark.parametrize("record", [
    {"name": "nul\u0000byte", "level": "cell"},
    {"name": "f_bad_prov", "level": "cell", "provenance": "someone"},
    "not an object",
])
def test_bad_records_are_422(client, record):
    r = client.post("/actions", json={"verb": "create", "object_type": "feature", "provenance": PROV,
                                      "records": [record]})
    assert r.status_code == 422, r.text


@pytest.mark.parametrize("geometry", [
    {"type": "point", "coordinates": [1]},
    {"type": "point", "coordinates": 5},
    {"type": "polygon", "coordinates": []},
    {"type": "polygon", "points_um": [[0, 0], [1], [2, 2]]},
    {"type": "polygon", "points_um": "abc"},
])
def test_malformed_geometry_is_422(client, geometry):
    r = client.post("/actions", json={"verb": "create", "object_type": "annotation", "provenance": PROV,
                                      "records": [{"annotation_id": "ann_bad_geom", "scan_id": "SCN-0001-A1-01",
                                                   "feature": "mitotic_figure", "label": "positive",
                                                   "geometry": geometry}]})
    assert r.status_code == 422, r.text


def test_empty_actor_is_refused(client):
    for actor in ("", "   "):
        r = client.post("/actions", json={"verb": "note", "provenance": {"actor": actor}})
        assert r.status_code == 422


def test_chi_in_a_text_value_is_refused(conn):
    chi = _synthetic_chi()
    for text in (chi, f"seen in clinic, CHI {chi}", f"{chi[:6]} {chi[6:]}", f"{chi[:6]}-{chi[6:]}"):
        with pytest.raises(EntipError, match="CHI check digit"):
            apply_action(conn, verb="note", object_type=None, provenance=PROV, detail={"note": text})
    with pytest.raises(EntipError, match="CHI check digit"):
        apply_action(conn, verb="create", object_type="specimen", provenance=PROV,
                     records=[{"specimen_id": "SP-chi-test", "study_id": "SYN-0001", "site": chi}])


def test_numbers_that_are_not_a_chi_are_accepted(conn):
    chi = _synthetic_chi()
    wrong_check = chi[:9] + str((int(chi[9]) + 1) % 10)
    sha = "a" * 27 + chi + "b" * 27                      # a 10-digit run inside a hash is not a CHI
    detail = {"n": int(chi), "wrong_check": wrong_check, "sha256": sha, "longer": chi + "7", "code": f"X{chi}"}
    assert apply_action(conn, verb="note", object_type=None, provenance=PROV, detail=detail)["changes"] == 0


# -- tokens -----------------------------------------------------------------

def test_open_when_no_tokens_are_set(seeded):
    c = TestClient(create_app(seeded, write_token="", read_token=""))
    assert c.get("/patients").status_code == 200


def test_write_token_guards_writes_only(seeded):
    c = TestClient(create_app(seeded, write_token="w-secret", read_token=""))
    body = {"verb": "note", "provenance": PROV}
    assert c.get("/patients").status_code == 200
    assert c.post("/actions", json=body).status_code == 401
    assert c.post("/actions", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.post("/actions", json=body, headers={"Authorization": "Basic w-secret"}).status_code == 401
    assert c.post("/actions", json=body, headers={"Authorization": "Bearer w-secret"}).status_code == 201


def test_read_token_guards_reads_and_cannot_write(seeded):
    c = TestClient(create_app(seeded, write_token="w-old, w-new", read_token="r-secret"))
    r, w = {"Authorization": "Bearer r-secret"}, {"Authorization": "Bearer w-new"}
    assert c.get("/health").status_code == 200           # liveness stays open
    assert c.get("/patients").status_code == 401
    assert c.get("/patients", headers=r).status_code == 200
    assert c.get("/patients", headers=w).status_code == 200
    assert c.get("/patients", headers={"Authorization": "Bearer w-old"}).status_code == 200
    body = {"verb": "note", "provenance": PROV}
    assert c.post("/actions", json=body, headers=r).status_code == 401
    assert c.post("/actions", json=body, headers=w).status_code == 201


def test_read_token_alone_turns_writes_off(seeded):
    c = TestClient(create_app(seeded, write_token="", read_token="r-secret"))
    r = c.post("/actions", json={"verb": "note", "provenance": PROV}, headers={"Authorization": "Bearer r-secret"})
    assert r.status_code == 403


def test_tokens_come_from_the_environment(seeded, monkeypatch):
    monkeypatch.setenv("ENTIP_DB_WRITE_TOKEN", "from-env")
    c = TestClient(create_app(seeded))
    assert c.post("/actions", json={"verb": "note", "provenance": PROV}).status_code == 401


def test_serve_refuses_a_public_address_without_a_token(seeded, monkeypatch, capsys):
    from entip_db.cli import main
    monkeypatch.delenv("ENTIP_DB_WRITE_TOKEN", raising=False)
    assert main(["--dsn", seeded, "serve", "--host", "0.0.0.0"]) == 2
    assert "ENTIP_DB_WRITE_TOKEN" in capsys.readouterr().err
