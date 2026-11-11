"""Hand-off of research-side objects to the central database (pillar 5).

Every registration is first appended to a JSON Lines outbox in the research
store, then posted to the database's ``POST /actions`` when a database URL is
given. The outbox is the durable queue: if the database is down, the fork still
succeeds and ``entip-fork replay`` posts what is waiting. Posting is safe to
repeat, because each registration carries its own ``action_id`` and the
database applies an ``action_id`` once.

Field names follow the database contract (``db/schema/README.md``): Scan
``width_px`` / ``height_px`` / ``format``, pyramid levels in ``meta``, positions
in the frame ``scan:<scan_id>``.

Nothing in these payloads is a direct identifier: no CHI, accession, name, date
of birth or original file name.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Protocol

from . import __version__

FORBIDDEN_KEYS = {"chi", "accession", "patient_name", "original_name", "clinical_uri", "dob", "name"}

# Parents in the chain are created once and then left alone; only the scan is refreshed on a re-fork.
_ID_FIELD = {"Patient": "study_id", "Specimen": "specimen_id", "Block": "block_id", "Section": "section_id",
             "Scan": "scan_id"}


def provenance(actor: str) -> dict:
    return {"kind": "program", "actor": actor, "tool": "entip-fork", "tool_version": __version__}


def build_objects(*, minted, block: str, section: str, stain: str, scan: dict, actor: str) -> list[dict]:
    prov = provenance(actor)
    objs = [
        {"type": "Patient", "study_id": minted.study_id},
        {"type": "Specimen", "specimen_id": minted.specimen_id, "study_id": minted.study_id},
        {"type": "Block", "block_id": minted.block_id, "specimen_id": minted.specimen_id, "label": block},
        {"type": "Section", "section_id": minted.section_id, "block_id": minted.block_id, "label": section, "stain": stain},
        {"type": "Scan", "scan_id": minted.link_code, "link_code": minted.link_code, "section_id": minted.section_id, **scan},
    ]
    for o in objs:
        o["provenance"] = prov
    return objs


def build_action(kind: str, actor: str, *, inputs: dict, outputs: dict) -> dict:
    return {
        "type": "Action",
        "action_id": str(uuid.uuid4()),
        "kind": kind,
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance(actor),
        "inputs": inputs,
        "outputs": outputs,
    }


def assert_no_identifiers(payload) -> None:
    if isinstance(payload, dict):
        bad = FORBIDDEN_KEYS & {str(k).lower() for k in payload}
        if bad:
            raise ValueError(f"identifier fields in research payload: {sorted(bad)}")
        for v in payload.values():
            assert_no_identifiers(v)
    elif isinstance(payload, list):
        for v in payload:
            assert_no_identifiers(v)


def to_db_action(objects: list[dict], action: dict) -> dict:
    """One ``POST /actions`` body: a mixed upsert of the objects, logged as the fork event."""
    return {
        "verb": "upsert",
        "object_type": None,
        "action_id": action["action_id"],
        "provenance": action["provenance"],
        "detail": {"event": action["kind"], "at": action["at"], "inputs": action["inputs"],
                   "outputs": action["outputs"]},
        "records": objects,
    }


class DbError(RuntimeError):
    """The central database refused a registration or could not be reached."""


class DbClient:
    """Minimal client for the central database API (Python standard library only)."""

    def __init__(self, base_url: str, token: str | None = None, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("ENTIP_DB_WRITE_TOKEN", "").split(",")[0].strip()
        self.timeout = timeout

    def _send(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                detail = json.loads(raw).get("detail")
            except (ValueError, AttributeError):
                detail = raw.decode("utf-8", "replace")[:300]
            return e.code, detail
        except (urllib.error.URLError, OSError) as e:
            raise DbError(f"cannot reach the database at {self.base}: {getattr(e, 'reason', e)}") from e

    def exists(self, object_type: str, object_id: str) -> bool:
        status, body = self._send("GET", f"/objects/{object_type}/{urllib.parse.quote(object_id, safe='')}")
        if status == 200:
            return True
        if status == 404:
            return False
        raise DbError(f"database answered {status} for {object_type} {object_id}: {body}")

    def post_action(self, body: dict) -> dict:
        status, out = self._send("POST", "/actions", body)
        if status not in (200, 201):
            raise DbError(f"database refused the action ({status}): {out}")
        return out


class Registry(Protocol):
    def register(self, objects: list[dict], action: dict) -> None: ...


class DbRegistry:
    """Posts a registration to the central database."""

    def __init__(self, client: DbClient):
        self.client = client

    def register(self, objects: list[dict], action: dict) -> dict:
        assert_no_identifiers({"objects": objects, "action": action})
        send = []
        for o in objects:
            if o["type"] != "Scan" and self.client.exists(o["type"].lower(), o[_ID_FIELD[o["type"]]]):
                continue  # already there: do not overwrite what others have added to it
            send.append(o)
        return self.client.post_action(to_db_action(send, action))


class OutboxRegistry:
    """Append-only JSON Lines outbox, one line per registration, plus a list of what has been posted."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sent_path = self.path.with_name(self.path.stem + ".sent.jsonl")

    def register(self, objects: list[dict], action: dict) -> None:
        record = {"objects": objects, "action": action}
        assert_no_identifiers(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def sent(self) -> set[str]:
        if not self.sent_path.exists():
            return set()
        return {json.loads(line)["action_id"] for line in self.sent_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}

    def mark_sent(self, action_id: str) -> None:
        with open(self.sent_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"action_id": action_id,
                                 "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}) + "\n")

    def pending(self) -> list[dict]:
        done = self.sent()
        return [r for r in self.records() if r["action"]["action_id"] not in done]

    def replay(self, db: DbRegistry) -> int:
        """Post every registration not yet posted, oldest first. Stops at the first failure."""
        n = 0
        for rec in self.pending():
            db.register(rec["objects"], rec["action"])
            self.mark_sent(rec["action"]["action_id"])
            n += 1
        return n
