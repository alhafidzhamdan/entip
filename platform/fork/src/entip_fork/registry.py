"""Hand-off of research-side objects to the central database (pillar 5).

Pillar 5 owns the schema. Until its API is up, objects go to an append-only
JSON Lines outbox in the research store; ``replay`` will post them once the API
exists. Field names here follow the shared ontology in docs/pillar_specs.md
(Patient > Specimen > Block > Section > Scan, provenance on every object, changes
as logged actions) and will be renamed to match the pillar 5 schema when it lands.

Nothing in these payloads is a direct identifier: no CHI, accession, name, date
of birth or original file name.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from pathlib import Path
from typing import Protocol

from . import __version__

FORBIDDEN_KEYS = {"chi", "accession", "patient_name", "original_name", "clinical_uri", "dob", "name"}


def provenance(actor: str) -> dict:
    return {"actor": actor, "tool": "entip-fork", "tool_version": __version__}


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
        bad = FORBIDDEN_KEYS & set(payload)
        if bad:
            raise ValueError(f"identifier fields in research payload: {sorted(bad)}")
        for v in payload.values():
            assert_no_identifiers(v)
    elif isinstance(payload, list):
        for v in payload:
            assert_no_identifiers(v)


class Registry(Protocol):
    def register(self, objects: list[dict], action: dict) -> None: ...


class OutboxRegistry:
    """Append-only JSON Lines outbox, one line per registration."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, objects: list[dict], action: dict) -> None:
        record = {"objects": objects, "action": action}
        assert_no_identifiers(record)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
