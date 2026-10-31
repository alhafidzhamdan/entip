"""NHS-side key store.

This is the only place where CHI and accession numbers sit next to research IDs.
It must live on NHS infrastructure, apart from the research store, and is never
copied into the research database (pillar 5).

SQLite (Python standard library) keeps it a single file with no server. Every
lookup that re-identifies a research ID is written to the ``audit`` table.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import chi as chi_mod
from . import ids

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    study_id    TEXT PRIMARY KEY,
    chi         TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    created_by  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS specimens (
    specimen_id TEXT PRIMARY KEY,
    study_id    TEXT NOT NULL REFERENCES patients(study_id),
    accession   TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blocks (
    block_id    TEXT PRIMARY KEY,
    specimen_id TEXT NOT NULL REFERENCES specimens(specimen_id),
    label       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (specimen_id, label)
);
CREATE TABLE IF NOT EXISTS sections (
    section_id  TEXT PRIMARY KEY,
    block_id    TEXT NOT NULL REFERENCES blocks(block_id),
    label       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (block_id, label)
);
CREATE TABLE IF NOT EXISTS scans (
    link_code        TEXT PRIMARY KEY,
    section_id       TEXT NOT NULL REFERENCES sections(section_id),
    clinical_sha256  TEXT NOT NULL UNIQUE,
    clinical_uri     TEXT NOT NULL,
    original_name    TEXT NOT NULL,
    research_uri     TEXT,
    research_sha256  TEXT,
    status           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    created_by       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at      TEXT NOT NULL,
    actor   TEXT NOT NULL,
    action  TEXT NOT NULL,
    detail  TEXT NOT NULL
);
"""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class RealIdentifierRefused(ValueError):
    """Raised when a CHI looks like a real person's and real identifiers are not allowed."""


@dataclass(frozen=True)
class Minted:
    study_id: str
    specimen_id: str
    block_id: str
    section_id: str
    link_code: str
    new_scan: bool


class KeyStore:
    def __init__(self, path: str | os.PathLike, *, allow_real_identifiers: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        if new:
            os.chmod(self.path, 0o600)
        self.allow_real_identifiers = allow_real_identifiers

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- helpers ---------------------------------------------------------
    def _audit(self, actor: str, action: str, detail: dict) -> None:
        self.db.execute(
            "INSERT INTO audit (at, actor, action, detail) VALUES (?, ?, ?, ?)",
            (_now(), actor, action, json.dumps(detail, sort_keys=True)),
        )

    def _get_or_mint(self, table: str, id_col: str, kind: str, match: dict, extra: dict) -> str:
        where = " AND ".join(f"{k} = ?" for k in match)
        row = self.db.execute(f"SELECT {id_col} FROM {table} WHERE {where}", tuple(match.values())).fetchone()
        if row:
            return row[0]
        for _ in range(10):
            new = ids.new_id(kind)
            cols = [id_col, *match, *extra]
            try:
                self.db.execute(
                    f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    (new, *match.values(), *extra.values()),
                )
                return new
            except sqlite3.IntegrityError:
                continue  # random collision on the new id; try again
        raise RuntimeError(f"could not mint a unique {kind} id")

    def check_chi(self, chi: str) -> str:
        chi = chi_mod.normalise(chi)
        if not chi_mod.is_valid(chi):
            raise ValueError("CHI fails the format or check-digit test")
        if chi_mod.looks_real(chi) and not self.allow_real_identifiers:
            raise RealIdentifierRefused(
                "This CHI has a real date of birth. Real identifiers are refused until governance "
                "approval; use synthetic CHIs (day 00) or pass allow_real_identifiers=True."
            )
        return chi

    # -- minting ---------------------------------------------------------
    def mint(
        self,
        *,
        chi: str,
        accession: str,
        block: str,
        section: str,
        clinical_sha256: str,
        clinical_uri: str,
        original_name: str,
        actor: str,
    ) -> Minted:
        """Get or create IDs for the Patient > Specimen > Block > Section > Scan chain."""
        chi = self.check_chi(chi)
        now = _now()
        with self.db:
            existing = self.db.execute(
                "SELECT link_code FROM scans WHERE clinical_sha256 = ?", (clinical_sha256,)
            ).fetchone()
            study_id = self._get_or_mint("patients", "study_id", "P", {"chi": chi}, {"created_at": now, "created_by": actor})
            spec_row = self.db.execute("SELECT study_id FROM specimens WHERE accession = ?", (accession,)).fetchone()
            if spec_row and spec_row[0] != study_id:
                raise ValueError("accession is already linked to a different patient")
            specimen_id = self._get_or_mint(
                "specimens", "specimen_id", "S", {"accession": accession}, {"study_id": study_id, "created_at": now}
            )
            block_id = self._get_or_mint(
                "blocks", "block_id", "B", {"specimen_id": specimen_id, "label": block}, {"created_at": now}
            )
            section_id = self._get_or_mint(
                "sections", "section_id", "T", {"block_id": block_id, "label": section}, {"created_at": now}
            )
            if existing:
                link_code = existing[0]
                row = self.db.execute("SELECT section_id FROM scans WHERE link_code = ?", (link_code,)).fetchone()
                if row[0] != section_id:
                    raise ValueError("this slide file is already forked under a different case/block/section")
                self._audit(actor, "mint.reuse", {"link_code": link_code})
                return Minted(study_id, specimen_id, block_id, section_id, link_code, False)
            link_code = self._get_or_mint(
                "scans",
                "link_code",
                "L",
                {"clinical_sha256": clinical_sha256},
                {
                    "section_id": section_id,
                    "clinical_uri": clinical_uri,
                    "original_name": original_name,
                    "status": "minted",
                    "created_at": now,
                    "created_by": actor,
                },
            )
            self._audit(actor, "mint", {"link_code": link_code, "study_id": study_id})
        return Minted(study_id, specimen_id, block_id, section_id, link_code, True)

    def set_research_copy(self, link_code: str, uri: str, sha256: str, actor: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE scans SET research_uri = ?, research_sha256 = ?, status = 'forked' WHERE link_code = ?",
                (uri, sha256, link_code),
            )
            self._audit(actor, "research_copy", {"link_code": link_code, "sha256": sha256})

    def set_status(self, link_code: str, status: str, actor: str, detail: dict | None = None) -> None:
        with self.db:
            self.db.execute("UPDATE scans SET status = ? WHERE link_code = ?", (status, link_code))
            self._audit(actor, f"status.{status}", {"link_code": link_code, **(detail or {})})

    def scan(self, link_code: str) -> dict | None:
        cur = self.db.execute("SELECT * FROM scans WHERE link_code = ?", (link_code,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip([c[0] for c in cur.description], row))

    # -- re-identification (NHS side only) -------------------------------
    def reidentify(self, code: str, *, actor: str, reason: str) -> dict:
        """Return the clinical identifiers behind a research ID. Always audited."""
        if not reason.strip():
            raise ValueError("a reason is required to re-identify")
        code = ids.normalise(code)
        if not ids.is_valid(code):
            raise ValueError("not a valid ENTIP id (check character mismatch)")
        q = {
            "L": "WHERE sc.link_code = ?",
            "T": "WHERE se.section_id = ?",
            "B": "WHERE b.block_id = ?",
            "S": "WHERE sp.specimen_id = ?",
            "P": "WHERE p.study_id = ?",
        }[code[0]]
        cur = self.db.execute(
            f"""
            SELECT p.study_id, p.chi, sp.specimen_id, sp.accession, b.block_id, b.label AS block,
                   se.section_id, se.label AS section, sc.link_code, sc.clinical_uri, sc.original_name
            FROM patients p
            LEFT JOIN specimens sp ON sp.study_id = p.study_id
            LEFT JOIN blocks b ON b.specimen_id = sp.specimen_id
            LEFT JOIN sections se ON se.block_id = b.block_id
            LEFT JOIN scans sc ON sc.section_id = se.section_id
            {q}
            """,
            (code,),
        )
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        with self.db:
            self._audit(actor, "reidentify", {"code": code, "reason": reason, "rows": len(rows)})
        return {"code": code, "rows": rows}

    def audit_log(self) -> list[dict]:
        cur = self.db.execute("SELECT at, actor, action, detail FROM audit ORDER BY id")
        return [dict(at=a, actor=b, action=c, detail=json.loads(d)) for a, b, c, d in cur.fetchall()]
