"""Local stand-in for the db/ pillar: SQLite (Python stdlib), same objects as the ontology.

When db/ publishes its schema and API, replace this class with a client that has the same methods.
Everything that changes state goes through ``log`` (an append-only action log); rows are never
edited except to mark status (e.g. an annotation retracted, a model promoted), and each such change
has a matching action.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

from .ontology import Annotation, Cell, Feature, Model, Region, Scan

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (scan_id TEXT PRIMARY KEY, path TEXT, mpp REAL, width_um REAL, height_um REAL,
                                  section_id TEXT);
CREATE TABLE IF NOT EXISTS features (name TEXT PRIMARY KEY, level TEXT, description TEXT, created_by TEXT,
                                     positives_only INTEGER);
CREATE TABLE IF NOT EXISTS cells (cell_id TEXT PRIMARY KEY, scan_id TEXT, x_um REAL, y_um REAL, source TEXT);
CREATE INDEX IF NOT EXISTS cells_scan ON cells(scan_id);
CREATE TABLE IF NOT EXISTS regions (region_id TEXT PRIMARY KEY, scan_id TEXT, x_um REAL, y_um REAL, w_um REAL,
                                    h_um REAL, source TEXT);
CREATE INDEX IF NOT EXISTS regions_scan ON regions(scan_id);
CREATE TABLE IF NOT EXISTS annotations (annotation_id TEXT PRIMARY KEY, scan_id TEXT, feature TEXT, label TEXT,
                                        geometry TEXT, author TEXT, created_at TEXT, locked INTEGER,
                                        retracted INTEGER);
CREATE INDEX IF NOT EXISTS ann_feature ON annotations(feature);
CREATE TABLE IF NOT EXISTS models (model_id TEXT PRIMARY KEY, feature TEXT, kind TEXT, encoder TEXT, status TEXT,
                                   trained_by TEXT, n_labels INTEGER, created_at TEXT, parent TEXT, metrics TEXT,
                                   artifact TEXT);
CREATE TABLE IF NOT EXISTS scores (object_id TEXT, feature TEXT, model_id TEXT, p REAL,
                                   PRIMARY KEY (object_id, feature, model_id));
CREATE TABLE IF NOT EXISTS actions (seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, actor TEXT, verb TEXT,
                                    object_type TEXT, object_id TEXT, detail TEXT);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    # ------------------------------------------------------------ action log
    def log(self, actor: str, verb: str, object_type: str, object_id: str, detail: Optional[dict] = None) -> None:
        with self.lock:
            self.db.execute("INSERT INTO actions (at, actor, verb, object_type, object_id, detail) VALUES (?,?,?,?,?,?)",
                            (now(), actor, verb, object_type, object_id, json.dumps(detail or {}, default=float)))
            self.db.commit()

    def actions(self, object_id: Optional[str] = None) -> List[dict]:
        q, a = "SELECT * FROM actions", ()
        if object_id:
            q, a = q + " WHERE object_id = ?", (object_id,)
        rows = self.db.execute(q + " ORDER BY seq", a).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    # ------------------------------------------------------------ scans, cells, regions
    def add_scan(self, scan: Scan, actor: str = "system") -> Scan:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO scans VALUES (?,?,?,?,?,?)", tuple(asdict(scan).values()))
            self.db.commit()
        self.log(actor, "register", "Scan", scan.scan_id, {"path": scan.path, "mpp": scan.mpp})
        return scan

    def scan(self, scan_id: str) -> Scan:
        r = self.db.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
        if r is None:
            raise KeyError(f"unknown scan {scan_id}")
        return Scan(**dict(r))

    def add_cells(self, cells: Iterable[Cell], actor: str = "system") -> int:
        rows = [tuple(asdict(c).values()) for c in cells]
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO cells VALUES (?,?,?,?,?)", rows)
            self.db.commit()
        if rows:
            self.log(actor, "add_cells", "Scan", rows[0][1], {"n": len(rows), "source": rows[0][4]})
        return len(rows)

    def cells(self, scan_id: str) -> List[Cell]:
        rows = self.db.execute("SELECT * FROM cells WHERE scan_id = ? ORDER BY rowid", (scan_id,)).fetchall()
        return [Cell(**dict(r)) for r in rows]

    def add_regions(self, regions: Iterable[Region], actor: str = "system") -> int:
        rows = [tuple(asdict(r).values()) for r in regions]
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO regions VALUES (?,?,?,?,?,?,?)", rows)
            self.db.commit()
        if rows:
            self.log(actor, "add_regions", "Scan", rows[0][1], {"n": len(rows)})
        return len(rows)

    def regions(self, scan_id: str) -> List[Region]:
        rows = self.db.execute("SELECT * FROM regions WHERE scan_id = ? ORDER BY rowid", (scan_id,)).fetchall()
        return [Region(**dict(r)) for r in rows]

    # ------------------------------------------------------------ features
    def add_feature(self, f: Feature) -> Feature:
        with self.lock:
            self.db.execute("INSERT INTO features VALUES (?,?,?,?,?)",
                            (f.name, f.level, f.description, f.created_by, int(f.positives_only)))
            self.db.commit()
        self.log(f.created_by or "system", "create", "Feature", f.name, {"level": f.level})
        return f

    def feature(self, name: str) -> Feature:
        r = self.db.execute("SELECT * FROM features WHERE name = ?", (name,)).fetchone()
        if r is None:
            raise KeyError(f"unknown feature {name}")
        d = dict(r)
        d["positives_only"] = bool(d["positives_only"])
        return Feature(**d)

    def features(self) -> List[Feature]:
        return [self.feature(r["name"]) for r in self.db.execute("SELECT name FROM features ORDER BY name")]

    # ------------------------------------------------------------ annotations
    def add_annotation(self, a: Annotation) -> Annotation:
        a.created_at = a.created_at or now()
        with self.lock:
            self.db.execute("INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?,?)",
                            (a.annotation_id, a.scan_id, a.feature, a.label, json.dumps(a.geometry), a.author,
                             a.created_at, int(a.locked), int(a.retracted)))
            self.db.commit()
        self.log(a.author, "annotate", "Annotation", a.annotation_id,
                 {"feature": a.feature, "label": a.label, "geometry": a.geometry, "locked": a.locked})
        return a

    def retract_annotation(self, annotation_id: str, actor: str) -> None:
        with self.lock:
            cur = self.db.execute("UPDATE annotations SET retracted = 1 WHERE annotation_id = ?", (annotation_id,))
            self.db.commit()
        if cur.rowcount == 0:
            raise KeyError(f"unknown annotation {annotation_id}")
        self.log(actor, "retract", "Annotation", annotation_id)

    def annotations(self, feature: str, scan_id: Optional[str] = None, locked: Optional[bool] = None,
                    include_retracted: bool = False) -> List[Annotation]:
        q, a = "SELECT * FROM annotations WHERE feature = ?", [feature]
        if scan_id is not None:
            q += " AND scan_id = ?"; a.append(scan_id)
        if locked is not None:
            q += " AND locked = ?"; a.append(int(locked))
        if not include_retracted:
            q += " AND retracted = 0"
        out = []
        for r in self.db.execute(q + " ORDER BY created_at, rowid", a).fetchall():
            d = dict(r)
            d.update(geometry=json.loads(d["geometry"]), locked=bool(d["locked"]), retracted=bool(d["retracted"]))
            out.append(Annotation(**d))
        return out

    # ------------------------------------------------------------ models and scores
    def add_model(self, m: Model) -> Model:
        m.created_at = m.created_at or now()
        with self.lock:
            self.db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (m.model_id, m.feature, m.kind, m.encoder, m.status, m.trained_by, m.n_labels,
                             m.created_at, m.parent, json.dumps(m.metrics, default=float), m.artifact))
            self.db.commit()
        self.log(m.trained_by, "train", "Model", m.model_id,
                 {"feature": m.feature, "kind": m.kind, "status": m.status, "n_labels": m.n_labels, "parent": m.parent})
        return m

    def set_model_status(self, model_id: str, status: str, actor: str, detail: Optional[dict] = None) -> None:
        with self.lock:
            self.db.execute("UPDATE models SET status = ? WHERE model_id = ?", (status, model_id))
            self.db.commit()
        self.log(actor, f"set_status:{status}", "Model", model_id, detail)

    def _model(self, r) -> Model:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"])
        return Model(**d)

    def model(self, model_id: str) -> Model:
        r = self.db.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if r is None:
            raise KeyError(f"unknown model {model_id}")
        return self._model(r)

    def models(self, feature: Optional[str] = None, status: Optional[str] = None) -> List[Model]:
        q, a = "SELECT * FROM models WHERE 1=1", []
        if feature:
            q += " AND feature = ?"; a.append(feature)
        if status:
            q += " AND status = ?"; a.append(status)
        return [self._model(r) for r in self.db.execute(q + " ORDER BY created_at, rowid", a).fetchall()]

    def latest_model(self, feature: str, status: Optional[str] = None) -> Optional[Model]:
        ms = self.models(feature, status)
        return ms[-1] if ms else None

    def put_scores(self, feature: str, model_id: str, object_ids: List[str], p: List[float]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
                                [(o, feature, model_id, float(v)) for o, v in zip(object_ids, p)])
            self.db.commit()
