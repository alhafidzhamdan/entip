"""Client for the db/ pillar's HTTP API with the same methods as ``store.Store``.

Use it instead of the SQLite stand-in when the central database is running:

    svc = TrainingService(DbStore("http://localhost:8000"), encoder, artifacts)

Reads scans, cells, regions, features, annotations and models from the database, and writes features,
annotations, models and predictions back through ``POST /actions`` (the database's only write path, so
every change lands in its action log). Cell and region ids are the database's own, and feature keys are
used exactly as the database stores them (e.g. ``mitotic_figure``).

Field mapping (from db/schema/README.md): ``author`` / ``created_by`` / ``trained_by`` become
``provenance``; Model ``artifact`` becomes ``uri`` and ``parent`` becomes ``parent_id``; scores go in
``prediction``. A mark on one object is stored as a point at the object's centre with
``target_type`` / ``target_id``.

HTTP uses the standard library only. Pass ``http=`` any object with ``get(path, params=)`` and
``post(path, json=)`` returning responses with ``status_code`` and ``json()`` (e.g. a FastAPI TestClient).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, List, Optional

from . import __version__
from .ontology import Annotation, Cell, Feature, Model, Region, Scan
from .store import new_id

PROGRAM = "entip-train"


class DbError(RuntimeError):
    pass


class _Resp:
    def __init__(self, status: int, body: bytes):
        self.status_code, self._body = status, body

    def json(self):
        return json.loads(self._body or b"null")

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", "replace")


class UrllibHttp:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base, self.timeout = base_url.rstrip("/"), timeout

    def _send(self, method: str, path: str, params=None, body=None) -> _Resp:
        url = self.base + path + ("?" + urllib.parse.urlencode(
            {k: v for k, v in (params or {}).items() if v is not None}) if params else "")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return _Resp(r.status, r.read())
        except urllib.error.HTTPError as e:
            return _Resp(e.code, e.read())

    def get(self, path: str, params=None) -> _Resp:
        return self._send("GET", path, params)

    def post(self, path: str, json=None) -> _Resp:
        return self._send("POST", path, body=json)


def _provenance(actor: str, **extra) -> dict:
    if actor.startswith("person:"):
        return {"kind": "person", "actor": actor, **extra}
    if actor.startswith("model:"):
        return {"kind": "model", "actor": actor, **extra}
    return {"kind": "program", "actor": actor or PROGRAM, **extra}


def _actor(prov: Optional[dict]) -> str:
    return (prov or {}).get("actor", "")


def _bbox(geojson: dict):
    xs = [p[0] for ring in geojson["coordinates"] for p in ring]
    ys = [p[1] for ring in geojson["coordinates"] for p in ring]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


class DbStore:
    def __init__(self, base_url: str = "http://localhost:8000", http=None, scan_ttl_s: float = 60.0):
        self.http = http or UrllibHttp(base_url)
        self.scan_ttl_s = scan_ttl_s
        self._scans: Optional[List[str]] = None
        self._scans_at = 0.0
        self._cells: dict = {}
        self._cross_scan: Optional[bool] = None

    # ------------------------------------------------------------ plumbing
    def _get(self, path: str, **params):
        r = self.http.get(path, params=params or None)
        if r.status_code == 404:
            raise KeyError(f"{path}: not found in the database")
        if r.status_code >= 400:
            raise DbError(f"GET {path} -> {r.status_code}: {r.text}")
        return r.json()

    def _act(self, verb: str, object_type: Optional[str], records: list, actor: str,
             detail: Optional[dict] = None) -> dict:
        body = {"verb": verb, "object_type": object_type, "records": records,
                "provenance": _provenance(actor, tool=PROGRAM, tool_version=__version__),
                "detail": detail or {}, "action_id": new_id("act")}
        r = self.http.post("/actions", json=body)
        if r.status_code >= 400:
            raise DbError(f"POST /actions {verb} {object_type} -> {r.status_code}: {r.text}")
        return r.json()

    # ------------------------------------------------------------ action log
    def log(self, actor: str, verb: str, object_type: str, object_id: str, detail: Optional[dict] = None) -> None:
        v = "".join(ch if ch.isalnum() else "_" for ch in verb.lower())
        self._act(v, None, [], actor, {"object_type": object_type, "object_id": object_id, **(detail or {})})

    def actions(self, object_id: Optional[str] = None) -> List[dict]:
        rows = self._get("/actions", limit=5000)
        if object_id:
            rows = [a for a in rows if object_id in json.dumps(a, default=str)]
        return rows

    # ------------------------------------------------------------ scans, cells, regions
    def scan(self, scan_id: str) -> Scan:
        d = self._get(f"/scans/{urllib.parse.quote(scan_id, safe='')}")
        return Scan(d["scan_id"], d.get("tile_source") or d.get("uri") or "", float(d["mpp"]),
                    float(d.get("width_um") or d["width_px"] * d["mpp"]),
                    float(d.get("height_um") or d["height_px"] * d["mpp"]), d.get("section_id"))

    def scan_ids(self) -> List[str]:
        """Every scan in the database (walks the patient tree; cached for ``scan_ttl_s``)."""
        if self._scans is None or time.time() - self._scans_at > self.scan_ttl_s:
            out = []
            for p in self._get("/patients", limit=1000):
                tree = self._get(f"/patients/{urllib.parse.quote(p['study_id'], safe='')}")
                for sp in tree.get("specimens", []):
                    for b in sp.get("blocks", []):
                        for se in b.get("sections", []):
                            out += [s["scan_id"] for s in se.get("scans", [])]
            self._scans, self._scans_at = out, time.time()
        return self._scans

    def add_scan(self, scan: Scan, actor: str = PROGRAM) -> Scan:
        """For tests and demos only: pillar 1 registers real scans. Needs an existing section."""
        if not scan.section_id:
            raise ValueError("the database needs section_id for a scan")
        self._act("upsert", "scan", [{"scan_id": scan.scan_id, "section_id": scan.section_id, "mpp": scan.mpp,
                                      "width_px": int(round(scan.width_um / scan.mpp)),
                                      "height_px": int(round(scan.height_um / scan.mpp)), "uri": scan.path}], actor)
        self._scans = None
        return scan

    def add_cells(self, cells: Iterable[Cell], actor: str = PROGRAM) -> int:
        recs = [{"cell_id": c.cell_id, "scan_id": c.scan_id, "x_um": c.x_um, "y_um": c.y_um, "source": c.source}
                for c in cells]
        if recs:
            self._act("upsert", "cell", recs, actor)
            self._cells.pop(recs[0]["scan_id"], None)
        return len(recs)

    def cells(self, scan_id: str) -> List[Cell]:
        if scan_id not in self._cells:
            out, after = [], None
            while True:
                page = self._get(f"/scans/{urllib.parse.quote(scan_id, safe='')}/cells", after=after, limit=50000)
                out += [Cell(c["cell_id"], c["scan_id"], float(c["x_um"]), float(c["y_um"]), c.get("source", ""))
                        for c in page]
                if len(page) < 50000:
                    break
                after = page[-1]["cell_id"]
            self._cells[scan_id] = out
        return self._cells[scan_id]

    def add_regions(self, regions: Iterable[Region], actor: str = PROGRAM) -> int:
        recs = [{"region_id": r.region_id, "scan_id": r.scan_id, "x_um": r.x_um, "y_um": r.y_um, "w_um": r.w_um,
                 "h_um": r.h_um, "kind": "tile", "source": r.source} for r in regions]
        if recs:
            self._act("upsert", "region", recs, actor)
        return len(recs)

    def regions(self, scan_id: str) -> List[Region]:
        """Tiles only: live tile heads train on the grid, not on hotspots or drawn polygons."""
        out = []
        for r in self._get(f"/scans/{urllib.parse.quote(scan_id, safe='')}/regions", kind="tile"):
            x, y, w, h = _bbox(r["geojson"])
            out.append(Region(r["region_id"], r["scan_id"], x, y, w, h, r.get("source", "")))
        return out

    # ------------------------------------------------------------ features
    def add_feature(self, f: Feature) -> Feature:
        self._act("upsert", "feature", [{"name": f.name, "level": f.level, "description": f.description,
                                         "positives_only": f.positives_only,
                                         "provenance": _provenance(f.created_by or PROGRAM)}], f.created_by or PROGRAM)
        return f

    def _feature(self, d: dict) -> Feature:
        return Feature(d["name"], d["level"], d.get("description", ""), _actor(d.get("provenance")),
                       bool(d.get("positives_only")))

    def feature(self, name: str) -> Feature:
        for d in self._get("/features"):
            if d["name"] == name:
                return self._feature(d)
        raise KeyError(f"unknown feature {name}")

    def features(self) -> List[Feature]:
        return [self._feature(d) for d in self._get("/features")]

    # ------------------------------------------------------------ annotations
    def _cell_xy(self, scan_id: str, cell_id: str):
        for c in self.cells(scan_id):
            if c.cell_id == cell_id:
                return c.x_um, c.y_um
        raise KeyError(f"unknown cell {cell_id} on scan {scan_id}")

    def get_annotation(self, scan_id: str, annotation_id: str) -> Optional[Annotation]:
        for a in self._annotations_on(scan_id, None, include_retracted=True):
            if a.annotation_id == annotation_id:
                return a
        return None

    def add_annotation(self, a: Annotation) -> Annotation:
        rec = {"annotation_id": a.annotation_id, "scan_id": a.scan_id, "feature": a.feature, "label": a.label,
               "locked": a.locked, "provenance": _provenance(a.author)}
        g = a.geometry
        if g.get("type") == "object":
            oid = g["object_id"]
            is_cell = any(c.cell_id == oid for c in self.cells(a.scan_id))
            if is_cell:
                x, y = self._cell_xy(a.scan_id, oid)
            else:
                reg = next((r for r in self.regions(a.scan_id) if r.region_id == oid), None)
                if reg is None:
                    raise KeyError(f"unknown object {oid} on scan {a.scan_id}")
                x, y = reg.centre
            rec.update(geometry={"type": "point", "x_um": x, "y_um": y},
                       target_type="cell" if is_cell else "region", target_id=oid)
        else:
            rec["geometry"] = g
        self._act("create", "annotation", [rec], a.author)
        return a

    def retract_annotation(self, annotation_id: str, actor: str) -> None:
        self._act("retract", "annotation", [{"annotation_id": annotation_id}], actor)

    def _annotations_on(self, scan_id: str, feature: Optional[str], include_retracted: bool) -> List[Annotation]:
        rows = self._get(f"/scans/{urllib.parse.quote(scan_id, safe='')}/annotations", feature=feature,
                         include_retracted=str(include_retracted).lower())
        return self._parse_annotations(rows)

    def _parse_annotations(self, rows: list) -> List[Annotation]:
        out = []
        for d in rows:
            g = d["geometry"]
            if d.get("target_id") and d.get("target_type") in ("cell", "region"):
                g = {"type": "object", "object_id": d["target_id"]}
            elif g.get("type") == "box":
                x, y, w, h = (float(g[k]) for k in ("x_um", "y_um", "w_um", "h_um"))
                g = {"type": "polygon", "points_um": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]}
            out.append(Annotation(d["annotation_id"], d["scan_id"], d["feature"], d["label"], g,
                                  _actor(d.get("provenance")), str(d.get("created_at", "")), bool(d.get("locked")),
                                  bool(d.get("retracted"))))
        return out

    def annotations(self, feature: str, scan_id: Optional[str] = None, locked: Optional[bool] = None,
                    include_retracted: bool = False) -> List[Annotation]:
        if scan_id is None and self._cross_scan is not False:     # GET /annotations (db/ PR #8 onwards)
            params = {"feature": feature, "include_retracted": str(include_retracted).lower(), "limit": 1000000}
            if locked is not None:
                params["locked"] = str(locked).lower()
            r = self.http.get("/annotations", params=params)
            if r.status_code in (404, 405):
                self._cross_scan = False                            # older db/: walk the scans instead
            elif r.status_code >= 400:
                raise DbError(f"GET /annotations -> {r.status_code}: {r.text}")
            else:
                self._cross_scan = True
                return sorted(self._parse_annotations(r.json()), key=lambda a: a.created_at)
        scans = [scan_id] if scan_id is not None else self.scan_ids()
        out = []
        for s in scans:
            out += [a for a in self._annotations_on(s, feature, include_retracted)
                    if locked is None or a.locked == locked]
        out.sort(key=lambda a: a.created_at)
        return out

    # ------------------------------------------------------------ models and scores
    def add_model(self, m: Model) -> Model:
        self._act("upsert", "model", [{"model_id": m.model_id, "feature": m.feature, "kind": m.kind,
                                       "encoder": m.encoder, "status": m.status, "parent_id": m.parent,
                                       "n_labels": m.n_labels, "metrics": m.metrics, "uri": m.artifact,
                                       "version": __version__,
                                       "provenance": _provenance(PROGRAM, version=__version__,
                                                                 requested_by=m.trained_by)}],
                  m.trained_by or PROGRAM)
        return m

    def set_model_status(self, model_id: str, status: str, actor: str, detail: Optional[dict] = None) -> None:
        self._act("update", "model", [{"model_id": model_id, "status": status}], actor, detail)

    def _model(self, d: dict) -> Model:
        prov = d.get("provenance") or {}
        return Model(d["model_id"], d.get("feature") or "", d["kind"], d.get("encoder") or "", d["status"],
                     prov.get("requested_by") or prov.get("actor", ""), int(d.get("n_labels") or 0),
                     str(d.get("created_at", "")), d.get("parent_id"), d.get("metrics") or {}, d.get("uri") or "")

    def model(self, model_id: str) -> Model:
        for d in self._get("/models"):
            if d["model_id"] == model_id:
                return self._model(d)
        raise KeyError(f"unknown model {model_id}")

    def models(self, feature: Optional[str] = None, status: Optional[str] = None) -> List[Model]:
        ms = [self._model(d) for d in self._get("/models", feature=feature, status=status)]
        return sorted(ms, key=lambda m: m.created_at)

    def latest_model(self, feature: str, status: Optional[str] = None) -> Optional[Model]:
        ms = self.models(feature, status)
        return ms[-1] if ms else None

    def put_scores(self, feature: str, model_id: str, object_ids: List[str], p: List[float],
                   scan_id: Optional[str] = None, target_type: str = "cell") -> None:
        recs = [{"model_id": model_id, "target_type": target_type, "target_id": o, "scan_id": scan_id,
                 "score": float(v)} for o, v in zip(object_ids, p)]
        if recs:
            self._act("upsert", "prediction", recs, PROGRAM, {"feature": feature, "n": len(recs)})
