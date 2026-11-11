"""HTTP API for the central database (FastAPI).

Reads: patients (one row each), the patient tree, scans and their cells /
regions / annotations / predictions, assays and spatial objects, transforms,
features, models, the action log.

Writes: only ``POST /actions``. One request is one logged action, applied in
one transaction.

Access: with no tokens set the API is open, which is only for localhost.
Set ``ENTIP_DB_WRITE_TOKEN`` (and optionally ``ENTIP_DB_READ_TOKEN``) and
clients must send ``Authorization: Bearer <token>``. A write token also reads.
With only a read token set, writes are refused. Several tokens may be given,
separated by commas, so one can be replaced without downtime. This is a
shared-secret gate for services, not sign-in for people: NHS single sign-on
belongs in the gateway in front.

Research use only.
"""
from __future__ import annotations

import hmac
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import SCHEMA_VERSION, __version__, queries
from .db import OBJECT_TYPES, connect, dsn, schema_version
from .db import apply_action as _apply
from .db_errors import EntipError
from .contract import contract


class Provenance(BaseModel):
    kind: Literal["person", "model", "program"] = "person"
    actor: str = Field(..., description="'person:<name>' or 'model:<id>' or a program name")
    version: Optional[str] = None
    tool: Optional[str] = None
    tool_version: Optional[str] = None

    model_config = {"extra": "allow"}


class ActionIn(BaseModel):
    verb: str = Field(..., description="create | upsert | update | retract | delete, or an event verb "
                                       "(e.g. fork, train, promote) with no records")
    object_type: Optional[str] = Field(None, description=f"one of {sorted(OBJECT_TYPES)}")
    records: list[dict[str, Any]] = Field(default_factory=list)
    provenance: Provenance
    detail: dict[str, Any] = Field(default_factory=dict, description="inputs, outputs, notes")
    action_id: Optional[str] = Field(None, description="client-made id, for idempotent retries")


class ActionOut(BaseModel):
    action_id: str
    verb: str
    object_type: Optional[str]
    n: int
    changes: int
    replayed: bool = Field(False, description="true when this action_id had already been applied")


def _tokens(value: str | None, env: str) -> list[str]:
    raw = os.environ.get(env, "") if value is None else value
    return [t.strip() for t in raw.split(",") if t.strip()]


def create_app(conninfo: str | None = None, *, write_token: str | None = None,
               read_token: str | None = None) -> FastAPI:
    """``write_token`` / ``read_token`` default to $ENTIP_DB_WRITE_TOKEN / $ENTIP_DB_READ_TOKEN."""
    writers, readers = _tokens(write_token, "ENTIP_DB_WRITE_TOKEN"), _tokens(read_token, "ENTIP_DB_READ_TOKEN")

    def _has(request: Request, allowed: list[str]) -> bool:
        scheme, _, given = request.headers.get("authorization", "").partition(" ")
        given = given.strip().encode()
        # compare against every token so the time taken does not show which one matched
        return scheme.lower() == "bearer" and any([hmac.compare_digest(given, t.encode()) for t in allowed])

    def gate(request: Request):
        if not (writers or readers) or request.url.path == "/health":
            return
        if request.method in ("GET", "HEAD"):
            if readers and not _has(request, readers + writers):
                raise HTTPException(401, "a read or write token is needed", {"WWW-Authenticate": "Bearer"})
        elif not writers:
            raise HTTPException(403, "writes are off: no write token is set on this server")
        elif not _has(request, writers):
            raise HTTPException(401, "a write token is needed", {"WWW-Authenticate": "Bearer"})

    app = FastAPI(dependencies=[Depends(gate)], title="ENTIP central database", version=__version__,
                  description="Pillar 5: one row per patient, linked to slides, assays and model outputs. "
                              "Research use only; synthetic or public data only.")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost", "http://127.0.0.1"],
                       allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?", allow_methods=["*"],
                       allow_headers=["*"])
    target = conninfo or dsn()

    @contextmanager
    def _conn():
        c = connect(target)
        try:
            yield c
        finally:
            c.close()

    def get_conn():
        with _conn() as c:
            yield c

    @app.exception_handler(EntipError)
    async def _bad(request: Request, exc: EntipError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def _bbox(bbox: Optional[str]):
        if not bbox:
            return None
        try:
            v = tuple(float(x) for x in bbox.split(","))
        except ValueError:
            raise HTTPException(422, "bbox is x0,y0,x1,y1 in micrometres") from None
        if len(v) != 4 or any(x != x for x in v):
            raise HTTPException(422, "bbox is x0,y0,x1,y1 in micrometres")
        return v

    def _404(obj, what):
        if obj is None:
            raise HTTPException(404, f"{what} not found")
        return obj

    @app.get("/health")
    def health(conn=Depends(get_conn)):
        return {"ok": True, "version": __version__, "schema_version": schema_version(conn)}

    @app.get("/schema")
    def schema(conn=Depends(get_conn)):
        return contract(conn)

    # -- patients ----------------------------------------------------------
    @app.get("/patients")
    def list_patients(request: Request, cohort: Optional[str] = None, diagnosis: Optional[str] = None,
                      synthetic: Optional[bool] = None, limit: int = Query(100, ge=1, le=1000),
                      offset: int = Query(0, ge=0),
                      conn=Depends(get_conn)):
        """One row per patient. Filter on a marker with ``marker.IDH=mutant`` (repeatable)."""
        markers = {k[7:]: v for k, v in request.query_params.items() if k.startswith("marker.")}
        return queries.patients(conn, cohort=cohort, diagnosis=diagnosis, marker=markers, synthetic=synthetic,
                                limit=limit, offset=offset)

    @app.get("/patients/{study_id}")
    def get_patient(study_id: str, conn=Depends(get_conn)):
        """The patient row plus specimens > blocks > sections > scans | assays, findings, transforms."""
        return _404(queries.patient_tree(conn, study_id), "patient")

    @app.get("/patients/{study_id}/actions")
    def patient_actions(study_id: str, since: Optional[datetime] = None, limit: int = Query(500, ge=1, le=5000),
                        conn=Depends(get_conn)):
        _404(queries.patient_row(conn, study_id), "patient")
        return queries.actions(conn, study_id=study_id, since=since, limit=limit)

    # -- scans -------------------------------------------------------------
    @app.get("/scans/{scan_id}")
    def get_scan(scan_id: str, conn=Depends(get_conn)):
        """The scan, its stain, and the patient's study_id, diagnosis, grade and markers."""
        return _404(queries.scan(conn, scan_id), "scan")

    @app.get("/scans/{scan_id}/cells")
    def scan_cells(scan_id: str, bbox: Optional[str] = Query(None, description="x0,y0,x1,y1 µm"),
                   model_id: Optional[str] = Query(None, description="join this model's scores"),
                   after: Optional[str] = Query(None, description="page: last cell_id seen"),
                   limit: int = Query(50000, ge=1, le=500000), conn=Depends(get_conn)):
        _404(queries.get_object(conn, "scan", scan_id), "scan")
        return queries.cells(conn, scan_id, bbox=_bbox(bbox), model_id=model_id, limit=limit, after=after)

    @app.get("/scans/{scan_id}/regions")
    def scan_regions(scan_id: str, bbox: Optional[str] = None, kind: Optional[str] = None,
                     label: Optional[str] = None, conn=Depends(get_conn)):
        return queries.regions(conn, scan_id, bbox=_bbox(bbox), kind=kind, label=label)

    @app.get("/scans/{scan_id}/annotations")
    def scan_annotations(scan_id: str, feature: Optional[str] = None, include_retracted: bool = False,
                         bbox: Optional[str] = None, conn=Depends(get_conn)):
        return queries.annotations(conn, scan_id, feature=feature, include_retracted=include_retracted,
                                   bbox=_bbox(bbox))

    @app.get("/scans/{scan_id}/predictions")
    def scan_predictions(scan_id: str, model_id: str, order: Literal["uncertain", "score", "id"] = "uncertain",
                         limit: int = Query(200, ge=1, le=100000), conn=Depends(get_conn)):
        return queries.predictions(conn, scan_id, model_id, order=order, limit=limit)

    # -- assays and registration ---------------------------------------------
    @app.get("/assays/{assay_id}")
    def get_assay(assay_id: str, conn=Depends(get_conn)):
        return _404(queries.get_object(conn, "assay", assay_id), "assay")

    @app.get("/assays/{assay_id}/objects")
    def assay_objects(assay_id: str, bbox: Optional[str] = None, frame: Optional[str] = None,
                      limit: int = Query(100000, ge=1, le=1000000), conn=Depends(get_conn)):
        return queries.spatial_objects(conn, assay_id, bbox=_bbox(bbox), frame=frame, limit=limit)

    @app.get("/transforms")
    def list_transforms(frame: Optional[str] = None, conn=Depends(get_conn)):
        return queries.transforms(conn, frame)

    # -- features and models -------------------------------------------------
    @app.get("/features")
    def list_features(level: Optional[str] = None, conn=Depends(get_conn)):
        return queries.simple_list(conn, "feature", {"level": level}, order="name")

    @app.get("/models")
    def list_models(feature: Optional[str] = None, status: Optional[str] = None, conn=Depends(get_conn)):
        return queries.simple_list(conn, "model", {"feature": feature, "status": status})

    # -- any object ----------------------------------------------------------
    @app.get("/objects/{object_type}/{object_id}")
    def get_any(object_type: str, object_id: str, conn=Depends(get_conn)):
        if object_type not in OBJECT_TYPES or OBJECT_TYPES[object_type] is None:
            raise HTTPException(404, f"unknown object type {object_type}")
        return _404(queries.get_object(conn, object_type, object_id), object_type)

    @app.get("/objects/{object_type}/{object_id}/history")
    def object_history(object_type: str, object_id: str, conn=Depends(get_conn)):
        return queries.actions(conn, object_type=object_type, object_id=object_id)

    # -- actions -------------------------------------------------------------
    @app.post("/actions", response_model=ActionOut, status_code=201)
    def post_action(body: ActionIn, conn=Depends(get_conn)):
        """Apply one action. A repeat of an ``action_id`` already applied returns 200 and ``replayed``."""
        out = _apply(conn, verb=body.verb, object_type=body.object_type, records=body.records,
                     provenance=body.provenance.model_dump(exclude_none=True), detail=body.detail,
                     action_id=body.action_id)
        return JSONResponse(out, status_code=200) if out["replayed"] else out

    @app.get("/actions")
    def list_actions(since: Optional[datetime] = None, verb: Optional[str] = None,
                     object_type: Optional[str] = None, limit: int = Query(200, ge=1, le=5000),
                     conn=Depends(get_conn)):
        return queries.actions(conn, since=since, verb=verb, object_type=object_type, limit=limit)

    @app.get("/actions/{action_id}")
    def get_action(action_id: str, conn=Depends(get_conn)):
        r = _404(conn.execute("SELECT * FROM entip.action WHERE action_id = %s", (action_id,)).fetchone(),
                 "action")
        ch = conn.execute("SELECT * FROM entip.change WHERE action_id = %s ORDER BY change_id",
                          (action_id,)).fetchall()
        return queries._clean([{**r, "changes": queries._clean(ch)}])[0]

    app.state.schema_version = SCHEMA_VERSION
    app.state.auth = {"write": bool(writers), "read": bool(readers)}
    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
