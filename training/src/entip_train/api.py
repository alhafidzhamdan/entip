"""HTTP API used by the clinician GUI (pillar 4). FastAPI (MIT).

Positions are micrometres in the frame ``scan:<scan_id>``. Authentication is the GUI's job; ``actor`` is
recorded as provenance on every mark and model.

  GET    /health
  POST   /scans/synthetic                 {scan_id?, seed?}            register a synthetic demo slide
  GET    /heads                                                        list heads
  POST   /heads                           {feature, level, scan_id?, positives_only?, actor?} -> {head_id}
  GET    /heads/{head_id}                                              counts, draft and champion models
  POST   /heads/{head_id}/labels          {scan_id, annotation_id | cell_id | object_id | x_um+y_um | polygon_um,
                                           label 1|0, actor}   (annotation_id: already in the database)
                                          -> {label_id, model_version, n_pos, n_neg, metrics, queue}
  DELETE /heads/{head_id}/labels/{label_id}?actor=                     retract a mark (logged)
  GET    /heads/{head_id}/queue?scan_id=&n=20&model=                   [{cell_id, p, reason, x_um, y_um}]
  GET    /heads/{head_id}/predictions?scan_id=&model=                  {model_version, predictions: [{cell_id, p}]}
  POST   /heads/{head_id}/retrain         {actor, kind?, force?}       -> {job_id}   gated champion/challenger
  GET    /jobs/{job_id}
  GET    /models?head_id=
  GET    /actions?object_id=
"""
from __future__ import annotations

from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__
from .service import TrainingService
from .slides import synthetic_slide


class HeadIn(BaseModel):
    feature: str
    level: str = "cell"
    scan_id: Optional[str] = None
    positives_only: bool = False
    description: str = ""
    actor: str = "system"
    locked_scans: List[str] = []


class LabelIn(BaseModel):
    scan_id: str
    label: Union[int, str]
    actor: str = "anonymous"
    cell_id: Optional[str] = None
    object_id: Optional[str] = None
    x_um: Optional[float] = None
    y_um: Optional[float] = None
    polygon_um: Optional[List[List[float]]] = None
    locked: Optional[bool] = None
    queue_n: int = 20
    annotation_id: Optional[str] = None     # mark already saved in the database: train on it, do not re-write


class RetrainIn(BaseModel):
    actor: str = "system"
    kind: Optional[str] = None
    force: bool = False


class SyntheticIn(BaseModel):
    scan_id: str = "synthetic-001"
    seed: int = 0


def create_app(svc: TrainingService, cors_origins: Optional[List[str]] = None) -> FastAPI:
    app = FastAPI(title="ENTIP training", version=__version__,
                  description="Live per-feature training and gated retraining. Research use only.")
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.state.svc = svc
    app.state.truth = {}

    def nf(e: Exception):
        raise HTTPException(404, str(e).strip("'\""))

    @app.get("/health")
    def health():
        return {"ok": True, "version": __version__, "encoder": svc.encoder.version}

    @app.post("/scans/synthetic")
    def add_synthetic(body: SyntheticIn):
        s = synthetic_slide(body.scan_id, body.seed)
        svc.register_scan(s.scan, s.reader, s.cells, s.regions, actor="system")
        app.state.truth[body.scan_id] = s
        return {"scan_id": s.scan.scan_id, "mpp": s.scan.mpp, "width_um": s.scan.width_um,
                "height_um": s.scan.height_um, "n_cells": len(s.cells), "n_regions": len(s.regions)}

    @app.get("/heads")
    def list_heads():
        return [svc.head_info(f.name) for f in svc.store.features()]

    @app.post("/heads")
    def create_head(body: HeadIn):
        try:
            hid = svc.create_head(body.feature, body.level, body.actor, body.description, body.positives_only,
                                  body.locked_scans)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if body.scan_id:
            try:
                svc.warm_cache(body.scan_id, body.level)
            except KeyError as e:
                nf(e)
        return {"head_id": hid}

    @app.get("/heads/{head_id}")
    def head(head_id: str):
        try:
            return svc.head_info(head_id)
        except KeyError as e:
            nf(e)

    @app.post("/heads/{head_id}/labels")
    def add_label(head_id: str, body: LabelIn):
        try:
            return svc.add_label(head_id, body.scan_id, body.label, body.actor, cell_id=body.cell_id,
                                 object_id=body.object_id, x_um=body.x_um, y_um=body.y_um,
                                 polygon_um=body.polygon_um, locked=body.locked, queue_n=body.queue_n,
                                 annotation_id=body.annotation_id)
        except KeyError as e:
            nf(e)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.delete("/heads/{head_id}/labels/{label_id}")
    def retract(head_id: str, label_id: str, actor: str = "anonymous"):
        try:
            return svc.retract_label(head_id, label_id, actor)
        except KeyError as e:
            nf(e)

    @app.get("/heads/{head_id}/queue")
    def queue(head_id: str, scan_id: str, n: int = 20, model: Optional[str] = None):
        try:
            return svc.queue(head_id, scan_id, n, model)
        except (KeyError, LookupError) as e:
            nf(e)

    @app.get("/heads/{head_id}/predictions")
    def predictions(head_id: str, scan_id: str, model: Optional[str] = None):
        try:
            return svc.predictions(head_id, scan_id, model)
        except (KeyError, LookupError) as e:
            nf(e)

    @app.post("/heads/{head_id}/retrain")
    def retrain(head_id: str, body: RetrainIn):
        try:
            svc.store.feature(head_id)
        except KeyError as e:
            nf(e)
        return {"job_id": svc.submit_retrain(head_id, body.actor, kind=body.kind, force=body.force)}

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in svc.jobs:
            raise HTTPException(404, "unknown job")
        return svc.jobs[job_id]

    @app.get("/models")
    def models(head_id: Optional[str] = None):
        return [m.to_dict() for m in svc.store.models(head_id)]

    @app.get("/actions")
    def actions(object_id: Optional[str] = None):
        return svc.store.actions(object_id)

    return app
