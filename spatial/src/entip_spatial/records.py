"""Hand-off to the central database (pillar 5).

Two writers, and nothing else in this package writes records:

* ``JsonlSink``: JSON Lines in a folder, one file per object type, plus
  ``actions.jsonl``. Always written, as the local record of a run.
* ``DbSink``: the central database's ``POST /actions``. One registration
  (Assay + Transform + its spots or nuclei) is one logged ``upsert`` action,
  so it is all there or not at all, and running it again replaces the
  positions instead of duplicating them. Field names follow
  ``db/schema/README.md``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np

from .ontology import Assay, Provenance, SpatialObject, Transform, new_id, now_iso


class Sink(Protocol):
    def create(self, object_type: str, records: Iterable[dict], provenance: Provenance) -> int: ...


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class JsonlSink:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, object_type: str, records: Iterable[dict], provenance: Provenance) -> int:
        n = 0
        with (self.root / f"{object_type}.jsonl").open("a") as f:
            for r in records:
                f.write(json.dumps(r, default=_default) + "\n")
                n += 1
        action = {"action_id": new_id("act"), "at": now_iso(), "verb": "create", "object_type": object_type,
                  "n": n, "provenance": provenance.__dict__}
        with (self.root / "actions.jsonl").open("a") as f:
            f.write(json.dumps(action, default=_default) + "\n")
        return n


class DbError(RuntimeError):
    """The central database refused a registration or could not be reached."""


def _prov(p: Provenance) -> dict:
    return {k: v for k, v in asdict(p).items() if v is not None}


def assay_record(a: Assay, uri: Optional[str] = None) -> dict:
    """Assay in the database's field names. The local source path is left out unless ``uri`` is given:
    folder names often carry laboratory numbers."""
    rec = {"type": "Assay", "assay_id": a.assay_id, "section_id": a.section_id, "kind": a.kind, "frame": a.frame,
           "n_obs": int(a.n_obs), "n_vars": int(a.n_vars), "sha256": a.source_sha256, "meta": a.meta,
           "provenance": _prov(a.provenance)}
    if uri:
        rec["uri"] = uri
    return rec


def transform_record(tf: Transform) -> dict:
    return {"type": "Transform", **{k: v for k, v in tf.to_record().items() if k != "provenance"},
            "provenance": _prov(tf.provenance)}


class DbSink:
    """Posts a registration to the central database (Python standard library only).

    ``chunk`` caps the spots or nuclei per request; a large assay goes as the first action (assay,
    transform, first chunk) followed by one action per further chunk.
    """

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 120.0, chunk: int = 50000):
        self.base = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get("ENTIP_DB_WRITE_TOKEN", "").split(",")[0].strip()
        self.timeout, self.chunk = timeout, chunk

    def post_action(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.base + "/actions", data=json.dumps(body, default=_default).encode(),
                                     method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                detail = json.loads(raw).get("detail")
            except (ValueError, AttributeError):
                detail = raw.decode("utf-8", "replace")[:300]
            raise DbError(f"database refused the action ({e.code}): {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            raise DbError(f"cannot reach the database at {self.base}: {getattr(e, 'reason', e)}") from e

    def register(self, assay: Assay, tf: Transform, objs: list[SpatialObject], provenance: Provenance,
                 uri: Optional[str] = None) -> list[dict]:
        """Assay, Transform and objects, in that order (each needs the one before). Returns the results."""
        run = uuid.uuid4().hex
        detail = {"event": "register_assay", "assay_id": assay.assay_id, "transform_id": tf.transform_id,
                  "n_objects": len(objs), "rmse_um": tf.rmse_um, "loo_rmse_um": tf.loo_rmse_um}
        records: list[dict] = [assay_record(assay, uri), transform_record(tf)]
        objects = [{"type": "SpatialObject", **o.to_record()} for o in objs]
        out = []
        for part, i in enumerate(range(0, max(len(objects), 1), self.chunk)):
            body = {"verb": "upsert", "object_type": None, "action_id": f"act_spatial_{run}_{part}",
                    "provenance": _prov(provenance), "detail": {**detail, "part": part},
                    "records": (records if part == 0 else []) + objects[i:i + self.chunk]}
            out.append(self.post_action(body))
        return out


def write_assay(sink: Sink, assay: Assay) -> None:
    sink.create("assay", [assay.to_record()], assay.provenance)


def write_transform(sink: Sink, tf: Transform) -> None:
    sink.create("transform", [tf.to_record()], tf.provenance)


def spatial_objects(adata, assay_id: str, tf: Transform, key: str = "spatial_he_um") -> list[SpatialObject]:
    meta = adata.uns["entip"]
    xy = adata.obsm[key]
    return [SpatialObject(object_id=f"{assay_id}:{bc}", assay_id=assay_id, kind=meta["object_kind"], barcode=str(bc),
                          frame=tf.dst_frame, x_um=float(x), y_um=float(y), radius_um=meta.get("radius_um"),
                          transform_id=tf.transform_id)
            for bc, (x, y) in zip(adata.obs_names, xy)]


def write_objects(sink: Sink, objs: list[SpatialObject], provenance: Provenance) -> None:
    sink.create("spatial_object", (o.to_record() for o in objs), provenance)


def read_transform(path: str | Path) -> Transform:
    d = json.loads(Path(path).read_text())
    d["provenance"] = Provenance(**d["provenance"])
    return Transform(**d)


def save_transform(tf: Transform, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(tf.to_record(), indent=2, default=_default))
    return p
