"""Hand-off to the central database (pillar 5).

Until the P5 schema and API land, objects are written as JSON Lines in a
folder, one file per object type, plus ``actions.jsonl``: every create is a
logged action (who, when, what), never an in-place edit. Swap ``JsonlSink``
for a P5 API sink later; nothing else in this package writes records.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

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
