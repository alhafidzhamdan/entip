"""Pillar 3 objects, shaped on the shared ENTIP ontology.

Patient > Specimen > Block > Section > Scan | Assay, plus Cell, Region,
Transform, Model, Annotation. This module owns Assay, Transform and the
spot / nucleus objects (SpatialObject). The central database (pillar 5) is
the contract; until its schema lands these dataclasses are the stand-in and
``records.py`` is the only place that needs changing to match it.

Positions are micrometres in a named frame:
  - ``scan:<scan_id>``   the H&E scan, level-0, micrometres
  - ``assay:<assay_id>`` the assay's own coordinates, converted to micrometres
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

AssayKind = Literal["visium", "xenium", "slide_tags", "anndata"]
ObjectKind = Literal["spot", "nucleus", "bin"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def scan_frame(scan_id: str) -> str:
    return f"scan:{scan_id}"


def assay_frame(assay_id: str) -> str:
    return f"assay:{assay_id}"


@dataclass(frozen=True)
class Provenance:
    """Who made an object: a person, or a model/program plus its version."""
    kind: Literal["person", "model"]
    actor: str
    version: Optional[str] = None
    at: str = field(default_factory=now_iso)

    @classmethod
    def program(cls, step: str) -> "Provenance":
        from . import __version__
        return cls(kind="model", actor=f"entip-spatial:{step}", version=__version__)


@dataclass
class Assay:
    assay_id: str
    section_id: str
    kind: AssayKind
    frame: str
    n_obs: int
    n_vars: int
    source_path: str
    source_sha256: Optional[str]
    provenance: Provenance
    meta: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class Transform:
    """Maps points from ``src_frame`` to ``dst_frame`` (both micrometres).

    ``matrix`` is 3x3 homogeneous (row-major). Errors are in micrometres in
    the destination frame: ``rmse_um`` on the landmarks used to fit,
    ``loo_rmse_um`` leave-one-out (an honest estimate of error on new points).
    """
    transform_id: str
    src_frame: str
    dst_frame: str
    model: Literal["affine", "similarity"]
    matrix: list[list[float]]
    rmse_um: float
    loo_rmse_um: Optional[float]
    max_err_um: float
    n_landmarks: int
    method: str
    provenance: Provenance
    meta: dict[str, Any] = field(default_factory=dict)

    def error_um(self) -> float:
        """Best estimate of placement error for one point: ICP residual when refined,
        else leave-one-out RMSE, else fit RMSE."""
        if self.meta.get("icp_median_residual_um") is not None:
            return float(1.5 * self.meta["icp_median_residual_um"])
        return float(self.loo_rmse_um if self.loo_rmse_um is not None else self.rmse_um)

    def to_record(self) -> dict:
        return asdict(self)


@dataclass
class SpatialObject:
    """A spot, bin or assay nucleus, placed in the H&E frame through a Transform."""
    object_id: str
    assay_id: str
    kind: ObjectKind
    barcode: str
    frame: str
    x_um: float
    y_um: float
    radius_um: Optional[float]
    transform_id: str
    label: Optional[str] = None
    label_score: Optional[float] = None

    def to_record(self) -> dict:
        return asdict(self)
