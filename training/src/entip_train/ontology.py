"""The ontology objects this pillar reads and writes.

These mirror the shared rules in docs/pillar_specs.md (Scan, Annotation, Cell, Region, Model)
until the db/ pillar lands its schema; then ``store.py`` is swapped for a client of that API and
these classes become thin views of its rows.

Conventions (from the shared rules):
* positions are micrometres in a named frame; here the frame is ``scan:<scan_id>``, origin at the
  top-left of the full-resolution image;
* every object carries provenance: a person (``person:<name>``) or a model (``model:<id>``);
* changes are logged actions, never edits in place (see ``Store.log``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

LEVELS = ("cell", "tile", "region")
POSITIVE, NEGATIVE = "positive", "negative"


def frame(scan_id: str) -> str:
    return f"scan:{scan_id}"


@dataclass
class Scan:
    scan_id: str
    path: str                      # research copy of the WSI (from pillar 1), or "synthetic:<seed>"
    mpp: float                     # micrometres per pixel at full resolution
    width_um: float
    height_um: float
    section_id: Optional[str] = None


@dataclass
class Feature:
    """A feature of interest that a live head learns, e.g. "mitotic figure" (cell) or "necrosis" (tile)."""
    name: str
    level: str                     # cell | tile | region
    description: str = ""
    created_by: str = ""
    positives_only: bool = False   # marks are mostly positives: sample background negatives

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")


@dataclass
class Cell:
    cell_id: str
    scan_id: str
    x_um: float
    y_um: float
    source: str                    # provenance, e.g. "model:cellpose-sam@4.0" or "synthetic"


@dataclass
class Region:
    """An axis-aligned tile; region-level features label the tiles whose centre falls in a polygon."""
    region_id: str
    scan_id: str
    x_um: float                    # top-left
    y_um: float
    w_um: float
    h_um: float
    source: str = "grid"

    @property
    def centre(self) -> Tuple[float, float]:
        return self.x_um + self.w_um / 2, self.y_um + self.h_um / 2


@dataclass
class Annotation:
    annotation_id: str
    scan_id: str
    feature: str
    label: str                     # positive | negative
    geometry: dict                 # {"type": "point", "x_um", "y_um"} or {"type": "polygon", "points_um": [[x, y], ...]}
    author: str                    # provenance: "person:<name>"
    created_at: str = ""
    locked: bool = False           # held out for the retrain gate; never trained on
    retracted: bool = False

    def __post_init__(self):
        if self.label not in (POSITIVE, NEGATIVE):
            raise ValueError("label must be 'positive' or 'negative'")


@dataclass
class Model:
    """One trained head or retrained model. ``status``: draft (live head), champion, challenger, rejected."""
    model_id: str
    feature: str
    kind: str                      # e.g. "live-head:logreg", "retrain:logreg"
    encoder: str                   # e.g. "h-optimus-0"
    status: str
    trained_by: str                # provenance of the request
    n_labels: int
    created_at: str = ""
    parent: Optional[str] = None   # warm-started from this model
    metrics: dict = field(default_factory=dict)
    artifact: str = ""             # path to the saved head

    def to_dict(self) -> dict:
        return asdict(self)


def point_in_polygon(x: float, y: float, pts: List[List[float]]) -> bool:
    """Ray casting; ``pts`` is a closed or open ring of [x, y]."""
    inside, n = False, len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
