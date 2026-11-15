"""Geometry in micrometres (SRID 0) between API JSON and PostGIS WKT.

Accepted shapes (all micrometres in the object's frame):
  {"type": "point", "x_um": 10.5, "y_um": 20.0}
  {"type": "polygon", "points_um": [[x, y], [x, y], ...]}       ring, closed or open
  {"type": "box", "x_um": .., "y_um": .., "w_um": .., "h_um": ..}  top-left + size
  GeoJSON Point / Polygon with "coordinates" in micrometres
"""
from __future__ import annotations

import math

from .db_errors import EntipError


def _num(v, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise EntipError(f"{what} must be a number") from None
    if not math.isfinite(f):
        raise EntipError(f"{what} must be finite")
    return f


def _ring(points) -> str:
    pts = [(_num(p[0], "x"), _num(p[1], "y")) for p in points]
    if len(pts) < 3:
        raise EntipError("a polygon needs at least 3 points")
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return "(" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in pts) + ")"


def box_wkt(x_um, y_um, w_um, h_um) -> str:
    x, y, w, h = (_num(v, k) for v, k in ((x_um, "x_um"), (y_um, "y_um"), (w_um, "w_um"), (h_um, "h_um")))
    if w <= 0 or h <= 0:
        raise EntipError("box width and height must be positive")
    return "POLYGON(" + _ring([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]) + ")"


def to_wkt(g: dict) -> str:
    if not isinstance(g, dict) or "type" not in g:
        raise EntipError("geometry must be an object with a 'type'")
    t = str(g["type"]).lower()
    if t == "point":
        if "coordinates" in g:
            x, y = g["coordinates"][:2]
        else:
            x, y = g.get("x_um"), g.get("y_um")
        return f"POINT({_num(x, 'x_um'):.6f} {_num(y, 'y_um'):.6f})"
    if t == "polygon":
        if "coordinates" in g:
            rings = g["coordinates"]
            return "POLYGON(" + ", ".join(_ring(r) for r in rings) + ")"
        return "POLYGON(" + _ring(g.get("points_um") or []) + ")"
    if t == "box":
        return box_wkt(g.get("x_um"), g.get("y_um"), g.get("w_um"), g.get("h_um"))
    raise EntipError(f"geometry type {g['type']!r} not supported (point, polygon, box)")
