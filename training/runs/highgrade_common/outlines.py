"""QuPath annotations (File > Export objects as GeoJSON, full-resolution pixels) as tile hits.

Each sub-step maps QuPath class names (any case) to its own labels, for example 3a ``{"mvp": "mvp", "vessel":
"vessel", "neither": "neither"}``. Other classes are ignored and counted.

* An outline (Polygon, MultiPolygon) hits a tile when the tile's centre is inside it (outside any hole), or when
  the outline's centre is inside the tile, so a vessel tuft smaller than a tile still hits the tile it sits in.
* A point (Point, MultiPoint: QuPath's point tool) hits the tile it falls in. Clicking each vessel is the quick way
  to label vessels for 3a.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _class(props: dict) -> str:
    c = props.get("classification")
    name = c.get("name", "") if isinstance(c, dict) else (c or props.get("name") or "")
    return str(name).strip().lower()


def read_geojson(path, mpp: float, classes: dict) -> tuple[list, dict]:
    """``[(label, "poly", exterior, holes) | (label, "point", xy, None)]`` in µm, and counts of other classes."""
    d = json.loads(Path(path).read_text())
    feats = d.get("features", []) if d.get("type") == "FeatureCollection" else (d if isinstance(d, list) else [d])
    shapes, other = [], {}
    for f in feats:
        g, cls = f.get("geometry") or {}, _class(f.get("properties") or {})
        lab = classes.get(cls)
        if lab is None:
            other[cls or "(none)"] = other.get(cls or "(none)", 0) + 1
            continue
        t, c = g.get("type"), g.get("coordinates")
        if t in ("Point", "MultiPoint"):
            for pt in ([c] if t == "Point" else c):
                shapes.append((lab, "point", np.asarray(pt, float)[:2] * mpp, None))
            continue
        for rings in ([c] if t == "Polygon" else c if t == "MultiPolygon" else []):
            rings = [np.asarray(r, float)[:, :2] * mpp for r in rings if len(r) >= 3]
            if rings:
                shapes.append((lab, "poly", rings[0], rings[1:]))
    return shapes, other


def inside(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for many points against one ring."""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    res = np.zeros(len(px), bool)
    for a, b, c, e in zip(x, y, x2, y2):
        if b == e:
            continue
        res ^= ((b > py) != (e > py)) & (px < (c - a) * (py - b) / (e - b) + a)
    return res


def hits(coords_um: np.ndarray, tile_um: float, shapes: list) -> dict:
    """{label: bool array over tiles}: which tiles each label's outlines and points touch."""
    cx, cy = coords_um[:, 0] + tile_um / 2, coords_um[:, 1] + tile_um / 2
    x0, y0 = coords_um[:, 0], coords_um[:, 1]
    out = {}
    for lab, kind, a, holes in shapes:
        def in_tile(gx, gy):
            return (x0 <= gx) & (gx < x0 + tile_um) & (y0 <= gy) & (gy < y0 + tile_um)
        if kind == "point":
            m = in_tile(*a)
        else:
            m = inside(cx, cy, a)
            for h in holes:
                m &= ~inside(cx, cy, h)
            m |= in_tile(*a.mean(0))
        out[lab] = out.get(lab, np.zeros(len(cx), bool)) | m
    return out


def draw_check(reader, coords_um: np.ndarray, tile_um: float, shapes: list, tile_label: dict, colours: dict,
               path, px: int = 1600):
    """The slide with the annotations and the tiles they label (``tile_label``: index -> label), to check that
    QuPath's coordinates line up with the bag's."""
    from PIL import Image, ImageDraw
    lo, hi = coords_um.min(0), coords_um.max(0) + tile_um
    c, side = (lo + hi) / 2, float((hi - lo).max()) * 1.05
    im = Image.fromarray(reader.read_um(float(c[0]), float(c[1]), side, px)).convert("RGBA")
    s, x0, y0 = px / side, c[0] - side / 2, c[1] - side / 2
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for i, lab in tile_label.items():
        x, y = coords_um[i]
        d.rectangle([(x - x0) * s, (y - y0) * s, (x + tile_um - x0) * s, (y + tile_um - y0) * s],
                    fill=(*colours.get(lab, (90, 90, 90)), 70))
    for lab, kind, a, holes in shapes:
        col = (*colours.get(lab, (90, 90, 90)), 255)
        if kind == "point":
            u, v = (a[0] - x0) * s, (a[1] - y0) * s
            d.ellipse([u - 5, v - 5, u + 5, v + 5], outline=col, width=2)
            continue
        for r in [a, *holes]:
            pts = [((x - x0) * s, (y - y0) * s) for x, y in r]
            d.line(pts + pts[:1], fill=col, width=2)
    Image.alpha_composite(im, over).convert("RGB").save(path, quality=88)
