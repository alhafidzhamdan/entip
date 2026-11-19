"""Tile labels from region outlines drawn in QuPath (File > Export objects as GeoJSON, full-resolution pixels).

Classes (QuPath's classification name, any case): ``Necrosis``; ``MVP`` or ``Microvascular proliferation``;
``Neither`` (assessable tissue with neither feature, for negatives). Other classes are ignored and counted.

A tile takes an outline's label when its centre is inside the outline (outside any hole), or when the outline's
centre is inside the tile, so a vessel tuft smaller than a tile still labels the tile it sits in (the same rule as
``highgrade.regions.tile_labels``). For each feature a positive outline wins over a ``Neither`` outline. A
``Necrosis`` outline says nothing about MVP, and an ``MVP`` outline nothing about necrosis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = {
    "necrosis": "necrosis", "tumour necrosis": "necrosis", "tumor necrosis": "necrosis",
    "mvp": "mvp", "microvascular proliferation": "mvp",
    "neither": "neither", "negative": "neither",
}


def _class(props: dict) -> str:
    c = props.get("classification")
    name = c.get("name", "") if isinstance(c, dict) else (c or props.get("name") or "")
    return str(name).strip().lower()


def read_geojson(path, mpp: float) -> tuple[list, dict]:
    """[(label, exterior, holes)] in µm, and the count of outlines with other classes."""
    d = json.loads(Path(path).read_text())
    feats = d.get("features", []) if d.get("type") == "FeatureCollection" else (d if isinstance(d, list) else [d])
    shapes, other = [], {}
    for f in feats:
        g, cls = f.get("geometry") or {}, _class(f.get("properties") or {})
        lab = CLASSES.get(cls)
        if lab is None:
            other[cls or "(none)"] = other.get(cls or "(none)", 0) + 1
            continue
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else \
            g["coordinates"] if g.get("type") == "MultiPolygon" else []
        for rings in polys:
            rings = [np.asarray(r, float)[:, :2] * mpp for r in rings if len(r) >= 3]
            if rings:
                shapes.append((lab, rings[0], rings[1:]))
    return shapes, other


def inside(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for many points against one ring."""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    res = np.zeros(len(px), bool)
    for a, b, c, e in zip(x, y, x2, y2):
        if b == e:
            continue
        cross = ((b > py) != (e > py)) & (px < (c - a) * (py - b) / (e - b) + a)
        res ^= cross
    return res


def tile_votes(coords_um: np.ndarray, tile_um: float, shapes: list) -> dict:
    """{tile index: {"n": 0|1|None, "m": 0|1|None}} for tiles any outline touches."""
    cx, cy = coords_um[:, 0] + tile_um / 2, coords_um[:, 1] + tile_um / 2
    hit = {"necrosis": np.zeros(len(cx), bool), "mvp": np.zeros(len(cx), bool), "neither": np.zeros(len(cx), bool)}
    for lab, ext, holes in shapes:
        m = inside(cx, cy, ext)
        for h in holes:
            m &= ~inside(cx, cy, h)
        gx, gy = ext.mean(0)
        m |= (coords_um[:, 0] <= gx) & (gx < coords_um[:, 0] + tile_um) & \
             (coords_um[:, 1] <= gy) & (gy < coords_um[:, 1] + tile_um)
        hit[lab] |= m
    out = {}
    for i in np.flatnonzero(hit["necrosis"] | hit["mvp"] | hit["neither"]):
        n = 1 if hit["necrosis"][i] else 0 if hit["neither"][i] else None
        v = 1 if hit["mvp"][i] else 0 if hit["neither"][i] else None
        out[int(i)] = {"n": n, "m": v}
    return out


def draw_check(reader, coords_um: np.ndarray, tile_um: float, shapes: list, votes: dict, path, px: int = 1600):
    """The slide with the outlines (necrosis red, MVP blue, neither grey) and the tiles they label, to check
    that QuPath's coordinates line up with the bag's."""
    from PIL import Image, ImageDraw
    lo, hi = coords_um.min(0), coords_um.max(0) + tile_um
    c, side = (lo + hi) / 2, float((hi - lo).max()) * 1.05
    im = Image.fromarray(reader.read_um(float(c[0]), float(c[1]), side, px)).convert("RGBA")
    s, x0, y0 = px / side, c[0] - side / 2, c[1] - side / 2
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    col = {"necrosis": (220, 40, 40), "mvp": (30, 90, 220), "neither": (90, 90, 90)}
    for i, v in votes.items():
        x, y = coords_um[i]
        k = "necrosis" if v["n"] == 1 else "mvp" if v["m"] == 1 else "neither"
        d.rectangle([(x - x0) * s, (y - y0) * s, (x + tile_um - x0) * s, (y + tile_um - y0) * s], fill=(*col[k], 70))
    for lab, ext, holes in shapes:
        for r in [ext, *holes]:
            pts = [((x - x0) * s, (y - y0) * s) for x, y in r]
            d.line(pts + pts[:1], fill=(*col[lab], 255), width=2)
    Image.alpha_composite(im, over).convert("RGB").save(path, quality=88)
