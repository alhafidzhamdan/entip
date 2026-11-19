"""Helpers on the shared tile bag (``entip_glioma.wsi.bags.Bag``, written by the step-1 pipeline).

A bag is one slide's tissue tiles and their frozen embeddings: ``features`` (N, D), ``coords_um`` (N, 2,
top-left of each tile in the frame ``scan:<scan_id>``), ``tile_um``, ``encoder`` (version string) and
``tissue_frac``. Build bags with ``wsi.bags.extract_bag`` (224 px tiles at 0.5 µm/px, H-optimus-0, Apache-2.0)
and load them with ``Bag.load``; step 3 never re-embeds a slide.
"""
from __future__ import annotations

import numpy as np

from ..wsi.bags import Bag, extract_bag  # noqa: F401  (re-exported)


def grid(bag: Bag) -> np.ndarray:
    """(N, 2) integer (row, col) of each tile."""
    return np.round(bag.coords_um / bag.tile_um).astype(int)[:, ::-1]


def tile_ids(bag: Bag) -> list:
    """Stable ids from grid position: ``<scan_id>:t<row>-<col>``."""
    return [f"{bag.scan_id}:t{r:04d}-{c:04d}" for r, c in grid(bag)]


def centres_um(bag: Bag) -> np.ndarray:
    return bag.coords_um + bag.tile_um / 2


def tile_area_mm2(bag: Bag) -> float:
    return (bag.tile_um / 1000.0) ** 2


def tissue_area_mm2(bag: Bag) -> float:
    """Tissue area from the tiles kept by the tissue mask, weighted by each tile's tissue fraction."""
    frac = bag.tissue_frac if len(bag.tissue_frac) == len(bag) else np.ones(len(bag))
    return float(frac.sum() * tile_area_mm2(bag))


def context_features(bag: Bag, radii=()) -> np.ndarray:
    """Each tile's embedding followed by the mean embedding of its (2r + 1) x (2r + 1) neighbourhood for each
    radius r in ``radii`` (tiles present only, the tile itself included). ``()`` returns the embeddings as they
    are. Gives a head the tissue around a tile without re-embedding the slide."""
    radii = sorted({int(r) for r in radii or ()})
    if not radii or len(bag) == 0:
        return bag.features
    rc = grid(bag)
    rc = rc - rc.min(0)
    R = radii[-1]
    idx = np.full(tuple(rc.max(0) + 1 + 2 * R), -1, np.int64)
    idx[rc[:, 0] + R, rc[:, 1] + R] = np.arange(len(bag))
    F = np.asarray(bag.features, np.float32)
    out = [F]
    for r in radii:
        s, n = np.zeros_like(F), np.zeros(len(F), np.float32)
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                j = idx[rc[:, 0] + R + dr, rc[:, 1] + R + dc]
                ok = j >= 0
                s[ok] += F[j[ok]]
                n[ok] += 1
        out.append(s / n[:, None])
    return np.concatenate(out, 1)
