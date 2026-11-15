"""Helpers on the shared tile bag (``entip_train.wsi.bags.Bag``, written by the step-1 pipeline).

A bag is one slide's tissue tiles and their frozen embeddings: ``features`` (N, D), ``coords_um`` (N, 2,
top-left of each tile in the frame ``scan:<scan_id>``), ``tile_um``, ``encoder`` (version string) and
``tissue_frac``. Build bags with ``wsi.bags.extract_bag`` (224 px tiles at 0.5 µm/px, H-optimus-0, Apache-2.0)
and load them with ``Bag.load``; step 2 never re-embeds a slide.
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
