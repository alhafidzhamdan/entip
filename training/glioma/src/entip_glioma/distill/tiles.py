"""Tile cache for distillation: each bag's tiles cut from its slide once, aligned with the teacher's numbers.

Per slide ``<scan_id>``, in the output folder:

* ``<scan_id>.tiles.npy``    (N, px, px, 3) uint8 RGB, read at the bag's positions and tile size
* ``<scan_id>.teacher.npy``  (N, D) float16, the teacher's embedding of the same tiles, row for row
* one line in ``index.jsonl`` with scan id, patient, label, tile count, teacher encoder and tile size

Plain ``.npy`` files are memory-mapped at training time, so the cache can be far larger than memory. At 224 px a
tile is 150 KB: 1,000 tiles per slide is 150 MB per slide, about 38 GB for 250 slides.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..wsi.bags import Bag

INDEX = "index.jsonl"


def extract_tiles(reader, bag: Bag, out_dir: str | Path, input_px: int = 224, max_tiles: Optional[int] = None,
                  seed: int = 0, batch: int = 64) -> dict:
    """Cut ``bag``'s tiles from ``reader`` (a slide reader with ``read_um``) and save them with the teacher's
    numbers. ``max_tiles`` keeps a seeded random subset of the slide's tiles."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.arange(len(bag))
    if max_tiles is not None and len(idx) > max_tiles:
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), max_tiles, replace=False))
    tiles = np.lib.format.open_memmap(out_dir / f"{bag.scan_id}.tiles.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(idx), input_px, input_px, 3))
    half = bag.tile_um / 2
    for k in range(0, len(idx), batch):
        rows = idx[k:k + batch]
        tiles[k:k + len(rows)] = np.stack([reader.read_um(bag.coords_um[i, 0] + half, bag.coords_um[i, 1] + half,
                                                          bag.tile_um, input_px) for i in rows])
    tiles.flush()
    del tiles
    np.save(out_dir / f"{bag.scan_id}.teacher.npy", bag.features[idx].astype(np.float16))
    return {"scan_id": bag.scan_id, "tiles": int(len(idx)), "input_px": input_px, "tile_um": bag.tile_um,
            "teacher": bag.encoder, "dim": int(bag.features.shape[1])}


def read_index(tile_dir: str | Path) -> list:
    p = Path(tile_dir) / INDEX
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def split_patients(entries: list, holdout: float = 0.2, seed: int = 0) -> tuple[set, set]:
    """Patients into train and held-out sets, per label, so every class is in both where it has 2+ patients.
    Splitting by patient (not tile or slide) keeps one person's tissue out of both sides."""
    by_label = defaultdict(set)
    for e in entries:
        by_label[e.get("label") or ""].add(e["patient"])
    rng = random.Random(seed)
    train, held = set(), set()
    for lab in sorted(by_label):
        ps = sorted(by_label[lab])
        rng.shuffle(ps)
        n = int(round(len(ps) * holdout)) if len(ps) > 1 else 0
        n = max(1, n) if len(ps) > 1 and holdout > 0 else n
        held.update(ps[:n])
        train.update(ps[n:])
    return train, held
