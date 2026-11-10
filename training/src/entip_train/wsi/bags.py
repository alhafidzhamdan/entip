"""Slide -> bag of tile embeddings, the shared feature pipeline for the glioma workflow models.

Tiles are ``tile_um`` square (default 112 µm, which is 224 px at 0.5 µm/px, the input H-optimus-0 was
pretrained on: https://huggingface.co/bioptimus/H-optimus-0). Each tile is read at the encoder's input size
whatever the scanner's pixel size, so slides from different scanners give the same physical field.

Tissue detection is deliberately simple (saturation and brightness on a 16 px preview of each tile). Run
HistoQC (BSD-3-Clause-Clear) upstream for pen, folds and blur; this only drops background.

A bag is saved as an ``.npz`` without pickles, so loading one cannot run code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_TILE_UM = 112.0
PREVIEW_PX = 16


@dataclass
class Bag:
    scan_id: str
    features: np.ndarray          # (N, D) float32
    coords_um: np.ndarray         # (N, 2) top-left x, y of each tile, µm in frame scan:<scan_id>
    tile_um: float
    encoder: str                  # encoder version string, e.g. "h-optimus-0@timm1.0.9"
    tissue_frac: np.ndarray       # (N,) share of tissue pixels in the preview

    def __len__(self) -> int:
        return len(self.features)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=self.features.astype(np.float16), coords_um=self.coords_um,
                            tissue_frac=self.tissue_frac, tile_um=np.float64(self.tile_um),
                            scan_id=np.str_(self.scan_id), encoder=np.str_(self.encoder))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Bag":
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["scan_id"]), z["features"].astype(np.float32), z["coords_um"].astype(np.float64),
                       float(z["tile_um"]), str(z["encoder"]), z["tissue_frac"].astype(np.float32))


def tissue_fraction(rgb: np.ndarray, min_sat: float = 0.07, max_val: float = 0.94, min_val: float = 0.08) -> float:
    """Share of pixels that look like stained tissue: coloured, not white glass, not black."""
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    tissue = (sat > min_sat) & (mx < max_val) & (mx > min_val)
    return float(tissue.mean())


def tile_grid(reader, width_um: float, height_um: float, tile_um: float = DEFAULT_TILE_UM,
              min_tissue: float = 0.5):
    """Top-left corners (µm) and tissue fractions of tiles with at least ``min_tissue`` tissue."""
    xy, frac = [], []
    for i in range(int(height_um // tile_um)):
        for j in range(int(width_um // tile_um)):
            x0, y0 = j * tile_um, i * tile_um
            f = tissue_fraction(reader.read_um(x0 + tile_um / 2, y0 + tile_um / 2, tile_um, PREVIEW_PX))
            if f >= min_tissue:
                xy.append((x0, y0))
                frac.append(f)
    return np.asarray(xy, np.float64).reshape(-1, 2), np.asarray(frac, np.float32)


def extract_bag(reader, scan_id: str, width_um: float, height_um: float, encoder,
                tile_um: float = DEFAULT_TILE_UM, min_tissue: float = 0.5, max_tiles: Optional[int] = None,
                batch: int = 64, seed: int = 0) -> Bag:
    """Embed every tissue tile of one slide with a frozen encoder. ``max_tiles`` subsamples at random
    (seeded) for very large slides; leave it None for the final runs."""
    xy, frac = tile_grid(reader, width_um, height_um, tile_um, min_tissue)
    if max_tiles is not None and len(xy) > max_tiles:
        keep = np.sort(np.random.default_rng(seed).choice(len(xy), max_tiles, replace=False))
        xy, frac = xy[keep], frac[keep]
    feats = []
    for k in range(0, len(xy), batch):
        crops = np.stack([reader.read_um(x + tile_um / 2, y + tile_um / 2, tile_um, encoder.input_px)
                          for x, y in xy[k:k + batch]])
        feats.append(encoder.embed(crops))
    f = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, encoder.dim), np.float32)
    return Bag(scan_id, f, xy, tile_um, encoder.version, frac)
