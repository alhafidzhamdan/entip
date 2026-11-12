"""Slide -> bag of tile embeddings, the shared feature pipeline for the glioma workflow models.

Tiles are ``tile_um`` square (default 112 µm, which is 224 px at 0.5 µm/px, the input H-optimus-0 was
pretrained on: https://huggingface.co/bioptimus/H-optimus-0). Each tile is read at the encoder's input size
whatever the scanner's pixel size, so slides from different scanners give the same physical field.

Tissue detection is deliberately simple (saturation and brightness on a 16 px preview of each tile). Run
HistoQC (BSD-3-Clause-Clear) upstream for pen and folds; this only drops background. Each tile also gets a
focus measure (``sharpness``), because some public scans are out of focus (seen on a TCGA-GBM slide).

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
    sharpness: Optional[np.ndarray] = None   # (N,) focus measure per tile (variance of the Laplacian), or None

    def __len__(self) -> int:
        return len(self.features)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=self.features.astype(np.float16), coords_um=self.coords_um,
                            tissue_frac=self.tissue_frac, tile_um=np.float64(self.tile_um),
                            sharpness=(self.sharpness if self.sharpness is not None else np.zeros(0, np.float32)),
                            scan_id=np.str_(self.scan_id), encoder=np.str_(self.encoder))
        return path

    @classmethod
    def load(cls, path: str | Path, max_tiles: Optional[int] = None, seed: int = 0, dtype=np.float32) -> "Bag":
        """``max_tiles`` keeps a seeded random subset (for training, where thousands of slides must fit in
        memory); ``dtype=np.float16`` halves memory again. Leave both at their defaults for inference."""
        with np.load(path, allow_pickle=False) as z:
            f, xy, tf = z["features"], z["coords_um"], z["tissue_frac"]
            sh = z["sharpness"] if "sharpness" in z.files and len(z["sharpness"]) == len(f) else None
            if max_tiles is not None and len(f) > max_tiles:
                keep = np.sort(np.random.default_rng(seed).choice(len(f), max_tiles, replace=False))
                f, xy, tf = f[keep], xy[keep], tf[keep]
                sh = sh[keep] if sh is not None else None
            return cls(str(z["scan_id"]), f.astype(dtype), xy.astype(np.float64),
                       float(z["tile_um"]), str(z["encoder"]), tf.astype(np.float32), sh)


def tissue_fraction(rgb: np.ndarray, min_sat: float = 0.07, max_val: float = 0.94, min_val: float = 0.08) -> float:
    """Share of pixels that look like stained tissue: coloured, not white glass, not black."""
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    tissue = (sat > min_sat) & (mx < max_val) & (mx > min_val)
    return float(tissue.mean())


def sharpness(rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grey image (0-255 scale): low on out-of-focus tiles. A relative
    measure: compare slides read at the same tile size and pixel count, and set any cut-off on local data."""
    g = rgb.astype(np.float32).mean(-1)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def tile_grid(reader, width_um: float, height_um: float, tile_um: float = DEFAULT_TILE_UM,
              min_tissue: float = 0.5, block: int = 32):
    """Top-left corners (µm) and tissue fractions of tiles with at least ``min_tissue`` tissue.

    The slide is previewed in blocks of ``block`` x ``block`` tiles (one low-resolution read each, 16 px per
    tile) rather than one read per tile: a 20 x 15 mm slide is about 24,000 tiles but only 24 reads."""
    ny, nx = int(height_um // tile_um), int(width_um // tile_um)
    xy, frac = [], []
    size_um, px = block * tile_um, block * PREVIEW_PX
    for bi in range(0, ny, block):
        for bj in range(0, nx, block):
            x0, y0 = bj * tile_um, bi * tile_um
            img = reader.read_um(x0 + size_um / 2, y0 + size_um / 2, size_um, px)
            for i in range(min(block, ny - bi)):
                for j in range(min(block, nx - bj)):
                    f = tissue_fraction(img[i * PREVIEW_PX:(i + 1) * PREVIEW_PX, j * PREVIEW_PX:(j + 1) * PREVIEW_PX])
                    if f >= min_tissue:
                        xy.append((x0 + j * tile_um, y0 + i * tile_um))
                        frac.append(f)
    order = np.lexsort((np.asarray(xy).reshape(-1, 2)[:, 0], np.asarray(xy).reshape(-1, 2)[:, 1]))
    return np.asarray(xy, np.float64).reshape(-1, 2)[order], np.asarray(frac, np.float32)[order]


def extract_bag(reader, scan_id: str, width_um: float, height_um: float, encoder,
                tile_um: float = DEFAULT_TILE_UM, min_tissue: float = 0.5, max_tiles: Optional[int] = None,
                batch: int = 64, seed: int = 0) -> Bag:
    """Embed every tissue tile of one slide with a frozen encoder. ``max_tiles`` subsamples at random
    (seeded) for very large slides; leave it None for the final runs."""
    xy, frac = tile_grid(reader, width_um, height_um, tile_um, min_tissue)
    if max_tiles is not None and len(xy) > max_tiles:
        keep = np.sort(np.random.default_rng(seed).choice(len(xy), max_tiles, replace=False))
        xy, frac = xy[keep], frac[keep]
    feats, sharp = [], []
    for k in range(0, len(xy), batch):
        crops = np.stack([reader.read_um(x + tile_um / 2, y + tile_um / 2, tile_um, encoder.input_px)
                          for x, y in xy[k:k + batch]])
        feats.append(encoder.embed(crops))
        sharp += [sharpness(c) for c in crops]
    f = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, encoder.dim), np.float32)
    return Bag(scan_id, f, xy, tile_um, encoder.version, frac, np.asarray(sharp, np.float32))
