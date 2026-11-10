"""Tile bags: one slide's tissue tiles and their frozen embeddings.

The file format is the one the step-1 (triage) thread's ``entip_train.wsi.bags`` writes: an ``.npz`` with
``features`` (N, D), ``coords_um`` (N, 2, top-left of each tile), ``tile_um``, ``mpp``, ``encoder_version`` and
``scan_id``. For real slides build the bag with that module (tissue mask, 224 px tiles at 0.5 µm/px,
H-optimus-0, Apache-2.0) and load it here with :func:`load_bag`. :func:`embed_bag` is a small stand-in for
tests and the synthetic demo; it reuses this package's encoders.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..encoders import Encoder
from ..ontology import Scan
from ..slides import Reader


@dataclass
class TileBag:
    scan_id: str
    features: np.ndarray        # (N, D) float32
    coords_um: np.ndarray       # (N, 2) top-left x, y in µm, frame scan:<scan_id>
    tile_um: float
    mpp: float
    encoder_version: str

    def __post_init__(self):
        self.features = np.asarray(self.features, np.float32)
        self.coords_um = np.asarray(self.coords_um, np.float64).reshape(-1, 2)
        if len(self.features) != len(self.coords_um):
            raise ValueError("features and coords_um differ in length")

    def __len__(self) -> int:
        return len(self.features)

    @property
    def tile_area_mm2(self) -> float:
        return (self.tile_um / 1000.0) ** 2

    @property
    def tissue_area_mm2(self) -> float:
        return len(self) * self.tile_area_mm2

    def tile_ids(self) -> list:
        """Stable ids from grid position: ``<scan_id>:t<row>-<col>`` at this tile size."""
        rc = np.round(self.coords_um / self.tile_um).astype(int)
        return [f"{self.scan_id}:t{r:04d}-{c:04d}" for c, r in rc]

    def grid(self) -> np.ndarray:
        """(N, 2) integer (row, col) of each tile."""
        rc = np.round(self.coords_um / self.tile_um).astype(int)
        return rc[:, ::-1]

    def centres_um(self) -> np.ndarray:
        return self.coords_um + self.tile_um / 2


def load_bag(path: str | Path) -> TileBag:
    z = np.load(path, allow_pickle=False)
    return TileBag(str(z["scan_id"]), z["features"], z["coords_um"], float(z["tile_um"]), float(z["mpp"]),
                   str(z["encoder_version"]))


def save_bag(bag: TileBag, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, scan_id=bag.scan_id, features=bag.features, coords_um=bag.coords_um, tile_um=bag.tile_um,
             mpp=bag.mpp, encoder_version=bag.encoder_version)
    return p


def embed_bag(scan: Scan, reader: Reader, encoder: Encoder, tile_um: float = 112.0,
              min_tissue: float = 0.25, white: int = 220, batch: int = 128) -> TileBag:
    """Grid the scan, keep tiles with at least ``min_tissue`` non-white pixels, embed them."""
    keep, crops = [], []
    for i in range(int(scan.height_um // tile_um)):
        for j in range(int(scan.width_um // tile_um)):
            x0, y0 = j * tile_um, i * tile_um
            c = reader.read_um(x0 + tile_um / 2, y0 + tile_um / 2, tile_um, encoder.input_px)
            if (c.min(-1) < white).mean() >= min_tissue:
                keep.append((x0, y0))
                crops.append(c)
    feats = [encoder.embed(np.stack(crops[k:k + batch])) for k in range(0, len(crops), batch)]
    F = np.concatenate(feats) if feats else np.zeros((0, getattr(encoder, "dim", 0)), np.float32)
    return TileBag(scan.scan_id, F, np.asarray(keep, float).reshape(-1, 2), tile_um, scan.mpp, encoder.version)
