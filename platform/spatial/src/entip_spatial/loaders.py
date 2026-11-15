"""Load spatial assays into AnnData with positions in micrometres.

Every loader returns an AnnData with:
  - ``obsm["spatial_um"]``: (n_obs, 2) x, y in micrometres, in the assay's own frame
  - ``uns["entip"]``: {"kind", "um_per_unit", "object_kind", "radius_um", "source"}

Native coordinates are kept in ``obsm["spatial"]`` (Visium: full-resolution
pixels, x = column, y = row; Xenium: already micrometres).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from .tenx import read_10x_h5

# Visium (v1/v2 standard slides): 55 um spot diameter, 100 um centre to centre.
VISIUM_SPOT_DIAMETER_UM = 55.0


def sha256(path: str | Path, chunk: int = 1 << 20) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _meta(adata, kind, um_per_unit, object_kind, radius_um, source):
    adata.uns["entip"] = {"kind": kind, "um_per_unit": float(um_per_unit), "object_kind": object_kind,
                          "radius_um": None if radius_um is None else float(radius_um), "source": str(source)}
    return adata


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    new = spatial_dir / "tissue_positions.csv"        # Space Ranger 2.0+: has a header
    old = spatial_dir / "tissue_positions_list.csv"   # Space Ranger 1.x: no header
    if new.exists():
        df = pd.read_csv(new)
    elif old.exists():
        df = pd.read_csv(old, header=None, names=cols)
    else:
        raise FileNotFoundError(f"no tissue_positions(.csv|_list.csv) in {spatial_dir}")
    return df.set_index("barcode")


def load_visium(outs: str | Path, in_tissue_only: bool = True) -> ad.AnnData:
    """Space Ranger ``outs/`` folder: filtered_feature_bc_matrix.h5 + spatial/.

    Micrometres per full-res pixel = 55 / spot_diameter_fullres (from
    scalefactors_json.json). Visium HD (square bins) is not handled here.
    """
    outs = Path(outs)
    adata = read_10x_h5(outs / "filtered_feature_bc_matrix.h5")
    pos = _read_tissue_positions(outs / "spatial")
    sf = json.loads((outs / "spatial" / "scalefactors_json.json").read_text())
    pos = pos.reindex(adata.obs_names)
    if pos["pxl_row_in_fullres"].isna().any():
        raise ValueError("some matrix barcodes have no tissue position")
    if in_tissue_only:
        keep = pos["in_tissue"].to_numpy() == 1
        adata, pos = adata[keep].copy(), pos[keep]
    adata.obs[["in_tissue", "array_row", "array_col"]] = pos[["in_tissue", "array_row", "array_col"]].to_numpy()
    px = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)
    um_per_px = VISIUM_SPOT_DIAMETER_UM / float(sf["spot_diameter_fullres"])
    adata.obsm["spatial"] = px
    adata.obsm["spatial_um"] = px * um_per_px
    adata.uns["scalefactors"] = sf
    return _meta(adata, "visium", um_per_px, "spot", VISIUM_SPOT_DIAMETER_UM / 2, outs)


def load_xenium(outs: str | Path) -> ad.AnnData:
    """Xenium output folder: cell_feature_matrix.h5 + cells.csv.gz (or cells.parquet).

    ``x_centroid`` / ``y_centroid`` are in micrometres. Control probes and
    codewords are dropped (only "Gene Expression" features kept).
    """
    outs = Path(outs)
    adata = read_10x_h5(outs / "cell_feature_matrix.h5")
    for name in ("cells.parquet", "cells.csv.gz", "cells.csv"):
        p = outs / name
        if p.exists():
            cells = pd.read_parquet(p) if name.endswith("parquet") else pd.read_csv(p)
            break
    else:
        raise FileNotFoundError(f"no cells.parquet / cells.csv(.gz) in {outs}")
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").reindex(adata.obs_names)
    if cells["x_centroid"].isna().any():
        raise ValueError("some matrix cells are missing from the cells table")
    for c in ("transcript_counts", "cell_area", "nucleus_area"):
        if c in cells:
            adata.obs[c] = cells[c].to_numpy()
    xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    adata.obsm["spatial"] = xy
    adata.obsm["spatial_um"] = xy.copy()
    return _meta(adata, "xenium", 1.0, "nucleus", None, outs)


def load_anndata(path: str | Path, kind: str = "anndata", spatial_key: str = "spatial",
                 um_per_unit: float = 1.0, object_kind: str = "nucleus",
                 radius_um: Optional[float] = None) -> ad.AnnData:
    """Any .h5ad with positions in ``obsm[spatial_key]``. For Slide-tags use
    ``kind="slide_tags"``; state the units with ``um_per_unit`` (no guessing)."""
    adata = ad.read_h5ad(path)
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found; has {list(adata.obsm)}")
    xy = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    adata.obsm["spatial_um"] = xy * um_per_unit
    return _meta(adata, kind, um_per_unit, object_kind, radius_um, path)


def load_assay(kind: str, path: str | Path, **kw) -> ad.AnnData:
    if kind == "visium":
        return load_visium(path, **kw)
    if kind == "xenium":
        return load_xenium(path, **kw)
    if kind in ("slide_tags", "anndata"):
        return load_anndata(path, kind=kind, **kw)
    raise ValueError(f"unknown assay kind {kind!r}")
