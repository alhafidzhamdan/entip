"""Per-spot tables: summarise H&E cells (e.g. MitoEd calls) under each spot.

The H&E cell table needs ``x_um, y_um`` in the scan frame and a class
column (MitoEd: ``call`` in {mitosis, mimic, uncertain, ...}; probabilities
such as ``p_mitosis``). Spots must already be in the scan frame
(``obsm["spatial_he_um"]`` after ``register.to_he``). The returned table is
indexed by spot barcode, so it joins straight onto ``adata.obs``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def spot_table(spot_xy_um, spot_ids: Sequence[str], cells: pd.DataFrame, radius_um: float,
               class_col: str = "call", prob_cols: Optional[Sequence[str]] = None,
               focus_class: Optional[str] = "mitosis", neighbourhood_um: Optional[float] = None) -> pd.DataFrame:
    spot_xy = np.asarray(spot_xy_um, dtype=np.float64)
    out = pd.DataFrame(index=pd.Index([str(s) for s in spot_ids], name="barcode"))
    xy = cells[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    if len(xy) == 0:
        out["n_cells"] = 0
        return out
    members = cKDTree(xy).query_ball_point(spot_xy, r=radius_um)
    out["n_cells"] = [len(m) for m in members]
    if class_col in cells:
        cls = cells[class_col].astype(str).to_numpy()
        for c in sorted(set(cls)):
            hit = cls == c
            out[f"n_{c}"] = [int(hit[m].sum()) for m in members]
    prob_cols = list(prob_cols) if prob_cols is not None else [c for c in cells.columns if c.startswith("p_")]
    for pc in prob_cols:
        p = cells[pc].to_numpy(dtype=np.float64)
        out[f"{pc}_max"] = [float(p[m].max()) if m else 0.0 for m in members]
        out[f"{pc}_mean"] = [float(p[m].mean()) if m else np.nan for m in members]
    if focus_class and class_col in cells:
        nb = neighbourhood_um or 2 * radius_um
        f_xy = xy[cells[class_col].astype(str).to_numpy() == focus_class]
        n_local = (np.array([len(m) for m in cKDTree(f_xy).query_ball_point(spot_xy, r=nb)])
                   if len(f_xy) else np.zeros(len(spot_xy), dtype=int))
        out[f"{focus_class}_local"] = n_local
        out[f"{focus_class}_local_per_mm2"] = n_local / (np.pi * (nb / 1000.0) ** 2)
    return out


def join_obs(adata, table: pd.DataFrame, prefix: str = "he_"):
    joined = table.add_prefix(prefix).reindex(adata.obs_names)
    for c in joined.columns:
        adata.obs[c] = joined[c].to_numpy()
    return adata
