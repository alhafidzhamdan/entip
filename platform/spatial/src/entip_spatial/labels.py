"""Label H&E nuclei from paired spatial data, as training data for pillar 2.

Idea (after CytoFormer; rebuilt here, no code copied): the spatial assay
knows what a cell *is* from its RNA; the H&E only shows what it looks like.
Placing assay cells on the H&E through a Transform lets each H&E nucleus
inherit a label, which then trains an image-only model.

Two routes:
  - single-cell assays (Xenium, Slide-tags): match each H&E nucleus to one
    assay cell (mutual nearest neighbour, distance gate tied to the
    Transform's error, plus an ambiguity test) and copy its label.
  - spot assays (Visium): every H&E nucleus in a spot gets the spot's
    dominant label, only when that label holds at least ``min_fraction`` of
    the spot. These are weak labels; their confidence is the fraction.

Unmatched or ambiguous nuclei get no label (they are left out, not called
negative).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .ontology import Transform
from .register import apply

COLUMNS = ["cell_id", "scan_id", "x_um", "y_um", "label", "confidence", "label_source", "assay_id",
           "assay_barcode", "match_dist_um", "transform_id", "transform_err_um"]


# ------------------------------------------------------------ assay cell labels

def marker_scores(adata, markers: Mapping[str, Sequence[str]], target_sum: float = 1e4) -> pd.DataFrame:
    """Per-cell score for each label = mean z-score of log-normalised marker genes."""
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1
    Xn = sparse.diags(target_sum / tot) @ X
    Xn = Xn.toarray()
    np.log1p(Xn, out=Xn)
    sd = Xn.std(0)
    sd[sd == 0] = 1
    Z = np.clip((Xn - Xn.mean(0)) / sd, -10, 10)
    genes = {g: i for i, g in enumerate(adata.var_names)}
    scores = {}
    for lab, gs in markers.items():
        idx = [genes[g] for g in gs if g in genes]
        if not idx:
            raise ValueError(f"no marker genes for {lab!r} found in the assay panel")
        scores[lab] = Z[:, idx].mean(1)
    return pd.DataFrame(scores, index=adata.obs_names)


def assign_labels(adata, markers: Optional[Mapping[str, Sequence[str]]] = None,
                  label_key: Optional[str] = None, min_margin: float = 0.25) -> pd.DataFrame:
    """Return per-assay-cell ``label`` and ``margin`` (best minus second-best score).

    Use ``label_key`` to take labels already in ``adata.obs`` (e.g. from a
    reference mapping done elsewhere); margin is then NaN (treated as sure).
    """
    if label_key is not None:
        return pd.DataFrame({"label": adata.obs[label_key].astype(str).to_numpy(), "margin": np.nan},
                            index=adata.obs_names)
    if not markers:
        raise ValueError("give markers or label_key")
    S = marker_scores(adata, markers)
    v = S.to_numpy()
    order = np.argsort(-v, axis=1)
    best = v[np.arange(len(v)), order[:, 0]]
    second = v[np.arange(len(v)), order[:, 1]] if v.shape[1] > 1 else np.full(len(v), -np.inf)
    lab = np.array(S.columns)[order[:, 0]].astype(object)
    margin = best - second
    lab[margin < min_margin] = None
    return pd.DataFrame({"label": lab, "margin": margin}, index=adata.obs_names)


# ------------------------------------------------------------ matching

def default_gate_um(tf: Transform) -> float:
    """3 x the Transform's error, kept between 4 and 12 um (about a nucleus apart)."""
    return float(np.clip(3.0 * tf.error_um(), 4.0, 12.0))


def match_nuclei(he_xy, assay_xy_he, max_dist_um: float, ratio: float = 0.7):
    """Mutual nearest neighbours within ``max_dist_um``; drop a match when the
    nucleus's second-nearest assay cell is nearly as close (d1 > ratio * d2)."""
    he_xy = np.asarray(he_xy, dtype=np.float64)
    a_xy = np.asarray(assay_xy_he, dtype=np.float64)
    if len(he_xy) == 0 or len(a_xy) < 2:
        return np.array([], int), np.array([], int), np.array([])
    d, j = cKDTree(a_xy).query(he_xy, k=2)
    _, back = cKDTree(he_xy).query(a_xy[j[:, 0]])
    ok = (d[:, 0] <= max_dist_um) & (back == np.arange(len(he_xy))) & (d[:, 0] <= ratio * d[:, 1])
    i = np.flatnonzero(ok)
    return i, j[i, 0], d[i, 0]


def _he_ids(cells: pd.DataFrame) -> np.ndarray:
    return (cells["cell_id"].astype(str).to_numpy() if "cell_id" in cells
            else np.array([f"cell_{k}" for k in range(len(cells))]))


def single_cell_labels(he_cells: pd.DataFrame, adata, tf: Transform, assay_labels: pd.DataFrame,
                       scan_id: str, assay_id: str, max_dist_um: Optional[float] = None,
                       ratio: float = 0.7) -> pd.DataFrame:
    gate = max_dist_um or default_gate_um(tf)
    a_he = apply(np.array(tf.matrix), adata.obsm["spatial_um"])
    he_xy = he_cells[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    i, j, d = match_nuclei(he_xy, a_he, gate, ratio)
    lab = assay_labels.reindex(adata.obs_names)
    labels = lab["label"].to_numpy()[j]
    margin = lab["margin"].to_numpy(dtype=np.float64)[j]
    keep = np.array([x is not None and x == x for x in labels], dtype=bool)
    m_conf = np.where(np.isnan(margin), 1.0, 1.0 - np.exp(-np.maximum(margin, 0.0)))
    conf = m_conf * (1.0 - d / gate)
    ids = _he_ids(he_cells)
    out = pd.DataFrame({
        "cell_id": ids[i], "scan_id": scan_id, "x_um": he_xy[i, 0], "y_um": he_xy[i, 1],
        "label": labels, "confidence": conf, "label_source": f"{adata.uns['entip']['kind']}-match",
        "assay_id": assay_id, "assay_barcode": np.asarray(adata.obs_names)[j], "match_dist_um": d,
        "transform_id": tf.transform_id, "transform_err_um": tf.error_um(),
    })
    return out[keep].reset_index(drop=True)[COLUMNS]


def spot_fractions_from_obs(adata, columns: Sequence[str]) -> pd.DataFrame:
    """Cell-type fractions per spot from obs columns (e.g. a deconvolution run elsewhere)."""
    F = adata.obs[list(columns)].to_numpy(dtype=np.float64)
    F = F / np.clip(F.sum(1, keepdims=True), 1e-12, None)
    return pd.DataFrame(F, index=adata.obs_names, columns=list(columns))


def spot_labels(he_cells: pd.DataFrame, adata, tf: Transform, fractions: pd.DataFrame, scan_id: str,
                assay_id: str, radius_um: Optional[float] = None, min_fraction: float = 0.8) -> pd.DataFrame:
    radius = radius_um or adata.uns["entip"].get("radius_um")
    if not radius:
        raise ValueError("spot radius unknown; pass radius_um")
    s_he = apply(np.array(tf.matrix), adata.obsm["spatial_um"])
    he_xy = he_cells[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    d, j = cKDTree(s_he).query(he_xy)
    F = fractions.reindex(adata.obs_names)
    top = F.to_numpy().argmax(1)
    frac = F.to_numpy()[np.arange(len(F)), top]
    i = np.flatnonzero((d <= radius) & (frac[j] >= min_fraction))
    ids = _he_ids(he_cells)
    return pd.DataFrame({
        "cell_id": ids[i], "scan_id": scan_id, "x_um": he_xy[i, 0], "y_um": he_xy[i, 1],
        "label": np.array(F.columns)[top[j[i]]], "confidence": frac[j[i]],
        "label_source": f"{adata.uns['entip']['kind']}-spot-dominant", "assay_id": assay_id,
        "assay_barcode": np.asarray(adata.obs_names)[j[i]], "match_dist_um": d[i],
        "transform_id": tf.transform_id, "transform_err_um": tf.error_um(),
    })[COLUMNS]
