"""Synthetic tissue with a paired H&E nucleus table, a Xenium-like run and a
Visium-like run, with known transforms. No real data; for tests and demos.

Truth is kept in ``truth.json`` and the ``true_type`` column of
``he_cells.csv`` so label transfer can be scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .register import apply, invert
from .tenx import write_10x_h5

TYPES = ["tumour", "t_cell", "macrophage", "endothelial"]
MARKERS = {
    "tumour": ["SOX2", "OLIG2", "EGFR", "MKI67", "NES"],
    "t_cell": ["CD3E", "CD2", "CD8A", "TRAC", "IL7R"],
    "macrophage": ["CD163", "CD68", "AIF1", "TYROBP", "C1QB"],
    "endothelial": ["PECAM1", "CLDN5", "VWF", "FLT1", "ESAM"],
}
HOUSEKEEPING = [f"HK{i:02d}" for i in range(20)]
GENES = [g for t in TYPES for g in MARKERS[t]] + HOUSEKEEPING


def _rigid(deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    a = np.deg2rad(deg)
    M = np.eye(3)
    M[:2, :2] = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M[:2, 2] = [tx, ty]
    return M


def _tissue(rng, w, h, spacing=12.0, keep=0.7):
    gx, gy = np.meshgrid(np.arange(spacing / 2, w, spacing), np.arange(spacing / 2, h, spacing))
    xy = np.c_[gx.ravel(), gy.ravel()] + rng.uniform(-3, 3, (gx.size, 2))
    xy = xy[rng.random(len(xy)) < keep]
    # tumour core on the left, immune rim, vessels as two horizontal bands
    cx, cy, r = 0.35 * w, 0.5 * h, 0.3 * min(w, h) + 0.15 * w
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    p = np.zeros((len(xy), 4))
    p[:, 0] = np.where(d < r, 0.85, 0.15)
    p[:, 1] = np.where(np.abs(d - r) < 80, 0.5, 0.1)
    p[:, 2] = 0.15
    vessel = (np.abs(xy[:, 1] - 0.25 * h) < 15) | (np.abs(xy[:, 1] - 0.75 * h) < 15)
    p[:, 3] = np.where(vessel, 3.0, 0.02)
    p /= p.sum(1, keepdims=True)
    t = np.array([rng.choice(4, p=row) for row in p])
    return xy, t


def _expression(rng, types):
    lam = np.full((len(types), len(GENES)), 0.3)
    lam[:, len(TYPES) * 5:] = 2.0
    for k in range(len(TYPES)):
        lam[types == k, k * 5:(k + 1) * 5] = 6.0
    return rng.poisson(lam).astype(np.int32)


def make(out: str | Path, seed: int = 0, width_um: float = 2000.0, height_um: float = 1500.0) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xy, t = _tissue(rng, width_um, height_um)
    n = len(xy)

    # H&E nuclei: centroid noise, 10% missed, 3% false detections; MitoEd-style calls
    det = rng.random(n) > 0.10
    he_xy = xy[det] + rng.normal(0, 1.0, (det.sum(), 2))
    he_type = np.array(TYPES)[t[det]]
    n_fp = int(0.03 * det.sum())
    he_xy = np.vstack([he_xy, rng.uniform([0, 0], [width_um, height_um], (n_fp, 2))])
    he_type = np.r_[he_type, ["none"] * n_fp]
    call = np.where(he_type == "tumour", rng.choice(["mitosis", "mimic", "other"], len(he_type), p=[.02, .03, .95]),
                    rng.choice(["mimic", "other"], len(he_type), p=[.01, .99]))
    p_mit = np.where(call == "mitosis", rng.uniform(.6, 1, len(call)), rng.uniform(0, .4, len(call)))
    he = pd.DataFrame({"cell_id": [f"he_{i}" for i in range(len(he_xy))], "x_um": he_xy[:, 0], "y_um": he_xy[:, 1],
                       "call": call, "p_mitosis": p_mit, "true_type": he_type})
    he.to_csv(out / "he_cells.csv", index=False)

    counts = _expression(rng, t)

    # Xenium-like: same section, assay frame = inverse(truth) of H&E frame
    T_x = _rigid(3.0, 1.01, 150.0, -80.0)  # assay um -> H&E um
    x_xy = apply(invert(T_x), xy) + rng.normal(0, 1.0, (n, 2))
    xd = out / "xenium"
    xd.mkdir(exist_ok=True)
    ids = [f"xc_{i}" for i in range(n)]
    pd.DataFrame({"cell_id": ids, "x_centroid": x_xy[:, 0], "y_centroid": x_xy[:, 1],
                  "transcript_counts": counts.sum(1)}).to_csv(xd / "cells.csv.gz", index=False)
    neg = rng.poisson(0.1, (n, 2)).astype(np.int32)  # control probes, should be dropped by the loader
    write_10x_h5(xd / "cell_feature_matrix.h5", np.hstack([counts, neg]), ids, GENES + ["NegControlProbe_1", "NegControlProbe_2"],
                 ["Gene Expression"] * len(GENES) + ["Negative Control Probe"] * 2)
    pick = rng.choice(n, 6, replace=False)
    lm = pd.DataFrame({"assay_x_um": x_xy[pick, 0], "assay_y_um": x_xy[pick, 1],
                       "he_x_um": xy[pick, 0] + rng.normal(0, 4, 6), "he_y_um": xy[pick, 1] + rng.normal(0, 4, 6)})
    lm.to_csv(out / "xenium_landmarks.csv", index=False)

    # Visium-like: 55 um spots on a 100 um hex grid, 0.25 um per full-res pixel
    T_v = _rigid(-5.0, 1.0, -60.0, 40.0)
    um_per_px = 0.25
    rows, cols, sp = [], [], []
    for r_i, y in enumerate(np.arange(50, height_um, 100 * np.sqrt(3) / 2)):
        for c_i, x in enumerate(np.arange(50 + (50 if r_i % 2 else 0), width_um, 100)):
            rows.append(r_i); cols.append(2 * c_i + (r_i % 2)); sp.append((x, y))
    sp_he = np.array(sp)
    sp_assay = apply(invert(T_v), sp_he)
    from scipy.spatial import cKDTree
    members = cKDTree(xy).query_ball_point(sp_he, r=27.5)
    vcounts = np.array([counts[m].sum(0) if m else np.zeros(len(GENES), int) for m in members])
    frac = np.array([np.bincount(t[m], minlength=4) / max(len(m), 1) for m in members])
    bcs = [f"SPOT{i:05d}-1" for i in range(len(sp))]
    vd = out / "visium"
    (vd / "spatial").mkdir(parents=True, exist_ok=True)
    write_10x_h5(vd / "filtered_feature_bc_matrix.h5", vcounts, bcs, GENES)
    px = sp_assay / um_per_px
    pd.DataFrame({"barcode": bcs, "in_tissue": 1, "array_row": rows, "array_col": cols,
                  "pxl_row_in_fullres": px[:, 1], "pxl_col_in_fullres": px[:, 0]}).to_csv(
        vd / "spatial" / "tissue_positions.csv", index=False)
    (vd / "spatial" / "scalefactors_json.json").write_text(json.dumps(
        {"spot_diameter_fullres": 55.0 / um_per_px, "fiducial_diameter_fullres": 89.0 / um_per_px,
         "tissue_hires_scalef": 0.1, "tissue_lowres_scalef": 0.03}))
    pick = rng.choice(len(sp), 6, replace=False)
    pd.DataFrame({"assay_x_um": sp_assay[pick, 0], "assay_y_um": sp_assay[pick, 1],
                  "he_x_um": sp_he[pick, 0] + rng.normal(0, 4, 6), "he_y_um": sp_he[pick, 1] + rng.normal(0, 4, 6)}
                 ).to_csv(out / "visium_landmarks.csv", index=False)
    pd.DataFrame(frac, index=pd.Index(bcs, name="barcode"), columns=TYPES).to_csv(out / "visium_fractions.csv")

    (out / "markers.json").write_text(json.dumps(MARKERS, indent=2))
    truth = {"xenium_to_he": T_x.tolist(), "visium_to_he": T_v.tolist(), "n_cells": int(n), "n_he": int(len(he)),
             "seed": seed}
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
