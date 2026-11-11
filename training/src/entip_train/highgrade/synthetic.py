"""A synthetic glioma-like slide for tests and the demo. Fully made up; no patient data.

Pink fibrillary background with dense purple nuclei ("tumour"), a white strip of glass, optional pale
necrosis patches with nuclear dust and a palisade of crowded nuclei at the edge, optional "microvascular
proliferation" tufts (dark multilayered rings of nuclei around open lumens), and a MitoEd-style candidate table
with a chosen number of true mitoses among low-scoring distractors. Truth is returned as polygons in µm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..ontology import Scan
from ..slides import ArrayReader


@dataclass
class SyntheticGBM:
    scan: Scan
    reader: ArrayReader
    necrosis_polys: List[list] = field(default_factory=list)
    mvp_polys: List[list] = field(default_factory=list)
    tumour_polys: List[list] = field(default_factory=list)   # areas with neither feature (for negatives)
    candidates: List[dict] = field(default_factory=list)
    n_true_mitoses: int = 0


def _disc(img, yy, xx, x, y, r, colour, ex=1.0, angle=0.0):
    y0, y1 = max(0, int(y - r - 2)), min(img.shape[0], int(y + r + 3))
    x0, x1 = max(0, int(x - r - 2)), min(img.shape[1], int(x + r + 3))
    if y1 <= y0 or x1 <= x0:
        return
    sx, sy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
    c, s = np.cos(angle), np.sin(angle)
    u, v = c * sx + s * sy, -s * sx + c * sy
    m = (u / r) ** 2 + (v / (r * ex)) ** 2 < 1
    img[y0:y1, x0:x1][m] = colour


def _square(cx, cy, h):
    return [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]


def synthetic_gbm(scan_id: str = "SCN-SYN-GBM-01", seed: int = 0, tiles: int = 20, tile_um: float = 112.0,
                  mpp: float = 0.5, necrosis: int = 2, mvp: int = 4, mitoses: int = 3, distractors: int = 40,
                  glass_cols: int = 2) -> SyntheticGBM:
    rng = np.random.default_rng(seed)
    tile_px = int(round(tile_um / mpp))
    H = W = tiles * tile_px
    img = np.empty((H, W, 3), np.float32)
    img[:] = (232, 168, 200)
    img += rng.normal(0, 7, (H, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    tissue_w = W - glass_cols * tile_px
    img[:, tissue_w:] = 250                                       # glass

    nec_polys, nec_mask = [], np.zeros((H, W), bool)
    for _ in range(necrosis):
        r = rng.uniform(1.4, 2.2) * tile_px
        cx, cy = rng.uniform(r, tissue_w - r), rng.uniform(r, H - r)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        nec_mask |= m
        img[m] = img[m] * 0.35 + np.array([236, 188, 200]) * 0.65
        for _ in range(int(r * r / 40)):                          # nuclear dust
            a, d = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform(0, 1))
            _disc(img, yy, xx, cx + d * np.cos(a), cy + d * np.sin(a), rng.uniform(0.8, 1.6), (90, 40, 110))
        for _ in range(int(2 * np.pi * r / 4)):                   # palisade at the edge
            a = rng.uniform(0, 2 * np.pi)
            d = r + rng.uniform(2, 14)
            _disc(img, yy, xx, cx + d * np.cos(a), cy + d * np.sin(a), 4.5, (95, 50, 140), ex=0.45, angle=a)
        n = 24
        nec_polys.append([[(cx + r * 0.8 * np.cos(t)) * mpp, (cy + r * 0.8 * np.sin(t)) * mpp]
                          for t in np.linspace(0, 2 * np.pi, n, endpoint=False)])

    mvp_polys, mvp_mask = [], np.zeros((H, W), bool)
    placed = 0
    while placed < mvp:
        R = rng.uniform(0.28, 0.38) * tile_px
        cx, cy = rng.uniform(R + 4, tissue_w - R - 4), rng.uniform(R + 4, H - R - 4)
        if nec_mask[max(0, int(cy - 2 * R)):int(cy + 2 * R), max(0, int(cx - 2 * R)):int(cx + 2 * R)].any() or \
                mvp_mask[int(cy), int(cx)]:
            continue
        mvp_mask |= (xx - cx) ** 2 + (yy - cy) ** 2 < (1.5 * R) ** 2
        _disc(img, yy, xx, cx, cy, R, (170, 80, 130))             # dense eosinophilic tuft
        for _ in range(int(R * 1.6)):                             # crowded multilayered nuclei
            a, d = rng.uniform(0, 2 * np.pi), R * np.sqrt(rng.uniform(0.1, 1))
            _disc(img, yy, xx, cx + d * np.cos(a), cy + d * np.sin(a), 5, (55, 25, 95), ex=0.5, angle=a + np.pi / 2)
        for _ in range(rng.integers(3, 6)):                       # open lumens
            a, d = rng.uniform(0, 2 * np.pi), R * rng.uniform(0, 0.6)
            _disc(img, yy, xx, cx + d * np.cos(a), cy + d * np.sin(a), rng.uniform(6, 10), (250, 238, 240))
        mvp_polys.append(_square(cx * mpp, cy * mpp, R * mpp))
        placed += 1

    # background tumour nuclei
    n_nuc = int(tissue_w * H / 450)
    for _ in range(n_nuc):
        x, y = rng.uniform(0, tissue_w), rng.uniform(0, H)
        if nec_mask[int(y), int(x)] or mvp_mask[int(y), int(x)]:
            continue
        _disc(img, yy, xx, x, y, rng.uniform(3.5, 6), np.array((120, 70, 160)) + rng.normal(0, 12, 3),
              ex=rng.uniform(0.6, 1.0), angle=rng.uniform(0, np.pi))

    rgb = np.clip(img, 0, 255).astype(np.uint8)
    scan = Scan(scan_id, f"synthetic:{seed}", mpp, W * mpp, H * mpp)

    # tumour polygons: tiles far from both features (for explicit negatives)
    tum = []
    for i in range(tiles):
        for j in range(tiles - glass_cols):
            y0, x0 = i * tile_px, j * tile_px
            blk = (slice(max(0, y0 - tile_px // 2), y0 + tile_px + tile_px // 2),
                   slice(max(0, x0 - tile_px // 2), x0 + tile_px + tile_px // 2))
            if not nec_mask[blk].any() and not mvp_mask[blk].any():
                cx, cy = (x0 + tile_px / 2) * mpp, (y0 + tile_px / 2) * mpp
                tum.append(_square(cx, cy, tile_um * 0.3))

    cands = []
    for k in range(mitoses):
        cands.append({"id": f"{scan_id}:m{k:06d}", "scan_id": scan_id, "x_um": float(rng.uniform(0, tissue_w * mpp)),
                      "y_um": float(rng.uniform(0, H * mpp)), "p": float(rng.uniform(0.9, 0.99))})
    for k in range(distractors):
        cands.append({"id": f"{scan_id}:m{mitoses + k:06d}", "scan_id": scan_id,
                      "x_um": float(rng.uniform(0, tissue_w * mpp)), "y_um": float(rng.uniform(0, H * mpp)),
                      "p": float(rng.beta(1, 60))})
    return SyntheticGBM(scan, ArrayReader(rgb, mpp), nec_polys, mvp_polys, tum, cands, mitoses)
