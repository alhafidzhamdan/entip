"""Make synthetic H&E-like slides and cell tables for the GUI mock.

Everything here is invented: no patient data, no real images. Each slide is
painted with a simple Beer-Lambert stain model (haematoxylin + eosin optical
densities), cut into a Deep Zoom (DZI) pyramid for OpenSeadragon, and written
with a cell table whose positions are in micrometres in frame "scan:<scan_id>".

Each cell also gets a short synthetic embedding. It stands in for the
H-optimus-0 embeddings that the training backend (P2) will serve, so the mock
live head has something to learn from.

Usage:  python gui/tools/make_mock_slides.py            (writes gui/web/mock/)
Needs:  numpy, Pillow (both BSD-style licences).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MPP = 0.25  # µm per pixel (a 40x scan)
TILE = 1024
EMB_DIM = 12

# Optical density per unit stain (Ruifrok & Johnston H&E vectors).
OD_H = np.array([0.65, 0.70, 0.29], dtype=np.float32)
OD_E = np.array([0.07, 0.99, 0.11], dtype=np.float32)

CLASSES = ["tumour", "mitosis", "apoptotic", "lymphocyte", "endothelial", "oligo"]

# One render per synthetic scan. Rates are per cell, not real incidences.
RENDERS = {
    "SCN-0001": dict(seed=11, w=6144, h=4608, density=3200, mitosis=0.018, apoptotic=0.02,
                     lymph=0.05, vessels=7, necrosis=True, oligo=0.0),
    "SCN-0002": dict(seed=22, w=6144, h=4608, density=2400, mitosis=0.006, apoptotic=0.012,
                     lymph=0.04, vessels=5, necrosis=False, oligo=0.0),
    "SCN-0003": dict(seed=33, w=6144, h=4608, density=2600, mitosis=0.003, apoptotic=0.006,
                     lymph=0.03, vessels=8, necrosis=False, oligo=0.85),
    "SCN-0004": dict(seed=44, w=6144, h=4608, density=1800, mitosis=0.001, apoptotic=0.004,
                     lymph=0.02, vessels=4, necrosis=False, oligo=0.0),
    "SCN-0005": dict(seed=55, w=6144, h=4608, density=3400, mitosis=0.022, apoptotic=0.025,
                     lymph=0.06, vessels=9, necrosis=True, oligo=0.0),
}


def smooth_noise(rng, w, h, scale, blur):
    """Low-frequency noise in [0, 1] at full size."""
    small = Image.fromarray((rng.random((max(2, h // scale), max(2, w // scale))) * 255).astype(np.uint8))
    small = small.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(small.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def blob(cx, cy, rx, ry, angle, rng, jitter=0.12, n=18):
    """An irregular ellipse as a polygon."""
    pts = []
    radial = 1 + jitter * rng.standard_normal(n)
    radial = np.convolve(np.r_[radial[-1], radial, radial[0]], [0.25, 0.5, 0.25], "valid")
    ca, sa = math.cos(angle), math.sin(angle)
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * radial[i] * math.cos(t), ry * radial[i] * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def paint(scan_id, p):
    rng = np.random.default_rng(p["seed"])
    w, h = p["w"], p["h"]
    um = lambda v: v / MPP  # micrometres to pixels

    # Tissue outline: a smooth blob with a ragged edge, white glass outside.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ell = ((xx - w / 2) / (w * 0.46)) ** 2 + ((yy - h / 2) / (h * 0.44)) ** 2
    tissue = (ell + 0.6 * (smooth_noise(rng, w, h, 192, 2) - 0.5)) < 1.0
    del yy, xx, ell

    # Eosin: fibrillary neuropil, stronger in streaks.
    fib = smooth_noise(rng, w, h, 6, 1.2)
    streak = smooth_noise(rng, w, h, 64, 3)
    dE = (0.18 + 0.20 * fib + 0.12 * streak) * tissue
    dH = np.zeros((h, w), np.float32)

    E = Image.new("L", (w, h), 0)
    H = Image.new("L", (w, h), 0)
    dE_draw, dH_draw = ImageDraw.Draw(E), ImageDraw.Draw(H)
    clear = Image.new("L", (w, h), 0)  # halos and lumens (remove stain)
    clear_draw = ImageDraw.Draw(clear)

    # Necrosis: a pale granular patch with nuclear dust.
    necro = np.zeros((h, w), bool)
    regions = []
    if p["necrosis"]:
        nx, ny, nr = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.3, 0.7), um(rng.uniform(180, 260))
        poly = blob(nx, ny, nr, nr * 0.8, rng.uniform(0, math.pi), rng, 0.2, 28)
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        necro = np.asarray(m) > 0
        regions.append({"region_id": f"{scan_id}-R1", "kind": "necrosis", "frame": f"scan:{scan_id}",
                        "polygon_um": [[round(x * MPP, 1), round(y * MPP, 1)] for x, y in poly],
                        "source": {"kind": "synthetic", "name": "make_mock_slides"}})
        for _ in range(int(nr * nr / 260)):
            a, r = rng.uniform(0, 2 * math.pi), nr * math.sqrt(rng.random()) * 0.8
            x, y = nx + r * math.cos(a), ny + r * math.sin(a)
            s = rng.uniform(1.5, 3.5)
            dH_draw.ellipse([x - s, y - s, x + s, y + s], fill=int(rng.uniform(60, 150)))

    # Vessels: rings of eosin with a pale lumen and red cells; endothelial nuclei on the wall.
    cells = []
    vessels = []
    for _ in range(p["vessels"]):
        for _try in range(50):
            vx, vy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
            if tissue[int(vy), int(vx)] and not necro[int(vy), int(vx)]:
                break
        vr = um(rng.uniform(14, 40))
        ang = rng.uniform(0, math.pi)
        asp = rng.uniform(0.5, 1)
        vessels.append((vx, vy, vr * 1.5))
        dE_draw.polygon(blob(vx, vy, vr * 1.35, vr * 1.35 * asp, ang, rng, 0.06), fill=150)
        clear_draw.polygon(blob(vx, vy, vr, vr * asp, ang, rng, 0.06), fill=230)
        for _ in range(int(vr / 6)):  # red cells in the lumen
            a, r = rng.uniform(0, 2 * math.pi), vr * 0.7 * math.sqrt(rng.random())
            x, y = vx + r * math.cos(a) * 1, vy + r * math.sin(a) * asp
            s = um(3.5)
            dE_draw.ellipse([x - s, y - s, x + s, y + s], fill=235)
        n_endo = max(3, int(2 * math.pi * vr / um(12)))
        for i in range(n_endo):
            t = 2 * math.pi * (i + rng.uniform(-0.2, 0.2)) / n_endo
            ex = vx + vr * 1.12 * (math.cos(t) * math.cos(ang) - asp * math.sin(t) * math.sin(ang))
            ey = vy + vr * 1.12 * (math.cos(t) * math.sin(ang) + asp * math.sin(t) * math.cos(ang))
            cells.append(("endothelial", ex, ey, t + ang + math.pi / 2))

    # Parenchymal cells by Poisson-disc-ish rejection on a coarse grid.
    area_mm2 = tissue.sum() * (MPP / 1000) ** 2
    n_target = int(p["density"] * area_mm2)
    min_d = um(7.5)
    grid = {}
    placed = 0
    attempts = 0
    while placed < n_target and attempts < n_target * 8:
        attempts += 1
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        if not tissue[int(y), int(x)] or necro[int(y), int(x)]:
            continue
        if any((x - vx) ** 2 + (y - vy) ** 2 < vr ** 2 for vx, vy, vr in vessels):
            continue
        gx, gy = int(x // min_d), int(y // min_d)
        if any((x - qx) ** 2 + (y - qy) ** 2 < min_d ** 2
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for qx, qy in grid.get((gx + dx, gy + dy), [])):
            continue
        grid.setdefault((gx, gy), []).append((x, y))
        u = rng.random()
        if u < p["mitosis"]:
            kind = "mitosis"
        elif u < p["mitosis"] + p["apoptotic"]:
            kind = "apoptotic"
        elif u < p["mitosis"] + p["apoptotic"] + p["lymph"]:
            kind = "lymphocyte"
        else:
            kind = "oligo" if rng.random() < p["oligo"] else "tumour"
        cells.append((kind, x, y, rng.uniform(0, math.pi)))
        placed += 1

    out = []
    for kind, x, y, ang in cells:
        if kind == "tumour":
            rx, ry = um(rng.uniform(5, 8)), um(rng.uniform(3.5, 5.5))
            dH_draw.polygon(blob(x, y, rx, ry, ang, rng, 0.14), fill=int(rng.uniform(95, 150)))
            r = max(rx, ry)
        elif kind == "oligo":
            r = um(rng.uniform(3.6, 4.4))
            clear_draw.ellipse([x - r * 1.9, y - r * 1.9, x + r * 1.9, y + r * 1.9], fill=170)
            dH_draw.polygon(blob(x, y, r, r * 0.95, ang, rng, 0.05), fill=int(rng.uniform(130, 170)))
        elif kind == "lymphocyte":
            r = um(rng.uniform(2.8, 3.4))
            dH_draw.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(200, 235)))
        elif kind == "endothelial":
            rx, ry = um(rng.uniform(6, 8)), um(rng.uniform(1.6, 2.3))
            dH_draw.polygon(blob(x, y, rx, ry, ang, rng, 0.08), fill=int(rng.uniform(140, 190)))
            r = rx
        elif kind == "apoptotic":
            r = um(rng.uniform(2.5, 4))
            clear_draw.ellipse([x - r * 2, y - r * 2, x + r * 2, y + r * 2], fill=150)
            dE_draw.ellipse([x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6], fill=200)
            for _ in range(int(rng.integers(1, 4))):
                ox, oy, s = rng.normal(0, r * 0.4), rng.normal(0, r * 0.4), r * rng.uniform(0.4, 0.8)
                dH_draw.ellipse([x + ox - s, y + oy - s, x + ox + s, y + oy + s], fill=int(rng.uniform(220, 250)))
        else:  # mitosis: metaphase plate or clumped chromosomes, with a clear zone
            r = um(rng.uniform(5, 7))
            clear_draw.ellipse([x - r * 1.4, y - r * 1.4, x + r * 1.4, y + r * 1.4], fill=70)
            if rng.random() < 0.6:  # metaphase: spiky dark band
                L, T = r, um(rng.uniform(1.4, 2.2))
                ca, sa = math.cos(ang), math.sin(ang)
                for _ in range(40):
                    t = rng.uniform(-L, L)
                    px, py = x + t * ca, y + t * sa
                    off = rng.normal(0, T * 0.6)
                    px, py = px - off * sa, py + off * ca
                    s = um(rng.uniform(0.7, 1.2))
                    dH_draw.ellipse([px - s, py - s, px + s, py + s], fill=int(rng.uniform(220, 255)))
            else:  # prophase / clumped: irregular dark clusters
                for _ in range(24):
                    ox, oy = rng.normal(0, r * 0.4), rng.normal(0, r * 0.4)
                    s = um(rng.uniform(0.7, 1.3))
                    dH_draw.ellipse([x + ox - s, y + oy - s, x + ox + s, y + oy + s], fill=int(rng.uniform(210, 250)))
        out.append((kind, x, y, r))

    # Compose stain densities into RGB.
    tex = 0.7 + 0.6 * smooth_noise(rng, w, h, 2, 0.6)
    Hd = np.asarray(H.filter(ImageFilter.GaussianBlur(0.9)), np.float32) / 255.0
    Hd = (1.1 * Hd + 0.9 * np.clip(Hd - 0.8, 0, None) * 5) * tex  # very dense chromatin gets darker
    Ed = dE + np.asarray(E.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0 * 0.55
    clr = np.asarray(clear.filter(ImageFilter.GaussianBlur(2.5)), np.float32) / 255.0
    Ed = Ed * (1 - 0.85 * clr) * np.where(necro, 0.75, 1.0)
    Ed = np.where(tissue, Ed, 0.0)
    od = Hd[..., None] * OD_H + Ed[..., None] * OD_E
    rgb = 248 * np.exp(-od) + rng.normal(0, 2.0, (h, w, 1))
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.5))
    return img, out, regions, float(area_mm2)


def embeddings(kinds, rng):
    """Class-conditional synthetic embeddings. Mitosis and apoptotic overlap on purpose."""
    base = np.random.default_rng(7)  # same class centres on every slide
    centres = {k: base.normal(0, 1, EMB_DIM) for k in CLASSES}
    centres["apoptotic"] = 0.55 * centres["mitosis"] + 0.45 * centres["apoptotic"]
    centres["lymphocyte"] = 0.6 * centres["lymphocyte"] + 0.4 * centres["apoptotic"]
    return np.stack([centres[k] + rng.normal(0, 0.75, EMB_DIM) for k in kinds])


def write_dzi(img, out_dir: Path):
    w, h = img.size
    max_level = math.ceil(math.log2(max(w, h)))
    files = out_dir / "slide_files"
    for level in range(max_level, -1, -1):
        scale = 2 ** (max_level - level)
        lw, lh = max(1, math.ceil(w / scale)), max(1, math.ceil(h / scale))
        li = img if scale == 1 else img.resize((lw, lh), Image.LANCZOS)
        d = files / str(level)
        d.mkdir(parents=True, exist_ok=True)
        for tx in range(math.ceil(lw / TILE)):
            for ty in range(math.ceil(lh / TILE)):
                box = (tx * TILE, ty * TILE, min(lw, (tx + 1) * TILE), min(lh, (ty + 1) * TILE))
                li.crop(box).save(d / f"{tx}_{ty}.jpg", quality=82)
    (out_dir / "slide.dzi").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{TILE}" Overlap="0" Format="jpg">'
        f'<Size Width="{w}" Height="{h}"/></Image>\n')
    thumb = img.copy()
    thumb.thumbnail((480, 360))
    thumb.save(out_dir / "thumb.jpg", quality=80)


def mock_detections(truth: dict, seed: int):
    """A stand-in for MitoEd's per-cell output: mostly right, with misses and mimic false positives."""
    rng = np.random.default_rng(seed)
    preds = []
    for cid, kind in truth.items():
        if kind == "mitosis":
            p = rng.beta(8, 2) if rng.random() > 0.15 else rng.beta(2, 5)
        elif kind == "apoptotic":
            p = rng.beta(5, 3) if rng.random() < 0.35 else rng.beta(1, 6)
        elif kind == "lymphocyte":
            p = rng.beta(2, 6) if rng.random() < 0.1 else rng.beta(1, 25)
        else:
            p = rng.beta(1, 40)
        if p >= 0.05:
            preds.append({"cell_id": cid, "p": round(float(p), 3)})
    return {"model": "MitoEd", "version": "0.5.4 (mock output)", "feature": "Mitotic figure", "threshold": 0.5,
            "note": "Synthetic detections for the GUI mock, derived from the hidden truth with noise.",
            "predictions": preds}


def write_detections(out_dir: Path, seed: int):
    truth = json.loads((out_dir / "truth.json").read_text())
    det = mock_detections(truth, seed)
    (out_dir / "mitoed.json").write_text(json.dumps(det, separators=(",", ":")))
    return sum(1 for x in det["predictions"] if x["p"] >= det["threshold"])


def make(scan_id, p, root: Path):
    out_dir = root / "slides" / scan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    img, cells, regions, area = paint(scan_id, p)
    rng = np.random.default_rng(p["seed"] + 1)
    emb = embeddings([c[0] for c in cells], rng)
    frame = f"scan:{scan_id}"
    table = {
        "scan_id": scan_id, "frame": frame, "mpp": MPP, "width_px": img.size[0], "height_px": img.size[1],
        "tissue_area_mm2": round(area, 3),
        "source": {"kind": "model", "name": "synthetic-segmenter", "version": "mock-1"},
        "embedding": {"name": "synthetic", "dim": EMB_DIM, "note": "stands in for H-optimus-0 (Apache-2.0)"},
        "cells": [{"cell_id": f"{scan_id}-C{i:05d}", "x_um": round(x * MPP, 2), "y_um": round(y * MPP, 2),
                   "r_um": round(r * MPP, 2), "emb": [round(float(v), 2) for v in emb[i]]}
                  for i, (_, x, y, r) in enumerate(cells)],
        "regions": regions,
    }
    (out_dir / "cells.json").write_text(json.dumps(table, separators=(",", ":")))
    # Hidden truth, for tests and the mock "MitoEd" output only. The GUI never shows it.
    (out_dir / "truth.json").write_text(json.dumps({c["cell_id"]: k for c, (k, *_) in zip(table["cells"], cells)}))
    write_dzi(img, out_dir)
    n_called = write_detections(out_dir, p["seed"] + 2)
    n_mit = sum(1 for c in cells if c[0] == "mitosis")
    return {"scan_id": scan_id, "n_cells": len(cells), "n_mitosis": n_mit, "n_mitoed_called": n_called,
            "tissue_area_mm2": round(area, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "web" / "mock"))
    ap.add_argument("--only", nargs="*", help="scan ids to render")
    ap.add_argument("--detections-only", action="store_true", help="redo mitoed.json from existing truth.json")
    a = ap.parse_args()
    root = Path(a.out)
    if a.detections_only:
        sm = json.loads((root / "slides" / "summary.json").read_text())
        for sid, p in RENDERS.items():
            if sid in sm and (not a.only or sid in a.only):
                sm[sid]["n_mitoed_called"] = write_detections(root / "slides" / sid, p["seed"] + 2)
                print(sid, sm[sid]["n_mitoed_called"])
        (root / "slides" / "summary.json").write_text(json.dumps(sm, indent=1))
        return
    summary = {}
    for sid, p in RENDERS.items():
        if a.only and sid not in a.only:
            continue
        s = make(sid, p, root)
        summary[sid] = s
        print(json.dumps(s))
    (root / "slides" / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
