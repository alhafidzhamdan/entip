"""Synthetic SVS-like slides for tests and demos.

The file mimics the layout of an Aperio SVS: tiled baseline, a stripped
thumbnail, tiled pyramid levels, then a label image and a macro image. The
baseline description and the label carry fake identifiers (synthetic CHI,
accession, a fake user name), so the fork has something to remove.

No real patient data is used. Pixels are random H&E-like blobs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from . import chi as chi_mod


@dataclass(frozen=True)
class SyntheticCase:
    path: Path
    chi: str
    accession: str
    block: str
    section: str
    stain: str
    patient_name: str
    user: str


def _tissue(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (242, 238, 240)  # glass
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, ry, rx = h / 2, w / 2, h * 0.4, w * 0.42
    tissue = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
    img[tissue] = (226, 156, 196)  # eosin
    n = int(tissue.sum() / 900)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    r = 6
    disk = (np.mgrid[-r : r + 1, -r : r + 1] ** 2).sum(0) <= r * r
    for y, x in zip(ys, xs):
        if not tissue[y, x]:
            continue
        y0, y1, x0, x1 = max(0, y - r), min(h, y + r + 1), max(0, x - r), min(w, x + r + 1)
        m = disk[y0 - y + r : y1 - y + r, x0 - x + r : x1 - x + r]
        img[y0:y1, x0:x1][m] = (92, 58, 140)  # haematoxylin
    noise = rng.integers(-8, 9, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _downsample(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    return img[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3)).astype(np.uint8)


def make_slide(
    out: str | Path,
    *,
    seed: int = 0,
    width: int = 2048,
    height: int = 1536,
    mpp: float = 0.25,
    accession: str | None = None,
    block: str = "A1",
    section: str = "L1",
    stain: str = "H&E",
) -> SyntheticCase:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    chi = chi_mod.synthetic(rnd)
    accession = accession or f"SYN26-{rnd.randint(10000, 99999)}"
    patient = f"SYNTHETIC^PATIENT{rnd.randint(100, 999)}"
    user = f"Dr Synthetic {rnd.choice(['Ash', 'Birch', 'Cedar', 'Rowan'])}"

    base = _tissue(height, width, rng)
    levels = [base, _downsample(base, 4), _downsample(base, 16)]
    head = "Aperio Image Library v12.0.15"
    fields = (
        f"AppMag = 40|StripeWidth = 2040|ScanScope ID = SYN0001|Filename = {accession} {block}"
        f"|Date = 10/30/26|Time = 09:15:00|User = {user}|Patient = {patient}|Barcode = {accession}^{chi}"
        f"|MPP = {mpp:.4f}|Left = 25.0|Top = 18.0"
    )
    label = np.full((300, 400, 3), 255, np.uint8)
    label[40:260:6, 30:370] = 0  # stand-in barcode stripes
    macro = _downsample(np.pad(base, ((0, 0), (width // 2, 0), (0, 0)), constant_values=250), 8)

    with tifffile.TiffWriter(out, bigtiff=False) as tif:
        common = dict(photometric="rgb", compression="zlib", metadata=None, software=False)
        tif.write(
            levels[0], tile=(256, 256),
            description=f"{head}\r\n{width}x{height} [0,0 {width}x{height}] (256x256) zlib/RGB|{fields}",
            **common,
        )
        thumb = _downsample(base, 8)
        tif.write(
            thumb, rowsperstrip=16,
            description=f"{head}\r\n{width}x{height} -> {thumb.shape[1]}x{thumb.shape[0]} - |{fields}",
            **common,
        )
        for lev in levels[1:]:
            tif.write(
                lev, tile=(256, 256),
                description=f"{head}\r\n{width}x{height} [0,0 {width}x{height}] -> {lev.shape[1]}x{lev.shape[0]} (256x256) zlib/RGB|{fields}",
                **common,
            )
        tif.write(
            label, rowsperstrip=16, subfiletype=1,
            description=f"{head}\r\nlabel {label.shape[1]}x{label.shape[0]}|Barcode = {accession}^{chi}",
            **common,
        )
        tif.write(
            macro, rowsperstrip=16, subfiletype=9,
            description=f"{head}\r\nmacro {macro.shape[1]}x{macro.shape[0]}|{fields}",
            **common,
        )
    return SyntheticCase(out, chi, accession, block, section, stain, patient, user)
