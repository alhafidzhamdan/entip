"""Slide readers and a synthetic H&E-like slide for tests and demos (no patient data).

A reader returns an RGB crop centred on a point given in micrometres, resampled to ``out_px``
pixels, so every encoder sees the same physical field whatever the scanner's pixel size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np

from .ontology import Cell, Region, Scan


class Reader(Protocol):
    mpp: float

    def read_um(self, x_um: float, y_um: float, size_um: float, out_px: int) -> np.ndarray: ...


def _resize_nn(img: np.ndarray, out_px: int) -> np.ndarray:
    """Area-ish resize without extra dependencies (block mean when shrinking, nearest when growing)."""
    h, w = img.shape[:2]
    if h == out_px and w == out_px:
        return img
    if h >= out_px and h % out_px == 0 and w % out_px == 0:
        f = h // out_px
        return img.reshape(out_px, f, out_px, f, 3).mean(axis=(1, 3)).astype(np.uint8)
    yi = (np.arange(out_px) * h / out_px).astype(int)
    xi = (np.arange(out_px) * w / out_px).astype(int)
    return img[yi][:, xi]


class ArrayReader:
    """An in-memory RGB image at a known pixel size; pads with white outside the image."""

    def __init__(self, rgb: np.ndarray, mpp: float):
        self.rgb, self.mpp = rgb, float(mpp)

    def read_um(self, x_um: float, y_um: float, size_um: float, out_px: int) -> np.ndarray:
        n = max(1, int(round(size_um / self.mpp)))
        x0, y0 = int(round(x_um / self.mpp - n / 2)), int(round(y_um / self.mpp - n / 2))
        out = np.full((n, n, 3), 255, np.uint8)
        H, W = self.rgb.shape[:2]
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + n), min(H, y0 + n)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.rgb[sy0:sy1, sx0:sx1]
        return _resize_nn(out, out_px)


class OpenSlideReader:
    """Whole-slide images via OpenSlide (openslide-python, LGPL-2.1: dynamically linked library;
    check with your information governance team before NHS deployment). Optional."""

    def __init__(self, path: str, mpp: float | None = None):
        import openslide  # optional dependency
        self.s = openslide.OpenSlide(path)
        self.mpp = float(mpp or self.s.properties.get("openslide.mpp-x"))

    def read_um(self, x_um: float, y_um: float, size_um: float, out_px: int) -> np.ndarray:
        n = max(1, int(round(size_um / self.mpp)))
        level = self.s.get_best_level_for_downsample(max(1.0, n / out_px))
        ds = self.s.level_downsamples[level]
        nl = max(1, int(round(n / ds)))
        x0, y0 = int(round(x_um / self.mpp - n / 2)), int(round(y_um / self.mpp - n / 2))
        img = np.asarray(self.s.read_region((x0, y0), level, (nl, nl)).convert("RGB"))
        return _resize_nn(img, out_px)


class DziReader:
    """Deep Zoom (DZI) pyramids, e.g. the GUI's tiles, from a local path or an http(s) URL.
    Needs Pillow (MIT-CMU licence, ``pip install entip-train[dzi]``). The pixel size comes from the scan
    record (DZI files do not carry one)."""

    def __init__(self, dzi: str, mpp: float, cache_tiles: int = 256):
        import xml.etree.ElementTree as ET
        from collections import OrderedDict
        self.dzi, self.mpp = dzi, float(mpp)
        root = ET.fromstring(self._fetch(dzi))
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        size = root.find(f"{ns}Size")
        self.tile, self.overlap = int(root.get("TileSize")), int(root.get("Overlap"))
        self.fmt = root.get("Format")
        self.W, self.H = int(size.get("Width")), int(size.get("Height"))
        self.max_level = int(np.ceil(np.log2(max(self.W, self.H))))
        base = dzi[: dzi.rfind(".")]
        self.tiles_base = base + "_files"
        self._tiles: "OrderedDict" = OrderedDict()
        self._cache_n = cache_tiles

    @staticmethod
    def _fetch(path: str) -> bytes:
        if path.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(path, timeout=30) as r:
                return r.read()
        with open(path, "rb") as f:
            return f.read()

    def _tile(self, level: int, col: int, row: int) -> np.ndarray:
        import io
        from PIL import Image
        k = (level, col, row)
        if k not in self._tiles:
            data = self._fetch(f"{self.tiles_base}/{level}/{col}_{row}.{self.fmt}")
            self._tiles[k] = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            if len(self._tiles) > self._cache_n:
                self._tiles.popitem(last=False)
        else:
            self._tiles.move_to_end(k)
        return self._tiles[k]

    def read_um(self, x_um: float, y_um: float, size_um: float, out_px: int) -> np.ndarray:
        n = max(1, int(round(size_um / self.mpp)))                 # full-resolution pixels
        down = 1
        while down * 2 <= max(1, n // out_px) and down * 2 <= 2 ** self.max_level:
            down *= 2
        level = self.max_level - int(np.log2(down))
        lw, lh = int(np.ceil(self.W / down)), int(np.ceil(self.H / down))
        m = max(1, int(round(n / down)))
        x0 = int(round(x_um / self.mpp / down - m / 2))
        y0 = int(round(y_um / self.mpp / down - m / 2))
        out = np.full((m, m, 3), 255, np.uint8)
        T, ov = self.tile, self.overlap
        for row in range(max(0, y0 // T), min((lh - 1) // T, (y0 + m - 1) // T) + 1):
            for col in range(max(0, x0 // T), min((lw - 1) // T, (x0 + m - 1) // T) + 1):
                t = self._tile(level, col, row)
                ox = ov if col > 0 else 0                           # tiles carry overlap on inner edges
                oy = ov if row > 0 else 0
                tx0, ty0 = col * T, row * T
                tw, th = min(T, lw - tx0), min(T, lh - ty0)
                core = t[oy:oy + th, ox:ox + tw]
                sx0, sy0 = max(x0, tx0), max(y0, ty0)
                sx1, sy1 = min(x0 + m, tx0 + tw), min(y0 + m, ty0 + th)
                if sx1 > sx0 and sy1 > sy0:
                    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = core[sy0 - ty0:sy1 - ty0, sx0 - tx0:sx1 - tx0]
        return _resize_nn(out, out_px)


def default_reader_factory(slides_dir: str | None = None):
    """Pick a reader for a scan record: ``<slides_dir>/<scan_id>/*.dzi`` or ``<slides_dir>/<scan_id>.dzi`` if
    present, else a ``.dzi`` path or URL in the scan's tile_source / uri, else OpenSlide on a file path.
    ``synthetic:`` scans have no pixels (KeyError)."""
    import glob
    import os

    def factory(scan: Scan) -> Reader:
        if slides_dir:
            hits = sorted(glob.glob(os.path.join(slides_dir, glob.escape(scan.scan_id), "*.dzi"))) + \
                [p for p in [os.path.join(slides_dir, scan.scan_id + ".dzi")] if os.path.exists(p)]
            if hits:
                return DziReader(hits[0], scan.mpp)
        path = scan.path or ""
        if path.endswith(".dzi"):
            return DziReader(path, scan.mpp)
        if not path or path.startswith("synthetic:"):
            raise KeyError(f"no pixels for scan {scan.scan_id} (path {path!r}); give --slides-dir or a tile_source")
        return OpenSlideReader(path, scan.mpp)

    return factory


# ------------------------------------------------------------------ synthetic slide

@dataclass
class SyntheticSlide:
    scan: Scan
    reader: ArrayReader
    cells: List[Cell]
    cell_truth: dict          # cell_id -> 1 if the cell has the synthetic feature ("atypical": big, dark)
    regions: List[Region]
    region_truth: dict        # region_id -> 1 if the tile lies in a pale "necrosis-like" patch


def synthetic_slide(scan_id: str = "synthetic-001", seed: int = 0, size_px: int = 1536, mpp: float = 0.5,
                    n_cells: int = 600, frac_positive: float = 0.08, tile_um: float = 64.0) -> SyntheticSlide:
    """Pink stroma, purple nuclei; a share of nuclei are larger and darker (the cell feature), and two
    pale patches with sparse nuclei form a tile feature. Fully synthetic."""
    rng = np.random.default_rng(seed)
    H = W = size_px
    img = np.empty((H, W, 3), np.float32)
    img[:] = (236, 170, 200)
    img += rng.normal(0, 8, (H, W, 1))

    # pale patches
    patches = [(rng.uniform(0.2, 0.4) * W, rng.uniform(0.2, 0.4) * H, 0.12 * W),
               (rng.uniform(0.6, 0.8) * W, rng.uniform(0.6, 0.8) * H, 0.10 * W)]
    yy, xx = np.mgrid[0:H, 0:W]
    pale = np.zeros((H, W), bool)
    for cx, cy, r in patches:
        pale |= (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
    img[pale] = img[pale] * 0.5 + np.array([250, 225, 235]) * 0.5

    cells, truth = [], {}
    margin = 12
    k = 0
    while k < n_cells:
        x, y = rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)
        if pale[int(y), int(x)] and rng.random() < 0.8:      # sparse nuclei in pale patches
            continue
        pos = rng.random() < frac_positive
        r = rng.uniform(7, 10) if pos else rng.uniform(3.5, 6)
        colour = np.array((60, 20, 90)) if pos else np.array((120, 70, 160)) + rng.normal(0, 10, 3)
        ex = rng.uniform(0.7, 1.0)
        y0, y1, x0, x1 = int(y - r - 1), int(y + r + 2), int(x - r - 1), int(x + r + 2)
        sub_y, sub_x = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        m = ((sub_x - x) / r) ** 2 + ((sub_y - y) / (r * ex)) ** 2 < 1
        img[y0:y1, x0:x1][m] = colour
        cid = f"{scan_id}-c{k:05d}"
        cells.append(Cell(cid, scan_id, x * mpp, y * mpp, "synthetic"))
        truth[cid] = int(pos)
        k += 1

    rgb = np.clip(img, 0, 255).astype(np.uint8)
    scan = Scan(scan_id, f"synthetic:{seed}", mpp, W * mpp, H * mpp)
    regions, rtruth = [], {}
    n_t = int(W * mpp // tile_um)
    for i in range(n_t):
        for j in range(n_t):
            rid = f"{scan_id}-t{i:03d}-{j:03d}"
            reg = Region(rid, scan_id, j * tile_um, i * tile_um, tile_um, tile_um)
            cx, cy = reg.centre
            regions.append(reg)
            rtruth[rid] = int(pale[min(H - 1, int(cy / mpp)), min(W - 1, int(cx / mpp))])
    return SyntheticSlide(scan, ArrayReader(rgb, mpp), cells, truth, regions, rtruth)


def grid_regions(scan: Scan, tile_um: float = 112.0) -> List[Region]:
    """Tiles covering a scan. 112 µm is 224 px at 0.5 µm/px; check the encoder's model card for the resolution
    it was pretrained at and set ``tile_um`` to match."""
    out: List[Region] = []
    for i in range(int(scan.height_um // tile_um)):
        for j in range(int(scan.width_um // tile_um)):
            out.append(Region(f"{scan.scan_id}-t{i:03d}-{j:03d}", scan.scan_id, j * tile_um, i * tile_um,
                              tile_um, tile_um))
    return out


def centres(items) -> List[Tuple[float, float]]:
    return [(c.x_um, c.y_um) if isinstance(c, Cell) else c.centre for c in items]
