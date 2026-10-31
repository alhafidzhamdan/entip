"""Write the pseudonymised research copy of a WSI and check it for identifiers.

What the research copy keeps:
- the image pyramid (series 0 in tifffile's reading of the file), tile by tile;
- pixel size (µm/px), objective magnification and the ICC colour profile;
- the linkage code and IDs from the key store.

What it drops: label and macro images, thumbnails, every vendor description field
not on the allow-list (file name, barcode, user, date, scanner ID...), private
tags, XMP, EXIF and dates. The file name becomes ``<link_code>.tiff``.

Formats: anything tifffile reads as a pyramidal series (Aperio SVS, generic
pyramidal TIFF, NDPI, OME-TIFF). JPEG-compressed vendor files need the optional
``imagecodecs`` package. Non-TIFF formats (MRXS, iSyntax) are not handled yet.

Limit: identifiers burned into the tissue pixels (pen marks, stickers on the
glass inside the scan area) are not detected. The label and macro images,
where printed identifiers normally sit, are removed whole.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import tifffile

from . import __version__
from . import chi as chi_mod

SOFTWARE = f"entip-fork {__version__}"
TILE = 256
_MAX_UNTILED_BYTES = 512 * 1024 * 1024  # stripped levels are read whole; refuse beyond this


@dataclass
class SourceInfo:
    width: int
    height: int
    levels: list[tuple[int, int]]  # (height, width) per level
    mpp: float | None
    magnification: float | None
    vendor: str
    dropped: list[str] = field(default_factory=list)


def _svs_fields(desc: str) -> dict[str, str]:
    out = {}
    for part in desc.split("|")[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _mpp_from_page(page: tifffile.TiffPage) -> float | None:
    tags = page.tags
    if "XResolution" not in tags or "ResolutionUnit" not in tags:
        return None
    num, den = tags["XResolution"].value
    if not num:
        return None
    per_unit = num / den  # pixels per unit
    unit = tags["ResolutionUnit"].value
    um = {2: 25400.0, 3: 10000.0}.get(int(unit))  # inch, centimetre
    return um / per_unit if um else None


def inspect(tif: tifffile.TiffFile) -> SourceInfo:
    series = tif.series[0]
    levels = [lev.shape[:2] for lev in series.levels]
    biggest = max(s.shape[0] * s.shape[1] for s in tif.series if len(s.shape) >= 2)
    if levels[0][0] * levels[0][1] != biggest:
        raise ValueError("series 0 is not the largest image; refusing to guess which image is the slide")
    page0 = series.levels[0].keyframe
    mpp = mag = None
    vendor = "tiff"
    if tif.is_svs:
        vendor = "svs"
        f = _svs_fields(page0.description)
        mpp = float(f["MPP"]) if "MPP" in f else None
        mag = float(f["AppMag"]) if "AppMag" in f else None
    elif tif.is_ndpi:
        vendor = "ndpi"
        mag = float(page0.tags["65421"].value) if "65421" in page0.tags else None
    if mpp is None:
        mpp = _mpp_from_page(page0)
    dropped = [f"{s.name or 'series'} {s.shape}" for s in tif.series[1:]]
    return SourceInfo(levels[0][1], levels[0][0], levels, mpp, mag, vendor, dropped)


def _tiles(page: tifffile.TiffPage) -> Iterator[np.ndarray]:
    """Yield TILE x TILE tiles of one level in row-major order, memory-bounded when the source is tiled."""
    samples = page.samplesperpixel
    if page.is_tiled and page.tilelength == TILE and page.tilewidth == TILE and page.imagedepth == 1:
        for seg, _idx, shape in page.segments():
            if seg is None:
                yield np.zeros((TILE, TILE, samples), page.dtype)
            else:
                yield seg.reshape(TILE, TILE, samples)
        return
    if page.is_tiled and page.imagedepth == 1:
        # retile from source tiles one tile-row band at a time
        th, tw = page.tilelength, page.tilewidth
        h, w = page.shaped[2], page.shaped[3]
        ntx = -(-w // tw)
        band: dict[int, np.ndarray] = {}
        segs = page.segments()
        for y0 in range(0, h, TILE):
            need_rows = range(y0 // th, min(-(-(y0 + TILE) // th), -(-h // th)))
            for r in need_rows:
                if r in band:
                    continue
                row = np.zeros((th, ntx * tw, samples), page.dtype)
                for c in range(ntx):
                    seg, _idx, _shape = next(segs)
                    if seg is not None:
                        row[:, c * tw : (c + 1) * tw] = seg.reshape(th, tw, samples)
                band[r] = row
            for r in list(band):
                if r < y0 // th:
                    del band[r]
            stack = np.concatenate([band[r] for r in need_rows], 0)
            off = y0 - need_rows[0] * th
            strip = stack[off : off + TILE]
            if strip.shape[0] < TILE:
                strip = np.pad(strip, ((0, TILE - strip.shape[0]), (0, 0), (0, 0)))
            for x0 in range(0, w, TILE):
                t = strip[:, x0 : x0 + TILE]
                if t.shape[1] < TILE:
                    t = np.pad(t, ((0, 0), (0, TILE - t.shape[1]), (0, 0)))
                yield t
        return
    nbytes = page.shaped[2] * page.shaped[3] * samples * page.dtype.itemsize
    if nbytes > _MAX_UNTILED_BYTES:
        raise ValueError("untiled level too large to read whole; convert the slide to tiled first")
    arr = page.asarray()
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w = arr.shape[:2]
    arr = np.pad(arr, ((0, -h % TILE), (0, -w % TILE), (0, 0)))
    for y0 in range(0, arr.shape[0], TILE):
        for x0 in range(0, arr.shape[1], TILE):
            yield arr[y0 : y0 + TILE, x0 : x0 + TILE]


def write_research_copy(
    src: str | Path,
    dst: str | Path,
    *,
    research_meta: dict,
    compression: str = "zlib",
) -> SourceInfo:
    """Write the pyramid of ``src`` to ``dst`` with rebuilt, allow-listed metadata only."""
    src, dst = Path(src), Path(dst)
    with tifffile.TiffFile(src) as tif:
        info = inspect(tif)
        series = tif.series[0]
        page0 = series.levels[0].keyframe
        photometric = "rgb" if page0.samplesperpixel >= 3 else "minisblack"
        icc = page0.tags["InterColorProfile"].value if "InterColorProfile" in page0.tags else None
        desc = {
            "entip": "research copy",
            "software": SOFTWARE,
            "mpp": info.mpp,
            "magnification": info.magnification,
            "levels": [list(lv) for lv in info.levels],
            **research_meta,
        }
        resolution = None
        if info.mpp:
            resolution = (1e4 / info.mpp, 1e4 / info.mpp)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with tifffile.TiffWriter(tmp, bigtiff=True) as out:
            for i, level in enumerate(series.levels):
                page = level.keyframe
                if len(level.pages) != 1:
                    raise ValueError("multi-page levels (z-stacks, multichannel) are not supported yet")
                h, w = level.shape[:2]
                scale = w / info.width
                kw = dict(
                    shape=(h, w, page.samplesperpixel) if page.samplesperpixel > 1 else (h, w),
                    dtype=page.dtype,
                    tile=(TILE, TILE),
                    photometric=photometric,
                    compression=compression,
                    metadata=None,
                    software=SOFTWARE if i == 0 else False,
                    resolution=(resolution[0] * scale, resolution[1] * scale) if resolution else None,
                    resolutionunit="CENTIMETER" if resolution else None,
                    iccprofile=icc if i == 0 else None,
                )
                tiles = _tiles(page)
                if page.samplesperpixel == 1:
                    tiles = (t[..., 0] for t in tiles)
                if i == 0:
                    out.write(tiles, description=json.dumps(desc, sort_keys=True),
                              subifds=len(series.levels) - 1, **kw)
                else:
                    out.write(tiles, subfiletype=1, **kw)
        tmp.replace(dst)
    return info


# -- identifier scan -------------------------------------------------------

def _encodings(term: str) -> list[bytes]:
    out = []
    for t in {term, term.upper(), term.lower()}:
        out += [t.encode("utf-8"), t.encode("utf-16-le"), t.encode("utf-16-be")]
    return out


def identifier_variants(identifiers: Iterable[str]) -> list[str]:
    """Each identifier plus common spellings (spaces, dashes, CHI split after the date)."""
    out: set[str] = set()
    for ident in identifiers:
        ident = str(ident).strip()
        if len(ident) < 4:
            continue  # too short to search for without false hits (e.g. block "A1")
        out.add(ident)
        squashed = re.sub(r"[\s_\-^/.]", "", ident)
        if len(squashed) >= 4:
            out.add(squashed)
        out.add(re.sub(r"[\s_\-^/.]+", " ", ident))
        out.add(re.sub(r"[\s_\-^/.]+", "-", ident))
        out.add(re.sub(r"[\s_\-^/.]+", "_", ident))
        n = chi_mod.normalise(ident)
        if re.fullmatch(r"\d{10}", n):
            out.update({n, f"{n[:6]} {n[6:]}", f"{n[:6]}-{n[6:]}"})
    return sorted(out)


def scan_for_identifiers(path: str | Path, identifiers: Iterable[str], chunk: int = 8 << 20) -> list[str]:
    """Return a list of findings; empty means clean.

    1. Byte search of the whole file (UTF-8 and UTF-16, case variants) for every
       identifier given.
    2. Every string tag in every page is checked for CHI-shaped numbers that pass
       the check-digit test, and for any page beyond the expected pyramid.
    """
    path = Path(path)
    findings: list[str] = []
    needles = {v: _encodings(v) for v in identifier_variants(identifiers)}
    longest = max((len(b) for bs in needles.values() for b in bs), default=1)
    hit: set[str] = set()
    with open(path, "rb") as fh:
        tail = b""
        while True:
            block = fh.read(chunk)
            if not block:
                break
            buf = tail + block
            for term, encs in needles.items():
                if term not in hit and any(e in buf for e in encs):
                    hit.add(term)
            tail = buf[-(longest - 1):] if longest > 1 else b""
    findings += [f"identifier found in file bytes: {t!r}" for t in sorted(hit)]

    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1:
            findings.append(f"expected one image series, found {len(tif.series)}")
        for page in tif.pages:
            pages = [page]
            for i, off in enumerate(page.subifds or ()):
                tif.filehandle.seek(off)
                pages.append(tifffile.TiffPage(tif, (page.index, i)))
            for p in pages:
                for tag in p.tags:
                    v = tag.value
                    if isinstance(v, bytes):
                        v = v.decode("latin-1")
                    if isinstance(v, str):
                        for c in chi_mod.find_candidates(v):
                            findings.append(f"CHI-shaped number in tag {tag.name}: {c[:2]}********")
                    if tag.name in {"XMP", "ImageJMetadata", "DateTime"} or tag.code in (34665, 34853):
                        findings.append(f"unexpected tag {tag.name}")
    return findings
