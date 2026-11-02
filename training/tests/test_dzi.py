import math

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from entip_train.slides import ArrayReader, DziReader, default_reader_factory  # noqa: E402
from entip_train.ontology import Scan  # noqa: E402


def write_dzi(rgb, root, name, tile=256, overlap=1):
    """Minimal Deep Zoom writer (PNG tiles, so pixels are exact)."""
    H, W = rgb.shape[:2]
    max_level = int(math.ceil(math.log2(max(W, H))))
    files = root / f"{name}_files"
    img = Image.fromarray(rgb)
    for level in range(max_level, -1, -1):
        d = 2 ** (max_level - level)
        lw, lh = max(1, math.ceil(W / d)), max(1, math.ceil(H / d))
        li = img if d == 1 else img.resize((lw, lh), Image.BOX)
        (files / str(level)).mkdir(parents=True, exist_ok=True)
        for col in range(math.ceil(lw / tile)):
            for row in range(math.ceil(lh / tile)):
                x0, y0 = max(0, col * tile - overlap), max(0, row * tile - overlap)
                x1, y1 = min(lw, (col + 1) * tile + overlap), min(lh, (row + 1) * tile + overlap)
                li.crop((x0, y0, x1, y1)).save(files / str(level) / f"{col}_{row}.png")
    p = root / f"{name}.dzi"
    p.write_text(f'<?xml version="1.0"?><Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="{tile}" '
                 f'Overlap="{overlap}" Format="png"><Size Width="{W}" Height="{H}"/></Image>')
    return p


def test_dzi_matches_array(tmp_path, slide):
    rgb = slide.reader.rgb
    d = write_dzi(rgb, tmp_path, "s")
    r, a = DziReader(str(d), slide.scan.mpp), ArrayReader(rgb, slide.scan.mpp)
    for x, y in [(10.0, 10.0), (128.3, 255.9), (700.0, 400.0), (767.0, 767.0)]:
        np.testing.assert_array_equal(r.read_um(x, y, 32, 64), a.read_um(x, y, 32, 64))   # full resolution
    big = r.read_um(384, 384, 512, 64)                                                    # read from a lower level
    assert big.shape == (64, 64, 3)
    assert np.abs(big.astype(int) - a.read_um(384, 384, 512, 64).astype(int)).mean() < 20


def test_factory_finds_slides_dir(tmp_path, slide):
    (tmp_path / "syn-a").mkdir()
    write_dzi(slide.reader.rgb[:300, :300], tmp_path / "syn-a", "slide")
    f = default_reader_factory(str(tmp_path))
    assert isinstance(f(Scan("syn-a", "synthetic:0", 0.5, 150, 150)), DziReader)
    with pytest.raises(KeyError):
        f(Scan("other", "synthetic:0", 0.5, 150, 150))
