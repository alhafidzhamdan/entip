"""Step 1 pre-training: slides -> frozen-encoder tile bags, on synthetic data only."""
import json

import numpy as np

import pytest

from entip_glioma.cli import main
from entip_train.slides import ArrayReader
from entip_glioma.wsi.bags import Bag


class _Reader(ArrayReader):
    @property
    def width_um(self):
        return self.rgb.shape[1] * self.mpp

    @property
    def height_um(self):
        return self.rgb.shape[0] * self.mpp


def test_extract_writes_bags_and_index(tmp_path, monkeypatch, slide):
    import entip_train.slides as slides
    rgb = slide.reader.rgb[:1024, :1024]
    monkeypatch.setattr(slides, "open_slide", lambda path, mpp=None: _Reader(rgb, 0.5))
    out = tmp_path / "bags"
    assert main(["embed", "extract", "a.svs", "b.svs", "--out", str(out), "--encoder", "handcrafted",
                 "--tile-um", "64"]) == 0
    index = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    assert [r["scan_id"] for r in index] == ["a", "b"]
    assert {r["tile_um"] for r in index} == {64.0} and index[0]["tiles"] > 0
    bag = Bag.load(out / "a.npz")
    assert bag.encoder == index[0]["encoder"] and len(bag) == index[0]["tiles"]


def test_extract_refuses_one_scan_id_for_many_slides(tmp_path):
    with pytest.raises(SystemExit):
        main(["embed", "extract", "a.svs", "b.svs", "--scan-id", "x", "--out", str(tmp_path),
              "--encoder", "handcrafted"])


@pytest.mark.parametrize("cmd", [["embed"], ["triage"]])     # triage keeps the old command as an alias
def test_build_bags_dry_run_counts_what_is_left(tmp_path, capsys, cmd):
    from entip_glioma.triage.manifest import FIELDS, write_manifest
    rows = [dict.fromkeys(FIELDS, "") | {"SeriesInstanceUID": f"1.2.{i}", "label": "glioma", "group": "g",
                                         "PatientID": f"p{i}", "series_size_MB": "500"} for i in range(4)]
    write_manifest(rows, tmp_path / "m.csv")
    (tmp_path / "bags").mkdir()
    (tmp_path / "bags" / "1.2.0.npz").write_bytes(b"")
    assert main(cmd + ["build-bags", "--manifest", str(tmp_path / "m.csv"), "--out", str(tmp_path / "bags"),
                       "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"slides": 4, "already_done": 1, "to_do": 3, "download_GB": 1.5}


def test_parallel_reading_gives_the_same_bag(slide):
    """Tiles read by worker processes (each opening its own copy of the slide) give exactly the serial bag."""
    import functools

    from entip_glioma.wsi.bags import extract_bag
    from entip_train.encoders import HandcraftedEncoder
    rgb = slide.reader.rgb[:1024, :1024].copy()
    r, enc = _Reader(rgb, 0.5), HandcraftedEncoder()
    t1, t4 = {}, {}
    serial = extract_bag(r, "a", r.width_um, r.height_um, enc, tile_um=64, batch=8, timings=t1)
    par = extract_bag(r, "a", r.width_um, r.height_um, enc, tile_um=64, batch=8, workers=2,
                      opener=functools.partial(ArrayReader, rgb, 0.5), timings=t4)
    assert len(serial) > 16 and np.array_equal(serial.coords_um, par.coords_um)
    assert np.allclose(serial.features, par.features) and np.allclose(serial.sharpness, par.sharpness)
    assert t1["workers"] == 1 and t4["workers"] == 2
    assert {"tissue", "read", "embed"} <= set(t4) and all(t4[k] >= 0 for k in ("tissue", "read", "embed"))


def test_index_records_timings(tmp_path, monkeypatch, slide):
    import entip_train.slides as slides
    rgb = slide.reader.rgb[:512, :512]
    monkeypatch.setattr(slides, "open_slide", lambda path, mpp=None: _Reader(rgb, 0.5))
    assert main(["embed", "extract", "a.svs", "--out", str(tmp_path), "--encoder", "handcrafted", "--tile-um", "64",
                 "--workers", "1"]) == 0
    rec = json.loads((tmp_path / "index.jsonl").read_text().splitlines()[0])
    assert rec["workers"] == 1 and set(rec["seconds"]) == {"open", "tissue", "read", "embed", "total"}
    assert rec["tiles_per_second"] > 0
