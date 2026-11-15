"""Distillation on synthetic slides: tile cache -> tiny student -> evaluation -> student as an encoder."""
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from entip_glioma.cli import main
from entip_glioma.distill.tiles import read_index, split_patients
from entip_glioma.wsi.bags import Bag, extract_bag
from entip_train.encoders import HandcraftedEncoder, StudentEncoder, get_encoder
from entip_train.slides import ArrayReader


class _Reader(ArrayReader):
    @property
    def width_um(self):
        return self.rgb.shape[1] * self.mpp

    @property
    def height_um(self):
        return self.rgb.shape[0] * self.mpp


@pytest.fixture
def run(tmp_path, monkeypatch, slide):
    """Six small 'slides' (crops of the synthetic slide) from six patients in two classes, with teacher bags."""
    import entip_train.slides as slides
    readers, rows = {}, []
    enc = HandcraftedEncoder()
    for i in range(6):
        y0, x0 = (i // 3) * 512, (i % 3) * 512
        r = _Reader(slide.reader.rgb[y0:y0 + 512, x0:x0 + 512], 0.5)
        sid = f"1.2.{i}"
        readers[sid] = r
        (tmp_path / "slides" / sid).mkdir(parents=True)
        extract_bag(r, sid, r.width_um, r.height_um, enc, tile_um=32.0).save(tmp_path / "bags" / f"{sid}.npz")
        rows.append({"SeriesInstanceUID": sid, "PatientID": f"p{i}", "label": "glioma" if i % 2 else "normal_brain",
                     "group": "g"})
    with open(tmp_path / "trial.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(slides, "open_slide", lambda path, mpp=None: readers[path.rsplit("/", 1)[-1]])
    return tmp_path


def test_split_patients_keeps_each_class_on_both_sides():
    entries = [{"patient": f"p{i}", "label": "a" if i < 5 else "b"} for i in range(10)]
    train, held = split_patients(entries, holdout=0.2, seed=1)
    assert not train & held and len(train | held) == 10
    assert {"a", "b"} == {("a" if int(p[1:]) < 5 else "b") for p in held}


def test_distill_end_to_end(run, capsys):
    t = run
    assert main(["distill", "tiles", "--bags", str(t / "bags"), "--slides", str(t / "slides"), "--manifest",
                 str(t / "trial.csv"), "--out", str(t / "tiles"), "--input-px", "32", "--max-tiles", "40"]) == 0
    idx = read_index(t / "tiles")
    assert len(idx) == 6 and all(e["teacher"] == "handcrafted@1" and e["patient"] for e in idx)
    tiles = np.load(t / "tiles" / f"{idx[0]['scan_id']}.tiles.npy")
    teacher = np.load(t / "tiles" / f"{idx[0]['scan_id']}.teacher.npy")
    assert tiles.shape[1:] == (32, 32, 3) and tiles.dtype == np.uint8 and len(teacher) == len(tiles)
    bag = Bag.load(t / "bags" / f"{idx[0]['scan_id']}.npz")
    assert teacher.shape[1] == bag.features.shape[1]

    # rerunning adds nothing
    assert main(["distill", "tiles", "--bags", str(t / "bags"), "--slides", str(t / "slides"), "--manifest",
                 str(t / "trial.csv"), "--out", str(t / "tiles"), "--input-px", "32"]) == 0
    assert len(read_index(t / "tiles")) == 6

    assert main(["distill", "train", "--tiles", str(t / "tiles"), "--out", str(t / "student"), "--arch", "tiny",
                 "--init", "scratch", "--epochs", "2", "--batch-size", "16", "--workers", "0", "--device", "cpu"]) == 0
    rep = json.loads((t / "student" / "report.json").read_text())
    assert len(rep["history"]) == 2 and np.isfinite(rep["best_heldout_cosine"])
    assert not set(rep["train_patients"]) & set(rep["heldout_patients"])

    capsys.readouterr()
    assert main(["distill", "evaluate", "--tiles", str(t / "tiles"), "--student", str(t / "student" / "student.pt"),
                 "--out", str(t / "eval.json")]) == 0
    ev = json.loads((t / "eval.json").read_text())
    assert ev["heldout_slides"] == len(rep["heldout_patients"]) and np.isfinite(ev["tile_cosine_mean"])

    enc = get_encoder(f"student:{t / 'student' / 'student.pt'}", device="cpu")
    assert isinstance(enc, StudentEncoder) and enc.version.startswith("student-tiny-from-handcrafted-")
    out = enc.embed(np.zeros((3, 32, 32, 3), np.uint8))
    assert out.shape == (3, bag.features.shape[1]) and out.dtype == np.float32

    # the student embeds a slide like any other encoder, and its bags say so
    assert main(["embed", "extract", str(t / "slides" / "1.2.0"), "--out", str(t / "student_bags"),
                 "--encoder", f"student:{t / 'student' / 'student.pt'}", "--tile-um", "32"]) == 0
    sb = Bag.load(t / "student_bags" / "1.2.0.npz")
    assert sb.encoder == enc.version and sb.features.shape[1] == bag.features.shape[1]
