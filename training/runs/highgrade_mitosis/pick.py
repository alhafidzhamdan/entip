"""Step 3c: pick MitoEd mitosis candidates for a pathologist to label as mitosis, not a mitosis, or not assessable.

    python pick.py --manifest trial.csv --cells mitoed/ --slides slides/ --out mit-r1 --round r1

The unit is one cell, so the page shows candidates, not tiles. ``--cells`` holds one ``mitoed predict`` CSV per
slide, ``<SeriesInstanceUID>.csv`` (``x_um, y_um, p_mitosis, call``), in the slide's µm frame (level 0 pixels x
pixel size, as the bags).

Candidates are sampled **at random within bands** of MitoEd's p, per slide, so the labels can recalibrate p on
glioma and estimate how many true mitoses each band holds (``export.py`` weights each label by its band's size):

* ``high`` (p >= 0.7), ``mid`` (0.3 to 0.7), ``low`` (counted, p < 0.3): candidates the step 3 count uses
  (MitoEd calls them mitosis or uncertain);
* ``mimic``: candidates MitoEd calls mimics, which the count leaves out. Labelling a few shows what the count
  misses.

Slides: ``--per-category`` patients per category. Mitoses separate grade 2 from grade 3, so the default leans on
``grade2`` and ``astro3``.

Writes ``labels.html``, ``cells/<code>.jpg`` (64 µm around the candidate) and ``cells/<code>-c.jpg`` (256 µm), and
``key.csv``.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import (find_slide, pick_slides, quotas, read_manifest, read_patients, read_seen,  # noqa: E402
                    write_round, xy, zoom)
from labelpage import page  # noqa: E402

CHOICES = [
    ("y", "Mitosis", "a mitotic figure in the circle, prophase to telophase"),
    ("n", "Not a mitosis", "apoptotic or pyknotic nucleus, lymphocyte, karyorrhectic debris, crush or anything else"),
    ("x", "Not assessable", "out of focus, folded, or too little of the cell to judge"),
]
INSTRUCTIONS = "Label the cell in the <b>circle</b>. Left: 64 µm around it; right: 256 µm."
BANDS = ("high", "mid", "low", "mimic")


def band(c: dict, min_p: float = 0.2) -> str:
    counted = c["call"] != "mimic" if c.get("call") else c["p"] >= min_p
    if not counted:
        return "mimic"
    return "high" if c["p"] >= 0.7 else "mid" if c["p"] >= 0.3 else "low"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cells", required=True, help="folder of mitoed predict CSVs, <SeriesInstanceUID>.csv")
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", default="r1")
    ap.add_argument("--per-category", nargs="*", default=["gbm=6", "astro4=4", "oligo3=6", "astro3=8", "grade2=8"])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--holdout", action="append", default=[])
    ap.add_argument("--high", type=int, default=6)
    ap.add_argument("--mid", type=int, default=8)
    ap.add_argument("--low", type=int, default=3)
    ap.add_argument("--mimic", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_train.slides import open_slide
    from entip_glioma.highgrade.mitoses import read_mitoed_cells
    rows = [r for r in read_manifest(a.manifest) if (Path(a.cells) / f"{r['SeriesInstanceUID']}.csv").exists()]
    out = Path(a.out)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    seen = read_seen(a.exclude)
    key, slides, n = [], [], 0
    for si, r in enumerate(pick_slides(rows, quotas(a.per_category), None, a.seed, read_patients(a.holdout)), 1):
        uid = r["SeriesInstanceUID"]
        cands = [c for c in read_mitoed_cells(Path(a.cells) / f"{uid}.csv", uid)
                 if xy(c["x_um"], c["y_um"]) not in seen[uid]]
        by = {b: [c for c in cands if band(c) == b] for b in BANDS}
        rng = random.Random(f"{a.seed}-{uid}")
        chosen = []
        for b in BANDS:
            k = min(getattr(a, b), len(by[b]))
            chosen += [(c, b, len(by[b]), k) for c in rng.sample(by[b], k)]
        if not chosen:
            continue
        rng.shuffle(chosen)
        rd = open_slide(str(find_slide(Path(a.slides), uid)))
        code_s, items = f"S{si:03d}", []
        for c, b, size, k in chosen:
            n += 1
            code = f"mit-{a.round}-C{n:05d}"
            zoom(rd, c["x_um"], c["y_um"], 64, 24, 256, circle=True).save(out / "cells" / f"{code}.jpg", quality=92)
            zoom(rd, c["x_um"], c["y_um"], 256, 24, 256, circle=True).save(out / "cells" / f"{code}-c.jpg",
                                                                          quality=88)
            items.append({"code": code, "images": [f"cells/{code}.jpg", f"cells/{code}-c.jpg"]})
            key.append({"tile_code": code, "round": a.round, "slide_code": code_s, "patient": r["PatientID"],
                        "slide": uid, "category": r["category"], "diagnosis": r.get("diagnosis", ""),
                        "source": b, "band_size": size, "band_picked": k, "candidate": c["id"],
                        "x_um": c["x_um"], "y_um": c["y_um"], "p": c["p"], "call": c.get("call") or ""})
        slides.append({"slide": code_s, "items": items})
        print(f"{code_s}  {r['PatientID']:<16} {r['category']:<7} {len(items)} candidates "
              f"({', '.join(f'{b} {len(by[b])}' for b in BANDS)})", file=sys.stderr)
    write_round(out, key, page(f"entip-mit-{a.round}-{a.seed}-{len(key)}", "Step 3c: mitosis labels", slides,
                               CHOICES, INSTRUCTIONS), len(slides))
    return 0


if __name__ == "__main__":
    sys.exit(main())
