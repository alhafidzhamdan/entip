"""Step 3a: a blinded high-power recheck of the MVP tile head's labels and likely mistakes.

    python recheck.py --oof tile-heads/mvp/oof.csv --locations tile-heads/tile-locations.csv --slides slides/ \
        --out mvp-recheck

The first ratings had no "vessel without MVP", so a head trained on them could be finding vessels rather than MVP.
This page shows, in one shuffled list:

* every tile rated MVP;
* the tiles rated not MVP that the head scored highest out of fold (``--hard``, at most ``--max-per-slide`` a slide);
* a random sample of the other tiles rated not MVP (``--controls``).

Each is labelled MVP, vessel without MVP, no vessel, or unsure / not assessable, using the keys of ``export.py``. The
page is one self-contained file and shows no slide, grade, earlier rating, score or reason for picking. Keep
``key.csv`` back: it holds all of those.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import crop, find_slide, write_round, zoom  # noqa: E402
from labelpage import page  # noqa: E402

CHOICES = [
    ("m", "MVP", "at least two layers of plump proliferating wall cells around a lumen, or a glomeruloid tuft, in the "
                 "outlined tile"),
    ("v", "Vessel, no MVP", "a vessel in the outlined tile without MVP: single-layer plump endothelium, thin-walled "
                            "capillaries, thick or hyalinised walls, tangential cuts, adjacent or reactive vessels"),
    ("0", "No vessel", "assessable tissue with no vessel in the outlined tile"),
    ("x", "Unsure / not assessable", "cannot decide, or blood, crush, folds, ink, out of focus or too little tissue"),
]
INSTRUCTIONS = ("Label the <b>outlined tile</b> (112 µm). Left: the tile at high power; right: the 3 × 3 field around "
                "it. A vessel counts if any part of it is inside the outline; MVP wins over vessel.")


def select(rows: list, hard: int, controls: int, max_per_slide: int, seed: int) -> list:
    """Every positive, the highest-scoring negatives, and random other negatives, each with its source."""
    pos = [dict(r, source="earlier-mvp") for r in rows if int(r["label"]) == 1]
    neg = sorted((r for r in rows if int(r["label"]) == 0), key=lambda r: (-float(r["p_out_of_fold"]), r["item"]))
    picked, per = [], Counter()
    for r in neg:
        if len(picked) == hard:
            break
        if per[r["slide"]] < max_per_slide:
            per[r["slide"]] += 1
            picked.append(r)
    used = {r["item"] for r in picked}
    rest = [r for r in neg if r["item"] not in used]
    ctrl = random.Random(seed).sample(rest, min(controls, len(rest)))
    return pos + [dict(r, source="hard-negative") for r in picked] + [dict(r, source="random-negative") for r in ctrl]


def data_uri(im, quality: int) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--oof", required=True, help="oof.csv from tile_heads.py train --feature mvp")
    ap.add_argument("--locations", required=True, help="tile-locations.csv from tile_heads.py locate")
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hard", type=int, default=50, help="negatives the head scored highest")
    ap.add_argument("--controls", type=int, default=40, help="random other negatives")
    ap.add_argument("--max-per-slide", type=int, default=6, help="most hard negatives from one slide")
    ap.add_argument("--round", default="recheck1", help="prefix for tile codes, unique per round")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_train.slides import open_slide
    loc = {r["item"]: r for r in csv.DictReader(open(a.locations, newline=""))}
    rows = list(csv.DictReader(open(a.oof, newline="")))
    missing = [r["item"] for r in rows if r["item"] not in loc or not int(loc[r["item"]]["ok"])]
    if missing:
        raise SystemExit(f"no clear location for {len(missing)} tiles, e.g. {missing[:3]}")
    chosen = select(rows, a.hard, a.controls, a.max_per_slide, a.seed)
    random.Random(f"{a.seed}-order").shuffle(chosen)
    for n, r in enumerate(chosen, start=1):
        r["tile_code"] = f"mvp-{a.round}-T{n:03d}"

    by = defaultdict(list)
    for r in chosen:
        by[r["slide"]].append(r)
    images, mpp = {}, {}
    for n, (slide, tiles) in enumerate(sorted(by.items()), start=1):
        rd = open_slide(str(find_slide(Path(a.slides), slide)))
        mpp[slide] = float(rd.mpp)
        for r in tiles:
            x, y, t = (float(loc[r["item"]][k]) for k in ("x_um", "y_um", "tile_um"))
            images[r["tile_code"]] = [data_uri(zoom(rd, x + t / 2, y + t / 2, 1.5 * t, t, 640), 88),
                                      data_uri(crop(rd, x, y, t, 384), 85)]
        print(f"{n}/{len(by)} slides", file=sys.stderr, flush=True)

    key = [{"tile_code": r["tile_code"], "round": a.round, "review_item": r["item"], "patient": r["patient"],
            "slide": r["slide"], "category": r["category"], "source": r["source"], "earlier_label": r["label"],
            "p_out_of_fold": r["p_out_of_fold"], "tile_index": loc[r["item"]]["tile_index"],
            "x_um": loc[r["item"]]["x_um"], "y_um": loc[r["item"]]["y_um"], "tile_um": loc[r["item"]]["tile_um"],
            "slide_mpp": mpp[r["slide"]]} for r in chosen]
    items = [{"code": r["tile_code"], "images": images[r["tile_code"]]} for r in chosen]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    html = page(f"entip-mvp-{a.round}-{a.seed}-{len(key)}", "MVP high-power recheck",
                [{"slide": "All tiles", "items": items}], CHOICES, INSTRUCTIONS, column_px=1000)
    write_round(out, key, html, len(by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
