"""Step 3b: choose slides to outline necrosis on in QuPath, and tiles to label on the page.

    python pick.py --manifest trial.csv --bags bags/ --slides slides/ --out nec-r1 --round r1 [--model triage/]
    python pick.py ... --out nec-r2 --round r2 --exclude nec-r1/key.csv --head export-r1/necrosis.joblib \
        --outline-per-category --attended 0 --diverse 0 --random 0 --uncertain 6 --likely 4

Necrosis is an area, and its edge decides most cases, so **outlines are the main labels**: they are quick for
large areas and give the edge. ``--outline-per-category`` writes ``outline-slides.csv`` (the slides to outline)
and ``fetch-slides.sh`` (copies them from Eddie to your Mac).

The page is for single tiles: look-alikes (haemorrhage, crush, cautery, fibrin) and the tiles a head is unsure of.
Round 1: ``--attended`` (triage model's most attended tiles), ``--grade`` (with ``--grade-model``: the tiles that
push a grade 4 vs grade 3 model most towards grade 4, see ``highgrade_common/grade_mil.py``), ``--diverse``,
``--random``. From round 2, with
``--head``: ``--uncertain`` (nearest 0.5), ``--likely`` (highest p), and on free-negative slides (``astro3``,
``grade2``: no necrosis by their grade) ``--audit``, the highest-p tiles: hard negatives, or a grading error.

Writes ``labels.html``, ``tiles/<code>.jpg`` (the tile) and ``tiles/<code>-c.jpg`` (its 5 x 5 field, since the
edge matters), and ``key.csv``. Turn the labels into training files with ``export.py``.
"""
from __future__ import annotations

import argparse
import csv
import random
import shlex
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import (FREE_NEGATIVE, Picker, diverse, find_slide, pick_slides, quotas, read_manifest,  # noqa: E402
                    read_patients, read_seen, write_round, xy, zoom)
from labelpage import page  # noqa: E402

CHOICES = [
    ("n", "Necrosis", "tumour necrosis in the outlined tile, palisading or not"),
    ("0", "Not necrosis", "assessable tissue without necrosis, including haemorrhage, crush, cautery and fibrin"),
    ("x", "Not assessable", "background, folds, ink, out of focus or too little tissue"),
]
INSTRUCTIONS = ("Label the <b>outlined tile</b> (112 µm). Left: the tile; right: the 5 × 5 field around it. "
                "Necrosis counts if any part of it is inside the outline.")


def choose(bag, a, cat, model, gm, head, rng, seen) -> list:
    pk = Picker(i for i, c in enumerate(bag.coords_um) if xy(*c) in seen)
    p = head.tile_p(bag) if head is not None else None
    if cat in FREE_NEGATIVE:
        if p is not None:
            pk.add(np.argsort(-p), "audit", a.audit)
        return pk.out
    if model is not None and a.attended:
        pk.add(np.argsort(-np.asarray(model.predict(bag)["attention"])), "attended", a.attended)
    if gm is not None and a.grade:
        from grade_mil import tile_scores
        pk.add(np.argsort(-tile_scores(gm, bag.features)), "grade", a.grade)
    if p is not None:
        pk.add(np.argsort(np.abs(p - 0.5)), "uncertain", a.uncertain)
        pk.add(np.argsort(-p), "likely", a.likely)
    pk.add(diverse(bag.features, a.diverse, a.seed), "diverse", a.diverse)
    pk.add(rng.permutation(len(bag)), "random", a.random)
    return pk.out


def outline_list(a, rows, holdout, out: Path) -> None:
    picked = pick_slides(rows, quotas(a.outline_per_category), a.bags, a.seed + 1, holdout)
    with (out / "outline-slides.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide", "patient", "category"])
        for r in picked:
            w.writerow([r["SeriesInstanceUID"], r["PatientID"], r["category"]])
    src = str(Path(a.slides).resolve())
    lines = ["#!/usr/bin/env bash", "# Copy the slides to outline from Eddie to this Mac. Run on the Mac.",
             "set -euo pipefail", 'HOST="${HOST:-mhamdan@eddie.ecdf.ed.ac.uk}"',
             'DEST="${DEST:-$HOME/highgrade-labels/necrosis-slides}"', 'mkdir -p "$DEST"']
    lines += [f'rsync -av "$HOST:{shlex.quote(src)}/{r["SeriesInstanceUID"]}" "$DEST/"' for r in picked]
    (out / "fetch-slides.sh").write_text("\n".join(lines) + "\n")
    print(f"{len(picked)} slides to outline: {out / 'outline-slides.csv'}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", default="r1")
    ap.add_argument("--per-category", nargs="*", default=["gbm=10", "astro4=8", "oligo3=10"],
                    help="patients per category for the page; free-negative slides (astro3, grade2) get audit "
                         "picks only, so add them from round 2")
    ap.add_argument("--outline-per-category", nargs="*", default=["gbm=4", "astro4=3", "oligo3=3"],
                    help="slides to outline in QuPath (none: give the flag with no values)")
    ap.add_argument("--model", help="triage model folder, for --attended")
    ap.add_argument("--head", help="necrosis head from an earlier round")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--holdout", action="append", default=[])
    ap.add_argument("--attended", type=int, default=4)
    ap.add_argument("--grade-model", help="grade-mil.npz from highgrade_common/grade_mil.py, for --grade")
    ap.add_argument("--grade", type=int, default=6,
                    help="tiles that push the grade model most towards grade 4")
    ap.add_argument("--diverse", type=int, default=6)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--uncertain", type=int, default=6)
    ap.add_argument("--likely", type=int, default=4)
    ap.add_argument("--audit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    from entip_glioma.highgrade.regions import RegionDetector
    model = None
    if a.model and a.attended:
        from entip_glioma.triage.model import TriageModel
        model = TriageModel.load(a.model)
    gm = None
    if a.grade_model and a.grade:
        from entip_glioma.wsi.abmil import ABMIL
        gm = ABMIL.load(a.grade_model)
    head = RegionDetector.load(a.head) if a.head else None

    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    rows, holdout = read_manifest(a.manifest), read_patients(a.holdout)
    if a.outline_per_category:
        outline_list(a, rows, holdout, out)
    seen, rng = read_seen(a.exclude), np.random.default_rng(a.seed)
    key, slides, n = [], [], 0
    for si, r in enumerate(pick_slides(rows, quotas(a.per_category), a.bags, a.seed, holdout), start=1):
        uid = r["SeriesInstanceUID"]
        bag = Bag.load(Path(a.bags) / f"{uid}.npz")
        if len(bag) == 0:
            continue
        tiles = choose(bag, a, r["category"], model, gm, head, rng, seen[uid])
        if not tiles:
            continue
        random.Random(f"{a.seed}-{uid}").shuffle(tiles)
        rd = open_slide(str(find_slide(Path(a.slides), uid)))
        code_s, items = f"S{si:03d}", []
        for i, source in tiles:
            n += 1
            code = f"nec-{a.round}-T{n:05d}"
            x, y = (float(v) for v in bag.coords_um[i])
            t = bag.tile_um
            zoom(rd, x + t / 2, y + t / 2, 1.5 * t, t, 448).save(out / "tiles" / f"{code}.jpg", quality=90)
            zoom(rd, x + t / 2, y + t / 2, 5 * t, t, 448).save(out / "tiles" / f"{code}-c.jpg", quality=88)
            items.append({"code": code, "images": [f"tiles/{code}.jpg", f"tiles/{code}-c.jpg"]})
            key.append({"tile_code": code, "round": a.round, "slide_code": code_s, "patient": r["PatientID"],
                        "slide": uid, "category": r["category"], "diagnosis": r.get("diagnosis", ""),
                        "source": source, "tile_index": i, "x_um": x, "y_um": y, "tile_um": t,
                        "encoder": bag.encoder})
        slides.append({"slide": code_s, "items": items})
        print(f"{code_s}  {r['PatientID']:<16} {r['category']:<7} {len(items)} tiles", file=sys.stderr)
    if key:
        write_round(out, key, page(f"entip-nec-{a.round}-{a.seed}-{len(key)}", "Step 3b: necrosis tile labels",
                                   slides, CHOICES, INSTRUCTIONS), len(slides))
    elif not a.outline_per_category:
        raise SystemExit("nothing picked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
