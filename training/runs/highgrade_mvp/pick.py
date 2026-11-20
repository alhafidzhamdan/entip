"""Step 3a: pick tiles for a pathologist to label as MVP, vessel without MVP, no vessel, or not assessable.

    python pick.py --manifest trial.csv --bags bags/ --slides slides/ --out mvp-r1 --round r1 [--model triage/]
    python pick.py ... --out mvp-r2 --round r2 --exclude mvp-r1/key.csv \
        --vessel-head export-r1/vessel.joblib --mvp-head export-r1/mvp.joblib

The decision for MVP is about one vessel, and the look-alikes are other vessels. So the labels say, for each tile,
whether it holds a vessel and whether that vessel shows MVP. That trains two heads: a vessel head (helper, used
only for picking) and the MVP head. From round 2 the picks come from the tiles the vessel head calls vessels:

* ``--uncertain``: vessels whose MVP probability is nearest 0.5 (what the MVP head learns most from);
* ``--likely``: vessels with the highest MVP probability (checks its positives);
* ``--vessels``: random vessels (keeps ordinary vessels in the labels);
* ``--audit``, on free-negative slides (``astro3``, ``grade2``: no MVP by their grade): the tiles with the highest
  MVP probability. These are the hardest negatives, or a grading error worth knowing about.

Round 1 has no heads: ``--attended`` (the triage model's most attended tiles, often vessels in a glioblastoma),
``--grade`` (with ``--grade-model``: the tiles that push a grade 4 vs grade 3 model most towards grade 4, see
``highgrade_common/grade_mil.py``), ``--diverse`` (one tile per cluster of the slide's embeddings) and ``--random``.

Slides: ``--per-category`` patients per category (see ``highgrade_common/common.py``), one slide each. The default
leans on lower-grade backgrounds with MVP (``oligo3``, ``astro4``) so the head learns MVP, not glioblastoma.

Writes ``labels.html``, ``tiles/<code>.jpg`` (the tile at high power) and ``tiles/<code>-c.jpg`` (its 3 x 3 field),
and ``key.csv``. Turn the labels into training files with ``export.py``.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import (FREE_NEGATIVE, Picker, crop, diverse, find_slide, pick_slides, quotas,  # noqa: E402
                    read_manifest, read_patients, read_seen, write_round, xy, zoom)
from labelpage import page  # noqa: E402

CHOICES = [
    ("m", "MVP", "microvascular proliferation in the outlined tile: multilayered vessel walls of proliferating "
                 "endothelial cells and pericytes, glomeruloid tufts or garlands"),
    ("v", "Vessel, no MVP", "a vessel in the outlined tile without MVP: thin or thick walled, hyalinised, "
                            "many vessels, reactive or cut tangentially"),
    ("0", "No vessel", "assessable tissue with no vessel in the outlined tile"),
    ("x", "Not assessable", "background, blood, crush, folds, ink, out of focus or too little tissue"),
]
INSTRUCTIONS = ("Label the <b>outlined tile</b> (112 µm). Left: the tile at high power; right: the 3 × 3 field "
                "around it. A vessel counts if any part of it is inside the outline; MVP wins over vessel.")


def choose(bag, a, cat, model, gm, vhead, mhead, rng, seen) -> list:
    pk = Picker(i for i, c in enumerate(bag.coords_um) if xy(*c) in seen)
    pm = mhead.tile_p(bag) if mhead is not None else None
    if cat in FREE_NEGATIVE and pm is not None:
        pk.add(np.argsort(-pm), "audit", a.audit)
        return pk.out
    if model is not None and a.attended:
        pk.add(np.argsort(-np.asarray(model.predict(bag)["attention"])), "attended", a.attended)
    if gm is not None and a.grade:
        from grade_mil import tile_scores
        pk.add(np.argsort(-tile_scores(gm, bag.features)), "grade", a.grade)
    if vhead is not None:
        pv = vhead.tile_p(bag)
        order = np.argsort(-pv)                          # vessels, or at least the most vessel-like tiles
        cand = order[:max(int((pv >= a.vessel_threshold).sum()), min(a.min_candidates, len(order)))]
        if pm is not None and len(cand):
            pk.add(cand[np.argsort(np.abs(pm[cand] - 0.5))], "uncertain", a.uncertain)
            pk.add(cand[np.argsort(-pm[cand])], "likely", a.likely)
        pk.add(rng.permutation(cand), "vessel", a.vessels)
    pk.add(diverse(bag.features, a.diverse, a.seed), "diverse", a.diverse)
    pk.add(rng.permutation(len(bag)), "random", a.random)
    return pk.out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round", default="r1", help="prefix for tile codes, unique per round")
    ap.add_argument("--per-category", nargs="*", default=["gbm=10", "astro4=8", "oligo3=12", "astro3=4", "grade2=4"])
    ap.add_argument("--model", help="triage model folder, for --attended")
    ap.add_argument("--vessel-head")
    ap.add_argument("--mvp-head")
    ap.add_argument("--vessel-threshold", type=float, default=0.5)
    ap.add_argument("--min-candidates", type=int, default=40,
                    help="per slide, at least this many of the most vessel-like tiles are candidates")
    ap.add_argument("--exclude", action="append", default=[], help="key.csv of an earlier round (repeat)")
    ap.add_argument("--holdout", action="append", default=[], help="CSV of held-out patients: never picked")
    ap.add_argument("--attended", type=int, default=4)
    ap.add_argument("--grade-model", help="grade-mil.npz from highgrade_common/grade_mil.py, for --grade")
    ap.add_argument("--grade", type=int, default=6,
                    help="tiles that push the grade model most towards grade 4")
    ap.add_argument("--diverse", type=int, default=8)
    ap.add_argument("--random", type=int, default=2)
    ap.add_argument("--uncertain", type=int, default=6)
    ap.add_argument("--likely", type=int, default=4)
    ap.add_argument("--vessels", type=int, default=0)
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
    vhead = RegionDetector.load(a.vessel_head) if a.vessel_head else None
    mhead = RegionDetector.load(a.mvp_head) if a.mvp_head else None

    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    seen, rng = read_seen(a.exclude), np.random.default_rng(a.seed)
    picked = pick_slides(read_manifest(a.manifest), quotas(a.per_category), a.bags, a.seed, read_patients(a.holdout))
    key, slides, n = [], [], 0
    for si, r in enumerate(picked, start=1):
        uid = r["SeriesInstanceUID"]
        bag = Bag.load(Path(a.bags) / f"{uid}.npz")
        if len(bag) == 0:
            continue
        tiles = choose(bag, a, r["category"], model, gm, vhead, mhead, rng, seen[uid])
        random.Random(f"{a.seed}-{uid}").shuffle(tiles)
        rd = open_slide(str(find_slide(Path(a.slides), uid)))
        code_s, items = f"S{si:03d}", []
        for i, source in tiles:
            n += 1
            code = f"mvp-{a.round}-T{n:05d}"
            x, y = (float(v) for v in bag.coords_um[i])
            t = bag.tile_um
            zoom(rd, x + t / 2, y + t / 2, 1.5 * t, t, 512).save(out / "tiles" / f"{code}.jpg", quality=90)
            crop(rd, x, y, t, 512).save(out / "tiles" / f"{code}-c.jpg", quality=88)
            items.append({"code": code, "images": [f"tiles/{code}.jpg", f"tiles/{code}-c.jpg"]})
            key.append({"tile_code": code, "round": a.round, "slide_code": code_s, "patient": r["PatientID"],
                        "slide": uid, "category": r["category"], "diagnosis": r.get("diagnosis", ""),
                        "source": source, "tile_index": i, "x_um": x, "y_um": y, "tile_um": t,
                        "encoder": bag.encoder})
        slides.append({"slide": code_s, "items": items})
        print(f"{code_s}  {r['PatientID']:<16} {r['category']:<7} {len(items)} tiles", file=sys.stderr)
    write_round(out, key, page(f"entip-mvp-{a.round}-{a.seed}-{len(key)}", "Step 3a: MVP tile labels", slides,
                               CHOICES, INSTRUCTIONS), len(slides))
    return 0


if __name__ == "__main__":
    sys.exit(main())
