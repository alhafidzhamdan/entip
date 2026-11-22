"""Step 3a: test a frozen MVP head on new patients, and collect more labels, in one blinded page.

    python new_patients.py --head tile-heads-recheck/mvp/head.joblib --manifest trial.csv --bags bags/ \
        --slides slides/ --exclude tile-review-key.csv --out mvp-new1

The head is copied to ``frozen-head.joblib`` and every tile is scored before anyone labels, so these scores are a test
on patients it never trained on. Patients in ``--exclude`` (the head's training patients) and ``--holdout`` are never
picked. One slide per patient, ``--per-category`` patients per category. From each slide:

* ``top``: the highest-scoring tiles;
* ``high``: random tiles from the rest of the slide's top 10%;
* ``rest``: random tiles from everything else.

``key.csv`` records how many tiles each pick stands for (``weight``), so the labels can be weighted back to the whole
slide. The page uses the keys of ``export.py`` and shows no slide, grade, score or reason for picking. Keep ``key.csv``
back until the labels are done.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
sys.path.insert(0, str(HERE))
from common import crop, find_slide, pick_slides, quotas, read_manifest, read_patients, write_round, zoom  # noqa: E402
from labelpage import page  # noqa: E402
from recheck import CHOICES, INSTRUCTIONS, data_uri  # noqa: E402


def select_tiles(p: np.ndarray, top: int, high: int, rest: int, rng) -> list:
    """``(tile index, source, tiles it stands for)``: three strata that together cover the slide."""
    order = np.argsort(-p, kind="stable")
    tops, others = order[:top], order[top:]
    n_high = min(len(others), max(high, math.ceil(0.1 * len(p)) - len(tops)))
    upper, lower = others[:n_high], others[n_high:]
    out = [(int(i), "top", 1.0) for i in tops]
    for stratum, k, name in ((upper, high, "high"), (lower, rest, "rest")):
        k = min(k, len(stratum))
        if k:
            out += [(int(i), name, len(stratum) / k) for i in rng.choice(stratum, size=k, replace=False)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", required=True, help="the MVP head to freeze and test")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--exclude", action="append", required=True, help="key CSV naming the head's training patients")
    ap.add_argument("--holdout", action="append", default=[], help="CSV of held-out patients: never picked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", nargs="*", default=["astro4=13", "gbm=10", "oligo3=8"])
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--high", type=int, default=2)
    ap.add_argument("--rest", type=int, default=2)
    ap.add_argument("--round", default="new1", help="prefix for tile codes, unique per round")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_glioma.highgrade.regions import RegionDetector
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    out = Path(a.out)
    if (out / "key.csv").exists():
        raise SystemExit(f"{out / 'key.csv'} already exists: a built round is never overwritten")
    trained = read_patients(a.exclude)
    if not trained:
        raise SystemExit("--exclude names no patients")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "frozen-head.joblib"
    shutil.copy2(a.head, frozen)
    head_sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    det = RegionDetector.load(frozen)
    if det.cfg.feature != "microvascular_proliferation":
        raise SystemExit(f"{a.head} is a {det.cfg.feature} head")

    picked = pick_slides(read_manifest(a.manifest), quotas(a.per_category), a.bags, a.seed,
                         trained | read_patients(a.holdout))
    rng, chosen = np.random.default_rng(a.seed), []
    for n, r in enumerate(picked, start=1):
        uid = r["SeriesInstanceUID"]
        bag = Bag.load(Path(a.bags) / f"{uid}.npz")
        if len(bag) == 0:
            continue
        p = det.tile_p(bag)
        rd = open_slide(str(find_slide(Path(a.slides), uid)))
        t = bag.tile_um
        for i, source, weight in select_tiles(p, a.top, a.high, a.rest, rng):
            x, y = (float(v) for v in bag.coords_um[i])
            chosen.append({"round": a.round, "patient": r["PatientID"], "slide": uid, "category": r["category"],
                           "diagnosis": r.get("diagnosis", ""), "source": source, "weight": round(weight, 4),
                           "p_frozen": round(float(p[i]), 6), "tile_index": i, "x_um": x, "y_um": y, "tile_um": t,
                           "slide_tiles": len(bag), "slide_mpp": float(rd.mpp), "head_id": det.model_id,
                           "head_sha256": head_sha,
                           "_images": [data_uri(zoom(rd, x + t / 2, y + t / 2, 1.5 * t, t, 640), 88),
                                       data_uri(crop(rd, x, y, t, 384), 85)]})
        print(f"{n}/{len(picked)} slides, {len(chosen)} tiles", file=sys.stderr, flush=True)
    if not chosen:
        raise SystemExit("nothing picked: check --bags, the manifest and --per-category")
    if {c["patient"] for c in chosen} & trained:
        raise SystemExit("a training patient was picked")
    random.Random(f"{a.seed}-order").shuffle(chosen)
    items, key = [], []
    for n, c in enumerate(chosen, start=1):
        code = f"mvp-{a.round}-T{n:03d}"
        items.append({"code": code, "images": c.pop("_images")})
        key.append({"tile_code": code, **c})
    html = page(f"entip-mvp-{a.round}-{a.seed}-{len(key)}", "MVP new-patient labels",
                [{"slide": "All tiles", "items": items}], CHOICES, INSTRUCTIONS, column_px=1000)
    write_round(out, key, html, len({k["slide"] for k in key}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
