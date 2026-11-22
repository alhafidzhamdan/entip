"""Separate MVP and necrosis tile heads from the blinded tile review (``rate_tiles.py``).

    python tile_heads.py locate --key tile-review-key.csv --tiles grade-mil/tiles --bags bags/ --slides slides/ \
        --out tile-heads
    python tile_heads.py train --feature mvp --key tile-review-key.csv --ratings ratings-AH.csv --rater AH \
        --locations tile-heads/tile-locations.csv --bags bags/ --out tile-heads
    python tile_heads.py train --feature necrosis ...

**locate.** The review key has no tile coordinates, so each saved image is matched back to its slide: a coarse search
over every tile in the bag, then an exact re-crop of the best candidates. Nothing is written unless all tiles match
clearly; otherwise ``tile-locations-failed.csv`` lists those that did not.

**train.** The rating applies to the outlined tile. MVP head: ``mvp`` / ``both`` are positive; necrosis head:
``necrosis`` / ``both`` are positive. ``neither``, ``gemistocytic``, ``haemorrhage`` and the other feature are
negative; artefact, can't assess, unsure and other are left out. The necrosis head adds the 3 x 3 and 5 x 5
neighbourhoods, as in ``highgrade_necrosis/export.py``.

Each head is scored out of fold, with all of a patient's slides on one side, then refitted on every labelled tile.
Writes ``<feature>/head.joblib``, ``oof.csv`` and ``metrics.json``. The review tiles were chosen by the grade model
and rated by the people named in ``--rater``, so this is a pilot: a head stays unvalidated until it is tested on
held-out patients whose tiles were chosen independently of the model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAINING_DATA, crop, find_slide, read_patients  # noqa: E402
from rate_tiles import FEATURE  # noqa: E402

HEADS = {"mvp": ("microvascular_proliferation", 0, ()), "necrosis": ("necrosis", 1, (1, 2))}
COARSE_PX = 16      # a 3 x 3 field in the coarse search
CANDIDATES = 12     # tiles re-cropped exactly for each image
MAX_MAD = 8.0       # largest mean absolute pixel difference (0-255) for a match; JPEG alone gives a few
MIN_GAP = 2.0       # the best match must beat the next by this much


def _unit(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)


def coarse_fields(rd, bag) -> np.ndarray:
    """Every tile's 3 x 3 field from one low-power read of the slide, as unit vectors."""
    t = bag.tile_um
    lo, hi = bag.coords_um.min(0) - 2 * t, bag.coords_um.max(0) + 3 * t
    span = float((hi - lo).max())
    cx, cy = (lo + hi) / 2
    scale = COARSE_PX / (3 * t)
    px = math.ceil(span * scale)
    im = np.asarray(rd.read_um(float(cx), float(cy), span, px), np.float32)
    x0 = np.clip(np.rint((bag.coords_um[:, 0] - t - (cx - span / 2)) * scale).astype(int), 0, im.shape[1] - COARSE_PX)
    y0 = np.clip(np.rint((bag.coords_um[:, 1] - t - (cy - span / 2)) * scale).astype(int), 0, im.shape[0] - COARSE_PX)
    fields = np.stack([im[y:y + COARSE_PX, x:x + COARSE_PX].ravel() for x, y in zip(x0, y0)])
    return _unit(fields)


def locate_slide(rd, bag, rows: list, tiles: Path) -> list:
    """Finds each review image in the bag; returns rows with tile index, coordinates and match quality."""
    from PIL import Image
    fields, out = coarse_fields(rd, bag), []
    for r in rows:
        with Image.open(tiles / r["image"]) as f:
            saved = f.convert("RGB")
        small = np.asarray(saved.resize((COARSE_PX, COARSE_PX), Image.BOX), np.float32).ravel()
        target = np.asarray(saved, np.int16)
        tried, mads = [], []
        for i in np.argsort(-(fields @ _unit(small))):     # best coarse match first; go on until one fits exactly
            x, y = bag.coords_um[int(i)]
            im = np.asarray(crop(rd, float(x), float(y), bag.tile_um, target.shape[0]), np.int16)
            tried.append(int(i))
            mads.append(float(np.abs(im - target).mean()))
            if len(tried) >= CANDIDATES and min(mads) <= MAX_MAD:
                break
        order = np.argsort(mads)
        i, mad = tried[order[0]], mads[order[0]]
        nxt = mads[order[1]] if len(order) > 1 else float("inf")
        out.append({"item": r["item"], "slide": r["slide"], "tile_index": i, "x_um": float(bag.coords_um[i, 0]),
                    "y_um": float(bag.coords_um[i, 1]), "tile_um": bag.tile_um, "mad": round(mad, 3),
                    "next_mad": round(nxt, 3), "ok": int(mad <= MAX_MAD and nxt - mad >= MIN_GAP)})
    used = [o["tile_index"] for o in out]
    if len(set(used)) < len(used):        # two images on one tile: neither can be trusted
        for o in out:
            if used.count(o["tile_index"]) > 1:
                o["ok"] = 0
    return out


def locate(a) -> int:
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    by = defaultdict(list)
    for r in csv.DictReader(open(a.key, newline="")):
        by[r["slide"]].append(r)
    found = []
    for n, (slide, rows) in enumerate(sorted(by.items()), start=1):
        found += locate_slide(open_slide(str(find_slide(Path(a.slides), slide))),
                              Bag.load(Path(a.bags) / f"{slide}.npz"), rows, Path(a.tiles))
        print(f"{n}/{len(by)} slides, {sum(f['ok'] for f in found)} of {len(found)} tiles matched", file=sys.stderr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = [f for f in found if not f["ok"]]
    name = "tile-locations-failed.csv" if bad else "tile-locations.csv"
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(found[0]))
        w.writeheader()
        w.writerows(bad or found)
    print(json.dumps({"tiles": len(found), "matched": len(found) - len(bad), "worst_mad": max(f["mad"] for f in found),
                      "written": str(out / name)}, indent=2))
    if bad:
        raise SystemExit(f"{len(bad)} tiles did not match clearly: no locations written")
    return 0


def read_items(a, column: int) -> list:
    """Labelled tiles for one feature, after checking the ratings, key and locations belong together."""
    key = {r["item"]: r for r in csv.DictReader(open(a.key, newline=""))}
    pages = {r["page_id"] for r in key.values()}
    rating = {}
    for r in csv.DictReader(open(a.ratings, newline="")):
        if r["rater"].upper() == a.rater.upper() and r["page_id"] in pages and r["item"] in key and r["rating"]:
            rating[r["item"]] = r["rating"]       # the latest row counts
    if set(rating) != set(key):
        raise SystemExit(f"{a.rater} has ratings for {len(rating)} of {len(key)} tiles on page {sorted(pages)}")
    loc = {r["item"]: r for r in csv.DictReader(open(a.locations, newline=""))}
    if set(loc) != set(key):
        raise SystemExit(f"{a.locations} covers {len(set(loc) & set(key))} of {len(key)} tiles")
    holdout = read_patients(a.holdout)
    items = []
    for item, k in key.items():
        if rating[item] in FEATURE and k["patient"] not in holdout:
            if loc[item]["slide"] != k["slide"]:
                raise SystemExit(f"{item}: key and locations name different slides")
            items.append({"item": item, "slide": k["slide"], "patient": k["patient"], "category": k["category"],
                          "kind": k["kind"], "i": int(loc[item]["tile_index"]), "x": float(loc[item]["x_um"]),
                          "y": float(loc[item]["y_um"]), "label": int(FEATURE[rating[item]][column])})
    return items


def slide_labels(items: list, bags: dict) -> dict:
    out = {s: np.full(len(b), -1, int) for s, b in bags.items()}
    for t in items:
        if not np.allclose(bags[t["slide"]].coords_um[t["i"]], [t["x"], t["y"]], atol=0.01):
            raise SystemExit(f"{t['item']}: the bag's tile {t['i']} is not at the located position")
        out[t["slide"]][t["i"]] = t["label"]
    return out


def fit_head(feature: str, context: tuple, seed: int, bags: dict, labels: dict, slides):
    from entip_glioma.highgrade.regions import RegionConfig, RegionDetector
    slides = sorted(slides)
    det = RegionDetector(RegionConfig(feature, training_data=TRAINING_DATA, context=tuple(context), seed=seed))
    return det.fit([bags[s] for s in slides], [labels[s] for s in slides])


def score(y, p) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = int(y.sum()), int((1 - y).sum())
    out = {"tiles": len(y), "positive": pos, "negative": neg}
    if pos and neg:
        call = p >= 0.5
        out.update(auroc=round(float(roc_auc_score(y, p)), 3), average_precision=round(float(
            average_precision_score(y, p)), 3), prevalence=round(pos / len(y), 3),
            sensitivity_at_0_5=round(float(call[y == 1].mean()), 3),
            specificity_at_0_5=round(float((~call)[y == 0].mean()), 3))
    return out


def patient_ci(y, p, patients, seed: int, n: int = 2000) -> dict:
    """95% interval for AUROC and average precision, resampling whole patients."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    by = defaultdict(list)
    for i, q in enumerate(patients):
        by[q].append(i)
    names, rng, au, ap = sorted(by), np.random.default_rng(seed), [], []
    for _ in range(n):
        idx = np.concatenate([by[names[j]] for j in rng.integers(len(names), size=len(names))])
        if 0 < y[idx].sum() < len(idx):
            au.append(roc_auc_score(y[idx], p[idx]))
            ap.append(average_precision_score(y[idx], p[idx]))
    return {"auroc_95ci": [round(float(v), 3) for v in np.percentile(au, [2.5, 97.5])],
            "average_precision_95ci": [round(float(v), 3) for v in np.percentile(ap, [2.5, 97.5])]}


def train(a) -> int:
    from sklearn.model_selection import StratifiedGroupKFold
    from entip_glioma.wsi.bags import Bag
    feature, column, context = HEADS[a.feature]
    items = read_items(a, column)
    bags = {s: Bag.load(Path(a.bags) / f"{s}.npz") for s in sorted({t["slide"] for t in items})}
    labels = slide_labels(items, bags)
    y = np.array([t["label"] for t in items])
    pats = np.array([t["patient"] for t in items])
    k = min(a.folds, len(set(pats[y == 1])), len(set(pats[y == 0])))
    if k < 2:
        raise SystemExit(f"{a.feature}: positives and negatives must each come from at least 2 patients")
    p, fold = np.full(len(y), np.nan), np.full(len(y), -1)
    splitter = StratifiedGroupKFold(k, shuffle=True, random_state=a.seed)
    for n, (tr, te) in enumerate(splitter.split(y, y, pats)):
        if len(set(y[tr])) < 2:
            raise SystemExit(f"{a.feature}: fold {n} has only one class to train on")
        det = fit_head(feature, context, a.seed, bags, labels, {items[i]["slide"] for i in tr})
        for s in {items[i]["slide"] for i in te}:
            tile_p = det.tile_p(bags[s])
            for i in te:
                if items[i]["slide"] == s:
                    p[i], fold[i] = tile_p[items[i]["i"]], n
    if np.isnan(p).any():
        raise SystemExit("some tiles were not scored out of fold")
    out = Path(a.out) / a.feature
    out.mkdir(parents=True, exist_ok=True)
    with (out / "oof.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "slide", "patient", "category", "kind", "label", "p_out_of_fold", "fold"])
        for t, q, n in zip(items, p, fold):
            w.writerow([t["item"], t["slide"], t["patient"], t["category"], t["kind"], t["label"], round(float(q), 5),
                        int(n)])
    cat, kind = np.array([t["category"] for t in items]), np.array([t["kind"] for t in items])
    groups = {"grade 4": cat == "astro4", "grade 3": cat == "astro3", "model-picked tiles": kind == "top",
              "random tiles": kind == "random"}
    metrics = {"feature": a.feature, "status": "pilot: not validated", "rater": a.rater.upper(),
               "patients": len(set(pats)), "slides": len(bags), "folds": k, "folds_grouped_by": "patient",
               "context": list(context), "out_of_fold": {**score(y, p), **patient_ci(y, p, pats, a.seed)},
               "by_group": {name: score(y[m], p[m]) for name, m in groups.items()},
               "positive_patients": len(set(pats[y == 1]))}
    final = fit_head(feature, context, a.seed, bags, labels, bags)
    final.save(out / "head.joblib")
    metrics["final_head"] = {"model_id": final.model_id, "n_labels": final.n_labels, "validated": False}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("locate", help="match each review image to its tile in the bag")
    b.add_argument("--key", required=True, help="tile-review-key.csv")
    b.add_argument("--tiles", required=True, help="the grade_mil --view tiles folder")
    b.add_argument("--bags", required=True)
    b.add_argument("--slides", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=locate)
    t = sub.add_parser("train", help="fit and score one head")
    t.add_argument("--feature", required=True, choices=sorted(HEADS))
    t.add_argument("--key", required=True, help="tile-review-key.csv")
    t.add_argument("--ratings", required=True, help="ratings CSV with rater, page_id, item and rating")
    t.add_argument("--rater", required=True)
    t.add_argument("--locations", required=True, help="tile-locations.csv from locate")
    t.add_argument("--bags", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--holdout", action="append", default=[], help="CSV of held-out patients: never used (repeat)")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=train)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
