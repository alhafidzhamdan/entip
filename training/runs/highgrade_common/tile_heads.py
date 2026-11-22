"""Separate MVP and necrosis tile heads from the blinded tile review (``rate_tiles.py``).

    python tile_heads.py locate --key tile-review-key.csv --tiles grade-mil/tiles --bags bags/ --slides slides/ \
        --out tile-heads
    python tile_heads.py train --feature mvp --key tile-review-key.csv --ratings ratings-AH.csv --rater AH \
        --locations tile-heads/tile-locations.csv --bags bags/ --out tile-heads
    python tile_heads.py train --feature necrosis ...

**locate.** The review key has no tile coordinates, so each saved image is matched back to its slide: a coarse search
over every tile in the bag, then an exact re-crop of the best candidates. Nothing is written unless all tiles match
clearly; otherwise ``tile-locations-failed.csv`` lists those that did not. ``--leave-out`` names tiles allowed not to
match; ``train`` refuses if any of them has a rating it would use.

**train.** The rating applies to the outlined tile. MVP head: ``mvp`` / ``both`` are positive; necrosis head:
``necrosis`` / ``both`` are positive. ``neither``, ``gemistocytic``, ``haemorrhage`` and the other feature are
negative; artefact, can't assess, unsure and other are left out. The necrosis head adds the 3 x 3 and 5 x 5
neighbourhoods, as in ``highgrade_necrosis/export.py``.

``--recheck-key`` and ``--recheck-labels`` replace the first MVP labels with those of the high-power recheck:
MVP is positive, vessel and no vessel are negative, and unsure is left out. ``--extra-key`` and ``--extra-labels`` add
the labelled patients of ``highgrade_mvp/new_patients.py`` in the same way.

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
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAINING_DATA, crop, find_slide, read_patients  # noqa: E402
from rate_tiles import FEATURE  # noqa: E402

HEADS = {"mvp": ("microvascular_proliferation", 0, ()), "necrosis": ("necrosis", 1, (1, 2))}
FIELD_PX = 48       # a 3 x 3 field in the one low-power read of the slide
COARSE_PX = 16      # the same field in the first pass over every tile
CANDIDATES = 40     # tiles compared at FIELD_PX for each image
EXACT = 3           # tiles re-cropped from the slide for each image: slide reads are the slow part
MAX_MAD = 12.0      # largest mean absolute pixel difference (0-255) for a match; JPEG alone gives up to about 9
MIN_RATIO = 2.0     # the next-best tile must differ at least this many times more than the match


def clear_match(mad: float, next_mad: float) -> bool:
    return mad <= MAX_MAD and next_mad >= MIN_RATIO * mad


def _unit(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)


def field_grid(rd, bag) -> tuple:
    """One low-power read of the slide, and where each tile's 3 x 3 field starts in it."""
    t = bag.tile_um
    lo, hi = bag.coords_um.min(0) - 2 * t, bag.coords_um.max(0) + 3 * t
    span = float((hi - lo).max())
    cx, cy = (lo + hi) / 2
    scale = FIELD_PX / (3 * t)
    px = math.ceil(span * scale)
    im = np.asarray(rd.read_um(float(cx), float(cy), span, px), np.float32)
    x0 = np.clip(np.rint((bag.coords_um[:, 0] - t - (cx - span / 2)) * scale).astype(int), 0, im.shape[1] - FIELD_PX)
    y0 = np.clip(np.rint((bag.coords_um[:, 1] - t - (cy - span / 2)) * scale).astype(int), 0, im.shape[0] - FIELD_PX)
    return im, x0, y0


def locate_slide(rd, bag, rows: list, tiles: Path) -> list:
    """Finds each review image in the bag; returns rows with tile index, coordinates and match quality."""
    from PIL import Image
    im, x0, y0 = field_grid(rd, bag)
    step = FIELD_PX // COARSE_PX
    coarse = _unit(np.stack([im[y:y + FIELD_PX:step, x:x + FIELD_PX:step].ravel() for x, y in zip(x0, y0)]))
    out = []
    for r in rows:
        with Image.open(tiles / r["image"]) as f:
            saved = f.convert("RGB")
        field = np.asarray(saved.resize((FIELD_PX, FIELD_PX), Image.BOX), np.float32)
        near = np.argsort(-(coarse @ _unit(field[::step, ::step].ravel())))[:CANDIDATES]
        fine = _unit(np.stack([im[y0[i]:y0[i] + FIELD_PX, x0[i]:x0[i] + FIELD_PX].ravel() for i in near]))
        tried = [int(i) for i in near[np.argsort(-(fine @ _unit(field.ravel())))[:EXACT]]]
        target = np.asarray(saved, np.int16)
        mads = []
        for i in tried:
            x, y = bag.coords_um[i]
            mads.append(float(np.abs(np.asarray(crop(rd, float(x), float(y), bag.tile_um, target.shape[0]),
                                                np.int16) - target).mean()))
        order = np.argsort(mads)
        i, mad = tried[order[0]], mads[order[0]]
        nxt = mads[order[1]] if len(order) > 1 else float("inf")
        out.append({"item": r["item"], "slide": r["slide"], "tile_index": i, "x_um": float(bag.coords_um[i, 0]),
                    "y_um": float(bag.coords_um[i, 1]), "tile_um": bag.tile_um, "mad": round(mad, 3),
                    "next_mad": round(nxt, 3), "ok": int(clear_match(mad, nxt))})
    used = [o["tile_index"] for o in out]
    if len(set(used)) < len(used):        # two images on one tile: neither can be trusted
        for o in out:
            if used.count(o["tile_index"]) > 1:
                o["ok"] = 0
    return out


def _locate_one(slide: str, rows: list, bags: str, slides: str, tiles: str) -> list:
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    return locate_slide(open_slide(str(find_slide(Path(slides), slide))), Bag.load(Path(bags) / f"{slide}.npz"), rows,
                        Path(tiles))


def locate(a) -> int:
    by = defaultdict(list)
    for r in csv.DictReader(open(a.key, newline="")):
        by[r["slide"]].append(r)
    workers = a.workers or int(os.environ.get("NSLOTS", 1))
    found = []
    with ProcessPoolExecutor(max(1, workers)) as pool:
        jobs = [pool.submit(_locate_one, s, rows, a.bags, a.slides, a.tiles) for s, rows in sorted(by.items())]
        for n, job in enumerate(as_completed(jobs), start=1):
            found += job.result()
            print(f"{n}/{len(by)} slides, {sum(f['ok'] for f in found)} of {len(found)} tiles matched",
                  file=sys.stderr, flush=True)
    found.sort(key=lambda f: f["item"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    unmatched = [f for f in found if not f["ok"]]
    bad = [f for f in unmatched if f["item"] not in set(a.leave_out)]
    name = "tile-locations-failed.csv" if bad else "tile-locations.csv"
    with (out / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(found[0]))
        w.writeheader()
        w.writerows(bad or found)
    print(json.dumps({"tiles": len(found), "matched": len(found) - len(unmatched),
                      "left_out": [f["item"] for f in unmatched if f not in bad],
                      "worst_matched_mad": max(f["mad"] for f in found if f["ok"]),
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
            if not int(loc[item]["ok"]):
                raise SystemExit(f"{item} is rated {rating[item]} but has no clear location: it cannot be left out")
            if loc[item]["slide"] != k["slide"]:
                raise SystemExit(f"{item}: key and locations name different slides")
            items.append({"item": item, "slide": k["slide"], "patient": k["patient"], "category": k["category"],
                          "kind": k["kind"], "i": int(loc[item]["tile_index"]), "x": float(loc[item]["x_um"]),
                          "y": float(loc[item]["y_um"]), "label": int(FEATURE[rating[item]][column])})
    items = apply_recheck(items, a) if getattr(a, "recheck_key", None) else items
    return add_new_patients(items, a) if getattr(a, "extra_key", None) else items


def add_new_patients(items: list, a) -> list:
    """Adds the labelled tiles of ``highgrade_mvp/new_patients.py``; refuses a patient already used or held out."""
    if a.feature != "mvp" or not a.extra_labels:
        raise SystemExit("--extra-key needs --extra-labels and --feature mvp")
    key = {r["tile_code"]: r for r in csv.DictReader(open(a.extra_key, newline=""))}
    labels = {r["tile_code"]: (r["label"] or "").strip().lower()
              for r in csv.DictReader(open(a.extra_labels, newline=""))}
    if set(key) != set(labels) or not set(labels.values()) <= {"m", "v", "0", "x"}:
        raise SystemExit("the new-patient labels do not cover their key with m, v, 0 or x")
    used, holdout = {t["patient"] for t in items}, read_patients(a.holdout)
    clash = ({k["patient"] for k in key.values()} & used) | ({k["patient"] for k in key.values()} & holdout)
    if clash:
        raise SystemExit(f"{len(clash)} new patients are already in training or held out")
    seen = set()
    for code, k in key.items():
        slot = (k["slide"], int(k["tile_index"]))
        if slot in seen:
            raise SystemExit(f"{code}: this tile is in the key twice")
        seen.add(slot)
        if labels[code] != "x":
            items.append({"item": code, "slide": k["slide"], "patient": k["patient"], "category": k["category"],
                          "kind": f"new-{k['source']}", "i": slot[1], "x": float(k["x_um"]), "y": float(k["y_um"]),
                          "label": int(labels[code] == "m"), "recheck": labels[code], "old_p": float(k["p_frozen"]),
                          "weight": 1.0, "cohort": "new"})
    return items


def apply_recheck(items: list, a) -> list:
    """High-power labels (``highgrade_mvp/recheck.py``) replace the first MVP labels of the tiles they cover."""
    if a.feature != "mvp" or not a.recheck_labels:
        raise SystemExit("--recheck-key needs --recheck-labels and --feature mvp")
    key = {r["tile_code"]: r for r in csv.DictReader(open(a.recheck_key, newline=""))}
    labels = {r["tile_code"]: (r["label"] or "").strip().lower()
              for r in csv.DictReader(open(a.recheck_labels, newline=""))}
    if set(key) != set(labels) or not set(labels.values()) <= {"m", "v", "0", "x"}:
        raise SystemExit("the recheck labels do not cover the recheck key with m, v, 0 or x")
    by = {t["item"]: t for t in items}
    negatives = sum(t["label"] == 0 for t in items)
    src = Counter(k["source"] for k in key.values())
    control_weight = (negatives - src["hard-negative"]) / src["random-negative"] if src["random-negative"] else 1.0
    for code, k in key.items():
        t = by.get(k["review_item"])
        if t is None or t["slide"] != k["slide"] or t["label"] != int(k["earlier_label"]):
            raise SystemExit(f"{code}: the recheck key does not match these ratings")
        t.update(recheck=labels[code], old_p=float(k["p_out_of_fold"]),
                 weight=control_weight if k["source"] == "random-negative" else 1.0, label=int(labels[code] == "m"))
    return [t for t in items if t.get("recheck") != "x"]


PAIRS = (("mvp_vs_vessel", ("m",), ("v",)), ("mvp_vs_no_vessel", ("m",), ("0",)),
         ("vessel_vs_no_vessel", ("v",), ("0",)), ("mvp_vs_rest", ("m",), ("v", "0")))


def checked_scores(items: list, p: np.ndarray, seed: int, n: int = 2000) -> dict:
    """Out-of-fold AUROC on every tile with a high-power label, overall and per category; 95% CI by patient."""
    from sklearn.metrics import roc_auc_score
    idx = np.array([i for i, t in enumerate(items) if "recheck" in t])
    lab, q = np.array([items[i]["recheck"] for i in idx]), p[idx]
    cats, pats = np.array([items[i]["category"] for i in idx]), np.array([items[i]["patient"] for i in idx])

    def auc(ids, pos, neg):
        ids = ids[np.isin(lab[ids], list(pos + neg))]
        y = np.isin(lab[ids], list(pos)).astype(int)
        return None if len(set(y)) < 2 else float(roc_auc_score(y, q[ids]))

    out, rng = {}, np.random.default_rng(seed)
    for group in ("all", *sorted(set(cats))):
        mask = np.ones(len(idx), bool) if group == "all" else cats == group
        by = defaultdict(list)
        for j in np.flatnonzero(mask):
            by[pats[j]].append(j)
        names, res = sorted(by), {"patients": len(by), "labels": {k: int((lab[mask] == k).sum()) for k in "mv0"}}
        for name, pos, neg in PAIRS:
            value = auc(np.flatnonzero(mask), pos, neg)
            draws = []
            for _ in range(n if value is not None else 0):
                got = auc(np.concatenate([by[names[j]] for j in rng.integers(len(names), size=len(names))]), pos, neg)
                if got is not None:
                    draws.append(got)
            res[name] = {"auroc": None if value is None else round(value, 3), "ci95": [
                round(float(v), 3) for v in np.percentile(draws, [2.5, 97.5])] if draws else None}
        out[group] = res
    return out


def recheck_scores(items: list, p: np.ndarray, seed: int, cohort: str = "first", category: str | None = None,
                   note: str = "hard negatives were chosen by the first head, so the comparison favours the retrained "
                               "head", n: int = 2000) -> dict:
    """Earlier and retrained head on the same tiles of one cohort, with the change's 95% CI by patient."""
    from sklearn.metrics import roc_auc_score
    idx = [i for i, t in enumerate(items) if "recheck" in t and t.get("cohort", "first") == cohort
           and (category is None or t["category"] == category)]
    lab = np.array([items[i]["recheck"] for i in idx])
    old, new = np.array([items[i]["old_p"] for i in idx]), p[idx]
    w = np.array([items[i]["weight"] for i in idx])
    by = defaultdict(list)
    for j, i in enumerate(idx):
        by[items[i]["patient"]].append(j)
    names, rng, out = sorted(by), np.random.default_rng(seed), {}

    def auc(ids, q, pos, neg):
        ids = ids[np.isin(lab[ids], list(pos + neg))]
        y = np.isin(lab[ids], list(pos)).astype(int)
        return None if len(set(y)) < 2 else float(roc_auc_score(y, q[ids], sample_weight=w[ids]))

    every = np.arange(len(idx))
    for name, pos, neg in PAIRS:
        first, again = auc(every, old, pos, neg), auc(every, new, pos, neg)
        diffs = []
        for _ in range(n if first is not None else 0):
            ids = np.concatenate([by[names[j]] for j in rng.integers(len(names), size=len(names))])
            a0, a1 = auc(ids, old, pos, neg), auc(ids, new, pos, neg)
            if a0 is not None and a1 is not None:
                diffs.append(a1 - a0)
        out[name] = {"first_head": None if first is None else round(first, 3),
                     "retrained": None if again is None else round(again, 3),
                     "change_95ci": [round(float(v), 3) for v in np.percentile(diffs, [2.5, 97.5])] if diffs else None}
    out["labels"] = {k: int((lab == k).sum()) for k in ("m", "v", "0", "x")}
    out["note"] = note
    return out


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
    groups = {{"astro4": "grade 4", "astro3": "grade 3"}.get(c, c): cat == c for c in sorted(set(cat))}
    groups.update({"model-picked tiles": kind == "top", "random tiles": kind == "random"})
    metrics = {"feature": a.feature, "status": "pilot: not validated", "rater": a.rater.upper(),
               "patients": len(set(pats)), "slides": len(bags), "folds": k, "folds_grouped_by": "patient",
               "context": list(context), "out_of_fold": {**score(y, p), **patient_ci(y, p, pats, a.seed)},
               "by_group": {name: score(y[m], p[m]) for name, m in groups.items()},
               "positive_patients": len(set(pats[y == 1]))}
    if any("recheck" in t and t.get("cohort", "first") == "first" for t in items):
        metrics["recheck"] = recheck_scores(items, p, a.seed)
    new_cats = sorted({t["category"] for t in items if t.get("cohort") == "new"})
    if new_cats:
        note = "both heads score these patients without training on them; the earlier head is the frozen one"
        metrics["new_patients"] = {"all": recheck_scores(items, p, a.seed, "new", note=note),
                                   **{c: recheck_scores(items, p, a.seed, "new", c, note=note) for c in new_cats}}
        metrics["all_high_power_labels"] = checked_scores(items, p, a.seed)
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
    b.add_argument("--workers", type=int, help="slides read at once (default: the job's cores)")
    b.add_argument("--leave-out", nargs="*", default=[], help="review items allowed not to match (rated unusable)")
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
    t.add_argument("--recheck-key", help="key.csv from highgrade_mvp/recheck.py (MVP only)")
    t.add_argument("--recheck-labels", help="labels CSV from the recheck page: these replace the first MVP labels")
    t.add_argument("--extra-key", help="key.csv from highgrade_mvp/new_patients.py (MVP only)")
    t.add_argument("--extra-labels", help="labels CSV for --extra-key: adds these patients to training")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=train)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
