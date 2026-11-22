"""Step 3a: find tiles that look like confirmed MVP across the already embedded slides, for a blinded review.

    python retrieve.py --labelled recheck/key.csv recheck/labels.csv --labelled new1/key.csv new1/labels.csv \
        --reviewed tile-locations.csv --manifest trial.csv --bags bags/ --slides slides/ --holdout reserved.csv \
        --out mvp-retrieve1

Every tile labelled ``m`` at high power is a reference. A tile's score is its mean cosine similarity to the closest MVP
reference from each of its ``--top-patients`` best-matching reference patients, never counting its own patient. So a
high score means it resembles MVP from several patients, not one patient's stain or tissue.

Held-out patients are never searched. Tiles already reviewed, and their neighbours, are skipped. The page holds the
``--near`` highest-scoring tiles (at most ``--per-patient`` a patient, not touching each other) and ``--controls`` random
tiles, shuffled, with the labels of ``export.py``. Nothing is labelled automatically. It shows no slide, grade, score or
reason for picking; keep ``key.csv`` back.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
sys.path.insert(0, str(HERE))
from common import crop, find_slide, read_manifest, read_patients, write_round, zoom  # noqa: E402
from labelpage import page  # noqa: E402
from recheck import CHOICES, INSTRUCTIONS, data_uri  # noqa: E402


def unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def read_references(pairs: list) -> tuple[list, dict]:
    """MVP references, and slide -> coordinates of every tile labelled in these rounds."""
    refs, seen = [], defaultdict(list)
    for key_path, label_path in pairs:
        key = {r["tile_code"]: r for r in csv.DictReader(open(key_path, newline=""))}
        labels = {r["tile_code"]: (r["label"] or "").strip().lower() for r in csv.DictReader(open(label_path, newline=""))}
        if set(key) != set(labels) or not set(labels.values()) <= {"m", "v", "0", "x"}:
            raise SystemExit(f"{label_path} does not cover {key_path} with m, v, 0 or x")
        for code, k in key.items():
            seen[k["slide"]].append((float(k["x_um"]), float(k["y_um"])))
            if labels[code] == "m":
                refs.append({"code": code, "slide": k["slide"], "patient": k["patient"], "i": int(k["tile_index"]),
                             "x": float(k["x_um"]), "y": float(k["y_um"])})
    return refs, seen


def similarity(x: np.ndarray, refs: np.ndarray, ref_patients: np.ndarray, own: str, top_patients: int) -> np.ndarray:
    """Mean of the best cosine similarity to each of the closest reference patients, excluding the tile's own."""
    others = sorted(set(ref_patients) - {own})
    if len(others) < top_patients:
        raise SystemExit(f"only {len(others)} reference patients besides {own}; --top-patients is {top_patients}")
    sim = x @ refs.T
    best = np.stack([sim[:, ref_patients == q].max(1) for q in others], axis=1)
    return np.partition(best, -top_patients, axis=1)[:, -top_patients:].mean(1)


def free_tiles(coords: np.ndarray, reviewed: list, tile_um: float) -> np.ndarray:
    """True for tiles that neither are nor touch an already reviewed tile."""
    ok = np.ones(len(coords), bool)
    for x, y in reviewed:
        ok &= ~((np.abs(coords[:, 0] - x) <= 1.01 * tile_um) & (np.abs(coords[:, 1] - y) <= 1.01 * tile_um))
    return ok


def spaced(order, coords: np.ndarray, tile_um: float, n: int) -> list:
    """The first ``n`` tiles of ``order`` that do not touch one another."""
    out = []
    for i in order:
        i = int(i)
        if all(max(abs(coords[i, 0] - coords[j, 0]), abs(coords[i, 1] - coords[j, 1])) > 1.01 * tile_um for j in out):
            out.append(i)
            if len(out) == n:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labelled", nargs=2, action="append", required=True, metavar=("KEY", "LABELS"),
                    help="a high-power round's key and labels (repeat)")
    ap.add_argument("--reviewed", action="append", default=[], help="other CSVs of reviewed tiles (slide, x_um, y_um)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--holdout", action="append", required=True, help="CSV of held-out patients: never searched")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-references", type=int, help="stop unless exactly this many MVP references are found")
    ap.add_argument("--top-patients", type=int, default=3)
    ap.add_argument("--near", type=int, default=150)
    ap.add_argument("--controls", type=int, default=30)
    ap.add_argument("--per-patient", type=int, default=6)
    ap.add_argument("--per-slide-pool", type=int, default=20, help="best spaced tiles kept from each slide")
    ap.add_argument("--round", default="retrieve1", help="prefix for tile codes, unique per round")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    out = Path(a.out)
    if (out / "key.csv").exists():
        raise SystemExit(f"{out / 'key.csv'} already exists: a built round is never overwritten")
    held = read_patients(a.holdout)
    if not held:
        raise SystemExit("--holdout names no patients")
    refs, reviewed = read_references(a.labelled)
    if a.expect_references is not None and len(refs) != a.expect_references:
        raise SystemExit(f"found {len(refs)} MVP references, expected {a.expect_references}")
    if {r["patient"] for r in refs} & held:
        raise SystemExit("an MVP reference comes from a held-out patient")
    for p in a.reviewed:
        for r in csv.DictReader(open(p, newline="")):
            reviewed[r["slide"]].append((float(r["x_um"]), float(r["y_um"])))

    samples, vectors = [], {}
    rng = np.random.default_rng(a.seed)
    by_slide = defaultdict(list)
    for r in refs:
        by_slide[r["slide"]].append(r)
    for slide, rows in sorted(by_slide.items()):
        bag = Bag.load(Path(a.bags) / f"{slide}.npz")
        for r in rows:
            if not np.allclose(bag.coords_um[r["i"]], [r["x"], r["y"]], atol=0.01):
                raise SystemExit(f"{r['code']}: the bag's tile is not at the labelled position")
            vectors[r["code"]] = bag.features[r["i"]].copy()
        samples.append(bag.features[rng.choice(len(bag), size=min(2000, len(bag)), replace=False)])
    centre = np.concatenate(samples).mean(0)
    ref_x = unit(np.stack([vectors[r["code"]] for r in refs]) - centre)
    ref_patients = np.array([r["patient"] for r in refs])

    rows = [r for r in read_manifest(a.manifest) if r.get("label", "glioma") == "glioma" and r["PatientID"] not in held
            and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists()]
    pool, randoms = [], []
    for n, r in enumerate(rows, start=1):
        uid = r["SeriesInstanceUID"]
        bag = Bag.load(Path(a.bags) / f"{uid}.npz")
        if len(bag) == 0:
            continue
        free = np.flatnonzero(free_tiles(bag.coords_um, reviewed.get(uid, []), bag.tile_um))
        if len(free) == 0:
            continue
        score = similarity(unit(bag.features[free] - centre), ref_x, ref_patients, r["PatientID"], a.top_patients)
        order = free[np.argsort(-score, kind="stable")]
        by_tile = dict(zip(free.tolist(), score.tolist()))
        make = lambda i, source: {"round": a.round, "patient": r["PatientID"], "slide": uid, "category": r["category"],
                                  "diagnosis": r.get("diagnosis", ""), "source": source,
                                  "similarity": round(by_tile[i], 5), "tile_index": i,
                                  "x_um": float(bag.coords_um[i, 0]), "y_um": float(bag.coords_um[i, 1]),
                                  "tile_um": bag.tile_um}
        pool += [make(i, "near") for i in spaced(order[:500], bag.coords_um, bag.tile_um, a.per_slide_pool)]
        randoms += [make(int(i), "control") for i in rng.choice(free, size=min(2, len(free)), replace=False)]
        print(f"searched {n}/{len(rows)} slides", file=sys.stderr, flush=True)
    if not pool:
        raise SystemExit("no tiles to search: check --bags, the manifest and --holdout")

    chosen, per = [], Counter()
    for c in sorted(pool, key=lambda c: (-c["similarity"], c["slide"], c["tile_index"])):
        if len(chosen) == a.near:
            break
        if per[c["patient"]] < a.per_patient:
            per[c["patient"]] += 1
            chosen.append(c)
    used = {(c["slide"], c["tile_index"]) for c in chosen}
    spare = [c for c in randoms if (c["slide"], c["tile_index"]) not in used]
    chosen += random.Random(a.seed).sample(spare, min(a.controls, len(spare)))
    if {c["patient"] for c in chosen} & held:
        raise SystemExit("a held-out patient was picked")
    random.Random(f"{a.seed}-order").shuffle(chosen)
    for n, c in enumerate(chosen, start=1):
        c["tile_code"] = f"mvp-{a.round}-T{n:03d}"

    images, mpp, groups = {}, {}, defaultdict(list)
    for c in chosen:
        groups[c["slide"]].append(c)
    for n, (slide, tiles) in enumerate(sorted(groups.items()), start=1):
        rd = open_slide(str(find_slide(Path(a.slides), slide)))
        mpp[slide] = float(rd.mpp)
        for c in tiles:
            x, y, t = c["x_um"], c["y_um"], c["tile_um"]
            images[c["tile_code"]] = [data_uri(zoom(rd, x + t / 2, y + t / 2, 1.5 * t, t, 640), 88),
                                      data_uri(crop(rd, x, y, t, 384), 85)]
        print(f"rendered {n}/{len(groups)} slides", file=sys.stderr, flush=True)
    key = [{"tile_code": c["tile_code"], **{k: v for k, v in c.items() if k != "tile_code"},
            "slide_mpp": mpp[c["slide"]]} for c in chosen]
    items = [{"code": c["tile_code"], "images": images[c["tile_code"]]} for c in chosen]
    out.mkdir(parents=True, exist_ok=True)
    html = page(f"entip-mvp-{a.round}-{a.seed}-{len(key)}", "MVP similar-tile review",
                [{"slide": "All tiles", "items": items}], CHOICES, INSTRUCTIONS, column_px=1000)
    write_round(out, key, html, len(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
