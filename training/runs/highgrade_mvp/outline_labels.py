"""Step 3a: turn the locked slide-pilot calls and outlines into tile labels for a new MVP head.

    python outline_labels.py --pilot mvp-pilot1 --calls mvp-pilot1/slide-calls-AH.csv --bags bags/ --out mvp-outlines1

**Positives** are tiles whose centre lies inside an outline, or that hold the outline's centroid (the rule of
``regions.tile_labels``), at most ``--max-positive`` per slide. **Negatives** are random tiles from slides called absent,
``--absent-tiles`` per slide. Outlines were not exhaustive, so other tissue on a present slide stays unlabelled, and
can't-tell slides are left out. The model's areas are never used as labels.

Writes ``key.csv`` and ``labels.csv`` for ``tile_heads.py train --similar``. Once these patients are trained on, the pilot
no longer tests the new head, which needs patients it has never seen.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_reveal import inside, sha256  # noqa: E402

SOURCES = {"present": "outline", "absent": "absent-slide"}


def positive_tiles(coords: np.ndarray, tile_um: float, poly: np.ndarray) -> set:
    """Tiles whose centre is inside the outline, or that contain its centroid."""
    (ax, ay), (bx, by) = poly.min(0), poly.max(0)
    gx, gy = poly.mean(0)
    near = np.flatnonzero((coords[:, 0] + tile_um >= ax) & (coords[:, 0] <= bx) &
                          (coords[:, 1] + tile_um >= ay) & (coords[:, 1] <= by))
    out = set()
    for i in near:
        x, y = coords[i]
        if (x <= gx < x + tile_um and y <= gy < y + tile_um) or inside(poly, x + tile_um / 2, y + tile_um / 2):
            out.add(int(i))
    return out


def main() -> int:
    from entip_glioma.wsi.bags import Bag
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", required=True, help="the pilot folder holding private/ and results/")
    ap.add_argument("--calls", required=True, help="the locked slide-calls CSV")
    ap.add_argument("--bags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-positive", type=int, default=30, help="most positive tiles from one slide")
    ap.add_argument("--absent-tiles", type=int, default=40, help="random negative tiles from each absent slide")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, pilot = Path(a.out), Path(a.pilot)
    if (out / "key.csv").exists():
        raise SystemExit(f"{out / 'key.csv'} already exists: these labels are never redrawn")
    done = pilot / "results" / "pilot-results.json"
    if not done.exists() or json.loads(done.read_text()).get("calls_sha256") != sha256(a.calls):
        raise SystemExit("these calls are not the ones the pilot comparison locked")
    keys = {r["case"]: r for r in csv.DictReader(open(pilot / "private" / "key.csv", newline=""))}
    calls = {r["case"]: r for r in csv.DictReader(open(a.calls, newline=""))}
    if set(keys) != set(calls):
        raise SystemExit("the calls do not cover the same cases as the key")
    rows, per_slide = [], {}
    for n, case in enumerate(sorted(keys)):
        k, call = keys[case], calls[case]["call"]
        if call not in SOURCES:
            continue
        bag = Bag.load(Path(a.bags) / f"{k['slide']}.npz")
        if float(k["tile_um"]) != bag.tile_um:
            raise SystemExit(f"{case}: the bag's tile size differs from the key")
        rng = np.random.default_rng([a.seed, n])
        if call == "present":
            origin = np.array([float(k["origin_x_um"]), float(k["origin_y_um"])])
            hit = set()
            for g in json.loads(calls[case].get("regions_um") or "[]"):
                hit |= positive_tiles(bag.coords_um, bag.tile_um, np.asarray(g, float) + origin)
            pool, cap, label = np.array(sorted(hit), int), a.max_positive, "m"
        else:
            pool, cap, label = np.arange(len(bag)), a.absent_tiles, "0"
        chosen = np.sort(rng.choice(pool, min(cap, len(pool)), replace=False)) if len(pool) else pool
        per_slide[case] = {"category": k["category"], "call": call, "eligible": int(len(pool)), "used": int(len(chosen))}
        for i in chosen:
            rows.append({"slide": k["slide"], "patient": k["patient"], "category": k["category"],
                         "source": SOURCES[call], "tile_index": int(i), "x_um": float(bag.coords_um[i, 0]),
                         "y_um": float(bag.coords_um[i, 1]), "label": label})
    if not any(r["label"] == "m" for r in rows) or not any(r["label"] == "0" for r in rows):
        raise SystemExit("the calls gave no positive or no negative tiles")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    for n, r in enumerate(rows, start=1):
        r["tile_code"] = f"mvp-outlines1-T{n:05d}"
    with (out / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_code", "slide", "patient", "category", "source", "tile_index", "x_um", "y_um"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_code", "label"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    used = [c for c in per_slide.values()]
    info = {"status": "training labels from one rater; these patients no longer test a head trained on them",
            "calls_sha256": sha256(a.calls), "slides_used": len(used), "patients": len({r["patient"] for r in rows}),
            "positive_tiles": sum(r["label"] == "m" for r in rows), "negative_tiles": sum(r["label"] == "0" for r in rows),
            "positive_tiles_before_cap": sum(c["eligible"] for c in used if c["call"] == "present"),
            "positive_slides_without_a_tile": sum(c["call"] == "present" and not c["used"] for c in used),
            "tiles_by_category_and_label": {f"{c}/{l}": v for (c, l), v in
                                            sorted(Counter((r["category"], r["label"]) for r in rows).items())},
            "max_positive_per_slide": a.max_positive, "absent_tiles_per_slide": a.absent_tiles}
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
