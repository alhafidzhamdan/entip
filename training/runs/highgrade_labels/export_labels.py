"""Turn tile labels from labels.html into step 3 training files, and report how well labellers agree.

    python export_labels.py --key labels-r1/key.csv [--key labels-r2/key.csv] labels-AH.csv labels-XY.csv \
        --out highgrade-labels [--rule agree|majority|any] [--bags bags/] \
        [--outlines qupath/ --slides slides/]

Tile labels and QuPath outlines (``--outlines``, see ``outlines.py``) can be used together or alone. Where both
label a tile, the tile label wins: it was judged on its own.

Each labeller's CSV holds one row per tile: ``n`` necrosis, ``m`` MVP, ``nm`` both, ``0`` neither, ``x`` not
assessable, blank not labelled. For each feature a tile is positive (1), negative (0) or left out: not assessable
or unlabelled tiles are left out, and so are tiles where labellers disagree under ``--rule agree`` (the default;
``majority`` keeps a strict majority, ``any`` makes a tile positive if anyone marked it). With one labeller their
label is used.

Writes, in ``--out``, for each feature (``necrosis``, ``microvascular_proliferation``):

* ``<feature>/<slide>.json``: ``{"positives": [...], "negatives": [...]}``, each tile as its square in µm, the
  input ``entip-train highgrade-train --labels`` takes (a tile's centre is inside its own square only);
* ``train-<feature>.sh``: the training command for those slides (needs ``--bags``);

and ``summary.json``: labelled tiles, positives and negatives per feature, Cohen's kappa for each pair of
labellers on the tiles both found assessable, and the counts by why the tile was picked.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

FEATURES = {"necrosis": "n", "microvascular_proliferation": "m"}


def vote(label: str, letter: str):
    """1 / 0 for this feature, or None (not assessable or not labelled)."""
    label = (label or "").strip().lower()
    if label in ("", "x"):
        return None
    return int(letter in label)


def consensus(votes: list, rule: str):
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    pos = sum(votes)
    if rule == "any":
        return int(pos > 0)
    if rule == "majority":
        return 1 if pos * 2 > len(votes) else 0 if pos * 2 < len(votes) else None
    return votes[0] if len(set(votes)) == 1 else None          # agree


def kappa(a: list, b: list):
    """Cohen's kappa for two lists of 0/1; None when undefined."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def square(x: float, y: float, s: float) -> list:
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


def outline_labels(a, labels: dict, tile_um: dict, summary: dict) -> None:
    """Add tile labels from QuPath outlines (``--outlines``: one ``<slide>.geojson`` per slide)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from outlines import draw_check, read_geojson, tile_votes
    from entip_glioma.wsi.bags import Bag
    if not a.bags:
        raise SystemExit("--outlines needs --bags (to find the tiles each outline covers)")
    if not (a.slides or a.mpp):
        raise SystemExit("--outlines needs --slides (for each slide's pixel size) or --mpp")
    files = sorted(Path(a.outlines).glob("*.geojson"))
    if not files:
        raise SystemExit(f"no .geojson files in {a.outlines}")
    info = {"slides": 0, "outlines": 0, "ignored_classes": Counter(), "tiles": 0}
    for f in files:
        slide = f.stem
        bag = Bag.load(Path(a.bags) / f"{slide}.npz")
        rd = None
        if a.slides:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attention_review"))
            from make_review import find_slide
            from entip_train.slides import open_slide
            rd = open_slide(str(find_slide(Path(a.slides), slide)))
        shapes, other = read_geojson(f, a.mpp or rd.mpp)
        votes = tile_votes(bag.coords_um, bag.tile_um, shapes)
        tile_um[slide] = bag.tile_um
        for i, v in votes.items():
            xy = (round(float(bag.coords_um[i][0]), 2), round(float(bag.coords_um[i][1]), 2))
            for feat, letter in FEATURES.items():
                if v[letter] is not None:
                    labels[feat][slide][xy] = v[letter]
        if rd is not None:
            (Path(a.out) / "check").mkdir(parents=True, exist_ok=True)
            draw_check(rd, bag.coords_um, bag.tile_um, shapes, votes, Path(a.out) / "check" / f"{slide}.jpg")
        info["slides"] += 1
        info["outlines"] += len(shapes)
        info["tiles"] += len(votes)
        info["ignored_classes"].update(other)
    info["ignored_classes"] = dict(info["ignored_classes"])
    summary["outlines"] = info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="*", help="labels-<initials>.csv from labels.html")
    ap.add_argument("--key", action="append", default=[], help="key.csv of each round (repeat)")
    ap.add_argument("--outlines", help="folder of QuPath GeoJSON exports, one <SeriesInstanceUID>.geojson per slide")
    ap.add_argument("--slides", help="slide folder: each outlined slide's pixel size, and a check image")
    ap.add_argument("--mpp", type=float, help="pixel size (µm) of the outlined slides, instead of --slides")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=["agree", "majority", "any"], default="agree")
    ap.add_argument("--bags", help="bag folder, to write the training commands (and needed for --outlines)")
    ap.add_argument("--training-data", default="ENTIP in-house tile labels on CC BY slides (TCGA, CPTAC via IDC)",
                    help="recorded in each head as the name and licence of its labels")
    a = ap.parse_args()
    if not (a.labels and a.key) and not a.outlines:
        raise SystemExit("give tile labels with their --key files, or --outlines, or both")
    if a.labels and not a.key:
        raise SystemExit("tile labels need the --key file of each round")

    key = {}
    for k in a.key:
        for r in csv.DictReader(open(k, newline="")):
            key[r["tile_code"]] = r
    raw = defaultdict(dict)                                     # tile -> rater -> label
    for p in a.labels:
        for r in csv.DictReader(open(p, newline="")):
            if r["tile_code"] not in key:
                raise SystemExit(f"{p}: tile {r['tile_code']} is not in any --key file")
            if (r.get("label") or "").strip():
                raw[r["tile_code"]][r["rater"]] = r["label"]
    raters = sorted({rt for d in raw.values() for rt in d})
    if a.labels and not raters:
        raise SystemExit("no labels in the CSV files")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"raters": raters, "rule": a.rule, "tiles_in_keys": len(key), "tiles_labelled": len(raw),
               "not_assessable": sum(all(v.strip().lower() == "x" for v in d.values()) for d in raw.values()),
               "features": {}}
    labels = {f: defaultdict(dict) for f in FEATURES}           # feature -> slide -> (x, y) -> 0/1
    tile_um = {}
    if a.outlines:
        outline_labels(a, labels, tile_um, summary)
    for feat, letter in FEATURES.items():
        from_outlines = sum(len(v) for v in labels[feat].values())
        by_source, dropped, overruled = Counter(), 0, 0
        for t, d in raw.items():
            c = consensus([vote(d.get(rt), letter) for rt in raters], a.rule)
            if c is None:
                dropped += any(vote(v, letter) is not None for v in d.values())
                continue
            r = key[t]
            xy = (round(float(r["x_um"]), 2), round(float(r["y_um"]), 2))
            old = labels[feat][r["slide"]].get(xy)
            overruled += old is not None and old != c
            labels[feat][r["slide"]][xy] = c                    # a tile judged on its own wins over an outline
            tile_um[r["slide"]] = float(r["tile_um"])
            by_source[(r["source"].split(":")[0], c)] += 1
        pairs = {}
        for r1, r2 in combinations(raters, 2):
            both = [(vote(d[r1], letter), vote(d[r2], letter)) for d in raw.values() if r1 in d and r2 in d]
            both = [(x, y) for x, y in both if x is not None and y is not None]
            pairs[f"{r1}-{r2}"] = {"tiles": len(both), "kappa": kappa([x for x, _ in both], [y for _, y in both])}
        per_slide = {}
        for slide, tl in labels[feat].items():
            s = tile_um[slide]
            per_slide[slide] = {"positives": [square(x, y, s) for (x, y), v in sorted(tl.items()) if v == 1],
                                "negatives": [square(x, y, s) for (x, y), v in sorted(tl.items()) if v == 0]}
        fdir = out / feat
        fdir.mkdir(parents=True, exist_ok=True)
        for old in fdir.glob("*.json"):
            old.unlink()
        for slide, lab in sorted(per_slide.items()):
            (fdir / f"{slide}.json").write_text(json.dumps({**lab, "training_data": a.training_data}) + "\n")
        npos = sum(len(v["positives"]) for v in per_slide.values())
        nneg = sum(len(v["negatives"]) for v in per_slide.values())
        summary["features"][feat] = {
            "slides": len(per_slide), "positive_tiles": npos, "negative_tiles": nneg,
            "slides_with_positives": sum(bool(v["positives"]) for v in per_slide.values()),
            "tiles_from_outlines": from_outlines, "outline_tiles_overruled_by_tile_labels": overruled,
            "left_out_disagreement": dropped, "agreement": pairs,
            "by_source": {f"{s}:{'pos' if c else 'neg'}": n for (s, c), n in sorted(by_source.items())}}
        if a.bags and per_slide:
            args = ["entip-train", "highgrade-train", "--feature", feat, "--training-data", a.training_data,
                    "--out", str(out / f"{feat}.joblib")]
            for slide in sorted(per_slide):
                args += ["--bag", str(Path(a.bags) / f"{slide}.npz"), "--labels", str(fdir / f"{slide}.json")]
            missing = [s for s in per_slide if not (Path(a.bags) / f"{s}.npz").exists()]
            if missing:
                print(f"{feat}: {len(missing)} bags not found in {a.bags}", file=sys.stderr)
            (out / f"train-{feat}.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                                                  + " ".join(shlex.quote(x) for x in args) + "\n")
        if npos == 0 or nneg == 0:
            print(f"{feat}: needs both positive and negative tiles before a head can be trained", file=sys.stderr)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
