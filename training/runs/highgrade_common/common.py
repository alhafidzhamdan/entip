"""Shared pieces of the step 3 training modules (3a MVP, 3b necrosis, 3c mitosis).

Slide categories (``category``) come from the manifest's WHO CNS5 ``diagnosis`` and ``who_cns5_basis``, which for
TCGA and CPTAC rest on IDH, 1p/19q and the source histological grade:

=========== ====================================================================================================
``gbm``     Glioblastoma, IDH-wildtype, with glioblastoma histology
``astro4``  Astrocytoma, IDH-mutant, grade 4 (TCGA's IDH-mutant glioblastomas): high grade in a lower-grade
            background
``oligo3``  Oligodendroglioma grade 3: MVP and necrosis can occur, in a lower-grade background
``astro3``  Astrocytic glioma, histological grade 3, unless the source diagnosis (``dx``) says oligoastrocytoma or
            mixed glioma (those go to ``other``: an anaplastic oligoastrocytoma could have MVP)
``grade2``  Diffuse glioma, histological grade 2
``other``   anything else (including IDH-wildtype glioblastomas called on molecular features alone: their
            histological grade and type are not known here)
=========== ====================================================================================================

**Free negatives.** Under the WHO grading TCGA used, necrosis or MVP made an astrocytic glioma a glioblastoma
(grade 4), and a grade 2 glioma had neither. So every slide in ``astro3`` or ``grade2`` (``FREE_NEGATIVE``) has no
necrosis and no MVP, at the specimen level, without labelling. The grade is of the case and the slide is a sample
of it, so a slide can only be wrong through a grading error, which the audit picks (``--audit``) look for. Not for
mitoses: a grade 2 glioma can have one.
"""
from __future__ import annotations

import csv
import json
import random
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attention_review"))
sys.path.insert(0, str(HERE))
from make_review import crop, find_slide  # noqa: E402,F401  (3 x 3 tile field, slide lookup)

CATEGORIES = ("gbm", "astro4", "oligo3", "astro3", "grade2", "other")
FREE_NEGATIVE = {"astro3", "grade2"}
TRAINING_DATA = "ENTIP in-house labels on CC BY slides (TCGA, CPTAC via IDC)"


def category(row: dict) -> str:
    d = (row.get("diagnosis") or "").lower()
    basis = (row.get("who_cns5_basis") or "").lower()
    m = re.search(r"grade (\d)", d)
    g = int(m.group(1)) if m else None
    if "glioblastoma" in d and "idh-wildtype" in d:
        return "other" if "lower-grade histology" in basis else "gbm"
    if "astrocytoma, idh-mutant" in d and g == 4:
        return "astro4"
    if "oligodendroglioma" in d and g == 3:
        return "oligo3"
    if g == 3 and "astrocyt" in d:
        src = (row.get("dx") or "").lower()
        return "other" if ("oligo" in src or "mixed" in src) else "astro3"
    if g == 2 and ("astrocytoma" in d or "oligodendroglioma" in d or "diffuse glioma" in d):
        return "grade2"
    return "other"


def quotas(items) -> dict:
    """``["gbm=12", "oligo3=10"]`` -> ``{"gbm": 12, "oligo3": 10}``."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if k not in CATEGORIES:
            raise SystemExit(f"unknown slide category {k!r}; use one of {', '.join(CATEGORIES)}")
        out[k] = int(v)
    return out


def read_manifest(path) -> list:
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        r["category"] = category(r)
    return rows


def read_patients(paths) -> set:
    """Patients from CSVs with a ``patient`` or ``PatientID`` column (held-out patients)."""
    out = set()
    for p in paths or []:
        for r in csv.DictReader(open(p, newline="")):
            out.add(r.get("patient") or r.get("PatientID"))
    return out - {None, ""}


def pick_slides(rows: list, want: dict, bags, seed: int, holdout: set = frozenset()) -> list:
    """One slide row per patient, ``want[category]`` patients per category, only slides with a bag, never a
    held-out patient. Shuffled, so slide order does not follow category."""
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("label", "glioma") == "glioma" and r["PatientID"] not in holdout \
                and (bags is None or (Path(bags) / f"{r['SeriesInstanceUID']}.npz").exists()):
            by[r["category"]][r["PatientID"]].append(r)
    picked = []
    for cat, n in want.items():
        pats = sorted(by[cat])
        rng.shuffle(pats)
        if len(pats) < n:
            print(f"{cat}: {len(pats)} patients with bags, {n} wanted", file=sys.stderr)
        for p in pats[:n]:
            picked.append(rng.choice(sorted(by[cat][p], key=lambda r: r["SeriesInstanceUID"])))
    rng.shuffle(picked)
    return picked


def xy(x, y) -> tuple:
    return round(float(x), 2), round(float(y), 2)


def read_seen(keys) -> dict:
    """slide -> {(x, y)} already in earlier rounds' key files."""
    seen = defaultdict(set)
    for k in keys or []:
        for r in csv.DictReader(open(k, newline="")):
            seen[r["slide"]].add(xy(r["x_um"], r["y_um"]))
    return seen


class Picker:
    """Collects (index, source) without repeats or earlier-round items."""

    def __init__(self, used=()):
        self.used, self.out = set(used), []

    def add(self, idx, source: str, n: int) -> None:
        k = 0
        for i in idx:
            if k >= n:
                break
            i = int(i)
            if i not in self.used:
                self.used.add(i)
                self.out.append((i, source))
                k += 1


def diverse(features: np.ndarray, k: int, seed: int) -> list:
    """The tile nearest each of k cluster centres of the slide's embeddings."""
    from sklearn.cluster import KMeans
    if k <= 0 or len(features) == 0:
        return []
    k = min(k, len(features))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(features), size=min(4000, len(features)), replace=False)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(features[sub])
    d = np.full((len(features), k), np.inf)
    d[sub] = ((features[sub][:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    return [int(i) for i in np.argmin(d, axis=0)]


def zoom(rd, cx_um: float, cy_um: float, span_um: float, box_um: float | None, px: int, circle: bool = False):
    """A field of ``span_um`` centred on (cx, cy), with the box (or circle) of ``box_um`` marked."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rd.read_um(cx_um, cy_um, span_um, px))
    if box_um:
        d, h = ImageDraw.Draw(im), px * box_um / span_um / 2
        c = px / 2
        shape = d.ellipse if circle else d.rectangle
        shape([c - h, c - h, c + h, c + h], outline=(0, 0, 0), width=3)
        shape([c - h - 3, c - h - 3, c + h + 3, c + h + 3], outline=(255, 255, 255), width=2)
    return im


def write_round(out: Path, key: list, page_html: str, picked_slides: int) -> None:
    if not key:
        raise SystemExit("nothing picked: check --bags, the manifest and the slide quotas")
    with (out / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    (out / "labels.html").write_text(page_html)
    src, cat = defaultdict(int), defaultdict(int)
    for k in key:
        src[k["source"]] += 1
        cat[k.get("category", "")] += 1
    print(json.dumps({"slides": picked_slides, "items": len(key), "by_source": dict(src),
                      "by_category": dict(cat), "out": str(out)}, indent=2))


# ---------------------------------------------------------------- reading labels

def read_labels(keys, label_files, holdout: set = frozenset()):
    """``key`` (code -> key row), ``raw`` (code -> rater -> label), raters. Held-out patients are dropped."""
    key = {}
    for k in keys:
        for r in csv.DictReader(open(k, newline="")):
            key[r["tile_code"]] = r
    raw = defaultdict(dict)
    for p in label_files:
        for r in csv.DictReader(open(p, newline="")):
            if r["tile_code"] not in key:
                raise SystemExit(f"{p}: {r['tile_code']} is not in any --key file")
            if (r.get("label") or "").strip() and key[r["tile_code"]]["patient"] not in holdout:
                raw[r["tile_code"]][r["rater"]] = r["label"].strip().lower()
    raters = sorted({rt for d in raw.values() for rt in d})
    return key, raw, raters


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
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def agreement(raw: dict, raters: list, vote) -> dict:
    """Cohen's kappa for each pair of labellers on items both labelled and both found assessable."""
    out = {}
    for i, r1 in enumerate(raters):
        for r2 in raters[i + 1:]:
            both = [(vote(d[r1]), vote(d[r2])) for d in raw.values() if r1 in d and r2 in d]
            both = [(x, y) for x, y in both if x is not None and y is not None]
            out[f"{r1}-{r2}"] = {"items": len(both), "kappa": kappa([x for x, _ in both], [y for _, y in both])}
    return out


# ---------------------------------------------------------------- writing training files

def square(x: float, y: float, s: float) -> list:
    return [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]


def write_feature(out: Path, name: str, feature: str, labels: dict, tile_um: dict, a, negative_bags=(),
                  context=()) -> dict:
    """``labels``: slide -> (x, y) -> 0/1. Writes ``<name>/<slide>.json`` and ``train-<name>.sh``; returns counts."""
    fdir = out / name
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob("*.json"):
        old.unlink()
    npos = nneg = 0
    for slide, tl in sorted(labels.items()):
        s = tile_um[slide]
        pos = [square(x, y, s) for (x, y), v in sorted(tl.items()) if v == 1]
        neg = [square(x, y, s) for (x, y), v in sorted(tl.items()) if v == 0]
        npos, nneg = npos + len(pos), nneg + len(neg)
        (fdir / f"{slide}.json").write_text(json.dumps({"positives": pos, "negatives": neg,
                                                        "training_data": a.training_data}) + "\n")
    info = {"slides": len(labels), "positive_tiles": npos, "negative_tiles": nneg,
            "slides_with_positives": sum(any(v == 1 for v in tl.values()) for tl in labels.values()),
            "free_negative_slides": len(negative_bags)}
    if a.bags and labels:
        args = ["entip-train", "highgrade-train", "--feature", feature, "--training-data", a.training_data,
                "--out", str(out / f"{name}.joblib")]
        for slide in sorted(labels):
            args += ["--bag", str(Path(a.bags) / f"{slide}.npz"), "--labels", str(fdir / f"{slide}.json")]
        for b in negative_bags:
            args += ["--negative-bag", str(b)]
        if negative_bags:
            args += ["--negatives-per-bag", str(a.negatives_per_bag)]
        if context:
            args += ["--context", *map(str, context)]
        missing = [s for s in labels if not (Path(a.bags) / f"{s}.npz").exists()]
        if missing:
            print(f"{name}: {len(missing)} bags not found in {a.bags}", file=sys.stderr)
        (out / f"train-{name}.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                                              + " ".join(shlex.quote(x) for x in args) + "\n")
    if npos == 0 or nneg + len(negative_bags) == 0:
        print(f"{name}: needs positive and negative tiles before a head can be trained", file=sys.stderr)
    return info


def free_negative_bags(manifest, bags, holdout: set, exclude_slides=()) -> list:
    """Bags of every ``FREE_NEGATIVE`` slide in the manifest, except held-out patients and ``exclude_slides``."""
    if not manifest or not bags:
        return []
    out = []
    for r in read_manifest(manifest):
        p = Path(bags) / f"{r['SeriesInstanceUID']}.npz"
        if r["category"] in FREE_NEGATIVE and r["PatientID"] not in holdout and p.exists() \
                and r["SeriesInstanceUID"] not in exclude_slides:
            out.append(p)
    return sorted(out)


def add_export_args(ap) -> None:
    ap.add_argument("labels", nargs="*", help="labels-<initials>.csv from labels.html")
    ap.add_argument("--key", action="append", default=[], help="key.csv of each round (repeat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=["agree", "majority", "any"], default="agree")
    ap.add_argument("--holdout", action="append", default=[],
                    help="CSV of held-out patients (patient column): never used for training (repeat)")
    ap.add_argument("--training-data", default=TRAINING_DATA, help="recorded in each head")


# ---------------------------------------------------------------- QuPath annotations

def outline_labels(a, classes: dict, decide, colours: dict) -> tuple[dict, dict, dict]:
    """Tile labels from ``--outlines`` (one ``<slide>.geojson`` per slide).

    ``decide(hit, i)`` turns the label hits of tile i into ``{name: 0 | 1 | None}``. Returns labels
    (name -> slide -> (x, y) -> 0/1), tile size per slide, and a summary. With ``--slides`` each slide also gets a
    check image, ``check/<slide>.jpg``."""
    from collections import Counter
    from outlines import draw_check, hits, read_geojson
    from entip_glioma.wsi.bags import Bag
    labels, tile_um = defaultdict(lambda: defaultdict(dict)), {}
    info = {"slides": 0, "annotations": 0, "ignored_classes": Counter(), "tiles": 0}
    if not a.outlines:
        return labels, tile_um, {}
    if not (a.slides or a.mpp):
        raise SystemExit("--outlines needs --slides (for each slide's pixel size) or --mpp")
    files = sorted(Path(a.outlines).glob("*.geojson"))
    if not files:
        raise SystemExit(f"no .geojson files in {a.outlines}")
    held = set()
    if a.holdout:
        if not a.manifest:
            raise SystemExit("--holdout with --outlines needs --manifest (to find the held-out slides)")
        pats = read_patients(a.holdout)
        held = {r["SeriesInstanceUID"] for r in read_manifest(a.manifest) if r["PatientID"] in pats}
    info["skipped_held_out"] = 0
    for f in files:
        slide = f.stem
        if slide in held:
            info["skipped_held_out"] += 1
            continue
        bag = Bag.load(Path(a.bags) / f"{slide}.npz")
        rd = None
        if a.slides:
            from entip_train.slides import open_slide
            rd = open_slide(str(find_slide(Path(a.slides), slide)))
        shapes, other = read_geojson(f, a.mpp or rd.mpp, classes)
        h = hits(bag.coords_um, bag.tile_um, shapes)
        touched = np.flatnonzero(np.any([v for v in h.values()], axis=0)) if h else []
        tile_um[slide] = bag.tile_um
        shown = {}
        for i in touched:
            for name, v in decide(h, int(i)).items():
                if v is not None:
                    labels[name][slide][xy(*bag.coords_um[i])] = v
            shown[int(i)] = next((lab for lab in colours if lab in h and h[lab][i]), None)
        if rd is not None:
            (Path(a.out) / "check").mkdir(parents=True, exist_ok=True)
            draw_check(rd, bag.coords_um, bag.tile_um, shapes, shown, colours, Path(a.out) / "check" / f"{slide}.jpg")
        info["slides"] += 1
        info["annotations"] += len(shapes)
        info["tiles"] += len(touched)
        info["ignored_classes"].update(other)
    info["ignored_classes"] = dict(info["ignored_classes"])
    return labels, tile_um, info


def add_outline_args(ap) -> None:
    ap.add_argument("--bags", required=True, help="bag folder (H-optimus-0 bags the heads train on)")
    ap.add_argument("--manifest", help="trial.csv, for the free-negative slides")
    ap.add_argument("--outlines", help="folder of QuPath GeoJSON exports, one <SeriesInstanceUID>.geojson per slide")
    ap.add_argument("--slides", help="slide folder: pixel size of outlined slides, and a check image each")
    ap.add_argument("--mpp", type=float, help="pixel size (µm) of the outlined slides, instead of --slides")
    ap.add_argument("--negatives-per-bag", type=int, default=200, help="random tiles per free-negative slide")
    ap.add_argument("--no-free-negatives", action="store_true", help="use labelled tiles only")


def merge_tile_labels(labels: dict, tile_um: dict, key: dict, raw: dict, raters: list, name: str, vote,
                      rule: str) -> dict:
    """Add consensus page labels for one feature (``vote(label) -> 0 | 1 | None``) to ``labels[name]``; a page
    label wins over an outline. Returns counts."""
    from collections import Counter
    by_source, dropped, overruled = Counter(), 0, 0
    from_outlines = sum(len(v) for v in labels[name].values())
    for t, d in raw.items():
        c = consensus([vote(d.get(rt)) if rt in d else None for rt in raters], rule)
        if c is None:
            dropped += any(vote(v) is not None for v in d.values())
            continue
        r = key[t]
        p = xy(r["x_um"], r["y_um"])
        old = labels[name][r["slide"]].get(p)
        overruled += old is not None and old != c
        labels[name][r["slide"]][p] = c
        tile_um[r["slide"]] = float(r["tile_um"])
        by_source[f"{r['source'].split(':')[0]}:{'pos' if c else 'neg'}"] += 1
    return {"tiles_from_outlines": from_outlines, "outline_tiles_overruled_by_page": overruled,
            "left_out_disagreement": dropped, "by_source": dict(sorted(by_source.items()))}
