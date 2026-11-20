"""Steps 3a and 3b: a slide-level grade 4 vs grade 3 model whose tiles point to MVP and necrosis, for picking.

    python grade_mil.py --manifest trial.csv --bags bags/ --out grade-mil [--positive astro4] [--negative astro3]

Under the WHO grading TCGA used, an IDH-mutant astrocytoma was graded 4 (then "glioblastoma") for MVP or necrosis,
and grade 3 for neither; mitoses were allowed in both. So within IDH-mutant astrocytomas, grade 4 against grade 3 is
close to "MVP or necrosis somewhere in the case" against "neither". An attention model (``wsi/abmil.py``) trained on
that slide-level contrast learns which tiles push a slide towards grade 4, with no tile labels.

It is used only to pick tiles for a pathologist to label (``pick.py --grade-model``), never as a label or a head:

* it cannot say which of MVP or necrosis a tile shows, or whether the slide (a sample of the case) has either;
* it can also pick up other grade 4 clues (cellularity, atypia), which the labeller then calls negative.

``--positive`` / ``--negative`` take slide categories (``common.py``). The default keeps the tumour type fixed;
adding ``gbm`` brings many more positives but also IDH-wildtype features unrelated to MVP or necrosis.

Writes ``grade-mil.npz`` (the model) and ``grade-mil.json``: slides per class and the out-of-fold AUC (folds
grouped by patient). An AUC near 0.5 means the model learnt nothing useful: do not pick with it.

**A look without labels** (``--view --slides slides/``): ``view.html`` shows, for up to ``--view-slides`` slides of
each class, the slide with its ``--top`` highest-scoring tiles numbered, those tiles as 3 x 3 fields, and
``--random-tiles`` random tiles for comparison. Every slide is scored by the fold model that did not train on it,
so what it shows is what the model would find on a new slide. If the top tiles on grade 4 slides are mostly MVP and
necrosis, and those on grade 3 slides are not, the approach works; nothing is labelled.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORIES, crop, find_slide, read_manifest, read_patients  # noqa: E402


def tile_scores(model, features: np.ndarray) -> np.ndarray:
    """Each tile's contribution to the slide's grade 4 logit: attention x (tile's own grade 4 - grade 3 logit)."""
    _, cache = model._forward(model._norm(features))
    h, att = cache[2], cache[5]
    return att * (h @ (model.p["Wc"][:, 1] - model.p["Wc"][:, 0]))


def view_slide(rd, bag, top: list, out: Path, code: str, rnd: list, px: int = 768) -> list:
    """Writes the overview (top tiles numbered) and 3 x 3 fields of the top and random tiles; returns image names."""
    from PIL import Image, ImageDraw
    t = bag.tile_um
    lo, hi = bag.coords_um.min(0), bag.coords_um.max(0) + t
    span = float((hi - lo).max())
    cx, cy = (lo + hi) / 2
    im = Image.fromarray(rd.read_um(float(cx), float(cy), span, px))
    d, k = ImageDraw.Draw(im), px / span
    for n, i in enumerate(top, start=1):
        x, y = (bag.coords_um[i] - [cx - span / 2, cy - span / 2]) * k
        d.rectangle([x, y, x + t * k, y + t * k], outline=(220, 30, 30), width=2)
        d.text((x + t * k + 2, y), str(n), fill=(220, 30, 30))
    im.save(out / f"{code}-overview.jpg", quality=85)
    names = [f"{code}-overview.jpg"]
    for tag, idx in (("top", top), ("random", rnd)):
        for n, i in enumerate(idx, start=1):
            x, y = (float(v) for v in bag.coords_um[i])
            crop(rd, x, y, t, 384).save(out / f"{code}-{tag}{n}.jpg", quality=85)
            names.append(f"{code}-{tag}{n}.jpg")
    return names


def view_page(cases: list, info: dict) -> str:
    blocks = []
    for c in cases:
        tops = "".join(f'<figure><img loading="lazy" src="tiles/{c["code"]}-top{n}.jpg" alt=""><figcaption>top {n}'
                       f'</figcaption></figure>' for n in range(1, c["top"] + 1))
        rnds = "".join(f'<figure><img loading="lazy" src="tiles/{c["code"]}-random{n}.jpg" alt=""><figcaption>random '
                       f'{n}</figcaption></figure>' for n in range(1, c["random"] + 1))
        blocks.append(f'<section><h2>{html.escape(c["code"])} <span>{html.escape(c["category"])}, '
                      f'{html.escape(c["diagnosis"])}; P(grade 4) out of fold {c["p"]:.2f}</span></h2>'
                      f'<div class="row"><img class="ov" loading="lazy" src="tiles/{c["code"]}-overview.jpg" alt="">'
                      f'<div class="grid">{tops}</div></div>'
                      + (f'<details><summary>random tiles, for comparison</summary><div class="grid">{rnds}</div>'
                         f'</details>' if rnds else "") + '</section>')
    auc_s = "n/a" if info["out_of_fold_auc"] is None else f'{info["out_of_fold_auc"]:.2f}'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grade model tiles</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}} h2{{font-size:16px;font-weight:500;margin:28px 0 8px}}
h2 span,p,figcaption,summary{{color:var(--muted);font-size:13px;font-weight:400}}
section{{border-top:1px solid var(--line)}}
.row{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:12px}}
@media (max-width:700px){{.row{{grid-template-columns:1fr}}}}
.ov{{width:100%;height:auto;border-radius:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}}
figure{{margin:0}} figure img{{width:100%;height:auto;display:block;border-radius:3px}}
</style></head><body>
<h1>Grade 4 vs grade 3: the tiles that make a slide grade 4</h1>
<p>{", ".join(info["positive"])} against {", ".join(info["negative"])}; {info["slides"]["positive"]} and
{info["slides"]["negative"]} slides; out-of-fold AUC {auc_s}. Each slide is scored by a model that did not train
on it. Red boxes on the overview are the numbered top tiles; each tile is shown as its 3 &times; 3 field, outlined.
Look for MVP and necrosis among the top tiles of grade 4 slides, and their absence on grade 3 slides.</p>
{''.join(blocks)}
</body></html>
"""


def auc(y, s) -> float | None:
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positive", nargs="+", default=["astro4"], choices=CATEGORIES)
    ap.add_argument("--negative", nargs="+", default=["astro3"], choices=CATEGORIES)
    ap.add_argument("--holdout", action="append", default=[], help="CSV of held-out patients: never used")
    ap.add_argument("--max-tiles", type=int, default=2048, help="tiles kept per slide in memory")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="write view.html: each slide's top tiles (needs --slides)")
    ap.add_argument("--slides", help="slide folder, for --view")
    ap.add_argument("--view-slides", type=int, default=15, help="slides per class in view.html")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--random-tiles", type=int, default=4)
    a = ap.parse_args()
    if a.view and not a.slides:
        raise SystemExit("--view needs --slides")

    from entip_glioma.wsi.abmil import ABMIL
    from entip_glioma.wsi.bags import Bag
    holdout = read_patients(a.holdout)
    rows = [r for r in read_manifest(a.manifest)
            if r["category"] in set(a.positive) | set(a.negative) and r.get("label", "glioma") == "glioma"
            and r["PatientID"] not in holdout and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists()]
    bags = [Bag.load(Path(a.bags) / f"{r['SeriesInstanceUID']}.npz", max_tiles=a.max_tiles, seed=a.seed).features
            for r in rows]
    keep = [i for i, b in enumerate(bags) if len(b)]                  # a slide with no tissue tiles has no bag to use
    if len(keep) < len(rows):
        print(f"{len(rows) - len(keep)} slides with no tissue tiles left out", file=sys.stderr)
    rows, bags = [rows[i] for i in keep], [bags[i] for i in keep]
    y = np.array([int(r["category"] in a.positive) for r in rows])
    if min(np.bincount(y, minlength=2)) < 2:
        raise SystemExit(f"need at least 2 slides of each class; have {np.bincount(y, minlength=2).tolist()} "
                         "(negative, positive): check --bags and the manifest's diagnosis columns")
    pats = np.array([r["PatientID"] for r in rows])

    up = np.unique(pats)
    rng = np.random.default_rng(a.seed)
    fold_of = dict(zip(rng.permutation(up), np.arange(len(up)) % max(2, min(a.folds, len(up)))))
    folds = np.array([fold_of[p] for p in pats])
    oof, fold_model = np.full(len(rows), np.nan), {}
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if len(set(y[tr])) < 2:
            continue
        m = ABMIL(2, epochs=a.epochs, seed=a.seed).fit([bags[i] for i in np.flatnonzero(tr)], y[tr])
        oof[te] = m.predict_proba([bags[i] for i in np.flatnonzero(te)])[:, 1]
        fold_model[k] = m
    ok = ~np.isnan(oof)
    model = ABMIL(2, epochs=a.epochs, seed=a.seed).fit(bags, y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "grade-mil.npz")
    info = {"positive": a.positive, "negative": a.negative, "slides": {"positive": int(y.sum()),
            "negative": int(len(y) - y.sum())}, "patients": len(up), "held_out_patients": len(holdout),
            "out_of_fold_auc": auc(y[ok], oof[ok]), "use": "picking tiles to label only"}
    (out / "grade-mil.json").write_text(json.dumps(info, indent=2) + "\n")
    if a.view:
        from entip_train.slides import open_slide
        (out / "tiles").mkdir(exist_ok=True)
        cases = []
        for cls in (1, 0):
            idx = [i for i in np.flatnonzero((y == cls) & ok) if folds[i] in fold_model]
            idx = sorted(rng.permutation(idx)[:a.view_slides], key=lambda i: -oof[i])
            for i in idx:
                r = rows[i]
                bag = Bag.load(Path(a.bags) / f"{r['SeriesInstanceUID']}.npz")
                sc = tile_scores(fold_model[folds[i]], bag.features)
                top = [int(j) for j in np.argsort(-sc)[:a.top]]
                rest = np.setdiff1d(np.arange(len(bag)), top)
                rnd = [int(j) for j in rng.choice(rest, size=min(a.random_tiles, len(rest)), replace=False)]
                code = f"{'G4' if cls else 'G3'}-{len(cases) + 1:03d}"
                rd = open_slide(str(find_slide(Path(a.slides), r["SeriesInstanceUID"])))
                view_slide(rd, bag, top, out / "tiles", code, rnd)
                cases.append({"code": code, "slide": r["SeriesInstanceUID"], "patient": r["PatientID"],
                              "category": r["category"], "diagnosis": r.get("diagnosis", ""), "p": float(oof[i]),
                              "top": len(top), "random": len(rnd)})
                print(f"{code}  {r['PatientID']:<16} {r['category']:<7} p={oof[i]:.2f}", file=sys.stderr)
        if not cases:
            raise SystemExit("no slide had an out-of-fold model to score it")
        (out / "view.html").write_text(view_page(cases, info))
        with (out / "view-key.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cases[0]))
            w.writeheader()
            w.writerows(cases)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
