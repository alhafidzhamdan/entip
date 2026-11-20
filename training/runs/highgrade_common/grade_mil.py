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
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CATEGORIES, read_manifest, read_patients  # noqa: E402


def tile_scores(model, features: np.ndarray) -> np.ndarray:
    """Each tile's contribution to the slide's grade 4 logit: attention x (tile's own grade 4 - grade 3 logit)."""
    _, cache = model._forward(model._norm(features))
    h, att = cache[2], cache[5]
    return att * (h @ (model.p["Wc"][:, 1] - model.p["Wc"][:, 0]))


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
    a = ap.parse_args()

    from entip_glioma.wsi.abmil import ABMIL
    from entip_glioma.wsi.bags import Bag
    holdout = read_patients(a.holdout)
    rows = [r for r in read_manifest(a.manifest)
            if r["category"] in set(a.positive) | set(a.negative) and r.get("label", "glioma") == "glioma"
            and r["PatientID"] not in holdout and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists()]
    y = np.array([int(r["category"] in a.positive) for r in rows])
    if min(np.bincount(y, minlength=2)) < 2:
        raise SystemExit(f"need at least 2 slides of each class; have {np.bincount(y, minlength=2).tolist()} "
                         "(grade 3, grade 4): check --bags and the manifest's diagnosis columns")
    bags = [Bag.load(Path(a.bags) / f"{r['SeriesInstanceUID']}.npz", max_tiles=a.max_tiles, seed=a.seed).features
            for r in rows]
    pats = np.array([r["PatientID"] for r in rows])

    up = np.unique(pats)
    rng = np.random.default_rng(a.seed)
    fold_of = dict(zip(rng.permutation(up), np.arange(len(up)) % max(2, min(a.folds, len(up)))))
    folds = np.array([fold_of[p] for p in pats])
    oof = np.full(len(rows), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if len(set(y[tr])) < 2:
            continue
        m = ABMIL(2, epochs=a.epochs, seed=a.seed).fit([bags[i] for i in np.flatnonzero(tr)], y[tr])
        oof[te] = m.predict_proba([bags[i] for i in np.flatnonzero(te)])[:, 1]
    ok = ~np.isnan(oof)
    model = ABMIL(2, epochs=a.epochs, seed=a.seed).fit(bags, y)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "grade-mil.npz")
    info = {"positive": a.positive, "negative": a.negative, "slides": {"positive": int(y.sum()),
            "negative": int(len(y) - y.sum())}, "patients": len(up), "held_out_patients": len(holdout),
            "out_of_fold_auc": auc(y[ok], oof[ok]), "use": "picking tiles to label only"}
    (out / "grade-mil.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
