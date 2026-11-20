"""Step 3b: turn necrosis outlines (QuPath) and page labels into training files for the necrosis head.

    python export.py --outlines qupath/ --slides slides/ --bags bags/ --manifest trial.csv --out export-r1 \
        [--key nec-r1/key.csv labels-AH.csv] [--holdout held_out.csv] [--context 1 2]

QuPath classes: ``Necrosis`` (or *Tumour necrosis*) and ``Neither`` (tumour without necrosis, for negatives);
outline some ``Neither`` on every slide, including look-alikes next to necrosis. Page labels: ``n`` -> 1, ``0`` -> 0,
``x`` left out. A page label wins over an outline: it was judged on its own.

**Free negatives.** ``--free-negatives free-r1 --free-review review-AH.csv``: the tiles of free-negative slides
(``astro3``, ``grade2``: no necrosis by their grade) picked by ``highgrade_common/free_negatives.py`` and checked by
eye on its page. A slide where anyone flagged necrosis there, or marked it on this page, is left out and listed in
``summary.json``.

The head gets each tile's neighbourhood (``--context``, default 1 2: the mean embeddings of the 3 x 3 and 5 x 5
fields), so it sees the edge of a necrosis and the tissue beside it without re-embedding.

Writes ``necrosis/<slide>.json``, ``train-necrosis.sh``, ``check/<slide>.jpg`` per outlined slide (necrosis red,
neither grey: look at one first) and ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import (add_export_args, add_outline_args, agreement, merge_tile_labels,  # noqa: E402
                    outline_labels, read_labels, read_patients, write_feature)

CLASSES = {"necrosis": "necrosis", "tumour necrosis": "necrosis", "tumor necrosis": "necrosis",
           "neither": "neither", "negative": "neither", "not necrosis": "neither"}
COLOURS = {"necrosis": (220, 40, 40), "neither": (90, 90, 90)}


def vote(lab):
    return None if lab in (None, "", "x") else int(lab == "n")


def decide(h, i):
    hit = {k: bool(v[i]) for k, v in h.items()}
    return {"necrosis": 1 if hit.get("necrosis") else 0 if hit.get("neither") else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_export_args(ap)
    add_outline_args(ap)
    ap.add_argument("--context", type=int, nargs="*", default=[1, 2], help="neighbourhood radii in tiles")
    a = ap.parse_args()
    if not (a.labels and a.key) and not a.outlines:
        raise SystemExit("give --outlines, or page labels with their --key files, or both")
    holdout = read_patients(a.holdout)
    key, raw, raters = read_labels(a.key, a.labels, holdout)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, tile_um, oinfo = outline_labels(a, CLASSES, decide, COLOURS)
    counts = merge_tile_labels(labels, tile_um, key, raw, raters, "necrosis", vote, a.rule)
    with_nec = sorted(s for s, tl in labels["necrosis"].items() if any(v == 1 for v in tl.values()))
    flagged = sorted({k["slide"] for k in key.values() if k.get("category") in ("astro3", "grade2")} & set(with_nec))
    finfo = None
    if a.free_negatives:
        from free_negatives import merge_free, read_review
        free, free_um, finfo = read_review(a.free_negatives, a.free_review, holdout)
        finfo["tiles_added"] = merge_free(labels["necrosis"], tile_um, free["necrosis"], free_um, exclude=with_nec)
    summary = {"raters": raters, "rule": a.rule, "items_labelled": len(raw), "held_out_patients": len(holdout),
               "outlines": oinfo, "context": a.context, "free_negative_slides_with_necrosis": flagged,
               "free_negatives": finfo,
               "features": {"necrosis": {
                   **write_feature(out, "necrosis", "necrosis", labels["necrosis"], tile_um, a, a.context),
                   **counts, "agreement": agreement(raw, raters, vote)}}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
