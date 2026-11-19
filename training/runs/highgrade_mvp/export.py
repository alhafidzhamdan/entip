"""Step 3a: turn MVP labels (page, QuPath points or outlines) into training files for the MVP and vessel heads.

    python export.py --key mvp-r1/key.csv labels-AH.csv labels-XY.csv --bags bags/ --manifest trial.csv \
        --out export-r1 [--outlines qupath/ --slides slides/] [--holdout held_out.csv]

From each tile label: MVP head ``m`` -> 1, ``v`` / ``0`` -> 0; vessel head ``m`` / ``v`` -> 1, ``0`` -> 0; ``x``
is left out of both. QuPath classes: ``MVP`` (or *Microvascular proliferation*), ``Vessel`` (a vessel without
MVP) and ``Neither``; points (one click per vessel) or outlines. ``Neither`` is a negative for MVP only, since
tumour with neither can still hold ordinary vessels.

**Free negatives.** With ``--manifest``, every free-negative slide (``astro3``, ``grade2``: no MVP by its grade)
adds ``--negatives-per-bag`` random tiles as MVP negatives, with no labelling. These are mostly ordinary
vessels and tissue in a lower-grade background, which balances the many glioblastoma positives. A free-negative
slide where a labeller found MVP is left out and listed in ``summary.json`` (``free_negative_slides_with_mvp``):
check its grade. Held-out patients (``--holdout``) are never used.

Writes ``mvp/`` and ``vessel/`` (``<slide>.json`` labels), ``train-mvp.sh``, ``train-vessel.sh`` and
``summary.json`` (counts, Cohen's kappa: for MVP on tiles both labellers called a vessel, which is where the
judgement lies).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import (add_export_args, add_outline_args, agreement, free_negative_bags, merge_tile_labels,  # noqa: E402
                    outline_labels, read_labels, read_patients, write_feature)

CLASSES = {"mvp": "mvp", "microvascular proliferation": "mvp", "vessel": "vessel", "ordinary vessel": "vessel",
           "vessel without mvp": "vessel", "neither": "neither", "negative": "neither"}
COLOURS = {"mvp": (30, 90, 220), "vessel": (40, 160, 70), "neither": (90, 90, 90)}


def mvp_vote(lab):
    return None if lab in (None, "", "x") else int(lab == "m")


def vessel_vote(lab):
    return None if lab in (None, "", "x") else int(lab in ("m", "v"))


def decide(h, i):
    hit = {k: bool(v[i]) for k, v in h.items()}
    mvp = 1 if hit.get("mvp") else 0 if (hit.get("vessel") or hit.get("neither")) else None
    vessel = 1 if (hit.get("mvp") or hit.get("vessel")) else None
    return {"mvp": mvp, "vessel": vessel}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_export_args(ap)
    add_outline_args(ap)
    a = ap.parse_args()
    if not (a.labels and a.key) and not a.outlines:
        raise SystemExit("give page labels with their --key files, or --outlines, or both")
    holdout = read_patients(a.holdout)
    key, raw, raters = read_labels(a.key, a.labels, holdout)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, tile_um, oinfo = outline_labels(a, CLASSES, decide, COLOURS)
    summary = {"raters": raters, "rule": a.rule, "items_in_keys": len(key), "items_labelled": len(raw),
               "held_out_patients": len(holdout), "outlines": oinfo, "features": {}}
    counts = {"mvp": merge_tile_labels(labels, tile_um, key, raw, raters, "mvp", mvp_vote, a.rule),
              "vessel": merge_tile_labels(labels, tile_um, key, raw, raters, "vessel", vessel_vote, a.rule)}

    with_mvp = sorted(s for s, tl in labels["mvp"].items() if any(v == 1 for v in tl.values()))
    neg = [] if a.no_free_negatives else free_negative_bags(a.manifest, a.bags, holdout, exclude_slides=with_mvp)
    flagged = sorted({k["slide"] for k in key.values() if k.get("category") in ("astro3", "grade2")} & set(with_mvp))
    summary["free_negative_slides_with_mvp"] = flagged

    both_vessel = {t: d for t, d in raw.items() if all(v in ("m", "v") for v in d.values())}
    summary["features"]["mvp"] = {
        **write_feature(out, "mvp", "microvascular_proliferation", labels["mvp"], tile_um, a, neg),
        **counts["mvp"], "agreement_on_vessels": agreement(both_vessel, raters, mvp_vote)}
    summary["features"]["vessel"] = {
        **write_feature(out, "vessel", "vessel", labels["vessel"], tile_um, a), **counts["vessel"],
        "agreement": agreement(raw, raters, vessel_vote)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
