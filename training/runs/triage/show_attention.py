"""Which tiles a triage model looked at on one slide: a heat map over the slide and its top tiles, cropped.

    python show_attention.py --bag bags-h-optimus-0/<uid>.npz --model model-h-optimus-0 \
        --slide slides/<uid> --out PBCJIG-teacher.png [--top 16]

Writes one PNG: on the left the slide thumbnail with each tile shaded by its attention weight (top tiles boxed and
numbered); on the right the top tiles at full resolution, captioned with rank and weight. Prints the call and the
share of attention the top tiles carry.

Caveat: the saved model is the final one, trained on every patient, so it has seen this slide with its label. The
out-of-fold call in the training report (mistakes.py) came from a fold model that had not; that model is not
saved, so its call and attention can differ from what is shown here.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--slide", required=True, help="slide file or DICOM series folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--thumb", type=int, default=1100, help="thumbnail size in pixels")
    a = ap.parse_args()

    from PIL import Image, ImageDraw
    from entip_glioma.triage.model import CLASSES, TriageModel
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide

    m = TriageModel.load(a.model)
    bag = Bag.load(a.bag)
    g = m.predict(bag)
    att = np.asarray(g["attention"], np.float64)
    top = np.argsort(-att)[:a.top]
    rd = open_slide(a.slide)

    # thumbnail: the tissue (every tile of the bag) in a square, with a margin
    lo, hi = bag.coords_um.min(0), bag.coords_um.max(0) + bag.tile_um
    cx, cy = (lo + hi) / 2
    side = float((hi - lo).max()) * 1.05
    thumb = Image.fromarray(rd.read_um(cx, cy, side, a.thumb)).convert("RGBA")
    s, x0, y0 = a.thumb / side, cx - side / 2, cy - side / 2
    heat = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    dh = ImageDraw.Draw(heat)
    w = np.sqrt(att / att.max())                      # square root so mid weights stay visible
    for (x, y), v in zip(bag.coords_um, w):
        px, py = (x - x0) * s, (y - y0) * s
        dh.rectangle([px, py, px + bag.tile_um * s, py + bag.tile_um * s], fill=(230, 80, 20, int(10 + 120 * v)))
    thumb = Image.alpha_composite(thumb, heat)
    dt = ImageDraw.Draw(thumb)
    for r, i in enumerate(top):
        x, y = bag.coords_um[i]
        px, py, q = (x - x0) * s, (y - y0) * s, max(6, bag.tile_um * s)
        dt.rectangle([px - 2, py - 2, px + q + 2, py + q + 2], outline=(20, 60, 200, 255), width=2)
        dt.text((px + q + 4, py - 2), str(r + 1), fill=(20, 60, 200, 255))

    # top tiles at full resolution, 4 per row
    tp, cols = 256, 4
    rows = -(-len(top) // cols)
    grid = Image.new("RGB", (cols * (tp + 8), rows * (tp + 22)), "white")
    dg = ImageDraw.Draw(grid)
    for r, i in enumerate(top):
        x, y = bag.coords_um[i]
        tile = Image.fromarray(rd.read_um(x + bag.tile_um / 2, y + bag.tile_um / 2, bag.tile_um, tp))
        cx, cy = (r % cols) * (tp + 8), (r // cols) * (tp + 22)
        grid.paste(tile, (cx, cy))
        dg.text((cx + 2, cy + tp + 4), f"#{r + 1}  weight {att[i]:.4f}  ({x:.0f}, {y:.0f}) um", fill="black")

    out = Image.new("RGB", (thumb.width + 16 + grid.width, max(thumb.height, grid.height) + 30), "white")
    p = g["probs"]
    call = CLASSES[int(p.argmax())]
    ImageDraw.Draw(out).text((4, 4), f"{bag.scan_id}  model {m.model_id}  call {call} {p.max():.3f}  "
                                     f"top {len(top)} of {len(att)} tiles carry {att[top].sum():.0%} of attention",
                             fill="black")
    out.paste(thumb.convert("RGB"), (0, 30))
    out.paste(grid, (thumb.width + 16, 30))
    out.save(a.out)
    print(json.dumps({"slide": bag.scan_id, "model": m.model_id, "call": call,
                      "probabilities": {c: round(float(v), 4) for c, v, t in zip(CLASSES, p, m.trained) if t},
                      "tiles": len(att), "top_share": round(float(att[top].sum()), 4),
                      "top": [{"rank": r + 1, "x_um": float(bag.coords_um[i][0]), "y_um": float(bag.coords_um[i][1]),
                               "weight": round(float(att[i]), 5)} for r, i in enumerate(top)],
                      "image": a.out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
