"""``entip-glioma distill ...`` (also ``entip-train distill ...``): tiles, train, evaluate."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _tiles(a) -> int:
    from entip_train.slides import open_slide

    from ..wsi.bags import Bag
    from .tiles import INDEX, extract_tiles, read_index
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {r[a.id_col]: r for r in csv.DictReader(open(a.manifest, newline=""))}
    done = {e["scan_id"] for e in read_index(out)} if (out / INDEX).exists() else set()
    n_new = n_skip = 0
    for f in sorted(Path(a.bags).glob("*.npz")):
        sid = f.stem
        if sid not in rows:
            continue                                       # a bag from another run or plan
        if sid in done:
            n_skip += 1
            continue
        bag = Bag.load(f)
        if len(bag) == 0:
            continue
        slide = Path(a.slides) / sid
        if not slide.exists():
            print(f"missing slide for {sid} in {a.slides}; skipped", file=sys.stderr)
            continue
        info = extract_tiles(open_slide(str(slide)), bag, out, input_px=a.input_px, max_tiles=a.max_tiles, seed=a.seed)
        info.update(patient=rows[sid]["PatientID"], label=rows[sid].get("label"), group=rows[sid].get("group"))
        with (out / INDEX).open("a") as fh:
            fh.write(json.dumps(info) + "\n")
        print(json.dumps(info), flush=True)
        n_new += 1
    print(json.dumps({"slides_added": n_new, "already_done": n_skip}))
    return 0


def _train(a) -> int:
    from .train import train_student
    rep = train_student(a.tiles, a.out, arch=a.arch, pretrained=a.init == "imagenet", epochs=a.epochs,
                        batch_size=a.batch_size, lr=a.lr, holdout=a.holdout, seed=a.seed, workers=a.workers,
                        device=a.device, resume=a.resume, max_steps=a.max_steps)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("history", "train_patients", "heldout_patients")},
                     indent=2))
    return 0


def _evaluate(a) -> int:
    from .train import evaluate_student
    rep = evaluate_student(a.tiles, a.student)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "slides"}, indent=2))
    return 0


def add_parser(sub) -> None:
    d = sub.add_parser("distill", help="distil H-optimus-0 into a small student encoder (tiles, train, evaluate)")
    ds = d.add_subparsers(dest="distill_cmd", required=True)
    p = ds.add_parser("tiles", help="cut each bag's tiles from its slide into a cache next to the teacher's numbers")
    p.add_argument("--bags", required=True, help="folder of teacher bags (<SeriesInstanceUID>.npz)")
    p.add_argument("--slides", required=True, help="folder of slides, one per bag: <slides>/<SeriesInstanceUID>")
    p.add_argument("--manifest", required=True, help="the run's trial.csv (patient and label per slide)")
    p.add_argument("--id-col", default="SeriesInstanceUID")
    p.add_argument("--out", required=True)
    p.add_argument("--input-px", type=int, default=224, help="student input size in pixels")
    p.add_argument("--max-tiles", type=int, default=1000, help="seeded random cap on tiles per slide")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_tiles)
    p = ds.add_parser("train", help="train the student on the tile cache (GPU)")
    p.add_argument("--tiles", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="vit_small_patch16_224", help="timm architecture, or 'tiny' for a smoke test")
    p.add_argument("--init", choices=["imagenet", "scratch"], default="imagenet",
                   help="start from timm's ImageNet weights (vit_small_patch16_224: Apache-2.0) or from scratch")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--holdout", type=float, default=0.2, help="share of patients held out, per class")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    p.add_argument("--resume", action="store_true", help="carry on from last.pt")
    p.set_defaults(fn=_train)
    p = ds.add_parser("evaluate", help="student vs teacher on the held-out patients")
    p.add_argument("--tiles", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--out", default=None, help="write the full per-slide report here")
    p.set_defaults(fn=_evaluate)
