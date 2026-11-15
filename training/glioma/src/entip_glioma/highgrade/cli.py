"""``highgrade-demo``, ``highgrade-train``, ``highgrade-calibrate`` and ``highgrade``: step 3 commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _synthetic_bag(scan_id, seed, enc, **kw):
    from .synthetic import synthetic_gbm
    from ..wsi.bags import extract_bag
    s = synthetic_gbm(scan_id, seed=seed, tiles=10, **kw)
    return s, extract_bag(s.reader, s.scan.scan_id, s.scan.width_um, s.scan.height_um, enc)


def highgrade_demo(args) -> int:
    """Train necrosis and MVP heads on synthetic slides, calibrate them on further synthetic slides, assess a
    synthetic specimen and print its records."""
    from entip_train.encoders import get_encoder
    from .actions import post_actions, write_jsonl
    from .assess import assess_specimen
    from .mitoses import MitosisConfig, MitosisCounter
    from .regions import RegionConfig, RegionDetector, tile_labels

    enc = get_encoder(args.encoder)
    train = [_synthetic_bag(f"SCN-SYN-TR-{i}", i, enc, necrosis=int(i < 3), mvp=3 * int(i < 3)) for i in range(4)]
    held = [_synthetic_bag(f"SCN-SYN-CAL-{i}", 50 + i, enc, necrosis=i % 2, mvp=2 * (i % 2)) for i in range(8)]
    dets = []
    for f, attr in (("necrosis", "necrosis_polys"), ("microvascular_proliferation", "mvp_polys")):
        labels = [tile_labels(b, getattr(s, attr), s.tumour_polys) for s, b in train]
        det = RegionDetector(RegionConfig(f, validated=args.validated)).fit([b for _, b in train], labels)
        det.calibrate_slides([[b] for _, b in held], [i % 2 for i in range(len(held))])
        dets.append(det)
    s, bag = _synthetic_bag(args.scan_id, args.seed, enc, necrosis=args.necrosis, mvp=args.mvp,
                            mitoses=args.mitoses)
    a = assess_specimen(args.study_id, [bag], dets, MitosisCounter(MitosisConfig(validated=args.validated)),
                        s.candidates)
    print(json.dumps(a.records(), indent=2, default=float))
    if args.actions:
        print(f"actions -> {write_jsonl(a.actions, args.actions)}", file=sys.stderr)
    if args.db:
        post_actions(args.db, a.actions)
        print(f"{len(a.actions)} actions posted to {args.db}", file=sys.stderr)
    return 0


def highgrade_train(args) -> int:
    """Fit one region head from tile bags and polygon labels ({"positives": [...], "negatives": [...]}, µm)."""
    from .regions import RegionConfig, RegionDetector, tile_labels
    from ..wsi.bags import Bag
    if len(args.bag) != len(args.labels):
        raise SystemExit("give one --labels file per --bag")
    bags = [Bag.load(p) for p in args.bag]
    labels = []
    for b, p in zip(bags, args.labels):
        d = json.loads(Path(p).read_text())
        labels.append(tile_labels(b, d.get("positives", []), d.get("negatives", [])))
    det = RegionDetector(RegionConfig(args.feature, threshold=args.threshold, training_data=args.training_data))
    det.fit(bags, labels).save(args.out)
    print(json.dumps(det.model_record(), indent=2, default=float))
    return 0


def highgrade_calibrate(args) -> int:
    """Calibrate a head's slide score on held-out specimens: a JSON list of {"bags": [paths], "present": 0|1}."""
    from .regions import RegionDetector
    from ..wsi.bags import Bag
    det = RegionDetector.load(args.head)
    rows = json.loads(Path(args.specimens).read_text())
    info = det.calibrate_slides([[Bag.load(p) for p in r["bags"]] for r in rows], [int(r["present"]) for r in rows])
    det.save(args.out or args.head)
    print(json.dumps(info, indent=2, default=float))
    return 0


def highgrade(args) -> int:
    """Assess one specimen: tile bags (one per scan), saved heads, optional MitoEd CSVs (scan_id=path)."""
    from .actions import post_actions, write_jsonl
    from .assess import assess_specimen
    from .mitoses import MitosisConfig, MitosisCounter, read_mitoed_cells
    from .regions import RegionDetector
    from ..wsi.bags import Bag
    bags = [Bag.load(p) for p in args.bag]
    dets = [RegionDetector.load(p) for p in args.head]
    cands = None
    if args.mitoed_cells:
        cands = []
        for item in args.mitoed_cells:
            sid, _, path = item.partition("=")
            cands += read_mitoed_cells(path, sid)
    counter = MitosisCounter(MitosisConfig(min_count=args.min_mitoses, small_biopsy_mm2=args.small_biopsy_mm2),
                             model_id=args.mitoed_model, version=args.mitoed_version)
    a = assess_specimen(args.study_id, bags, dets, counter if cands is not None else None, cands)
    out = json.dumps(a.records(), indent=2, default=float)
    if args.out:
        Path(args.out).write_text(out + "\n")
    print(out)
    if args.actions:
        write_jsonl(a.actions, args.actions)
    if args.db:
        post_actions(args.db, a.actions)
    return 0


def add_parser(sub) -> None:
    h = sub.add_parser("highgrade-demo", help="step 3 on a synthetic specimen: necrosis, MVP, mitotic activity")
    h.add_argument("--encoder", default="handcrafted")
    h.add_argument("--study-id", default="SYN-0001")
    h.add_argument("--scan-id", default="SCN-0001-A1-01", help="must exist in the db when posting (seeded ids)")
    h.add_argument("--seed", type=int, default=7)
    h.add_argument("--necrosis", type=int, default=1)
    h.add_argument("--mvp", type=int, default=2)
    h.add_argument("--mitoses", type=int, default=3)
    h.add_argument("--validated", action="store_true", help="treat models as validated (demo only)")
    h.add_argument("--actions", help="write the db actions as JSON lines here")
    h.add_argument("--db", help="post the actions to this db API, e.g. http://127.0.0.1:8705")
    h.set_defaults(fn=highgrade_demo)

    t = sub.add_parser("highgrade-train", help="fit a necrosis or MVP head from tile bags and polygon labels")
    t.add_argument("--feature", required=True, choices=["necrosis", "microvascular_proliferation"])
    t.add_argument("--bag", action="append", required=True, help="tile bag .npz (repeat)")
    t.add_argument("--labels", action="append", required=True, help="polygon labels .json per bag (repeat)")
    t.add_argument("--threshold", type=float, default=0.8)
    t.add_argument("--training-data", default="", help="name and licence of the labels")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=highgrade_train)

    c = sub.add_parser("highgrade-calibrate", help="calibrate a head's slide score on held-out specimens")
    c.add_argument("--head", required=True)
    c.add_argument("--specimens", required=True, help='JSON list of {"bags": [paths], "present": 0|1}')
    c.add_argument("--out", help="default: overwrite --head")
    c.set_defaults(fn=highgrade_calibrate)

    g = sub.add_parser("highgrade", help="assess one specimen's high-grade features")
    g.add_argument("--study-id", required=True)
    g.add_argument("--bag", action="append", default=[], help="tile bag .npz per scan (repeat)")
    g.add_argument("--head", action="append", default=[], help="saved region head .joblib (repeat)")
    g.add_argument("--mitoed-cells", action="append", default=[], help="SCAN_ID=cells.csv from mitoed predict")
    g.add_argument("--mitoed-model", default="mitoed")
    g.add_argument("--mitoed-version", default="")
    g.add_argument("--min-mitoses", type=int, default=2)
    g.add_argument("--small-biopsy-mm2", type=float, default=None)
    g.add_argument("--out")
    g.add_argument("--actions")
    g.add_argument("--db")
    g.set_defaults(fn=highgrade)
