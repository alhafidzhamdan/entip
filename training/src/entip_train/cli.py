"""Command line: ``entip-train demo`` (milestone check on a synthetic slide), ``entip-train serve``, and the
step-2 high-grade feature commands ``highgrade-demo``, ``highgrade-train`` and ``highgrade``."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def build_service(data_dir: Path, encoder: str):
    from .encoders import get_encoder
    from .service import TrainingService
    from .store import Store
    return TrainingService(Store(data_dir / "training.sqlite"), get_encoder(encoder), data_dir / "artifacts")


def demo(args) -> int:
    """~20 clicks on a synthetic slide -> a head and a ranked review queue; scored against hidden truth."""
    from sklearn.metrics import roc_auc_score
    from .slides import synthetic_slide

    data = Path(args.data or tempfile.mkdtemp(prefix="entip-train-"))
    svc = build_service(data, args.encoder)
    s = synthetic_slide("synthetic-001", seed=args.seed)
    svc.register_scan(s.scan, s.reader, s.cells, s.regions)
    hid = svc.create_head("atypical nucleus", "cell", actor="demo")
    svc.warm_cache(s.scan.scan_id, "cell")
    rng = np.random.default_rng(args.seed)
    pos = [c for c in s.cells if s.cell_truth[c.cell_id]]
    neg = [c for c in s.cells if not s.cell_truth[c.cell_id]]
    clicks = [(c, 1) for c in rng.choice(pos, args.clicks // 2, replace=False)] + \
             [(c, 0) for c in rng.choice(neg, args.clicks - args.clicks // 2, replace=False)]
    rng.shuffle(clicks)
    for c, y in clicks:                                   # clicks land near, not on, the cell centre
        out = svc.add_label(hid, s.scan.scan_id, y, "demo", x_um=c.x_um + rng.normal(0, 1.5),
                            y_um=c.y_um + rng.normal(0, 1.5))
    _, ids, _, p = svc.predict(hid, s.scan.scan_id)
    truth = np.array([s.cell_truth[i] for i in ids])
    q = svc.queue(hid, s.scan.scan_id, 10)
    report = {"data_dir": str(data), "encoder": svc.encoder.version, "clicks": len(clicks),
              "model_version": out.get("model_version"), "metrics": out.get("metrics"),
              "auroc_vs_hidden_truth": float(roc_auc_score(truth, p)), "queue_top10": q}
    print(json.dumps(report, indent=2, default=float))
    return 0


def serve(args) -> int:
    import uvicorn
    from .api import create_app
    data = Path(args.data or "entip-train-data")
    app = create_app(build_service(data, args.encoder), cors_origins=args.cors or None)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def highgrade_demo(args) -> int:
    """Train necrosis and MVP heads on synthetic slides, assess a synthetic specimen, print its records."""
    from .encoders import get_encoder
    from .highgrade.actions import post_actions, write_jsonl
    from .highgrade.assess import assess_specimen
    from .highgrade.bags import embed_bag
    from .highgrade.mitoses import MitosisConfig, MitosisCounter
    from .highgrade.regions import RegionConfig, RegionDetector, tile_labels
    from .highgrade.synthetic import synthetic_gbm

    enc = get_encoder(args.encoder)
    train = [synthetic_gbm(f"SCN-SYN-TR-{i}", seed=i, tiles=12, necrosis=int(i < 3), mvp=3 * int(i < 3))
             for i in range(4)]
    tbags = [embed_bag(s.scan, s.reader, enc) for s in train]
    dets = []
    for f, attr in (("necrosis", "necrosis_polys"), ("microvascular_proliferation", "mvp_polys")):
        labels = [tile_labels(b, getattr(s, attr), s.tumour_polys) for s, b in zip(train, tbags)]
        dets.append(RegionDetector(RegionConfig(f, validated=args.validated)).fit(tbags, labels))
    s = synthetic_gbm(args.scan_id, seed=args.seed, tiles=12, necrosis=args.necrosis, mvp=args.mvp,
                      mitoses=args.mitoses)
    a = assess_specimen(args.study_id, [embed_bag(s.scan, s.reader, enc)], dets,
                        MitosisCounter(MitosisConfig(validated=args.validated)), s.candidates)
    print(json.dumps(a.records(), indent=2, default=float))
    if args.actions:
        print(f"actions -> {write_jsonl(a.actions, args.actions)}", file=sys.stderr)
    if args.db:
        post_actions(args.db, a.actions)
        print(f"{len(a.actions)} actions posted to {args.db}", file=sys.stderr)
    return 0


def highgrade_train(args) -> int:
    """Fit one region head from tile bags and polygon labels ({"positives": [...], "negatives": [...]}, µm)."""
    from .highgrade.bags import load_bag
    from .highgrade.regions import RegionConfig, RegionDetector, tile_labels
    if len(args.bag) != len(args.labels):
        raise SystemExit("give one --labels file per --bag")
    bags = [load_bag(p) for p in args.bag]
    labels = []
    for b, p in zip(bags, args.labels):
        d = json.loads(Path(p).read_text())
        labels.append(tile_labels(b, d.get("positives", []), d.get("negatives", [])))
    det = RegionDetector(RegionConfig(args.feature, threshold=args.threshold, training_data=args.training_data))
    det.fit(bags, labels).save(args.out)
    print(json.dumps(det.model_record(), indent=2, default=float))
    return 0


def highgrade(args) -> int:
    """Assess one specimen: tile bags (one per scan), saved heads, optional MitoEd CSVs (scan_id=path)."""
    from .highgrade.actions import post_actions, write_jsonl
    from .highgrade.assess import assess_specimen
    from .highgrade.bags import load_bag
    from .highgrade.mitoses import MitosisConfig, MitosisCounter, read_mitoed_cells
    from .highgrade.regions import RegionDetector
    bags = [load_bag(p) for p in args.bag]
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("entip-train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="train a head from ~20 clicks on a synthetic slide")
    d.add_argument("--clicks", type=int, default=20)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--encoder", default="handcrafted", help="handcrafted (CPU) or h-optimus-0")
    d.add_argument("--data", default=None)
    d.set_defaults(fn=demo)
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--encoder", default="h-optimus-0")
    s.add_argument("--data", default=None)
    s.add_argument("--cors", nargs="*", default=[], help="allowed GUI origins, e.g. http://localhost:8080")
    s.set_defaults(fn=serve)

    h = sub.add_parser("highgrade-demo", help="step 2 on a synthetic specimen: necrosis, MVP, mitotic activity")
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
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
