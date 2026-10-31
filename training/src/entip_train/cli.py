"""Command line: ``entip-train demo`` (milestone check on a synthetic slide) and ``entip-train serve``."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def build_service(data_dir: Path, encoder: str, db: str | None = None, slides_dir: str | None = None):
    """``db``: URL of the db/ pillar's API; without it a local SQLite stand-in is used."""
    from .encoders import get_encoder
    from .service import TrainingService
    from .slides import default_reader_factory
    from .store import Store
    if db:
        from .dbstore import DbStore
        store, write_predictions = DbStore(db), True
    else:
        store, write_predictions = Store(data_dir / "training.sqlite"), False
    return TrainingService(store, get_encoder(encoder), data_dir / "artifacts",
                           reader_factory=default_reader_factory(slides_dir), write_predictions=write_predictions)


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
    app = create_app(build_service(data, args.encoder, args.db, args.slides_dir), cors_origins=args.cors or None)
    uvicorn.run(app, host=args.host, port=args.port)
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
    s.add_argument("--db", default=None, help="db/ API URL, e.g. http://localhost:8000 (default: local SQLite)")
    s.add_argument("--slides-dir", default=None, help="folder of DZI pyramids by scan_id (e.g. gui/web/mock/slides)")
    s.add_argument("--cors", nargs="*", default=[], help="allowed GUI origins, e.g. http://localhost:8080")
    s.set_defaults(fn=serve)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
