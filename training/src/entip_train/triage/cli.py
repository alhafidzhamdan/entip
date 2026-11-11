"""``entip-train triage ...``: manifest, build-bags, extract, train, predict, demo."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _manifest(a) -> int:
    from .manifest import from_idc, label_rows, write_manifest
    rows, dropped = label_rows(from_idc(), preps=tuple(a.prep) if a.prep else None)
    write_manifest(rows, a.out)
    print(json.dumps({"out": a.out, "kept": len(rows), "dropped": dropped,
                      "slides_per_class": Counter(r["label"] for r in rows),
                      "patients_per_class": {c: len({r["PatientID"] for r in rows if r["label"] == c})
                                             for c in sorted({r["label"] for r in rows})}}, indent=2))
    return 0


def _embed(path: str, scan_id: str, enc, a, out_dir: Path) -> dict:
    from ..slides import open_slide
    from ..wsi.bags import extract_bag
    r = open_slide(path, mpp=a.mpp)
    bag = extract_bag(r, scan_id, r.width_um, r.height_um, enc, tile_um=a.tile_um, min_tissue=a.min_tissue,
                      max_tiles=a.max_tiles)
    p = bag.save(out_dir / f"{scan_id}.npz")
    return {"scan_id": scan_id, "tiles": len(bag), "reader": type(r).__name__, "mpp": round(r.mpp, 4),
            "encoder": bag.encoder, "out": str(p)}


def _extract(a) -> int:
    from ..encoders import get_encoder
    enc = get_encoder(a.encoder)
    if a.scan_id and len(a.slides) > 1:
        raise SystemExit("--scan-id names one slide; pass one slide, or leave it out to use file names")
    for path in a.slides:
        p = Path(path)
        sid = a.scan_id or (p.name if p.is_dir() else p.stem)
        print(json.dumps(_embed(path, sid, enc, a, Path(a.out))), flush=True)
    return 0


def _build_bags(a) -> int:
    """Manifest -> bags, one slide at a time: download, embed, delete. Safe to stop and rerun."""
    import shutil
    import tempfile
    from ..encoders import get_encoder
    from .manifest import read_manifest
    rows = read_manifest(a.manifest)
    if a.per_class:
        seen, pick = Counter(), []
        for r in sorted(rows, key=lambda r: float(r.get("series_size_MB") or 0)):
            if seen[(r["label"], r["group"])] < a.per_class:
                seen[(r["label"], r["group"])] += 1
                pick.append(r)
        rows = pick
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (out / f"{r['SeriesInstanceUID']}.npz").exists()]
    if a.limit:
        todo = todo[:a.limit]
    gb = sum(float(r.get("series_size_MB") or 0) for r in todo) / 1000
    print(json.dumps({"slides": len(rows), "already_done": len(rows) - len(todo) if not a.limit else None,
                      "to_do": len(todo), "download_GB": round(gb, 1)}), flush=True)
    if a.dry_run or not todo:
        return 0
    from idc_index import IDCClient
    client, enc = IDCClient(), get_encoder(a.encoder)
    failed = 0
    for r in todo:
        uid = r["SeriesInstanceUID"]
        tmp = Path(tempfile.mkdtemp(prefix="idc-", dir=a.tmp))
        try:
            client.download_from_selection(downloadDir=str(tmp), seriesInstanceUID=[uid], quiet=True,
                                           show_progress_bar=False)
            dcm = next(iter(tmp.rglob("*.dcm")), None)
            if dcm is None:
                raise RuntimeError("download returned no DICOM files")
            info = _embed(str(dcm.parent), uid, enc, a, out)
            info["label"] = r["label"]
            print(json.dumps(info), flush=True)
        except Exception as e:                      # one bad slide must not stop a run of thousands
            failed += 1
            with (out / "failed.jsonl").open("a") as f:
                f.write(json.dumps({"SeriesInstanceUID": uid, "error": repr(e)[:500]}) + "\n")
            print(json.dumps({"scan_id": uid, "error": repr(e)[:200]}), flush=True)
        finally:
            if not a.keep_slides:
                shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"done": len(todo) - failed, "failed": failed}))
    return 0


def _train(a) -> int:
    import numpy as np
    from ..wsi.bags import Bag
    from .manifest import read_manifest
    from .model import TriageConfig, TriageModel
    rows = read_manifest(a.manifest)
    bags, labels, groups, patients, missing = [], [], [], [], 0
    for r in rows:
        f = Path(a.bags) / f"{r[a.id_col]}.npz"
        if not f.exists():
            missing += 1
            continue
        # a seeded sample of tiles per slide, as float16: ~3 MB per slide at 1,024 tiles of 1,536 numbers
        bags.append(Bag.load(f, max_tiles=a.train_tiles, seed=a.seed, dtype=np.float16))
        labels.append(r["label"])
        groups.append(r["group"])
        patients.append(r["PatientID"])
    if not bags:
        raise SystemExit(f"no bags found in {a.bags} for the {len(rows)} manifest rows")
    cfg = TriageConfig(epochs=a.epochs, folds=a.folds, max_error=a.max_error, seed=a.seed)
    m = TriageModel(cfg)
    rep = m.fit(bags, labels, groups, patients, loso=not a.no_loso)
    rep["slides_missing_bags"] = missing
    m.report = rep
    m.save(a.out)
    if a.db:
        from .db import post_actions, register_actions
        post_actions(a.db, register_actions(m, uri=str(Path(a.out).resolve())))
    print(json.dumps(rep, indent=2, default=float))
    return 0


def _predict(a) -> int:
    from ..wsi.bags import Bag
    from .db import build_actions, post_patient
    from .model import TriageModel
    m = TriageModel.load(a.model)
    bags = [Bag.load(p) for p in a.bags]
    if a.db:
        post_patient(a.db, m, bags, a.study_id, uri=str(Path(a.model).resolve()))
    print(json.dumps(build_actions(m, bags, a.study_id)[-1]["detail"], indent=2, default=float))
    return 0


def _demo(a) -> int:
    import tempfile
    from .db import build_actions
    from .model import TriageConfig, TriageModel
    from .synthetic import synthetic_cohort
    bags, labels, groups, patients = synthetic_cohort(seed=a.seed)
    m = TriageModel(TriageConfig(epochs=a.epochs, folds=3, max_error=0.1, min_accepted=10, seed=a.seed))
    rep = m.fit(bags, labels, groups, patients)
    d = m.save(Path(tempfile.mkdtemp(prefix="entip-triage-")))
    i = labels.index("glioma")
    from .db import register_actions
    acts = register_actions(m) + build_actions(m, [bags[i]], "SYN-DEMO")
    cv = rep["cv"]["calibrated"]
    print(json.dumps({"model_dir": str(d), "classes_trained": rep["classes_trained"],
                      "classes_excluded": rep["classes_excluded"],
                      "cv_balanced_accuracy": cv["balanced_accuracy"], "cv_ece": cv["ece"],
                      "auroc_tumour": cv.get("auroc_tumour"), "auroc_glial": cv.get("auroc_glial"),
                      "gate": rep["cv"]["gate"], "novelty": rep["novelty"],
                      "unseen_classes": rep["unseen_classes"], "source_check": rep["source_check"],
                      "leave_one_source_out": rep.get("leave_one_source_out"),
                      "actions": [a["verb"] + ":" + (a.get("object_type") or "event") for a in acts],
                      "records": acts[-1]["detail"]["records"]}, indent=2, default=float)[:20000])
    return 0


def add_parser(sub) -> None:
    t = sub.add_parser("triage", help="step 1: tumour / glial triage on H&E")
    ts = t.add_subparsers(dest="triage_cmd", required=True)
    p = ts.add_parser("manifest", help="labelled slide list from IDC (needs internet and idc-index)")
    p.add_argument("--out", default="triage_manifest.csv")
    p.add_argument("--prep", nargs="*", default=None, help="keep only these: ffpe frozen paxgene_paraffin unknown")
    p.set_defaults(fn=_manifest)
    def embed_args(p):
        p.add_argument("--out", required=True, help="folder for <id>.npz bags")
        p.add_argument("--encoder", default="h-optimus-0")
        p.add_argument("--mpp", type=float, default=None, help="override the slide's pixel size (µm/px)")
        p.add_argument("--tile-um", type=float, default=112.0)
        p.add_argument("--min-tissue", type=float, default=0.5)
        p.add_argument("--max-tiles", type=int, default=None, help="random cap on tiles per slide")

    p = ts.add_parser("extract", help="local slides -> tile embedding bags (.npz)")
    p.add_argument("slides", nargs="+", help="slide files or DICOM series folders")
    p.add_argument("--scan-id", default=None)
    embed_args(p)
    p.set_defaults(fn=_extract)
    p = ts.add_parser("build-bags", help="manifest -> bags: download each IDC slide, embed it, delete it")
    p.add_argument("--manifest", required=True)
    embed_args(p)
    p.add_argument("--per-class", type=int, default=None,
                   help="trial run: only the N smallest slides of each class and source")
    p.add_argument("--limit", type=int, default=None, help="stop after this many slides")
    p.add_argument("--tmp", default=None, help="folder for the slide being processed")
    p.add_argument("--keep-slides", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="only say how many slides and GB are left")
    p.set_defaults(fn=_build_bags)
    p = ts.add_parser("train", help="train, cross-validate, calibrate and save a triage model")
    p.add_argument("--manifest", required=True)
    p.add_argument("--bags", required=True, help="folder of <id>.npz bags")
    p.add_argument("--id-col", default="SeriesInstanceUID")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-error", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-loso", action="store_true", help="skip leave-one-source-out runs")
    p.add_argument("--train-tiles", type=int, default=1024, help="tiles kept per slide in memory for training")
    p.add_argument("--db", default=None, help="db API base URL: register the model there (status draft)")
    p.set_defaults(fn=_train)
    p = ts.add_parser("predict", help="score one patient's bags; optionally write to the db")
    p.add_argument("bags", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--study-id", required=True)
    p.add_argument("--db", default=None, help="db API base URL, e.g. http://localhost:8000")
    p.set_defaults(fn=_predict)
    p = ts.add_parser("demo", help="synthetic cohort end to end (CPU, under a minute)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_demo)
