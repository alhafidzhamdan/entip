"""``entip-train triage ...``: manifest, train, predict, demo. ``extract`` and ``build-bags`` moved to step 1
(``entip-train embed``) and stay here as aliases."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .manifest import CORRECTIONS


def _manifest(a) -> int:
    from .manifest import from_idc, label_rows, write_manifest
    rows, dropped = label_rows(from_idc(), preps=tuple(a.prep) if a.prep else None,
                               non_cns=a.non_cns)
    write_manifest(rows, a.out)
    print(json.dumps({"out": a.out, "kept": len(rows), "dropped": dropped,
                      "slides_per_class": Counter(r["label"] for r in rows),
                      "patients_per_class": {c: len({r["PatientID"] for r in rows if r["label"] == c})
                                             for c in sorted({r["label"] for r in rows})}}, indent=2))
    return 0


def _train(a) -> int:
    import numpy as np
    from ..wsi.bags import Bag
    from .manifest import apply_corrections, read_corrections, read_manifest
    from .model import TriageConfig, TriageModel
    rows, corrected = apply_corrections(read_manifest(a.manifest),
                                        read_corrections(None if a.no_corrections else a.corrections or CORRECTIONS))
    bags, labels, groups, patients, missing, empty = [], [], [], [], 0, []
    for r in rows:
        f = Path(a.bags) / f"{r[a.id_col]}.npz"
        if not f.exists():
            missing += 1
            continue
        # a seeded sample of tiles per slide, as float16: ~3 MB per slide at 1,024 tiles of 1,536 numbers
        bag = Bag.load(f, max_tiles=a.train_tiles, seed=a.seed, dtype=np.float16)
        if len(bag) == 0:                       # no tissue tile kept (blank or unreadable slide): nothing to learn from
            empty.append(r[a.id_col])
            print(f"skipping {r[a.id_col]}: bag has no tiles", file=sys.stderr)
            continue
        bags.append(bag)
        labels.append(r["label"])
        groups.append(r["group"])
        patients.append(r["PatientID"])
    if not bags:
        raise SystemExit(f"no usable bags in {a.bags} for the {len(rows)} manifest rows "
                         f"({missing} missing, {len(empty)} with no tiles)")
    cfg = TriageConfig(epochs=a.epochs, folds=a.folds, max_error=a.max_error, seed=a.seed)
    m = TriageModel(cfg)
    from .model import default_jobs
    rep = m.fit(bags, labels, groups, patients, loso=not a.no_loso, n_jobs=a.jobs or default_jobs())
    rep["slides_missing_bags"] = missing
    rep["label_corrections"] = corrected
    rep["slides_empty_bags"] = empty
    m.report = rep
    m.save(a.out)
    if a.db:
        from .db import post_actions, register_actions
        post_actions(a.db, register_actions(m, uri=str(Path(a.out).resolve())))
    print(json.dumps(rep, indent=2, default=float))
    return 0


def _evaluate(a) -> int:
    """Score a trained model on an external set it never trained on (see evaluate.py)."""
    from .evaluate import evaluate
    from .manifest import read_manifest
    from .model import TriageModel
    rep = evaluate(TriageModel.load(a.model), read_manifest(a.manifest), a.bags, id_col=a.id_col)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, default=float))
    short = {k: v for k, v in rep.items() if k not in ("patients", "per_diagnosis")}
    print(json.dumps(short, indent=2, default=float))
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
    t = sub.add_parser("triage", help="step 2: tumour / glial triage on H&E")
    ts = t.add_subparsers(dest="triage_cmd", required=True)
    p = ts.add_parser("manifest", help="labelled slide list from IDC (needs internet and idc-index)")
    p.add_argument("--out", default="triage_manifest.csv")
    p.add_argument("--prep", nargs="*", default=None, help="keep only these: ffpe frozen paxgene_paraffin unknown")
    p.add_argument("--non-cns", action="store_true",
                   help="add TCGA lung, breast, melanoma, kidney and colorectal primaries to the non-CNS / metastasis class")
    p.set_defaults(fn=_manifest)
    from ..embed.cli import add_commands
    add_commands(ts, " (alias of `entip-train embed`, step 1)")
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
    p.add_argument("--corrections", default=None, help="label corrections CSV (default: the one shipped with ENTIP)")
    p.add_argument("--no-corrections", action="store_true", help="train on the manifest's labels as they are")
    p.add_argument("--jobs", type=int, default=None,
                   help="processes for the cross-validation folds (default: the job's cores, NSLOTS, else 1)")
    p.add_argument("--db", default=None, help="db API base URL: register the model there (status draft)")
    p.set_defaults(fn=_train)
    p = ts.add_parser("evaluate", help="score a trained model on an external set it never trained on")
    p.add_argument("--manifest", required=True, help="CSV with SeriesInstanceUID (or --id-col), PatientID, label[, diagnosis]")
    p.add_argument("--bags", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--id-col", default="SeriesInstanceUID")
    p.add_argument("--out", default=None, help="write the full report (every patient, per diagnosis) here")
    p.set_defaults(fn=_evaluate)
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
