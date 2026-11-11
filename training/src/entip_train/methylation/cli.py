"""``entip-train methyl ...``: the step-4 commands.

  demo       train and predict on synthetic bags (CPU, about a minute)
  pair       pair TCGA diagnostic slides with 450K methylation files by case
  label      label cases from methylation: --report CSV, or --ref-betas/--ref-labels (own reference classifier)
  train      train from a labels CSV and a folder of bags (<slide name>.npz, from wsi.bags.extract_bag)
  predict    one patient's bags -> a finding record; --db URL also writes it through POST /actions
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def add_parser(sub) -> None:
    p = sub.add_parser("methyl", help="step 4: methylation class from H&E")
    s = p.add_subparsers(dest="mcmd", required=True)

    d = s.add_parser("demo")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=demo)

    a = s.add_parser("pair")
    a.add_argument("--slides", required=True, help="text file, one slide file name per line")
    a.add_argument("--methylation", required=True, help="text file, one methylation file name per line")
    a.add_argument("--methylation-sheet", help="GDC sample sheet for the methylation files (names lack barcodes)")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=pair)

    lb = s.add_parser("label")
    lb.add_argument("--report", help="CSV case_id,class,score from a classifier")
    lb.add_argument("--ref-betas", help="folder of reference beta files (<sample>.txt)")
    lb.add_argument("--ref-labels", help="CSV sample_id,class for the reference set")
    lb.add_argument("--target-betas", help="folder of target beta files")
    lb.add_argument("--target-sheet", help="GDC sample sheet mapping target file names to barcodes")
    lb.add_argument("--exclude-probes", help="text file of probe ids to leave out (sex chromosomes, SNP-affected)")
    lb.add_argument("--min-score", type=float, default=0.9)
    lb.add_argument("--out", required=True)
    lb.set_defaults(fn=label)

    t = s.add_parser("train")
    t.add_argument("--labels", required=True, help="CSV case_id,class,score (from 'label')")
    t.add_argument("--pairs", help="CSV from 'pair' (case_id,slide,...); else labels need a 'slides' column")
    t.add_argument("--bags", required=True, help="folder of <slide>.npz bags")
    t.add_argument("--taxonomy", help="CSV class,family (default: built-in glioma list)")
    t.add_argument("--model-id", default="methylation-class-abmil")
    t.add_argument("--version", default="0.1.0")
    t.add_argument("--max-error", type=float, default=0.05)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--min-per-class", type=int, default=10, help="drop classes with fewer labelled cases")
    t.add_argument("--min-tiles", type=int, default=50, help="fewer tissue tiles -> not assessed")
    t.add_argument("--out", required=True)
    t.set_defaults(fn=train_cmd)

    pr = s.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--bags", nargs="+", required=True)
    pr.add_argument("--study-id", required=True)
    pr.add_argument("--triage-call", default=None)
    pr.add_argument("--db", default=None, help="db API base URL, e.g. http://127.0.0.1:8705")
    pr.set_defaults(fn=predict_cmd)


def demo(args) -> int:
    from .classes import DEFAULT_CLASSES, Taxonomy
    from .synthetic import synthetic_cases
    from .train import Case, train
    from .db import build_actions
    data = synthetic_cases(DEFAULT_CLASSES, n_per_class=12, seed=args.seed)
    cases = [Case(cid, [bag], c) for cid, bag, c in data]
    m = train(cases, Taxonomy(), "methylation-class-demo", "0.0.1", k=5, min_tiles=20,
              mil_kw={"epochs": 15}, seed=args.seed, log=print)
    rec = m.predict([data[0][1]], study_id="SYN-0001")
    print(json.dumps({"metrics": {k: m.metrics[k] for k in ("class", "family")}, "record": rec,
                      "actions": [a["verb"] for a in build_actions(rec, m)]}, indent=2, default=float))
    return 0


def pair(args) -> int:
    from .tcga import pair_slides, read_sample_sheet
    sheet = read_sample_sheet(args.methylation_sheet) if args.methylation_sheet else None
    rows = pair_slides(Path(args.slides).read_text().split(), Path(args.methylation).read_text().split(),
                       sample_of=sheet)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "slide", "methylation", "sample_type", "group"])
        for r in rows:
            w.writerow([r.case_id, r.slide, r.methylation, r.sample_type, r.site])
    print(f"{len(rows)} cases paired -> {args.out}")
    return 0


def _read_folder(folder: str, sheet: str | None = None):
    """{case id (TCGA) or file stem: betas}. With a GDC sample sheet, names are mapped to barcodes first."""
    from .labels import read_beta_file
    from .tcga import parse, read_sample_sheet
    names = read_sample_sheet(sheet) if sheet else {}
    out = {}
    for p in sorted(Path(folder).rglob("*.txt")):
        b = parse(names.get(p.name, p.name))
        if b and b[1] != "01":
            continue                                   # primary tumour only, to match the diagnostic slide
        out.setdefault(b[0] if b else p.stem, read_beta_file(p))
    return out


def label(args) -> int:
    from .labels import ReferenceClassifier, from_report_csv, stack_betas
    if args.report:
        labs = from_report_csv(args.report, args.min_score)
    else:
        ref = _read_folder(args.ref_betas)
        with open(args.ref_labels, newline="") as f:
            y = {r["sample_id"]: r["class"] for r in csv.DictReader(f)}
        ids, cpgs, X = stack_betas({k: v for k, v in ref.items() if k in y})
        ex = Path(args.exclude_probes).read_text().split() if args.exclude_probes else None
        clf = ReferenceClassifier(exclude=ex).fit(X, cpgs, [y[i] for i in ids])
        print(f"reference classifier, out-of-bag: {json.dumps(clf.oob)}")
        tids, tcpgs, TX = stack_betas(_read_folder(args.target_betas, args.target_sheet), cpgs)
        labs = clf.label(tids, TX, tcpgs, args.min_score)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["case_id", "class", "score", "source"])
        w.writeheader()
        for lab in labs:
            w.writerow(lab.to_dict())
    print(f"{len(labs)} labelled cases -> {args.out}")
    return 0


def train_cmd(args) -> int:
    from ..wsi.bags import Bag
    from .classes import Taxonomy
    from .train import Case, train
    tax = Taxonomy.from_csv(args.taxonomy) if args.taxonomy else Taxonomy()
    slides_of = {}
    if args.pairs:
        with open(args.pairs, newline="") as f:
            for r in csv.DictReader(f):
                slides_of.setdefault(r["case_id"], []).append(r["slide"])
    cases, missing = [], 0
    with open(args.labels, newline="") as f:
        for r in csv.DictReader(f):
            names = slides_of.get(r["case_id"]) or [s for s in (r.get("slides") or "").split(";") if s]
            paths = [Path(args.bags) / f"{_stem(s)}.npz" for s in names]
            bags = [Bag.load(p) for p in paths if p.exists()]
            if bags:
                sc = float(r.get("score") or "nan")
                cases.append(Case(r["case_id"], bags, r["class"], 1.0 if sc != sc else sc))
            else:
                missing += 1
    print(f"{len(cases)} cases with bags; {missing} labelled cases had no bag in {args.bags}")
    m = train(cases, tax, args.model_id, args.version, k=args.folds, max_error=args.max_error,
              min_per_class=args.min_per_class, min_tiles=args.min_tiles, log=print)
    m.save(args.out)
    print(json.dumps(m.metrics, indent=2, default=float))
    return 0


def _stem(slide: str) -> str:
    """Slide file name without its extension (.svs, .tif, .ndpi, ...), as used for bag file names."""
    name = Path(slide).name
    return name[: -len(Path(name).suffix)] if Path(name).suffix else name


def predict_cmd(args) -> int:
    from ..wsi.bags import Bag
    from .db import build_actions, post_actions
    from .model import MethylationModel
    m = MethylationModel.load(args.model)
    rec = m.predict([Bag.load(p) for p in args.bags], args.study_id, triage_call=args.triage_call)
    print(json.dumps(rec, indent=2, default=float))
    if args.db:
        post_actions(args.db, build_actions(rec, m))
    return 0
