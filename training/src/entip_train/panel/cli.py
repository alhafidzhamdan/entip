"""``entip-panel``: train and run the glioma panel model (step 3 of the AI glioma workflow). Research use only.

  entip-panel demo                                    # synthetic cohort: train, gate, predict (CPU, about a minute)
  entip-panel labels --cbioportal samples.tsv [--long extra.csv] --out labels.json
  entip-panel train --bags bags/ --labels labels.json --out model/ [--config configs/glioma_panel.json]
  entip-panel predict --model model/ --bags bags/ [--scan-map scans.csv] [--db http://127.0.0.1:8705]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


def _patient_fn(scan_map):
    from .dataset import read_scan_map
    from .labels import tcga_patient
    if scan_map:
        m = read_scan_map(scan_map)
        return m.get
    return tcga_patient


def cmd_labels(a) -> int:
    from . import labels as L
    rep: Counter = Counter()
    tabs = []
    if a.cbioportal:
        tabs.append(L.from_cbioportal_pan_glioma(a.cbioportal, rep))
    for p in a.long or []:
        tabs.append(L.from_long_table(p, rep))
    lab = L.merge(*tabs)
    Path(a.out).write_text(json.dumps(lab, indent=1, sort_keys=True))
    print(json.dumps({"patients": len(lab), "per_marker": L.counts(L.apply_requirements(lab)),
                      "skipped": dict(rep)}, indent=1))
    return 0


def cmd_train(a) -> int:
    from .dataset import load_patients
    from .labels import tcga_site
    from .markers import load_config
    from .model import PanelModel
    labels = json.loads(Path(a.labels).read_text())
    group = None if a.scan_map else tcga_site
    pats = load_patients(a.bags, _patient_fn(a.scan_map), group)
    pats = [p for p in pats if p.study_id in labels]
    print(f"[panel] {len(pats)} patients with bags and labels")
    m = PanelModel(load_config(a.config)).fit(pats, labels, source=a.source)
    m.save(a.out)
    print(f"[panel] saved to {a.out}")
    return 0


def cmd_predict(a) -> int:
    from .dataset import load_patients
    from .db import model_actions, post_actions, step_action, validate
    from .model import PanelModel
    from .predict import predict_patient, summarise
    m = PanelModel.load(a.model)
    pats = load_patients(a.bags, _patient_fn(a.scan_map))
    out = []
    for p in pats:
        recs = predict_patient(m, p)
        validate(recs)
        out.append({"study_id": p.study_id, "summary": summarise(recs), "records": recs})
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    for o in out:
        print(o["study_id"], json.dumps(o["summary"]))
    if a.db:
        acts = [model_actions(m, uri=a.model_uri)] + [step_action(m, o["study_id"], o["records"]) for o in out]
        res = post_actions(a.db, acts)
        print(f"[panel] wrote {len(res)} actions to {a.db}")
    return 0


def cmd_demo(a) -> int:
    from .db import validate
    from .markers import load_config
    from .model import PanelModel
    from .predict import predict_patient, summarise
    from .synthetic import synthetic_cohort
    train, labels, _ = synthetic_cohort(a.n, seed=0)
    test, tlabels, _ = synthetic_cohort(100, seed=1, prefix="TST")
    cfg = load_config(a.config, abmil={"hidden": 32, "attn": 16, "epochs": 15, "max_tiles": 128})
    m = PanelModel(cfg).fit(train, labels, source="synthetic")
    right = called = total = 0
    for p in test:
        recs = predict_patient(m, p)
        validate(recs)
        for r in recs:
            truth = tlabels[p.study_id].get(r["finding"])
            if truth is None:
                continue
            total += 1
            if r["status"] == "confident":
                called += 1
                right += r["call"] == truth
    print(json.dumps(summarise(predict_patient(m, test[0])), indent=1))
    print(f"[panel] held-out synthetic patients: {called}/{total} marker calls accepted, "
          f"{right}/{called} of those correct; the rest order the real test")
    out = Path(a.out or tempfile.mkdtemp(prefix="entip-panel-"))
    m.save(out)
    print(f"[panel] model saved to {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="entip-panel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--config")
    d.add_argument("--out")
    d.set_defaults(fn=cmd_demo)
    lb = sub.add_parser("labels")
    lb.add_argument("--cbioportal", help="cBioPortal lgggbm_tcga_pub sample clinical table")
    lb.add_argument("--long", action="append", help="CSV patient_id,finding,call (repeatable)")
    lb.add_argument("--out", required=True)
    lb.set_defaults(fn=cmd_labels)
    t = sub.add_parser("train")
    t.add_argument("--bags", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config")
    t.add_argument("--scan-map", help="CSV scan_id,study_id (default: TCGA barcodes)")
    t.add_argument("--source", default="TCGA-GBM+LGG")
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--bags", required=True)
    p.add_argument("--scan-map")
    p.add_argument("--out")
    p.add_argument("--db", help="database API base URL, e.g. http://127.0.0.1:8705")
    p.add_argument("--model-uri", default="")
    p.set_defaults(fn=cmd_predict)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
