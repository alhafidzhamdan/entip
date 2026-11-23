"""Step 3b: choose the TCGA FFPE slides of every patient whose report clearly states necrosis or no necrosis.

    python necrosis_cohort.py pick --manifest manifest.csv --idh idh.tsv --labels report-labels.csv --out run/
    python necrosis_cohort.py check --trial run/trial.csv --bags run/bags-h-optimus-0

**pick** writes ``trial.csv`` and ``cohort.json``: patients, slides and download size per tumour category and report
label, with no patient names. It downloads nothing, so the size can be checked first.

**record** saves a pathologist's reading of the low-grade reports that the rule called present. Those calls replace the
rule's necrosis label. Case grades are not changed: a different diagnosis is stored beside the original category.

**check** stops training if more than ``--max-missing`` of the chosen slides have no embeddings.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import category  # noqa: E402

CATEGORIES = ("gbm", "astro4", "oligo3", "astro3", "grade2")
COLLECTIONS = ("tcga_gbm", "tcga_lgg")
CLEAR = ("present", "absent")


def choose(rows: list, labels: dict, min_mb: float, annotate) -> list:
    keep = [r for r in rows if r.get("label") == "glioma" and r.get("collection_id") in COLLECTIONS
            and r.get("prep") == "ffpe" and labels.get(r["PatientID"]) in CLEAR
            and float(r.get("series_size_MB") or 0) >= min_mb]
    return [r for r in annotate(keep) if category(r) in CATEGORIES]


def describe(rows: list, labels: dict) -> dict:
    out = {c: {k: {"patients": set(), "slides": 0, "gb": 0.0} for k in CLEAR} for c in CATEGORIES}
    for r in rows:
        cell = out[category(r)][labels[r["PatientID"]]]
        cell["patients"].add(r["PatientID"])
        cell["slides"] += 1
        cell["gb"] += float(r["series_size_MB"]) / 1000
    table = {c: {k: {"patients": len(v["patients"]), "slides": v["slides"], "gb": round(v["gb"], 1)}
                 for k, v in cells.items()} for c, cells in out.items()}
    return {"patients": len({r["PatientID"] for r in rows}), "slides": len(rows),
            "download_gb": round(sum(float(r["series_size_MB"]) for r in rows) / 1000, 1), "by_category": table}


def pick(a) -> int:
    from entip_glioma.triage import who_cns5
    from entip_glioma.triage.manifest import read_corrections
    out = Path(a.out)
    if (out / "trial.csv").exists():
        raise SystemExit(f"{out / 'trial.csv'} already exists: a selection is never redone")
    labels = {r["patient"]: r["label"] for r in csv.DictReader(open(a.labels, newline=""))}
    rows = list(csv.DictReader(open(a.manifest, newline="")))
    molecular, corrections = who_cns5.read_molecular(a.idh), read_corrections()
    chosen = choose(rows, labels, a.min_mb, lambda keep: who_cns5.annotate(keep, molecular, corrections))
    if not chosen:
        raise SystemExit("no slides match: check the manifest, IDH table and report labels")
    fields = list(rows[0]) + [k for k in ("diagnosis", "who_cns5_basis") if k not in rows[0]]
    chosen.sort(key=lambda r: r["SeriesInstanceUID"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trial.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(chosen)
    info = describe(chosen, labels)
    (out / "cohort.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


def check(a) -> int:
    rows = list(csv.DictReader(open(a.trial, newline="")))
    missing = sum(not (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists() for r in rows)
    print(json.dumps({"slides": len(rows), "missing_embeddings": missing}))
    if not rows or missing / len(rows) > a.max_missing:
        raise SystemExit(f"{missing} of {len(rows)} slides have no embeddings: not training")
    return 0


def record(a) -> int:
    """Saves a pathologist's reading of the low-grade reports the rule called present, in their case order."""
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists: a review is recorded once")
    labels = {r["patient"]: r["label"] for r in csv.DictReader(open(a.labels, newline=""))}
    cats = {r["PatientID"]: category(r) for r in csv.DictReader(open(a.trial, newline=""))}
    chosen = sorted(p for p, c in cats.items() if c in ("astro3", "grade2") and labels.get(p) == "present")
    calls = {}
    for item in a.calls:
        m = re.fullmatch(r"case(\d+)=(present|absent)(?::(\w+))?", item)
        if not m or (m[3] and m[3] not in CATEGORIES):
            raise SystemExit(f"cannot read {item!r}: use case01=present:oligo3 or case03=absent")
        calls[int(m[1])] = (m[2], m[3] or "")
    if set(calls) != set(range(1, len(chosen) + 1)):
        raise SystemExit(f"give one call for each of the {len(chosen)} cases")
    rows = [{"patient": p, "necrosis": calls[n][0], "diagnosis": calls[n][1]} for n, p in enumerate(chosen, 1)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "necrosis", "diagnosis"])
        w.writeheader()
        w.writerows(rows)
    out.chmod(0o600)
    print(json.dumps({"reviewed": len(rows), "necrosis": dict(Counter(r["necrosis"] for r in rows)),
                      "category_disagreements": sum(bool(r["diagnosis"]) and r["diagnosis"] != cats[r["patient"]]
                                                    for r in rows)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="save the reviewed necrosis calls; case grades are not changed")
    r.add_argument("--trial", required=True)
    r.add_argument("--labels", required=True, help="report-labels.csv")
    r.add_argument("--out", required=True)
    r.add_argument("--calls", nargs="+", required=True, help="case01=present:oligo3 case03=absent ...")
    r.set_defaults(fn=record)
    p = sub.add_parser("pick", help="choose the slides and report the download size")
    p.add_argument("--manifest", required=True)
    p.add_argument("--idh", required=True)
    p.add_argument("--labels", required=True, help="report-labels.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--min-mb", type=float, default=100)
    p.set_defaults(fn=pick)
    c = sub.add_parser("check", help="confirm the chosen slides were embedded")
    c.add_argument("--trial", required=True)
    c.add_argument("--bags", required=True)
    c.add_argument("--max-missing", type=float, default=0.05)
    c.set_defaults(fn=check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
