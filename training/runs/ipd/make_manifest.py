"""IPD-Brain (Indian Pathology Dataset, NIMS Hyderabad) -> an external test manifest for `entip-glioma triage evaluate`.

    python make_manifest.py --metadata IPD_Brain.csv --slides /path/to/ipd_download --out runs/ipd

* ``--metadata``: the dataset's clinical table (one row per patient: Case Number, Age, Sex, Diagnosis, WHO Grade,
  Subtype, IDH1R132H, ATRX, p53, ...). A patient with several slides lists them all in Case Number, one per line
  ("IN Brain-0004(a)" and "IN Brain-0004(b)").
* ``--slides``: the folder you downloaded the 40x TIFF files into (searched recursively; matched by slide name,
  ignoring case, spaces and punctuation).

Every labelled patient is a glioma (glioblastoma, astrocytoma or oligodendroglioma), so every slide is labelled
``glioma``. The diagnosis column records the subtype and the IHC result (IDH1 R132H positive or negative): the
stain misses the rarer IDH mutations, so a negative stain is not a confirmed IDH-wildtype. The unlabelled slides
in the download have no table row and are left out.

Writes, in ``--out``: ``manifest.csv`` (SeriesInstanceUID = a file-safe slide id such as IPD-0004a, PatientID =
IPD-0004, label, diagnosis, ...), ``left_out.csv`` (every table slide without a file, and every file without a
table row) and ``slides/<id>.<ext>`` links, so each bag is named by its id. IPD-Brain is used under the India Data
Data User Agreement: research use, no commercial use without the provider's written consent. It is used here to
test, never to train.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

EXTS = (".tif", ".tiff", ".svs", ".ndpi")
CASE = re.compile(r"^IN\s*Brain-(\d+)\s*(?:\(([a-z])\))?$", re.I)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slide_ids(case_cell: str) -> list:
    """'IN Brain-0004(a)\\nIN Brain-0004(b)' -> [('IN Brain-0004(a)', 'IPD-0004a', 'IPD-0004'), ...]"""
    out = []
    for raw in re.split(r"[\n;]+", case_cell):
        raw = raw.strip()
        if not raw:
            continue
        m = CASE.match(raw)
        if not m:
            raise SystemExit(f"unexpected Case Number {raw!r}: expected like 'IN Brain-0004' or 'IN Brain-0004(a)'")
        out.append((raw, f"IPD-{m.group(1)}{(m.group(2) or '').lower()}", f"IPD-{m.group(1)}"))
    return out


def diagnosis(r: dict) -> str:
    sub = (r.get("Subtype") or "").strip().lower() or "glioma"
    idh = {"1": "IDH1 R132H positive", "0": "IDH1 R132H negative"}.get(str(r.get("IDH1R132H", "")).strip(),
                                                                       "IDH1 R132H unknown")
    return f"{sub} ({idh})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.metadata, newline="", encoding="utf-8-sig", errors="replace")))
    need = {"Case Number", "Subtype", "IDH1R132H"}
    if not rows or not need <= set(rows[0]):
        raise SystemExit(f"metadata needs columns {sorted(need)}; has {sorted(rows[0]) if rows else 'no rows'}")
    files = {}
    for p in Path(a.slides).rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            files.setdefault(norm(p.stem), p)
    out = Path(a.out)
    (out / "slides").mkdir(parents=True, exist_ok=True)
    kept, left, used = [], [], set()
    for r in rows:
        for raw, sid, pid in slide_ids(r["Case Number"]):
            f = files.get(norm(raw))
            if f is None:
                left.append({"slide": raw, "patient": pid, "why": "no slide file found for this case number"})
                continue
            used.add(norm(raw))
            link = out / "slides" / f"{sid}{f.suffix.lower()}"
            if not link.exists():
                os.symlink(f.resolve(), link)
            kept.append({"SeriesInstanceUID": sid, "PatientID": pid, "label": "glioma", "diagnosis": diagnosis(r),
                         "who_grade": (r.get("WHO Grade") or "").strip(), "idh1_r132h_ihc": r.get("IDH1R132H", ""),
                         "group": "ipd|Morphle|ffpe", "collection_id": "ipd_brain",
                         "series_size_MB": round(f.stat().st_size / 1e6, 1), "file": f.name})
    for k, f in sorted(files.items()):
        if k not in used:
            left.append({"slide": f.name, "patient": "", "why": "no table row (unlabelled slide)"})
    if not kept:
        raise SystemExit("no slides matched: check --slides and the metadata")
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    with (out / "left_out.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slide", "patient", "why"])
        w.writeheader()
        w.writerows(left)
    print(f"{len(kept)} slides from {len({k['PatientID'] for k in kept})} patients in the manifest; "
          f"{len(left)} left out ({out / 'left_out.csv'})")
    print("  by diagnosis:", dict(Counter(k["diagnosis"] for k in kept)))
    print("  left out:", dict(Counter(x["why"] for x in left)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
