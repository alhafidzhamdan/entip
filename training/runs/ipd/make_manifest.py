"""IPD-Brain (Indian Pathology Dataset, NIMS Hyderabad) -> an external test manifest for `entip-glioma triage evaluate`.

    python make_manifest.py --metadata IPD_Brain.csv --slides /path/to/ipd_download --out runs/ipd

* ``--metadata``: the dataset's clinical table (one row per patient: Case Number, Age, Sex, Diagnosis, WHO Grade,
  Subtype, IDH1R132H, ATRX, p53, ...). A patient with several slides lists them all in Case Number, one per line
  ("IN Brain-0004(a)" and "IN Brain-0004(b)").
* ``--slides``: the folder you downloaded the 40x TIFF files into (searched recursively; matched by slide name,
  ignoring case, spaces and punctuation).

Every labelled patient is a glioma (glioblastoma, astrocytoma or oligodendroglioma), so every slide is labelled
``glioma``. The diagnosis column is the WHO CNS5 term from the subtype, grade, IDH1 R132H stain and age, with
``who_cns5_basis`` saying what it rests on. The stain misses the rarer IDH mutations and 1p/19q was not tested,
so many cases are NOS: oligodendrogliomas always, and stain-negative glioblastomas in patients aged 55 or under. The unlabelled slides
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


GRADES = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def _grade(v) -> int | None:
    m = re.search(r"\b(iv|iii|ii|i|[1-4])\b", str(v or "").strip().lower())
    return GRADES.get(m.group(1)) if m else None


def _age(v) -> float | None:
    try:
        return float(str(v).strip())
    except ValueError:
        return None


def diagnosis(r: dict) -> tuple[str, str]:
    """WHO CNS5 term and its basis from the subtype, the WHO grade, the IDH1 R132H stain and age.

    IDH comes from IDH1 R132H IHC only and 1p/19q was not tested. WHO CNS5 accepts a negative stain as
    IDH-wildtype for a glioblastoma in a patient over 55; younger patients need sequencing, so those are NOS."""
    sub = (r.get("Subtype") or "").strip().lower()
    ihc = str(r.get("IDH1R132H", "")).strip()
    g, age = _grade(r.get("WHO Grade")), _age(r.get("Age"))
    stain = {"1": "IDH1 R132H IHC positive", "0": "IDH1 R132H IHC negative"}.get(ihc, "IDH1 R132H IHC unknown")
    basis = f"IPD-Brain: {sub or 'no subtype'}, WHO grade {g or '?'}, {stain}, age {age if age is not None else '?'}"
    gr = f", CNS WHO grade {g}" if g else ""
    if sub == "astrocytoma" and g == 1:
        return "Astrocytoma, NOS", basis + "; grade 1 recorded: probably circumscribed (e.g. pilocytic), not diffuse"
    if ihc == "1" and sub in ("glioblastoma", "astrocytoma"):
        return "Astrocytoma, IDH-mutant" + (", CNS WHO grade 4" if sub == "glioblastoma" else gr), \
            basis + "; CDKN2A/B not tested (a homozygous deletion would make it grade 4)"
    if sub == "glioblastoma" and ihc == "0":
        if age is not None and age > 55:
            return "Glioblastoma, IDH-wildtype, CNS WHO grade 4", basis + "; negative stain accepted over 55"
        return "High-grade glioma, NOS", basis + "; under 55 a negative stain needs IDH sequencing"
    if sub == "oligodendroglioma":
        return "Oligodendroglioma, NOS", basis + "; 1p/19q not tested"
    if sub == "astrocytoma":
        return "Diffuse glioma, NOS", basis + "; IDH not confirmed"
    return "Diffuse glioma, NOS", basis


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
            dx, basis = diagnosis(r)
            kept.append({"SeriesInstanceUID": sid, "PatientID": pid, "label": "glioma", "diagnosis": dx,
                         "who_cns5_basis": basis, "ipd_subtype": (r.get("Subtype") or "").strip(),
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
