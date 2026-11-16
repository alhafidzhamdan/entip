"""Digital Brain Tumour Atlas (DBTA) -> an external test manifest for `entip-glioma triage evaluate`.

    python make_manifest.py --metadata dbta_metadata.csv --slides /path/to/dbta_download --out runs/dbta

* ``--metadata``: the DBTA sample table (one row per slide: uuid, pat_id, diagnosis, ..., control, comment), from
  the dataset's EBRAINS page (Roetzer-Pejrimovsky et al. 2022, Scientific Data).
* ``--slides``: the folder you downloaded the .ndpi files into (searched recursively; matched by uuid in the name).
* ``dbta_classes.csv`` (next to this script) maps each DBTA diagnosis to a triage class, or to none (left out).
  It is a judgement table: review it, and set the controls (normal_brain or reactive) there or per slide.

Writes, in ``--out``: ``manifest.csv`` (SeriesInstanceUID = uuid, PatientID = pat_id, label, diagnosis, ...),
``left_out.csv`` (every slide not in the manifest and why) and ``slides/<uuid>.ndpi`` links to the files, so each
bag is named by its uuid. The DBTA is research-only and pseudonymised personal data (EBRAINS Data Use Agreement):
it is used here to test, never to train.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_table(path: str) -> list:
    text = Path(path).read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    return list(csv.DictReader(text.splitlines(), dialect=dialect))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=str(HERE / "dbta_classes.csv"))
    a = ap.parse_args()

    classes = {r["diagnosis"].strip(): r for r in csv.DictReader(open(a.classes, newline=""))}
    meta = read_table(a.metadata)
    need = {"uuid", "pat_id", "diagnosis"}
    if not meta or not need <= set(meta[0]):
        raise SystemExit(f"metadata needs columns {sorted(need)}; has {sorted(meta[0]) if meta else 'no rows'}")
    files = {}
    for p in Path(a.slides).rglob("*.ndpi"):
        files.setdefault(p.stem.lower(), p)
    out = Path(a.out)
    (out / "slides").mkdir(parents=True, exist_ok=True)
    kept, left = [], []
    for r in meta:
        uid, diag = r["uuid"].strip(), (r.get("diagnosis") or "").strip()
        is_control = str(r.get("control", "")).strip().lower() in ("1", "true", "yes")
        key = "control" if is_control else diag
        c = classes.get(key)
        why = None
        if c is None:
            why = f"diagnosis not in dbta_classes.csv: {key!r}"
        elif not c["triage_class"]:
            why = c["note"] or "no triage class"
        f = files.get(uid.lower()) or next((p for s, p in files.items() if uid.lower() in s), None)
        if why is None and f is None:
            why = "no .ndpi file found for this uuid"
        if why:
            left.append({"uuid": uid, "pat_id": r.get("pat_id"), "diagnosis": diag, "control": int(is_control), "why": why})
            continue
        link = out / "slides" / f"{uid}.ndpi"
        if not link.exists():
            os.symlink(f.resolve(), link)
        kept.append({"SeriesInstanceUID": uid, "PatientID": f"DBTA-{r['pat_id'].strip()}", "label": c["triage_class"],
                     "diagnosis": "control" if is_control else diag, "group": "dbta|Hamamatsu|ffpe",
                     "collection_id": "dbta", "series_size_MB": round(f.stat().st_size / 1e6, 1),
                     "recurrence": r.get("recurrence", ""), "note": c["note"]})
    if not kept:
        raise SystemExit("no slides matched: check --slides and the metadata")
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    with (out / "left_out.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["uuid", "pat_id", "diagnosis", "control", "why"])
        w.writeheader()
        w.writerows(left)
    print(f"{len(kept)} slides from {len({k['PatientID'] for k in kept})} patients in the manifest; "
          f"{len(left)} left out ({out / 'left_out.csv'})")
    print("  by class:", dict(Counter(k["label"] for k in kept)))
    print("  left out:", dict(Counter(x["why"].split(":")[0] for x in left)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
