"""Step 3a: test whether TCGA pathology reports can give case-level MVP labels.

    python report_labels.py fetch --out mvp-reports
    python report_labels.py extract --out mvp-reports --workers 8
    python report_labels.py check --out mvp-reports --labelled key.csv labels.csv --manifest trial.csv --bags bags/

**fetch** downloads open-access TCGA-GBM and TCGA-LGG report PDFs from the NCI GDC and verifies each checksum.
**extract** reads each PDF's text layer, uses OCR for scanned pages, and labels each case by rule, with no external AI
service:

* ``present``: the report states MVP;
* ``absent``: the report explicitly states there is none;
* ``unclear``: hedged, general, form-like or conflicting wording;
* ``not_mentioned``: never treated as absent.

**check** reports coverage, compares report labels with patients who have reviewed MVP tiles, and draws a blinded random
sample for slide review. A report describes the case and a slide is one sample, so labels are unproven until that review.
Report text and snippets stay in ``private/``. Do not try to identify participants; acknowledge TCGA and the GDC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

API = "https://api.gdc.cancer.gov"
PROJECTS = ("TCGA-GBM", "TCGA-LGG")
LABELS = ("present", "absent", "unclear", "not_mentioned")
HEADERS = {"User-Agent": "research-data-client", "Content-Type": "application/json"}

TERM = re.compile(
    r"(?:micro\s*-?\s*vascular|vascular|vessel|capillary|endothelial(?:\s+cell)?)\s+(?:cell\s+)?(?:prolif\w*|hyperplasia)"
    r"|glomeruloid(?:\s+\w+)?|\bmvp\b"
    r"|prolif\w+\s+(?:of\s+)?(?:small\s+)?(?:blood\s+)?(?:vessels|endothelium|endothelial cells|capillaries)")
BREAK = re.compile(r"\b(?:but|however|although|though|except|yet|while|whereas)\b|,\s*(?:with|and\s+(?:prominent|marked|"
                   r"focal|florid))\b")
NEG_BEFORE = re.compile(r"\b(?:no|not|without|absent|absence of|negative for|lacks?|lacking|none|neither|nor|free of|"
                        r"devoid of|fails? to show)\b")
NEG_AFTER = re.compile(r"^\W*(?:is|are|was|were|:)?\s*(?:not\b|absent\b|none\b|negative\b|no\b)")
HEDGE = re.compile(r"\b(?:cannot|can't|could not|suspicious|suggest\w*|possibl\w*|probabl\w*|questionabl\w*|equivocal|"
                   r"early|incipient|borderline|rule[sd]? out|not excluded|may|might)\b")
GENERIC = re.compile(r"\b(?:criteria|criterion|defin\w+|required|requires|would|characteri[sz]\w+|typically|usually|"
                     r"grading|grade is based|if)\b")
FORM = re.compile(r"\b(?:present|yes)\b.{0,25}\b(?:absent|no)\b|\b(?:absent|no)\b.{0,25}\b(?:present|yes)\b")


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def mention(text: str, m) -> str:
    stop = max(text.rfind(s, 0, m.start()) for s in (". ", "; ", "? ", "! "))
    start = stop + 2 if stop >= 0 else 0
    ends = [i for i in (text.find(s, m.end()) for s in (". ", "; ", "? ", "! ")) if i >= 0]
    end = min(ends) if ends else len(text)
    before, after = text[start:m.start()], text[m.end():end]
    if FORM.search(after[:50]) or GENERIC.search(before + " " + after) or HEDGE.search(before + " " + after):
        return "unclear"
    before = BREAK.split(before)[-1]
    return "absent" if NEG_BEFORE.search(before) or NEG_AFTER.search(after[:45]) else "present"


def classify(text: str) -> tuple[str, list]:
    """Case-level label from one or more reports, with each mention and its surrounding text."""
    text, found = clean(text), []
    for m in TERM.finditer(text):
        found.append((mention(text, m), text[max(0, m.start() - 70):m.end() + 70]))
    calls = {c for c, _ in found}
    if not calls:
        return "not_mentioned", found
    if "unclear" in calls or len(calls) > 1:
        return "unclear", found
    return calls.pop(), found


def api_files(api: str) -> list:
    body = {"filters": {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": list(PROJECTS)}},
        {"op": "in", "content": {"field": "data_type", "value": ["Pathology Report"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}}]},
        "fields": "file_id,file_name,md5sum,file_size,access,cases.submitter_id,cases.project.project_id",
        "format": "JSON", "size": 5000}
    req = urllib.request.Request(f"{api}/files", json.dumps(body).encode(), HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["data"]["hits"]


def download(api: str, row: dict, folder: Path, tries: int = 4) -> str | None:
    path = folder / f"{row['file_id']}.pdf"
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest() == row["md5"]:
        return None
    for n in range(tries):
        try:
            req = urllib.request.Request(f"{api}/data/{row['file_id']}", headers={"User-Agent": HEADERS["User-Agent"]})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if hashlib.md5(data).hexdigest() != row["md5"]:
                raise ValueError("checksum mismatch")
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return None
        except Exception as e:  # network errors and bad checksums are retried
            if n == tries - 1:
                return f"{row['file_id']}: {e}"
            time.sleep(2 ** n)
    return None


def fetch(a) -> int:
    out = Path(a.out) / "private"
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    rows = []
    for f in api_files(a.api):
        cases = f.get("cases") or []
        if f.get("access") != "open" or len(cases) != 1:
            continue
        rows.append({"file_id": f["file_id"], "patient": cases[0]["submitter_id"],
                     "project": cases[0]["project"]["project_id"], "md5": f["md5sum"], "size": f["file_size"]})
    if not rows:
        raise SystemExit("GDC returned no reports to download")
    with ThreadPoolExecutor(a.workers) as pool:
        errors = [e for e in pool.map(lambda r: download(a.api, r, out / "pdf"), rows) if e]
    if errors:
        raise SystemExit(f"{len(errors)} downloads failed, e.g. {errors[0]}")
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"files": len(rows), "patients": len({r["patient"] for r in rows}),
                      "mb": round(sum(int(r["size"]) for r in rows) / 1e6, 1)}, indent=2))
    return 0


def pdf_text(path: str, min_chars: int = 40, dpi: int = 250) -> tuple[str, int, int]:
    """Text of a PDF, with OCR for pages without a usable text layer. Returns text, pages and OCR pages."""
    import pypdfium2 as pdfium
    import pytesseract
    doc, parts, ocr = pdfium.PdfDocument(path), [], 0
    try:
        for i in range(len(doc)):
            page = doc[i]
            tp = page.get_textpage()
            text = tp.get_text_range()
            tp.close()
            if len(text.strip()) < min_chars:
                image = page.render(scale=dpi / 72).to_pil().convert("L")
                text = pytesseract.image_to_string(image, config="--psm 4")
                ocr += 1
            page.close()
            parts.append(text)
        return "\n".join(parts), len(doc), ocr
    finally:
        doc.close()


def read_one(row: dict, base: str) -> dict:
    path = Path(base) / "text" / f"{row['file_id']}.txt"
    try:
        text, pages, ocr = pdf_text(str(Path(base) / "pdf" / f"{row['file_id']}.pdf"))
        path.write_text(text)
        return {**row, "pages": pages, "ocr_pages": ocr, "chars": len(text), "error": ""}
    except Exception as e:  # one unreadable report must not stop the run; it is counted below
        return {**row, "pages": 0, "ocr_pages": 0, "chars": 0, "error": str(e)[:120]}


def extract(a) -> int:
    base = Path(a.out) / "private"
    rows = list(csv.DictReader(open(base / "manifest.csv", newline="")))
    (base / "text").mkdir(exist_ok=True)
    done = []
    with ProcessPoolExecutor(max(1, a.workers or int(os.environ.get("NSLOTS", 1)))) as pool:
        jobs = [pool.submit(read_one, r, str(base)) for r in rows]
        for n, job in enumerate(as_completed(jobs), start=1):
            done.append(job.result())
            if n % 50 == 0 or n == len(rows):
                print(f"read {n}/{len(rows)} reports", file=sys.stderr, flush=True)
    by = defaultdict(list)
    for r in done:
        by[r["patient"]].append(r)
    labels, audit = [], []
    for patient, files in sorted(by.items()):
        readable = [r for r in files if not r["error"] and r["chars"] >= a.min_chars]
        if not readable:
            labels.append({"patient": patient, "project": files[0]["project"], "label": "unreadable", "files": len(files),
                           "mentions": 0, "chars": 0})
            continue
        text = "\n".join((base / "text" / f"{r['file_id']}.txt").read_text() for r in readable)
        label, found = classify(text)
        labels.append({"patient": patient, "project": files[0]["project"], "label": label, "files": len(files),
                       "mentions": len(found), "chars": len(text)})
        audit += [{"patient": patient, "mention_call": c, "snippet": s} for c, s in found]
    for name, data, cols in (("labels.csv", labels, ["patient", "project", "label", "files", "mentions", "chars"]),
                             ("audit.csv", audit, ["patient", "mention_call", "snippet"])):
        with (base / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print(json.dumps({"reports": len(done), "patients": len(labels), "read_errors": sum(bool(r["error"]) for r in done),
                      "reports_using_ocr": sum(int(r["ocr_pages"]) > 0 for r in done),
                      "labels": dict(Counter(r["label"] for r in labels))}, indent=2))
    return 0


def check(a) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
    from common import read_manifest
    base = Path(a.out) / "private"
    out = base / "review-sample.csv"
    if out.exists():
        raise SystemExit(f"{out} already exists: the review sample is never redrawn")
    labels = {r["patient"]: r for r in csv.DictReader(open(base / "labels.csv", newline=""))}
    tile_mvp, tile_seen = set(), set()
    for key_path, label_path in a.labelled:
        key = {r["tile_code"]: r for r in csv.DictReader(open(key_path, newline=""))}
        for r in csv.DictReader(open(label_path, newline="")):
            tile_seen.add(key[r["tile_code"]]["patient"])
            if r["label"].strip().lower() == "m":
                tile_mvp.add(key[r["tile_code"]]["patient"])
    usable = [r for r in labels.values() if r["label"] in LABELS]
    summary = {"patients": len(labels), "unreadable": len(labels) - len(usable), "by_project": {}}
    for p in PROJECTS:
        c = Counter(r["label"] for r in usable if r["project"] == p)
        n = sum(c.values())
        summary["by_project"][p] = {"patients": n, **{k: c[k] for k in LABELS},
                                    "explicit_share": round((c["present"] + c["absent"]) / n, 3) if n else None}
    explicit = sum(r["label"] in ("present", "absent") for r in usable)
    summary["explicit_share_all"] = round(explicit / len(usable), 3) if usable else None
    summary["patients_with_reviewed_mvp_tiles"] = dict(Counter(labels[p]["label"] for p in tile_mvp if p in labels))
    summary["reviewed_patients_with_no_mvp_tile"] = dict(Counter(labels[p]["label"] for p in tile_seen - tile_mvp
                                                                 if p in labels))
    slides = defaultdict(list)
    for r in read_manifest(a.manifest):
        if r.get("label", "glioma") == "glioma" and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists():
            slides[r["PatientID"]].append(r["SeriesInstanceUID"])
    rng, sample = random.Random(a.seed), []
    for label in LABELS:
        pool = sorted(p for p, r in labels.items() if r["label"] == label and p in slides)
        for p in rng.sample(pool, min(a.per_label, len(pool))):
            sample.append({"patient": p, "slide": rng.choice(sorted(slides[p])), "project": labels[p]["project"],
                           "report_label": label})
    rng.shuffle(sample)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "slide", "project", "report_label"])
        w.writeheader()
        w.writerows(sample)
    summary["review_sample"] = dict(Counter(r["report_label"] for r in sample))
    summary["go_rule_set_in_advance"] = ("most reports explicit, then stated-present and stated-absent each at least 90% "
                                         "correct on blinded slide review")
    (Path(a.out) / "report-check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download open-access report PDFs")
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=4)
    f.add_argument("--api", default=API)
    f.set_defaults(fn=fetch)
    e = sub.add_parser("extract", help="read reports and label MVP by rule")
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int)
    e.add_argument("--min-chars", type=int, default=100, help="fewer characters than this counts as unreadable")
    e.set_defaults(fn=extract)
    c = sub.add_parser("check", help="coverage, agreement with reviewed tiles, and a blinded review sample")
    c.add_argument("--out", required=True)
    c.add_argument("--labelled", nargs=2, action="append", required=True, metavar=("KEY", "LABELS"))
    c.add_argument("--manifest", required=True)
    c.add_argument("--bags", required=True)
    c.add_argument("--per-label", type=int, default=16)
    c.add_argument("--seed", type=int, default=20261122)
    c.set_defaults(fn=check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
