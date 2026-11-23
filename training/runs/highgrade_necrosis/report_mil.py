"""Step 3b: find TCGA cases whose report states necrosis or no necrosis, and train a first attention model on them.

    python report_mil.py --reports mvp-reports/private --manifest trial.csv --bags bags/ --out necrosis-report1 \
        [--holdout held_out.csv] [--test-fraction 0.2]

**Labels.** Each case is labelled by rule from the text of its report, with no external AI service: ``present``,
``absent``, ``unclear`` or ``not_mentioned``. Only ``present`` and ``absent`` are used. Hedged, general, form-like,
conflicting or treatment-related wording is ``unclear``, and "not mentioned" is never treated as absent.

**Training.** A report describes the case, so all of a patient's slides form one bag. Patients in ``--holdout`` and a
random ``--test-fraction`` are never loaded for training or scored here. The rest are cross-validated by patient.

**Reading the result.** Necrosis tracks grade, so ``summary.json`` also scores a baseline that knows only the tumour
category, and gives the AUC within each category. A model that does not beat the baseline has learnt tumour type. The
labels are unproven until compared with blinded slide review, so the saved model is a draft and makes no clinical call.

Report text, snippets and patient lists stay in ``private/``. Acknowledge TCGA and the GDC.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import read_manifest, read_patients  # noqa: E402
from grade_mil import auc  # noqa: E402

CATEGORIES = ("gbm", "astro4", "oligo3", "astro3", "grade2")
NOT_THIS = re.compile(r"\btumou?r necrosis factor\b|\btnf\b")
TERM = re.compile(r"\bnecro(?:sis|ses|tic)\b|\bpseudo-? ?palisad\w*")
BREAK = re.compile(r"\b(?:but|however|although|though|except|yet|while|whereas)\b")
NEG_BEFORE = re.compile(r"\b(?:no|not|without|absent|absence of|negative for|lacks?|lacking|none|neither|nor|free of|"
                        r"devoid of|fails? to show)\b")
NEG_AFTER = re.compile(r"^\s*:?\s*(?:(?:is|are|was|were)\s+)?(?:not|absent|none|negative|no)\b")
STATED_AFTER = re.compile(r"^\s*:?\s*(?:(?:is|are|was|were)\s+)?(?:present|seen|identified|noted|evident)\b")
HEDGE = re.compile(r"\b(?:cannot|can't|could not|suspicious|suggest\w*|possibl\w*|probabl\w*|questionabl\w*|equivocal|"
                   r"early|incipient|borderline|rule[sd]? out|not excluded|may|might)\b")
GENERIC = re.compile(r"\b(?:criteria|criterion|defin\w+|required|requires|would|characteri[sz]\w+|typically|usually|"
                     r"grading|grade is based|if)\b")
TREATED = re.compile(r"\b(?:radiat\w*|irradiat\w*|radiotherapy|treatment|treated|therapy|chemo\w*|recurr\w*|prior|"
                     r"previous\w*|history|embol\w*|infarct\w*)\b")
FORM = re.compile(r"\b(?:present|yes)\b.{0,25}\b(?:absent|no)\b|\b(?:absent|no)\b.{0,25}\b(?:present|yes)\b")


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text.lower())
    return NOT_THIS.sub(" ", re.sub(r"\s+", " ", text)).strip()


def mention(text: str, m) -> str:
    stop = max(text.rfind(s, 0, m.start()) for s in (". ", "; ", "? ", "! "))
    start = stop + 2 if stop >= 0 else 0
    ends = [i for i in (text.find(s, m.end()) for s in (". ", "; ", "? ", "! ")) if i >= 0]
    end = min(ends) if ends else len(text)
    before, after = text[start:m.start()], text[m.end():end]
    around = before + " " + after
    if FORM.search(after[:50]) or GENERIC.search(around) or HEDGE.search(around) or TREATED.search(around):
        return "unclear"
    before = BREAK.split(before)[-1]
    negs = list(NEG_BEFORE.finditer(before))
    if negs and "," in before[negs[-1].end():] and STATED_AFTER.search(after[:30]):
        return "unclear"  # "no mitoses, necrosis present"
    return "absent" if negs or NEG_AFTER.search(after[:45]) else "present"


def classify(text: str) -> tuple[str, list]:
    text, found = clean(text), []
    for m in TERM.finditer(text):
        found.append((mention(text, m), text[max(0, m.start() - 70):m.end() + 70]))
    calls = {c for c, _ in found}
    if not calls:
        return "not_mentioned", found
    return calls.pop() if len(calls) == 1 and "unclear" not in calls else "unclear", found


def write_csv(path: Path, rows: list, cols: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def label_reports(base: Path, private: Path, min_chars: int) -> dict:
    files = defaultdict(list)
    for r in csv.DictReader(open(base / "manifest.csv", newline="")):
        files[r["patient"]].append(r["file_id"])
    labels, audit = {}, []
    for patient, ids in sorted(files.items()):
        paths = [base / "text" / f"{i}.txt" for i in ids]
        text = "\n".join(p.read_text() for p in paths if p.exists())
        if len(text) < min_chars:
            labels[patient] = "unreadable"
            continue
        labels[patient], found = classify(text)
        audit += [{"patient": patient, "mention_call": c, "snippet": s} for c, s in found]
    write_csv(private / "report-labels.csv", [{"patient": p, "label": v} for p, v in labels.items()], ["patient", "label"])
    write_csv(private / "audit.csv", audit, ["patient", "mention_call", "snippet"])
    return labels


def split_patients(cases: dict, fraction: float, seed: int) -> dict:
    """Reserves the same share of each category and label."""
    groups, out = defaultdict(list), {}
    for p, c in sorted(cases.items()):
        groups[(c["category"], c["y"])].append(p)
    rng = random.Random(f"{seed}-split")
    for key in sorted(groups):
        pats = groups[key]
        rng.shuffle(pats)
        n = int(round(fraction * len(pats)))
        out.update({p: "reserved" if i < n else "train" for i, p in enumerate(pats)})
    return out


def folds_for(pats: list, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng, out = random.Random(f"{seed}-folds"), np.zeros(len(pats), int)
    for cls in (0, 1):
        idx = [i for i in range(len(pats)) if y[i] == cls]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            out[i] = n % k
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reports", required=True, help="the private report folder: manifest.csv and text/")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="append", default=[], help="CSV of patients never used here (repeat)")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="share of labelled patients to reserve unseen")
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--min-per-class", type=int, default=10, help="stop below this many training patients per class")
    ap.add_argument("--max-tiles", type=int, default=2048, help="tiles kept per slide in memory")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261123)
    a = ap.parse_args()

    out = Path(a.out)
    private = out / "private"
    if (out / "summary.json").exists():
        raise SystemExit(f"{out / 'summary.json'} already exists: a run is never overwritten")
    private.mkdir(parents=True, exist_ok=True)
    labels = label_reports(Path(a.reports), private, a.min_chars)
    holdout = read_patients(a.holdout)
    slides, category = defaultdict(list), {}
    for r in read_manifest(a.manifest):
        if r["category"] in CATEGORIES and r.get("label", "glioma") == "glioma" \
                and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists():
            slides[r["PatientID"]].append(r["SeriesInstanceUID"])
            category[r["PatientID"]] = r["category"]
    matched = [p for p in slides if p in labels]
    cases = {p: {"category": category[p], "y": int(labels[p] == "present")} for p in matched
             if labels[p] in ("present", "absent") and p not in holdout}
    split = split_patients(cases, a.test_fraction, a.seed)
    write_csv(private / "cases.csv", [{"patient": p, "category": c["category"], "report_label": labels[p],
                                       "split": split[p], "slides": " ".join(sorted(slides[p]))}
                                      for p, c in sorted(cases.items())],
              ["patient", "category", "report_label", "split", "slides"])
    by_category = {c: dict(Counter(labels[p] for p in matched if category[p] == c)) for c in CATEGORIES}
    train = sorted(p for p in cases if split[p] == "train")
    y = np.array([cases[p]["y"] for p in train], int)
    summary = {"label_source": "TCGA pathology reports, by rule; unproven until blinded slide review",
               "report_patients": len(labels), "report_labels": dict(Counter(labels.values())),
               "patients_with_report_and_bag": len(matched), "labels_by_category": by_category,
               "held_out_patients_left_out": sum(p in holdout for p in matched),
               "reserved_unseen": {"present": sum(split[p] == "reserved" and cases[p]["y"] == 1 for p in cases),
                                   "absent": sum(split[p] == "reserved" and cases[p]["y"] == 0 for p in cases)},
               "training": {"present": int(y.sum()), "absent": int(len(y) - y.sum())}}
    if min(summary["training"].values()) < a.min_per_class:
        summary["status"] = f"stopped: fewer than {a.min_per_class} training patients in a class; no model trained"
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise SystemExit(json.dumps(summary, indent=2))

    from entip_glioma.wsi.abmil import ABMIL
    from entip_glioma.wsi.bags import Bag
    bags = [np.concatenate([Bag.load(Path(a.bags) / f"{s}.npz", max_tiles=a.max_tiles, seed=a.seed).features
                            for s in sorted(slides[p])]) for p in train]
    cats = np.array([cases[p]["category"] for p in train])
    folds = folds_for(train, y, max(2, a.folds), a.seed)
    oof, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        model = ABMIL(2, epochs=a.epochs, seed=a.seed).fit([bags[i] for i in tr], y[tr])
        oof[te] = model.predict_proba([bags[i] for i in te])[:, 1]
        for i in te:  # the share of positives in this category among the other folds
            same = tr[cats[tr] == cats[i]]
            base[i] = (y[same].sum() + 1) / (len(same) + 2)
    within = {}
    for c in CATEGORIES:
        m = cats == c
        within[c] = {"present": int(y[m].sum()), "absent": int(m.sum() - y[m].sum()), "auc": auc(y[m], oof[m])}
    final = ABMIL(2, epochs=a.epochs, seed=a.seed).fit(bags, y)
    final.save(out / "necrosis-mil.npz")
    summary.update(status="draft model trained", out_of_fold_auc=auc(y, oof), category_only_baseline_auc=auc(y, base),
                   auc_within_category=within)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "model-record.json").write_text(json.dumps({
        "model": "necrosis-mil.npz", "status": "draft", "label_source": summary["label_source"],
        "commercial_use": "project assumption, no licence confirms it; retrain if overturned",
        "acknowledge": ["TCGA", "NCI GDC"], "training_patients": len(train), "seed": a.seed}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
