"""Step 3c: score a frozen finder on reviewed glioma fields, and turn the review into glioma training data.

    python glioma_retrain.py score --round round1 --finder mitosis-finder2 --slides slides/ --groups held_out development
    python glioma_retrain.py data --round round1 --source mitosis-finder2 --slides slides/ --reference ref/private/patients.csv --out mitosis-finder3
    python glioma_retrain.py score --round round1 --finder mitosis-finder3 --slides slides/

**score** runs the frozen finder on the fields marked in full and compares its detections at its frozen cut-off with
the reader's marks. A detection matches a marked mitosis within 7.5 µm, each used once. A detection on an uncertain
mark is neither right nor wrong. It refuses any field of a patient the finder was trained on, so a retrained finder
is scored on the held-out patients only. Each finder is scored once per round.

**data** copies the source finder's split and adds the development patients' review as training images:

- Each development field marked in full becomes one image. Mitoses are positives, look-alikes negatives and uncertain
  marks are ignored. Unmarked tissue counts as background, because the field was searched in full.
- Each reviewed call becomes a 128 µm image labelled only at its centre: mitosis or granular is positive; prophase,
  look-alike or not mitosis is negative; uncertain or unusable calls are left out. Unmarked tissue is not used.

Marks are points, so each is drawn as a disc of 4 µm radius. Held-out and reference patients are never used, and
the tune images stay the public ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
import finder as F  # noqa: E402
import glioma_review as G  # noqa: E402
import reference_set as R  # noqa: E402

DISC_UM = 4.0
CALL_UM, CALL_PX = 128.0, 512
CALL_ROLE = {"mitosis": "pos", "granular": "pos", "prophase": "neg", "look_alike": "neg", "not_mitosis": "neg"}
MARK_ROLE = {"mitosis": "pos", "look_alike": "neg", "uncertain": "ignore"}
CURVE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def disc(x: float, y: float, r: float, n: int = 16) -> list:
    return [round(v, 2) for k in range(n) for v in (x + r * math.cos(2 * math.pi * k / n),
                                                     y + r * math.sin(2 * math.pi * k / n))]


def read_round(rnd: Path, rater: str) -> tuple:
    """The round's full fields and complete marks, and its calls with one label each. Refuses anything incomplete."""
    private = rnd / "private"
    fields = {r["field"]: r for r in csv.DictReader(open(private / "full-key.csv", newline=""))}
    _, frows, points = R.read_marks(str(private / f"mitosis-marks-{rater}.csv"), set(fields))
    calls = {r["call"]: r for r in csv.DictReader(open(private / "calls-key.csv", newline=""))}
    labels = {}
    for r in csv.DictReader(open(private / f"mitosis-call-labels-{rater}.csv", newline="")):
        if r["label"] not in G.CALL_LABELS:
            raise SystemExit(f"{r['call']}: unknown call label {r['label']!r}")
        if r["call"] in labels:
            raise SystemExit(f"{r['call']} is labelled twice")
        labels[r["call"]] = r["label"]
    if set(labels) != set(calls):
        raise SystemExit(f"{len(set(calls) - set(labels))} calls are unlabelled and {len(set(labels) - set(calls))} "
                         "labels are for calls not in this round")
    return fields, frows, points, calls, labels


def trained_patients(finder: Path, rnd: Path) -> set:
    """Patients of this round whose review the finder was trained on."""
    meta = json.load(open(finder / "index.json")).get("glioma")
    if not meta:
        return set()
    if Path(meta["round"]).resolve() != rnd.resolve():
        raise SystemExit("this finder was trained on another glioma round: score it on that round's held-out patients")
    private = rnd / "private"
    fields = {r["field"]: r["patient"] for r in csv.DictReader(open(private / "full-key.csv", newline=""))}
    calls = {r["call"]: r["patient"] for r in csv.DictReader(open(private / "calls-key.csv", newline=""))}
    return {fields[f] for f in meta["fields"]} | {calls[c] for c in meta["calls"]}


def score(a) -> int:
    rnd, finder = Path(a.round), Path(a.finder)
    out = rnd / f"score-{finder.resolve().name}.json"
    if out.exists():
        raise SystemExit(f"{out} already exists: a finder is scored once per round")
    fields, frows, points, _, _ = read_round(rnd, a.rater)
    used = trained_patients(finder, rnd)
    todo = sorted(f for f, r in fields.items() if r["group"] in a.groups)
    clash = sorted(f for f in todo if fields[f]["patient"] in used)
    if clash:
        raise SystemExit(f"{len(clash)} fields belong to patients this finder was trained on: score held_out only")
    if not todo:
        raise SystemExit("no fields in those groups")
    model, dev, cutoff, sha = G.load_finder(finder)
    half, px = R.FIELD_UM / 2, 1 / F.UM_PER_PX
    rows, per_field = defaultdict(list), []
    curve = {g: {t: [] for t in CURVE + (cutoff,)} for g in a.groups}
    for n, f in enumerate(todo, start=1):
        k = fields[f]
        cx, cy = float(k["x_um"]) + half, float(k["y_um"]) + half
        arr = np.asarray(R.open_reader(a.slides, k["slide"]).read_um(cx, cy, R.IMAGE_UM, R.IMAGE_PX))
        det = F.peaks(F.predict_map(model, arr, dev))
        det = det.copy()
        det[:, 0] = (det[:, 0] - R.IMAGE_PX / 2) * F.UM_PER_PX + half      # field µm
        det[:, 1] = (det[:, 1] - R.IMAGE_PX / 2) * F.UM_PER_PX + half
        det = det[(det[:, 0] >= 0) & (det[:, 0] <= R.FIELD_UM) & (det[:, 1] >= 0) & (det[:, 1] <= R.FIELD_UM)]
        det[:, :2] *= px                                                    # field pixels, for the match radius
        im = {"w": R.FIELD_UM * px, "h": R.FIELD_UM * px,
              "mm2": R.FIELD_MM2 * float(frows[f]["assessable_pct"]) / 100,
              "objects": [{"role": MARK_ROLE[lab], "x": x * px, "y": y * px} for x, y, lab, _ in points.get(f, [])]}
        r = F.score_image(det, im, cutoff)
        rows[k["group"]].append(r)
        per_field.append({"field": f, "group": k["group"], "true": r["n_true"], "found": r["tp"], "called": r["n_pred"]})
        for t in curve[k["group"]]:
            curve[k["group"]][t].append(F.score_image(det, im, t))
        print(f"scored {n}/{len(todo)} fields", file=sys.stderr, flush=True)

    def per_mm2(rs: list) -> dict:
        mm2 = sum(x["mm2"] for x in rs)
        return {"marked_per_mm2": round(sum(x["n_true"] for x in rs) / mm2, 2),
                "called_per_mm2": round(sum(x["n_pred"] for x in rs) / mm2, 2)}

    result = {"finder": str(finder.resolve()), "finder_sha256": sha, "cutoff": cutoff, "cutoff_chosen_on": "public tune",
              "rater": a.rater, "fields": len(todo),
              "by_group": {g: {**F.summarise(v), **per_mm2(v)} for g, v in sorted(rows.items())},
              "by_cutoff_for_information": {g: {str(t): {k: F.summarise(v)[k] for k in ("precision", "recall", "f1")}
                                                for t, v in sorted(c.items()) if v} for g, c in curve.items() if any(c.values())},
              "per_field": per_field,
              "note": "The cut-off was chosen on public tune images. The table by cut-off describes these fields and "
                      "must not be used to choose a cut-off."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_field", "by_cutoff_for_information")}, indent=2))
    return 0


def data(a) -> int:
    from PIL import Image
    rnd, src, out = Path(a.round), Path(a.source), Path(a.out)
    if (out / "index.json").exists():
        raise SystemExit(f"{out / 'index.json'} already exists: the split is fixed once")
    if not (src / "frozen.json").exists():
        raise SystemExit("the source finder must be frozen first")
    fields, frows, points, calls, labels = read_round(rnd, a.rater)
    reserved = G.reserved_patients(a.reference)
    held = {r["patient"] for r in fields.values() if r["group"] == "held_out"}
    dev_fields = sorted(f for f, r in fields.items() if r["group"] == "development")
    for name, pats in (("development fields", {fields[f]["patient"] for f in dev_fields}),
                       ("calls", {r["patient"] for r in calls.values()})):
        if pats & reserved:
            raise SystemExit(f"{name} include sealed reference patients")
        if pats & held:
            raise SystemExit(f"{name} include held-out patients")
    index = json.load(open(src / "index.json"))
    if any(im["source"] == F.GLIOMA for im in index["images"]):
        raise SystemExit("the source split already holds reviewed glioma images")
    folder = out / "glioma"
    (folder / "fields").mkdir(parents=True, exist_ok=True)
    (folder / "calls").mkdir(parents=True, exist_ok=True)
    px, fpx = 1 / F.UM_PER_PX, int(round(R.FIELD_UM / F.UM_PER_PX))
    added, roles = [], Counter()
    for n, f in enumerate(dev_fields, start=1):
        k = fields[f]
        cx, cy = float(k["x_um"]) + R.FIELD_UM / 2, float(k["y_um"]) + R.FIELD_UM / 2
        arr = np.asarray(R.open_reader(a.slides, k["slide"]).read_um(cx, cy, R.FIELD_UM, fpx))
        path = folder / "fields" / f"{f}.png"
        Image.fromarray(arr).save(path)
        s = fpx / R.FIELD_UM
        objs = [{"role": MARK_ROLE[lab], "x": x * s, "y": y * s, "rings": [disc(x * s, y * s, DISC_UM * s)]}
                for x, y, lab, _ in points.get(f, [])]
        roles.update(f"field_{o['role']}" for o in objs)
        added.append({"id": f"glioma-{f}", "path": str(path.resolve()), "source": F.GLIOMA, "tumour": "glioma",
                      "split": "train", "w": fpx, "h": fpx, "mm2": R.FIELD_MM2, "objects": objs})
        print(f"field {n}/{len(dev_fields)}", file=sys.stderr, flush=True)
    by_slide = defaultdict(list)
    for c, r in calls.items():
        if labels[c] in CALL_ROLE:
            by_slide[r["slide"]].append(r)
    skipped = sum(labels[c] not in CALL_ROLE for c in calls)
    for s_, rs in sorted(by_slide.items()):
        rd = R.open_reader(a.slides, s_)
        for r in sorted(rs, key=lambda r: r["call"]):
            arr = np.asarray(rd.read_um(float(r["x_um"]), float(r["y_um"]), CALL_UM, CALL_PX))
            path = folder / "calls" / f"{r['call']}.png"
            Image.fromarray(arr).save(path)
            role, c0 = CALL_ROLE[labels[r["call"]]], CALL_PX / 2
            roles[f"call_{role}"] += 1
            added.append({"id": f"glioma-{r['call']}", "path": str(path.resolve()), "source": F.GLIOMA,
                          "tumour": "glioma", "split": "train", "w": CALL_PX, "h": CALL_PX, "background": 0.0,
                          "objects": [{"role": role, "x": c0, "y": c0, "rings": [disc(c0, c0, DISC_UM * px)]}]})
    index["images"] += added
    index["glioma"] = {"round": str(rnd.resolve()), "rater": a.rater, "fields": dev_fields,
                       "calls": sorted(c for c in calls if labels[c] in CALL_ROLE), "disc_um": DISC_UM}
    info = {"source_run": str(src.resolve()), "glioma_images": len(added), "development_fields": len(dev_fields),
            "calls_used": len(added) - len(dev_fields), "calls_left_out": skipped, "glioma_objects": dict(sorted(roles.items())),
            "held_out_fields_unused": sum(r["group"] == "held_out" for r in fields.values())}
    with (out / "index.json").open("w") as fh:
        json.dump(index, fh, separators=(",", ":"))
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="score a frozen finder on the round's fields marked in full")
    s.add_argument("--round", required=True)
    s.add_argument("--finder", required=True)
    s.add_argument("--slides", required=True)
    s.add_argument("--groups", nargs="+", default=["held_out"], choices=["held_out", "development"])
    s.add_argument("--rater", default="AH")
    s.set_defaults(fn=score)
    d = sub.add_parser("data", help="add the development patients' review to a copy of a finder's split")
    d.add_argument("--round", required=True)
    d.add_argument("--source", required=True, help="the frozen finder whose split is copied")
    d.add_argument("--slides", required=True)
    d.add_argument("--reference", required=True, help="the sealed reference set's private/patients.csv")
    d.add_argument("--out", required=True)
    d.add_argument("--rater", default="AH")
    d.set_defaults(fn=data)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
