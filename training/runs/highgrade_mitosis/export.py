"""Step 3c: recalibrate MitoEd's p on glioma from the candidate labels, and report what the count gets right.

    python export.py --key mit-r1/key.csv labels-AH.csv labels-XY.csv --out export-r1 [--holdout held_out.csv]

Labels: ``y`` -> 1, ``n`` -> 0, ``x`` left out; several labellers by ``--rule``. Each label stands for its band on
its slide (``pick.py`` samples at random within bands of p), so it is weighted by the band's size over the number
labelled in it. With those weights:

* ``mitosis-calibration.json``: a sigmoid on logit(p) (``a``, ``b``), for ``entip-train highgrade
  --mitosis-calibration``. Needs 5 or more mitoses and 5 or more non-mitoses. Its error is checked in folds grouped
  by slide (weighted Brier score, before and after);
* ``summary.json``: per band, the estimated share of true mitoses and the estimated number of them per slide;
  the share of the counted candidates that are mitoses (the count's precision); the mitoses the count misses
  (in the ``mimic`` band); Cohen's kappa for each pair of labellers.

The calibration fixes MitoEd's p on glioma; it does not make MitoEd usable commercially (its segmenter was trained
on CC BY-NC data), and the specimen cut-off is still a setting.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "highgrade_common"))
from common import TRAINING_DATA, agreement, consensus, read_labels, read_patients  # noqa: E402


def vote(lab):
    return None if lab in (None, "", "x") else int(lab == "y")


def logit(p):
    q = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def brier(y, p, w):
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--key", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=["agree", "majority", "any"], default="agree")
    ap.add_argument("--holdout", action="append", default=[])
    ap.add_argument("--training-data", default=TRAINING_DATA)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    key, raw, raters = read_labels(a.key, a.labels, read_patients(a.holdout))
    rows = []
    for t, d in raw.items():
        c = consensus([vote(d.get(rt)) if rt in d else None for rt in raters], a.rule)
        if c is not None:
            rows.append({**key[t], "y": c})
    if not rows:
        raise SystemExit("no usable labels")
    n_in = defaultdict(int)
    for r in rows:
        n_in[(r["round"], r["slide"], r["source"])] += 1
    for r in rows:
        r["w"] = float(r["band_size"]) / n_in[(r["round"], r["slide"], r["source"])]
    y = np.array([r["y"] for r in rows])
    p = np.array([float(r["p"]) for r in rows])
    w = np.array([r["w"] for r in rows])
    slides = np.array([r["slide"] for r in rows])
    n_slides = len(set(slides))

    bands = {}
    for b in ("high", "mid", "low", "mimic"):
        m = np.array([r["source"] == b for r in rows])
        if m.any():
            bands[b] = {"labelled": int(m.sum()), "mitoses_labelled": int(y[m].sum()),
                        "share_mitoses": round(float(np.sum(w[m] * y[m]) / np.sum(w[m])), 3),
                        "est_mitoses_per_slide": round(float(np.sum(w[m] * y[m])) / n_slides, 2)}
    counted = np.array([r["source"] != "mimic" for r in rows])
    summary = {"raters": raters, "rule": a.rule, "labelled": len(rows), "slides": n_slides,
               "mitoses": int(y.sum()), "bands": bands,
               "count_precision": (round(float(np.sum(w[counted] * y[counted]) / np.sum(w[counted])), 3)
                                   if counted.any() else None),
               "missed_mitoses_per_slide": bands.get("mimic", {}).get("est_mitoses_per_slide"),
               "agreement": agreement(raw, raters, vote)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if min(y.sum(), (1 - y).sum()) >= 5:
        z = logit(p)[:, None]
        lr = LogisticRegression(C=100.0).fit(z, y, sample_weight=w)
        cal = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n_labels": len(rows),
               "n_mitoses": int(y.sum()), "n_slides": n_slides, "training_data": a.training_data,
               "note": "sigmoid on logit(MitoEd p), weighted by band size; not a specimen-level validation"}
        if n_slides >= 3:
            from sklearn.model_selection import GroupKFold
            q = np.zeros(len(y))
            for tr, te in GroupKFold(3).split(z, y, slides):
                if len(set(y[tr])) < 2:
                    q[te] = p[te]
                    continue
                q[te] = LogisticRegression(C=100.0).fit(z[tr], y[tr], sample_weight=w[tr]).predict_proba(z[te])[:, 1]
            cal["brier_before"] = round(brier(y, p, w), 4)
            cal["brier_after_slide_folds"] = round(brier(y, q, w), 4)
        (out / "mitosis-calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
        summary["calibration"] = cal
    else:
        print("fewer than 5 mitoses or non-mitoses: no calibration written", file=sys.stderr)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
