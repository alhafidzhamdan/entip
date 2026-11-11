"""Turn a patient's bag into ``entip.glioma_step/1`` finding records, one per panel marker.

Each record is a call with a calibrated confidence. It is ``confident`` only when every gate passes:

* the marker was trained (enough labelled patients of both classes) and is switched on;
* the bag was made with the same encoder and tile size as the training bags;
* the patient has at least ``min_tiles`` tissue tiles;
* the bag is not out of distribution (mean-embedding distance within the training ``ood_quantile``);
* the marker's prerequisite holds (1p/19q needs IDH-mutant, from a real test or a confident IDH call);
* the calibrated confidence reaches the accept threshold for that call (each call has its own, see ``gate.py``).

Otherwise the record is ``uncertain`` and names the real test to order, or ``not_assessed`` when the marker is
switched off or its prerequisite is ruled out. ``reasons`` says which gate failed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .dataset import PatientBag
from .markers import CONTRACT, STEP
from .model import PanelModel

INDETERMINATE = "indeterminate"


def _evidence(model: PanelModel, finding: str, pb: PatientBag, k: int) -> List[dict]:
    a = model.attention(finding, pb.features)
    top = np.argsort(-a)[:k]
    t = pb.tile_um
    out = []
    for i in top:
        sid = pb.scan_ids[int(pb.scan_index[i])]
        x, y = (float(v) for v in pb.coords_um[i])
        out.append({"kind": "tile", "id": f"{sid}:tile:{x:.0f}_{y:.0f}", "x_um": x, "y_um": y,
                    "w_um": t, "h_um": t, "weight": float(a[i])})
    return out


def predict_patient(model: PanelModel, pb: PatientBag, known: Optional[Dict[str, str]] = None) -> List[dict]:
    """Records for every configured marker, in panel order. ``known``: real results already in hand
    ({finding: call}), used only for prerequisites (the engine reads real results from the database itself)."""
    known = dict(known or {})
    cfg = model.config
    common_gates: List[str] = []
    if pb.encoder != model.encoder:
        common_gates.append(f"bag encoder {pb.encoder!r} differs from training encoder {model.encoder!r}")
    if abs(pb.tile_um - model.tile_um) > 1e-6:
        common_gates.append(f"tile size {pb.tile_um} µm differs from training {model.tile_um} µm")
    if len(pb) < cfg.min_tiles:
        common_gates.append(f"only {len(pb)} tissue tiles (need {cfg.min_tiles})")
    ood = None
    if not common_gates:            # distances are meaningless across encoders
        ood = model.ood_distance(pb.features)
        if ood > model.ood["limit"]:
            common_gates.append(f"slides look unlike the training slides (distance {ood:.2f} > "
                                f"{model.ood['limit']:.2f}, the {cfg.ood_quantile:.0%} training quantile)")

    records: Dict[str, dict] = {}
    # prerequisites first (IDH before 1p/19q), output in panel order
    for m in sorted(cfg.markers, key=lambda m: m.requires is not None):
        fit = model.fits.get(m.finding)
        rec = {
            "contract": CONTRACT, "step": STEP, "finding": m.finding, "study_id": pb.study_id,
            "scan_ids": list(pb.scan_ids), "source": "ai",
            "model": {"model_id": f"{model.model_id}:{m.finding}", "version": model.version,
                      "licence": model.licence},
            "provenance": {"kind": "model", "actor": f"{model.model_id}:{m.finding}", "version": model.version},
            "reasons": [],
        }
        if m.label_proxy:
            rec["reasons"].append(m.label_proxy)
        if m.finding in cfg.disabled or fit is None:
            rec.update(call=INDETERMINATE, confidence=0.0, status="not_assessed", order_test=None)
            rec["reasons"].insert(0, "marker switched off in the panel configuration")
            records[m.finding] = rec
            continue
        if not fit.trained:
            rec.update(call=INDETERMINATE, confidence=0.0, status="uncertain", order_test=m.order_test)
            rec["reasons"].insert(0, f"no model for this marker: {fit.reason}")
            records[m.finding] = rec
            continue

        gates = list(common_gates)
        if m.requires:
            req_f, req_call = m.requires
            have, have_src = known.get(req_f), "test"
            up = records.get(req_f)
            if have is None and up is not None and up["status"] == "confident":
                have, have_src = up["call"], "ai"
            if have is not None and have != req_call:
                rec.update(call=INDETERMINATE, confidence=0.0, status="not_assessed", order_test=None)
                rec["reasons"].insert(0, f"only assessed when {req_f} is {req_call}; {req_f} is {have}")
                records[m.finding] = rec
                continue
            if have is None:
                gates.append(f"{req_f} status not established; this model only applies when {req_f} is {req_call}")

        if any(g.startswith("bag encoder") or g.startswith("tile size") for g in gates):
            rec.update(call=INDETERMINATE, confidence=0.0, status="uncertain", order_test=m.order_test)
            rec["reasons"] = gates + rec["reasons"]
            records[m.finding] = rec
            continue

        p = model.predict_proba(m.finding, pb.features)
        c = int(np.argmax(p))
        conf = float(p[c])
        raw_thr = fit.thresholds[c]
        thr = float(min(raw_thr, 1.0))
        acc = fit.metrics.get("accept", {}).get(m.calls[c], {})
        rec.update(call=m.calls[c], confidence=conf, threshold=thr,
                   probabilities={m.calls[0]: float(p[0]), m.calls[1]: float(p[1])},
                   evidence=_evidence(model, m.finding, pb, cfg.top_evidence),
                   value={"auroc_oof": fit.metrics.get("auroc_oof"), "coverage_oof": fit.metrics.get("coverage_oof"),
                          "call_error_upper_oof": acc.get("error_upper"), "call_miss_upper_oof": acc.get("miss_upper"),
                          "n_tiles": int(len(pb)), **({"ood_distance": ood} if ood is not None else {})})
        if raw_thr > 1.0:
            gates.append(f"a '{m.calls[c]}' call never met the error and miss limits in cross-validation "
                         f"({acc.get('n_other_class', 0)} patients of the other class seen)")
        elif conf < raw_thr:
            gates.append(f"confidence {conf:.3f} below the accept threshold {thr:.3f} for '{m.calls[c]}'")
        if m.requires and have is not None:
            rec["value"]["prerequisite"] = {"finding": m.requires[0], "call": have, "source": have_src}
            if have_src == "ai":
                rec["reasons"].append(f"{m.requires[0]} {have} is itself an AI call, not a test result")
        if gates:
            rec.update(status="uncertain", order_test=m.order_test)
            rec["reasons"] = gates + rec["reasons"]
        else:
            rec.update(status="confident", order_test=None)
        records[m.finding] = rec
    return [records[m.finding] for m in cfg.markers]


def summarise(records: List[dict]) -> dict:
    """{confident: {finding: call}, order: [tests], not_assessed: [findings]} for a quick look."""
    return {"confident": {r["finding"]: r["call"] for r in records if r["status"] == "confident"},
            "order": sorted({r["order_test"] for r in records if r["status"] == "uncertain" and r.get("order_test")}),
            "not_assessed": [r["finding"] for r in records if r["status"] == "not_assessed"]}
