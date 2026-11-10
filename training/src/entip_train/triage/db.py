"""Write triage results to the central database (db/ pillar) through ``POST /actions``.

One patient's run becomes these actions, in order (all idempotent upserts, so a rerun replaces nothing it
should not):

1. ``feature`` cns_tissue_class (level region) and the ``model`` row (status draft; promotion is separate);
2. ``region`` rows: one whole-tissue ``roi`` per scan labelled with the call, and the top tiles by attention
   (kind ``tile``, label ``attention``, meta weight/rank) for the GUI heatmap;
3. ``prediction`` rows on those regions (score = calibrated confidence for the slide, attention for tiles);
4. an event ``glioma_step`` with ``detail = {study_id, records}``, which the step 5 engine reads from
   ``GET /patients/{study_id}/actions``.
"""
from __future__ import annotations

import json
import urllib.request
from typing import List, Sequence

import numpy as np

from ..wsi.bags import Bag
from .model import FEATURE, TriageModel, slide_region_id, tile_region_id


def build_actions(model: TriageModel, bags: Sequence[Bag], study_id: str, uri: str | None = None) -> List[dict]:
    prov = {"kind": "model", "actor": model.model_id, "version": model.version}
    recs = model.records(bags, study_id)
    tc = next((r for r in recs if r["finding"] == "tissue_class"), None)
    rep = model.report.get("cv", {}).get("calibrated", {})
    actions = [
        {"verb": "upsert", "object_type": "feature", "provenance": prov,
         "records": [{"name": FEATURE, "level": "region", "positives_only": False,
                      "description": "Step 1 triage: normal brain / reactive / glioma / other primary CNS / "
                                     "metastasis / lymphoma"}]},
        {"verb": "upsert", "object_type": "model", "provenance": prov,
         "records": [{"model_id": model.model_id, "feature": FEATURE, "kind": "abmil", "encoder": model.encoder,
                      "status": "draft", "version": model.version, "n_labels": model.report.get("n_slides"),
                      "metrics": {k: rep[k] for k in ("balanced_accuracy", "ece") if k in rep},
                      "licence": "MIT (code)", "uri": uri,
                      "meta": {"temperature": model.temperature, "threshold": model.threshold,
                               "classes_trained": model.report.get("classes_trained", [])}}]},
    ]
    regions, preds = [], []
    for b in bags:
        if len(b) == 0:
            continue
        p = model.predict(b)
        x0, y0 = b.coords_um.min(0)
        x1, y1 = b.coords_um.max(0) + b.tile_um
        rid = slide_region_id(b.scan_id, model.model_id)
        call = tc["call"] if tc else None
        regions.append({"region_id": rid, "scan_id": b.scan_id, "kind": "roi", "label": call,
                        "x_um": float(x0), "y_um": float(y0), "w_um": float(x1 - x0), "h_um": float(y1 - y0),
                        "source": "model", "model_id": model.model_id,
                        "meta": {"probabilities": tc["probabilities"] if tc else {}, "n_tiles": len(b)}})
        preds.append({"model_id": model.model_id, "target_type": "region", "target_id": rid,
                      "scan_id": b.scan_id, "score": float(p["probs"].max()), "label": call})
        top = np.argsort(-p["attention"])[:model.cfg.top_k]
        amax = float(p["attention"][top[0]])
        for r, i in enumerate(top):
            tid = tile_region_id(b.scan_id, model.model_id, r)
            x, y = (float(v) for v in b.coords_um[i])
            w = float(p["attention"][i])
            regions.append({"region_id": tid, "scan_id": b.scan_id, "kind": "tile", "label": "attention",
                            "x_um": x, "y_um": y, "w_um": b.tile_um, "h_um": b.tile_um, "source": "model",
                            "model_id": model.model_id, "meta": {"weight": w, "rank": r}})
            preds.append({"model_id": model.model_id, "target_type": "region", "target_id": tid,
                          "scan_id": b.scan_id, "score": w / amax if amax > 0 else 0.0, "label": "attention"})
    if regions:
        actions.append({"verb": "upsert", "object_type": "region", "provenance": prov, "records": regions})
        actions.append({"verb": "upsert", "object_type": "prediction", "provenance": prov, "records": preds})
    actions.append({"verb": "glioma_step", "provenance": prov, "detail": {"study_id": study_id, "records": recs}})
    return actions


def post_actions(base_url: str, actions: Sequence[dict], timeout: float = 30.0) -> List[dict]:
    """POST each action in order; stops at the first error (the db writes nothing for a refused action)."""
    out = []
    for a in actions:
        req = urllib.request.Request(base_url.rstrip("/") + "/actions", data=json.dumps(a).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:   # raises on HTTP 4xx/5xx
            out.append(json.loads(r.read() or b"{}"))
    return out
