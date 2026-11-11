"""Write triage results to the central database (db/ pillar) through ``POST /actions``.

Registering a model (once, after training: ``register_actions``):

* ``feature`` cns_tissue_class (level region) and the ``model`` row with status ``draft``.

Scoring one patient (``build_actions``), all idempotent upserts so a rerun replaces only its own rows:

1. ``region`` rows: one whole-tissue ``roi`` per scan labelled with that scan's call, and the top tiles by
   attention (kind ``tile``, label ``attention``, meta weight/rank) for the GUI heatmap;
2. ``prediction`` rows on those regions (score = calibrated confidence for the scan, attention for tiles);
3. an event ``glioma_step`` with ``detail = {study_id, records}``, which the step 5 engine reads from
   ``GET /patients/{study_id}/actions``.

Scoring never rewrites the model row: an upsert there would set a promoted model back to ``draft``.
``post_patient`` registers the model only if the database does not have it yet.
"""
from __future__ import annotations

import json
import urllib.request
from typing import List, Sequence

import numpy as np

from ..wsi.bags import Bag
from .labels import CLASSES
from .model import FEATURE, TriageModel, slide_region_id, tile_region_id


def _prov(model: TriageModel) -> dict:
    return {"kind": "model", "actor": model.model_id, "version": model.version}


def register_actions(model: TriageModel, uri: str | None = None) -> List[dict]:
    rep = model.report.get("cv", {})
    met = {k: rep.get("per_patient", {}).get(k) for k in ("balanced_accuracy", "ece", "auroc_tumour", "auroc_glial")}
    met["gate"] = rep.get("gate")
    return [
        {"verb": "upsert", "object_type": "feature", "provenance": _prov(model),
         "records": [{"name": FEATURE, "level": "region", "positives_only": False,
                      "description": "Step 1 triage: normal brain / reactive / glioma / other primary CNS / "
                                     "metastasis / lymphoma"}]},
        {"verb": "upsert", "object_type": "model", "provenance": _prov(model),
         "records": [{"model_id": model.model_id, "feature": FEATURE, "kind": "abmil", "encoder": model.encoder,
                      "status": "draft", "version": model.version, "n_labels": model.report.get("n_slides"),
                      "metrics": {k: v for k, v in met.items() if v is not None},
                      "licence": "MIT (code)", "uri": uri,
                      "meta": {"temperature": model.temperature, "threshold": model.threshold,
                               "classes_trained": model.report.get("classes_trained", []),
                               "classes_excluded": model.report.get("classes_excluded", [])}}]},
    ]


def build_actions(model: TriageModel, bags: Sequence[Bag], study_id: str) -> List[dict]:
    prov = _prov(model)
    recs = model.records(bags, study_id)
    regions, preds = [], []
    for b in bags:
        if len(b) == 0:
            continue
        p = model.predict(b)
        call = CLASSES[int(p["probs"].argmax())]
        x0, y0 = b.coords_um.min(0)
        x1, y1 = b.coords_um.max(0) + b.tile_um
        rid = slide_region_id(b.scan_id, model.model_id)
        regions.append({"region_id": rid, "scan_id": b.scan_id, "kind": "roi", "label": call,
                        "x_um": float(x0), "y_um": float(y0), "w_um": float(x1 - x0), "h_um": float(y1 - y0),
                        "source": "model", "model_id": model.model_id,
                        "meta": {"probabilities": {c: round(float(v), 4) for c, v, t in
                                                   zip(CLASSES, p["probs"], model.trained) if t},
                                 "n_tiles": len(b), "novel": p["novel"]}})
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
    actions = []
    if regions:
        actions.append({"verb": "upsert", "object_type": "region", "provenance": prov, "records": regions})
        actions.append({"verb": "upsert", "object_type": "prediction", "provenance": prov, "records": preds})
    actions.append({"verb": "glioma_step", "provenance": prov, "detail": {"study_id": study_id, "records": recs}})
    return actions


def model_registered(base_url: str, model_id: str, timeout: float = 30.0) -> bool:
    import urllib.error
    import urllib.parse
    url = base_url.rstrip("/") + "/objects/model/" + urllib.parse.quote(model_id, safe="")
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def post_patient(base_url: str, model: TriageModel, bags: Sequence[Bag], study_id: str,
                 uri: str | None = None) -> List[dict]:
    """Register the model if the database lacks it, then write this patient's results."""
    actions = [] if model_registered(base_url, model.model_id) else register_actions(model, uri)
    return post_actions(base_url, actions + build_actions(model, bags, study_id))


def post_actions(base_url: str, actions: Sequence[dict], timeout: float = 30.0) -> List[dict]:
    """POST each action in order; stops at the first error (the db writes nothing for a refused action)."""
    out = []
    for a in actions:
        req = urllib.request.Request(base_url.rstrip("/") + "/actions", data=json.dumps(a).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:   # raises on HTTP 4xx/5xx
            out.append(json.loads(r.read() or b"{}"))
    return out
