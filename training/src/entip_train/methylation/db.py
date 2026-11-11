"""Write a step-4 result to the central database (pillar 5) through ``POST /actions``.

Three actions per patient, each with its own ``action_id`` so retries are safe:

1. ``upsert`` the model row (status ``draft``: a model only becomes champion through a gated round).
   ``post_actions`` skips this when the model is already registered, so a later prediction never resets a
   promoted model back to ``draft``.
2. ``create`` a ``genomic_finding`` with marker ``methylation_class_ai`` (kind ``class``) and the evidence tiles
   as ``region`` rows labelled ``methylation_attention``. The ``_ai`` marker keeps the AI value apart from
   a real array result in ``patient_row``; the rules engine ignores ``_ai`` markers.
3. The ``glioma_step`` event with the finding record, which is what the step-5 engine reads.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import List

from .model import MethylationModel

MARKER = "methylation_class_ai"
REGION_LABEL = "methylation_attention"


def _id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def build_actions(record: dict, model: MethylationModel) -> List[dict]:
    prov = record["provenance"]
    sid = record["study_id"]
    key = _id(sid, model.model_id, model.version, json.dumps(record, sort_keys=True))
    model_row = {"model_id": model.model_id, "kind": "slide-mil:abmil", "encoder": model.encoder,
                 "status": "draft", "version": model.version, "metrics": model.metrics, "licence": model.licence,
                 "meta": {"step": "methylation_class", "label_source": model.label_source,
                          "class_threshold": model.class_threshold, "family_threshold": model.family_threshold,
                          "temperatures": model.temperatures, "ensemble": len(model.members)}}
    acts = [{"action_id": f"act_mc_model_{_id(model.model_id, model.version)}", "verb": "upsert",
             "object_type": "model", "provenance": prov, "records": [model_row], "detail": {}}]

    recs: List[dict] = []
    ev_out = []
    for i, ev in enumerate(record.get("evidence", [])):
        scan_id = ev["id"].split(":tile@")[0]
        rid = f"{scan_id}:mc{key[:8]}-{i:02d}"
        recs.append({"type": "Region", "region_id": rid, "scan_id": scan_id, "kind": "tile", "label": REGION_LABEL,
                     "x_um": ev["x_um"], "y_um": ev["y_um"], "w_um": ev["w_um"], "h_um": ev["h_um"],
                     "source": f"model:{model.model_id}@{model.version}", "model_id": model.model_id,
                     "meta": {"attention": ev["weight"]}})
        ev_out.append({**ev, "kind": "region", "id": rid})
    rec = {**record, "evidence": ev_out}
    if record["status"] != "not_assessed":
        recs.insert(0, {"type": "GenomicFinding", "finding_id": f"gf_mc_{key}", "study_id": sid,
                        "marker": MARKER, "kind": "class", "result": record["call"],
                        "detail": (record.get("value") or {}).get("class"), "value": record["confidence"],
                        "method": "predicted from H&E (AI; not a methylation test)",
                        "meta": {"status": record["status"], "threshold": record["threshold"],
                                 "order_test": record.get("order_test"), "scan_ids": record["scan_ids"]}})
    if recs:
        acts.append({"action_id": f"act_mc_out_{key}", "verb": "create", "object_type": None,
                     "provenance": prov, "records": recs, "detail": {"step": "methylation_class"}})
    acts.append({"action_id": f"act_mc_step_{key}", "verb": "glioma_step", "provenance": prov,
                 "detail": {"study_id": sid, "records": [rec]}})
    return acts


def _exists(base_url: str, object_type: str, object_id: str, timeout: float) -> bool:
    url = f"{base_url.rstrip('/')}/objects/{object_type}/{urllib.parse.quote(object_id, safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def post_actions(base_url: str, actions: List[dict], timeout: float = 30.0) -> List[dict]:
    out = []
    for a in actions:
        if a.get("object_type") == "model" and _exists(base_url, "model", a["records"][0]["model_id"], timeout):
            continue
        body = {k: v for k, v in a.items() if v is not None}
        req = urllib.request.Request(base_url.rstrip("/") + "/actions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out.append(json.loads(r.read() or b"{}"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"db refused action {a.get('action_id')} ({a.get('verb')}): "
                               f"{e.code} {e.read().decode(errors='replace')[:500]}") from e
    return out
