"""Write panel results to the central database (pillar 5) through ``POST /actions``, its only write path.

Two kinds of action:

* ``upsert`` of ``model`` rows, one per trained marker (``glioma-panel:<finding>``), with its metrics and accept
  threshold, so the GUI and the audit trail can see what made each call;
* an event action, verb ``glioma_step``, with ``detail = {study_id, records}``: the finding records the step 5
  rules engine reads from ``GET /patients/{study_id}/actions``.

AI calls are never written to ``genomic_finding``: ``patient_row.molecular`` shows the latest finding per marker,
so a prediction there would mask the real test result.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Iterable, List, Optional

from .model import PanelModel


def model_actions(model: PanelModel, uri: str = "", actor: str = "program:entip-panel") -> dict:
    recs = []
    for name, f in model.fits.items():
        if not f.trained:
            continue
        recs.append({
            "model_id": f"{model.model_id}:{name}", "kind": "slide-mil:abmil", "encoder": model.encoder,
            "status": "draft", "version": model.version, "n_labels": f.metrics.get("n"),
            "metrics": json.loads(json.dumps(f.metrics, default=float)), "licence": model.licence,
            "uri": f"{uri.rstrip('/')}/{name}.npz" if uri else None,
            "meta": {"step": "glioma_panel", "finding": name, "temperature": f.temperature,
                     "thresholds": f.thresholds, "tile_um": model.tile_um, "trained_on": model.trained_on},
        })
    return {"verb": "upsert", "object_type": "model", "records": recs,
            "provenance": {"kind": "program", "actor": actor, "version": model.version}}


def step_action(model: PanelModel, study_id: str, records: List[dict]) -> dict:
    return {"verb": "glioma_step", "detail": {"study_id": study_id, "records": records},
            "provenance": {"kind": "model", "actor": model.model_id, "version": model.version}}


def post_actions(base_url: str, actions: Iterable[dict], timeout: float = 30.0) -> List[dict]:
    """POST each action in turn; stops at the first error (the database writes nothing for a failed action)."""
    out = []
    for a in actions:
        req = urllib.request.Request(base_url.rstrip("/") + "/actions", data=json.dumps(a).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out.append(json.loads(r.read() or b"{}"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"POST /actions ({a['verb']}) failed: {e.code} {e.read().decode(errors='replace')}")
    return out


def validate(records: List[dict]) -> Optional[int]:
    """Check records with the step 5 validator when ``entip_dx`` is installed; returns how many were checked,
    or None if the validator isn't available."""
    try:
        from entip_dx.contract import validate_finding
    except ImportError:
        return None
    for r in records:
        validate_finding(r)
    return len(records)
