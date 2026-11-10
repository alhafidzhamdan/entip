"""Read inputs from, and post drafts to, the ENTIP database API (pillar 5). Standard library only."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from .contract import FINDINGS, STEP_CONTRACT, normalise_result

ENGINE_PROV = {"kind": "program", "actor": "entip-dx"}


def _get(api: str, path: str):
    with urllib.request.urlopen(api.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())


def records_from_findings(study_id: str, findings: list[dict]) -> list[dict]:
    """Turn db genomic_finding rows (real results) into source="test" finding records."""
    out = []
    for g in sorted(findings, key=lambda g: (g.get("reported_day") is None, g.get("reported_day") or 0)):
        marker = g.get("marker", "")
        if marker.endswith("_ai"):
            continue  # AI values live in glioma_step actions, never in place of a test
        if marker == "methylation_class":
            fam = (g.get("meta") or {}).get("family") or g.get("result")
            if fam in FINDINGS["methylation_class"]["methylation_class"]:
                out.append(_test_rec(study_id, "methylation_class", "methylation_class", fam, g,
                                     {"class": g.get("detail") or g.get("result")}))
            continue
        hit = normalise_result(marker, g.get("result", ""))
        if hit is None:
            continue
        finding, call = hit
        out.append(_test_rec(study_id, "glioma_panel", finding, call, g, {"detail": g.get("detail")}))
    return out


def _test_rec(study_id, step, finding, call, g, value) -> dict:
    value = {k: v for k, v in {**value, "method": g.get("method")}.items() if v}
    return {"contract": STEP_CONTRACT, "step": step, "finding": finding, "study_id": study_id, "call": call,
            "source": "test", "confidence": 1.0, "status": "confident", "value": value,
            "evidence": [{"kind": "genomic_finding", "id": g["finding_id"]}],
            "provenance": g.get("provenance") or {"kind": "program", "actor": "lab"}}


def records_from_actions(study_id: str, actions: list[dict]) -> list[dict]:
    out = []
    for a in actions:
        if a.get("verb") != "glioma_step":
            continue
        d = a.get("detail") or {}
        for r in d.get("records", []):
            if r.get("study_id") == study_id:
                out.append(r)
    return out


def fetch_records(api: str, study_id: str) -> list[dict]:
    sid = urllib.parse.quote(study_id)
    tree = _get(api, f"/patients/{sid}")
    actions = _get(api, f"/patients/{sid}/actions?limit=5000")
    # AI first, tests after: resolve() prefers tests anyway, and later records win ties.
    return records_from_actions(study_id, actions) + records_from_findings(study_id, tree.get("genomic_findings", []))


def post_dx(api: str, dx: dict) -> dict:
    body = {"verb": "integrated_dx", "provenance": {**ENGINE_PROV, "version": dx["engine"]["version"]},
            "detail": dx}
    req = urllib.request.Request(api.rstrip("/") + "/actions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())
