"""Turn step-2 results into ``POST /actions`` payloads for the central database (db/, pillar 5).

Order of actions for one specimen:
  1. ``upsert feature``  necrosis, microvascular_proliferation (tile level), mitotic_figure (cell level)
  2. ``upsert model``    one row per detector (status ``draft`` until a gated retrain promotes it)
  3. ``upsert region``   tiles above the tile threshold (kind ``tile``) and each focus (kind ``roi``), labelled
                         with the finding; ``upsert cell`` for mitosis candidates
  4. ``upsert prediction`` scores on those regions and cells
  5. ``glioma_step``     event with ``detail = {study_id, records}`` in the step-5 contract; the rules engine
                         reads these from ``GET /patients/{study_id}/actions``

Region and focus ids carry the model id and the tile's grid position, so re-running one model replaces its
rows and a new model version never overwrites an older one's.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..wsi.bags import Bag
from .bags import tile_ids
from .contract import FeatureResult

PROGRAM = {"kind": "program", "actor": "entip-train highgrade"}

FEATURE_ROWS = [
    {"name": "necrosis", "level": "tile", "positives_only": False,
     "description": "Necrosis in a diffuse glioma (any pattern), tile level"},
    {"name": "microvascular_proliferation", "level": "tile", "positives_only": False,
     "description": "Microvascular proliferation (multilayered / glomeruloid vessels), tile level"},
    {"name": "mitotic_figure", "level": "cell", "positives_only": True, "description": "Mitotic figure"},
]


def _act(verb, object_type, records, provenance, detail=None) -> dict:
    return {"verb": verb, "object_type": object_type, "records": records, "provenance": provenance,
            "detail": detail or {}}


def region_records(result: FeatureResult, bag: Bag, tile_p: np.ndarray, tile_threshold: float):
    """Tile and focus regions plus their predictions, for one scan."""
    regions, preds = [], []
    ids = tile_ids(bag)
    src = f"model:{result.model_id}"
    for i in np.flatnonzero(tile_p >= tile_threshold):
        rid = f"{ids[i]}:{result.model_id}"
        x, y = bag.coords_um[i]
        regions.append({"region_id": rid, "scan_id": bag.scan_id, "kind": "tile", "label": result.finding,
                        "x_um": float(x), "y_um": float(y), "w_um": bag.tile_um, "h_um": bag.tile_um,
                        "source": src, "model_id": result.model_id})
        preds.append({"model_id": result.model_id, "target_type": "region", "target_id": rid,
                      "scan_id": bag.scan_id, "score": float(tile_p[i]), "label": result.finding})
    for e in result.evidence:
        if e.get("kind") != "region" or e.get("scan_id") != bag.scan_id:
            continue
        regions.append({"region_id": e["id"], "scan_id": bag.scan_id, "kind": "roi", "label": result.finding,
                        "x_um": e["x_um"], "y_um": e["y_um"], "w_um": e["w_um"], "h_um": e["h_um"],
                        "source": src, "model_id": result.model_id,
                        "meta": {"n_tiles": e["n_tiles"], "area_mm2": e["area_mm2"], "score": e["weight"]}})
        preds.append({"model_id": result.model_id, "target_type": "region", "target_id": e["id"],
                      "scan_id": bag.scan_id, "score": float(e["weight"]), "label": result.finding})
    return regions, preds


def cell_records(result: FeatureResult, candidates: Sequence[dict]):
    cells, preds = [], []
    for c in candidates:
        cells.append({"cell_id": c["id"], "scan_id": c["scan_id"], "x_um": float(c["x_um"]), "y_um": float(c["y_um"]),
                      "source": f"model:{result.model_id}", "model_id": result.model_id,
                      "meta": {"kind": "mitosis_candidate"}})
        preds.append({"model_id": result.model_id, "target_type": "cell", "target_id": c["id"],
                      "scan_id": c["scan_id"], "score": float(c["p"]), "label": "mitotic_figure"})
    return cells, preds


def build_actions(study_id: str, results: Sequence[FeatureResult], models: Sequence[dict],
                  regions: Iterable[dict] = (), cells: Iterable[dict] = (), predictions: Iterable[dict] = ()
                  ) -> List[dict]:
    regions, cells, predictions = list(regions), list(cells), list(predictions)
    acts = [_act("upsert", "feature", FEATURE_ROWS, PROGRAM)]
    for m in models:
        acts.append(_act("upsert", "model", [m], {"kind": "model", "actor": f"model:{m['model_id']}",
                                                   "version": m.get("version") or ""}))
    if regions:
        acts.append(_act("upsert", "region", regions, PROGRAM))
    if cells:
        acts.append(_act("upsert", "cell", cells, PROGRAM))
    if predictions:
        acts.append(_act("upsert", "prediction", predictions, PROGRAM))
    acts.append(_act("glioma_step", None, [], PROGRAM,
                     {"study_id": study_id, "records": [r.to_record(study_id) for r in results]}))
    return acts


def post_actions(db_url: str, actions: Sequence[dict], timeout: float = 60.0) -> List[dict]:
    """POST each action to ``<db_url>/actions`` in order; stops at the first error (HTTP 422 = nothing written)."""
    out = []
    for a in actions:
        req = urllib.request.Request(db_url.rstrip("/") + "/actions", data=json.dumps(a).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out.append(json.loads(r.read()))
    return out


def write_jsonl(actions: Sequence[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(a) + "\n" for a in actions))
    return p
