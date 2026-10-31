"""Load the GUI's synthetic patients and slides into the central database (db/ pillar) over its HTTP API.

For trying the GUI in "http" mode against a real database. Everything loaded is synthetic.
Run make_mock_slides.py and make_mock_patients.py first, start the database API (entip-db serve), then:

    python gui/tools/load_mock_into_db.py --db http://127.0.0.1:8705

Scans get tile_source "mock/slides/<scan_id>/slide.dzi", relative to the GUI page, so serve the GUI from
gui/web as usual. Safe to run twice (upserts; cells and regions are replaced per scan).
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

MOCK = Path(__file__).resolve().parents[1] / "web" / "mock"
PROV = {"kind": "program", "actor": "gui:load_mock_into_db", "version": "1"}
MODEL_ID = "mdl-mitoed-mock"

# The GUI mock uses display names; the database keys markers as below (see db/schema/README.md).
MARKER = {"1p/19q": "1p19q", "CDKN2A/B": "CDKN2A", "TERT promoter": "TERT", "+7/-10": "chr7gain_chr10loss"}
KIND = {"IDH": "mutation", "1p19q": "codeletion", "MGMT": "methylation", "CDKN2A": "deletion", "TERT": "mutation",
        "EGFR": "amplification", "ATRX": "expression", "TP53": "mutation", "chr7gain_chr10loss": "other"}
SEX = {"F": "female", "M": "male"}


def post(db: str, body: dict) -> dict:
    req = urllib.request.Request(f"{db}/actions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{body['verb']} {body.get('object_type')}: {e.code} {e.read().decode()[:400]}")


def act(db, verb, object_type, records, **kw):
    return post(db, {"verb": verb, "object_type": object_type, "provenance": PROV, "records": records, **kw})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="http://127.0.0.1:8705")
    a = ap.parse_args()
    db = a.db.rstrip("/")
    data = json.loads((MOCK / "patients.json").read_text())

    act(db, "upsert", "feature", [{"name": "mitotic_figure", "level": "cell", "positives_only": True,
                                   "description": "Mitotic figure (MitoEd)"}])
    act(db, "upsert", "model", [{"model_id": MODEL_ID, "feature": "mitotic_figure", "kind": "classifier:mitoed",
                                 "status": "draft", "version": "0.5.4-mock", "licence": "MIT (synthetic output)",
                                 "meta": {"display_name": "MitoEd", "threshold": 0.5,
                                          "note": "Synthetic detections from the GUI mock"}}])
    for p in data["patients"]:
        sid = p["study_id"]
        act(db, "upsert", "patient", [{"study_id": sid, "cohort": "gui-mock", "sex": SEX.get(p["sex"], "unknown"),
                                       "age_band": p["age_band"], "diagnosis": p["diagnosis"],
                                       "who_grade": p["who_grade"], "synthetic": True,
                                       "clinical": {"review_status": p["review_status"]}}])
        for sp in p["specimens"]:
            act(db, "upsert", "specimen", [{"specimen_id": sp["specimen_id"], "study_id": sid,
                                            "kind": sp["procedure"].lower(), "site": sp["site"], "day_offset": 0}])
            panel = f"{sp['specimen_id']}-PANEL"
            act(db, "upsert", "assay", [{"assay_id": panel, "specimen_id": sp["specimen_id"], "kind": "panel",
                                         "meta": {"panel": "synthetic glioma panel"}}])
            findings = []
            for k, v in p["molecular"].items():
                m = MARKER.get(k, k)
                findings.append({"finding_id": f"GF-{sid}-{m}", "study_id": sid, "assay_id": panel, "marker": m,
                                 "kind": KIND.get(m, "other"), "result": v, "method": "synthetic", "reported_day": 14})
            act(db, "upsert", "genomic_finding", findings)
            for bl in sp["blocks"]:
                act(db, "upsert", "block", [{"block_id": bl["block_id"], "specimen_id": sp["specimen_id"],
                                             "label": bl["label"]}])
                act(db, "upsert", "section", [{"section_id": se["section_id"], "block_id": bl["block_id"],
                                               "label": se["section_id"].split("-")[-1], "stain": se["stain"],
                                               "thickness_um": se["thickness_um"], "serial_index": se["level"]}
                                              for se in bl["sections"]])
                for se in bl["sections"]:
                    for sc in se["scans"]:
                        summary = {}
                        cells_path = MOCK / "slides" / sc["scan_id"] / "cells.json"
                        if sc.get("tile_source") and cells_path.exists():
                            summary = json.loads(cells_path.read_text())
                        act(db, "upsert", "scan", [{
                            "scan_id": sc["scan_id"], "section_id": se["section_id"], "link_code": sc["scan_id"],
                            "mpp": sc["mpp"], "width_px": sc.get("width_px") or 6144, "height_px": sc.get("height_px") or 4608,
                            "magnification": 40, "format": "dzi" if sc.get("tile_source") else "none",
                            "uri": f"synthetic:{sc['scan_id']}",
                            "tile_source": f"mock/{sc['tile_source']}" if sc.get("tile_source") else None,
                            "meta": {"tissue_area_mm2": summary.get("tissue_area_mm2")} if summary else {}}])
                        if summary:
                            load_slide(db, sc["scan_id"], summary)
                    for asy in se["assays"]:
                        kind = "methylation_array" if "methylation" in asy["kind"].lower() else "visium"
                        meta = {k: asy[k] for k in ("platform", "status", "result") if asy.get(k)}
                        rec = {"assay_id": asy["assay_id"], "kind": kind, "meta": meta}
                        rec["section_id" if kind == "visium" else "specimen_id"] = se["section_id"] if kind == "visium" else sp["specimen_id"]
                        act(db, "upsert", "assay", [rec])
        print("loaded", sid)


def load_slide(db, scan_id, table):
    act(db, "delete", "cell", [{"scan_id": scan_id}])
    cells = [{"cell_id": c["cell_id"], "scan_id": scan_id, "x_um": c["x_um"], "y_um": c["y_um"],
              "area_um2": round(math.pi * c["r_um"] ** 2, 1), "source": "synthetic-segmenter",
              "model_id": None} for c in table["cells"]]
    for i in range(0, len(cells), 5000):
        act(db, "create", "cell", cells[i:i + 5000])
    act(db, "delete", "region", [{"scan_id": scan_id}])
    for r in table.get("regions", []):
        act(db, "upsert", "region", [{"region_id": r["region_id"], "scan_id": scan_id, "kind": "polygon",
                                      "meta": {"label": r["kind"]}, "source": "synthetic",
                                      "geometry": {"type": "polygon", "points_um": r["polygon_um"]}}])
    det = MOCK / "slides" / scan_id / "mitoed.json"
    if det.exists():
        preds = json.loads(det.read_text())["predictions"]
        act(db, "upsert", "prediction", [{"model_id": MODEL_ID, "target_type": "cell", "target_id": x["cell_id"],
                                          "scan_id": scan_id, "score": x["p"]} for x in preds])


if __name__ == "__main__":
    main()
