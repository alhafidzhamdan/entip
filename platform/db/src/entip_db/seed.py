"""Synthetic demo patients. Every value here is made up; none comes from a person.

Three CNS tumour patients with WHO CNS5-style diagnoses and the markers those
diagnoses rest on, one H&E scan each (plus Ki67 and a Visium section for the
first), synthetic cells, a feature, a model and a few annotations.
"""
from __future__ import annotations

import random

from .db import apply_action

SEEDER = {"kind": "program", "actor": "entip-db:seed", "version": "0.1.0"}

PATIENTS = [
    {
        "study_id": "SYN-0001", "sex": "male", "age_band": "60-69", "who_grade": 4,
        "diagnosis": "Glioblastoma, IDH-wildtype",
        "findings": [("IDH", "mutation", "wildtype", None, "NGS panel"),
                     ("MGMT", "methylation", "methylated", None, "pyrosequencing"),
                     ("EGFR", "amplification", "amplified", None, "NGS panel"),
                     ("TERT", "mutation", "mutant", "TERT promoter C228T", "NGS panel"),
                     ("CDKN2A", "deletion", "homozygous deletion", None, "NGS panel")],
        "spatial": True,
    },
    {
        "study_id": "SYN-0002", "sex": "female", "age_band": "40-49", "who_grade": 2,
        "diagnosis": "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted",
        "findings": [("IDH", "mutation", "mutant", "IDH1 R132H", "IHC"),
                     ("1p19q", "codeletion", "codeleted", None, "NGS panel"),
                     ("TERT", "mutation", "mutant", "TERT promoter C250T", "NGS panel")],
        "spatial": False,
    },
    {
        "study_id": "SYN-0003", "sex": "female", "age_band": "30-39", "who_grade": 3,
        "diagnosis": "Astrocytoma, IDH-mutant",
        "findings": [("IDH", "mutation", "mutant", "IDH1 R132H", "IHC"),
                     ("ATRX", "mutation", "loss", None, "IHC"),
                     ("1p19q", "codeletion", "not codeleted", None, "NGS panel"),
                     ("CDKN2A", "deletion", "retained", None, "NGS panel")],
        "spatial": False,
    },
]

MPP = 0.25


def _ids(study_id: str) -> dict:
    n = study_id.split("-")[1]
    return {"specimen": f"SPC-{n}-1", "block": f"BLK-{n}-A1", "he": f"SEC-{n}-A1-01", "ki67": f"SEC-{n}-A1-02",
            "vis": f"SEC-{n}-A1-03", "scan_he": f"SCN-{n}-A1-01", "scan_ki67": f"SCN-{n}-A1-02",
            "panel": f"ASY-{n}-PANEL", "visium": f"ASY-{n}-VIS"}


def seed(conn, n_cells: int = 400, rng_seed: int = 7) -> dict:
    """Write the demo patients. Safe to run twice (uses upsert)."""
    rng = random.Random(rng_seed)
    out = {"patients": [], "actions": []}

    def act(verb, object_type, records, **kw):
        r = apply_action(conn, verb=verb, object_type=object_type, records=records, provenance=SEEDER, **kw)
        out["actions"].append(r["action_id"])
        return r

    act("upsert", "feature", [{"name": "mitotic_figure", "level": "cell", "positives_only": True,
                               "description": "Mitotic figure (MitoEd)"},
                              {"name": "necrosis", "level": "tile", "description": "Necrosis"}])
    act("upsert", "model", [{"model_id": "mdl-cellpose-sam-synthetic", "kind": "segmenter:cellpose-sam",
                             "status": "draft", "version": "synthetic", "licence": "BSD-3-Clause code; "
                             "weights trained on CC-BY-NC data (research only)"}])

    for p in PATIENTS:
        sid, ids = p["study_id"], _ids(p["study_id"])
        act("upsert", "patient", [{"study_id": sid, "cohort": "synthetic-demo", "sex": p["sex"],
                                   "age_band": p["age_band"], "diagnosis": p["diagnosis"],
                                   "who_grade": p["who_grade"], "synthetic": True}])
        act("upsert", "specimen", [{"specimen_id": ids["specimen"], "study_id": sid, "kind": "resection",
                                    "site": "cerebrum", "day_offset": 0}])
        act("upsert", "block", [{"block_id": ids["block"], "specimen_id": ids["specimen"], "label": "A1"}])
        sections = [{"section_id": ids["he"], "block_id": ids["block"], "label": "A1-01", "stain": "H&E",
                     "thickness_um": 4.0, "serial_index": 1}]
        if p["spatial"]:
            sections += [{"section_id": ids["ki67"], "block_id": ids["block"], "label": "A1-02", "stain": "Ki67",
                          "thickness_um": 4.0, "serial_index": 2},
                         {"section_id": ids["vis"], "block_id": ids["block"], "label": "A1-03", "stain": "H&E",
                          "thickness_um": 10.0, "serial_index": 3}]
        act("upsert", "section", sections)
        scans = [{"scan_id": ids["scan_he"], "section_id": ids["he"], "link_code": ids["scan_he"], "mpp": MPP,
                  "width_px": 40000, "height_px": 30000, "magnification": 40, "format": "tiff",
                  "uri": f"synthetic:{ids['scan_he']}", "tile_source": None}]
        if p["spatial"]:
            scans.append({"scan_id": ids["scan_ki67"], "section_id": ids["ki67"], "link_code": ids["scan_ki67"],
                          "mpp": MPP, "width_px": 40000, "height_px": 30000, "magnification": 40,
                          "format": "tiff", "uri": f"synthetic:{ids['scan_ki67']}"})
        act("upsert", "scan", scans)
        act("upsert", "assay", [{"assay_id": ids["panel"], "specimen_id": ids["specimen"], "kind": "panel",
                                 "meta": {"panel": "synthetic glioma panel"}}])
        act("upsert", "genomic_finding",
            [{"finding_id": f"GF-{sid}-{m}", "study_id": sid, "assay_id": ids["panel"], "marker": m, "kind": k,
              "result": r, "detail": d, "method": meth, "reported_day": 14} for m, k, r, d, meth in p["findings"]])

        # Synthetic cells on the H&E scan (clustered, in micrometres).
        w_um, h_um = 40000 * MPP, 30000 * MPP
        centres = [(rng.uniform(0.2, 0.8) * w_um, rng.uniform(0.2, 0.8) * h_um) for _ in range(4)]
        cells = []
        for i in range(n_cells):
            cx, cy = centres[i % 4]
            cells.append({"cell_id": f"{ids['scan_he']}:c{i:06d}", "scan_id": ids["scan_he"],
                          "x_um": round(rng.gauss(cx, 150.0), 2), "y_um": round(rng.gauss(cy, 150.0), 2),
                          "area_um2": round(rng.uniform(20, 80), 1), "source": "synthetic",
                          "model_id": "mdl-cellpose-sam-synthetic"})
        act("delete", "cell", [{"scan_id": ids["scan_he"]}])
        act("create", "cell", cells)
        act("upsert", "region", [{"region_id": f"{ids['scan_he']}:hotspot1", "scan_id": ids["scan_he"],
                                  "kind": "hotspot", "source": "synthetic",
                                  "geometry": {"type": "box", "x_um": centres[0][0] - 500,
                                               "y_um": centres[0][1] - 500, "w_um": 1000, "h_um": 1000}}])
        picks = rng.sample(cells, 6)
        act("upsert", "annotation",
            [{"annotation_id": f"ANN-{sid}-{j:03d}", "scan_id": ids["scan_he"], "feature": "mitotic_figure",
              "label": "positive" if j < 4 else "negative",
              "geometry": {"type": "point", "x_um": c["x_um"], "y_um": c["y_um"]},
              "target_type": "cell", "target_id": c["cell_id"],
              "provenance": {"kind": "person", "actor": "person:synthetic-annotator"}}
             for j, c in enumerate(picks)])

        if p["spatial"]:
            act("upsert", "assay", [{"assay_id": ids["visium"], "section_id": ids["vis"], "kind": "visium",
                                     "n_obs": 50, "n_vars": 200, "uri": f"synthetic:{ids['visium']}"}])
            tid = f"TF-{ids['visium']}-to-{ids['scan_he']}"
            act("upsert", "transform", [{"transform_id": tid, "src_frame": f"assay:{ids['visium']}",
                                         "dst_frame": f"scan:{ids['scan_he']}", "model": "affine",
                                         "matrix": [[1.02, 0.01, 120.0], [-0.01, 0.99, 85.0], [0, 0, 1]],
                                         "rmse_um": 6.5, "loo_rmse_um": 8.1, "max_err_um": 14.0,
                                         "n_landmarks": 12, "method": "synthetic landmarks"}])
            spots = [{"object_id": f"{ids['visium']}:s{i:04d}", "assay_id": ids["visium"], "kind": "spot",
                      "barcode": f"SYNBC{i:04d}-1", "frame": f"scan:{ids['scan_he']}",
                      "x_um": 2000 + (i % 10) * 100.0, "y_um": 2000 + (i // 10) * 100.0, "radius_um": 27.5,
                      "transform_id": tid, "label": rng.choice(["tumour", "tumour", "microglia", "endothelium"])}
                     for i in range(50)]
            act("delete", "spatial_object", [{"assay_id": ids["visium"]}])
            act("create", "spatial_object", spots)
        out["patients"].append(sid)
    return out
