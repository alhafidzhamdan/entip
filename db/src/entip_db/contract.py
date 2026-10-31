"""The schema contract other pillars code against, read from the live database."""
from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .db import DELETABLE, FORBIDDEN_KEYS, OBJECT_TYPES, VERBS_WRITE, tables

RULES = {
    "identifiers": "No direct identifiers. Patients are known by study_id only; the CHI-to-study_id key stays "
                   "in pillar 1's NHS-side key store. Payloads with these keys are refused: "
                   + ", ".join(sorted(FORBIDDEN_KEYS)) + ". A bare 10-digit number in an id is refused.",
    "dates": "No calendar dates for patients. Use day offsets (specimen.day_offset, genomic_finding.reported_day).",
    "units": "Positions and sizes in micrometres. PostGIS SRID 0 means micrometres in the object's frame.",
    "frames": "Frame names: 'scan:<scan_id>' (level-0 image, origin top-left), 'assay:<assay_id>', "
              "'section:<section_id>'. Moving between frames needs a stored transform with rmse_um.",
    "provenance": "Every object carries provenance {kind: person|model|program, actor, version?}. "
                  "If a record has none, the action's provenance is used.",
    "actions": "Every write is a logged action (POST /actions). The database refuses writes outside an action "
               "and records each change in entip.change. action and change are append-only.",
    "deletes": "Only " + ", ".join(sorted(DELETABLE)) + " rows can be deleted (to replace a run). "
               "Annotations are retracted, never deleted.",
    "data": "Research use only. Synthetic or public data only in any repository or test.",
}

GEOMETRY = {
    "point": {"type": "point", "x_um": 0.0, "y_um": 0.0},
    "polygon": {"type": "polygon", "points_um": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]},
    "box": {"type": "box", "x_um": 0.0, "y_um": 0.0, "w_um": 256.0, "h_um": 256.0},
    "note": "GeoJSON Point/Polygon with 'coordinates' in micrometres is also accepted.",
}

EXAMPLE_ACTION = {
    "verb": "create",
    "object_type": "annotation",
    "provenance": {"kind": "person", "actor": "person:trainee-01"},
    "records": [{"annotation_id": "ann_0001", "scan_id": "SCN-SYN0001-A1-01", "feature": "mitotic_figure",
                 "label": "positive", "geometry": {"type": "point", "x_um": 1520.0, "y_um": 880.5},
                 "target_type": "cell", "target_id": "SCN-SYN0001-A1-01:c000123"}],
}


def contract(conn) -> dict:
    tbls = tables(conn)
    out_tables = {}
    for name, t in sorted(tbls.items()):
        if name in ("action", "change"):
            role = "log"
        elif name in OBJECT_TYPES:
            role = "object"
        else:
            role = "other"
        out_tables[name] = {
            "role": role,
            "comment": t.comment,
            "primary_key": t.pk,
            "columns": {c.name: {"type": c.type, "nullable": c.nullable, "default": c.has_default,
                                 "generated": c.generated} for c in t.columns.values()},
        }
    fks = conn.execute("""
        SELECT tc.table_name, kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'entip'
        ORDER BY 1, 2""").fetchall()
    for fk in fks:
        col = out_tables[fk["table_name"]]["columns"][fk["column_name"]]
        col["references"] = f"{fk['ref_table']}.{fk['ref_column']}"
    view_cols = conn.execute("""SELECT column_name, data_type FROM information_schema.columns
                                WHERE table_schema = 'entip' AND table_name = 'patient_row'
                                ORDER BY ordinal_position""").fetchall()
    return {
        "name": "ENTIP central database contract",
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "rules": RULES,
        "object_types": {k: {"id": v} for k, v in OBJECT_TYPES.items()},
        "write_verbs": sorted(VERBS_WRITE),
        "geometry": GEOMETRY,
        "example_action": EXAMPLE_ACTION,
        "patient_row": {r["column_name"]: r["data_type"] for r in view_cols},
        "tables": out_tables,
    }


def export(conn, out_dir: str | Path, app=None) -> list[Path]:
    """Write schema.json, schema.sql and (with an app) openapi.json to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    p = out / "schema.json"
    p.write_text(json.dumps(contract(conn), indent=2) + "\n")
    written.append(p)
    src = resources.files("entip_db") / "migrations"
    for f in sorted(src.iterdir()):
        if f.name.endswith(".sql"):
            dst = out / f.name
            with resources.as_file(f) as fp:
                shutil.copyfile(fp, dst)
            written.append(dst)
    if app is not None:
        p = out / "openapi.json"
        p.write_text(json.dumps(app.openapi(), indent=2) + "\n")
        written.append(p)
    return written
