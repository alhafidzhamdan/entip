"""Read side: the patient row, the patient tree, slide objects, the action log."""
from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from .db import OBJECT_TYPES, tables

# Columns never returned as-is (PostGIS binary); geometry comes back as GeoJSON instead.
_GEOM = {"geom", "outline"}


def _cols(conn, table: str, alias: str = "t") -> str:
    cols = [c for c in tables(conn)[table].columns if c not in _GEOM]
    return ", ".join(f'{alias}."{c}"' for c in cols)


def _clean(rows: list[dict]) -> list[dict]:
    for r in rows:
        for k, v in list(r.items()):
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    return rows


def patients(conn, *, cohort: str | None = None, diagnosis: str | None = None,
             marker: dict[str, str] | None = None, synthetic: bool | None = None,
             limit: int = 100, offset: int = 0) -> list[dict]:
    where, args = [], []
    if cohort:
        where.append("cohort = %s"); args.append(cohort)
    if diagnosis:
        where.append("diagnosis ILIKE %s"); args.append(f"%{diagnosis}%")
    if synthetic is not None:
        where.append("synthetic = %s"); args.append(synthetic)
    for m, v in (marker or {}).items():
        where.append("molecular->>%s = %s"); args += [m, v]
    q = "SELECT * FROM entip.patient_row"
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY study_id LIMIT %s OFFSET %s"
    return _clean(conn.execute(q, [*args, limit, offset]).fetchall())


def patient_row(conn, study_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM entip.patient_row WHERE study_id = %s", (study_id,)).fetchone()
    return _clean([r])[0] if r else None


def patient_tree(conn, study_id: str) -> dict | None:
    row = patient_row(conn, study_id)
    if row is None:
        return None
    specimens = _clean(conn.execute(f"SELECT {_cols(conn, 'specimen')} FROM entip.specimen t "
                                    "WHERE study_id = %s ORDER BY day_offset NULLS LAST, specimen_id",
                                    (study_id,)).fetchall())
    sp_ids = [s["specimen_id"] for s in specimens]
    blocks = _clean(conn.execute(f"SELECT {_cols(conn, 'block')} FROM entip.block t "
                                 "WHERE specimen_id = ANY(%s) ORDER BY label, block_id", (sp_ids,)).fetchall())
    b_ids = [b["block_id"] for b in blocks]
    sections = _clean(conn.execute(f"SELECT {_cols(conn, 'section')} FROM entip.section t "
                                   "WHERE block_id = ANY(%s) ORDER BY serial_index NULLS LAST, section_id",
                                   (b_ids,)).fetchall())
    se_ids = [s["section_id"] for s in sections]
    scans = _clean(conn.execute(f"SELECT {_cols(conn, 'scan')}, se.stain FROM entip.scan t "
                                "JOIN entip.section se USING (section_id) "
                                "WHERE t.section_id = ANY(%s) ORDER BY t.scan_id", (se_ids,)).fetchall())
    assays = _clean(conn.execute(f"SELECT {_cols(conn, 'assay')} FROM entip.assay t "
                                 "WHERE section_id = ANY(%s) OR specimen_id = ANY(%s) ORDER BY assay_id",
                                 (se_ids, sp_ids)).fetchall())
    findings = _clean(conn.execute(f"SELECT {_cols(conn, 'genomic_finding')} FROM entip.genomic_finding t "
                                   "WHERE study_id = %s ORDER BY marker, reported_day NULLS LAST",
                                   (study_id,)).fetchall())
    a_ids = [a["assay_id"] for a in assays]
    s_ids = [s["scan_id"] for s in scans]
    frames = [f"scan:{i}" for i in s_ids] + [f"assay:{i}" for i in a_ids]
    transforms = _clean(conn.execute(f"SELECT {_cols(conn, 'transform')} FROM entip.transform t "
                                     "WHERE src_frame = ANY(%s) OR dst_frame = ANY(%s) ORDER BY transform_id",
                                     (frames, frames)).fetchall())
    counts = {r["scan_id"]: r for r in conn.execute("""
        SELECT s.scan_id,
               (SELECT count(*) FROM entip.cell c WHERE c.scan_id = s.scan_id)::int AS n_cells,
               (SELECT count(*) FROM entip.annotation a WHERE a.scan_id = s.scan_id AND NOT a.retracted)::int
                   AS n_annotations
        FROM unnest(%s::text[]) AS s(scan_id)""", (s_ids,))}

    for s in scans:
        s.update({k: counts[s["scan_id"]][k] for k in ("n_cells", "n_annotations")})
    by = lambda rows, key: {v: [r for r in rows if r[key] == v] for v in {r[key] for r in rows}}  # noqa: E731
    scans_by, assays_by_se = by(scans, "section_id"), by([a for a in assays if a["section_id"]], "section_id")
    for se in sections:
        se["scans"] = scans_by.get(se["section_id"], [])
        se["assays"] = assays_by_se.get(se["section_id"], [])
    sections_by = by(sections, "block_id")
    for b in blocks:
        b["sections"] = sections_by.get(b["block_id"], [])
    blocks_by = by(blocks, "specimen_id")
    assays_by_sp = by([a for a in assays if a["specimen_id"] and not a["section_id"]], "specimen_id")
    for sp in specimens:
        sp["blocks"] = blocks_by.get(sp["specimen_id"], [])
        sp["assays"] = assays_by_sp.get(sp["specimen_id"], [])
    return {**row, "specimens": specimens, "genomic_findings": findings, "transforms": transforms}


def get_object(conn, object_type: str, object_id: str) -> dict | None:
    key = OBJECT_TYPES.get(object_type)
    if key is None:
        return None
    extra = ""
    if object_type in ("cell", "region", "annotation"):
        col = "outline" if object_type == "cell" else "geom"
        extra = f", ST_AsGeoJSON(t.{col})::jsonb AS geojson"
    r = conn.execute(f"SELECT {_cols(conn, object_type)}{extra} FROM entip.{object_type} t "
                     f'WHERE t."{key}" = %s', (object_id,)).fetchone()
    if r is None:
        return None
    out = _clean([r])[0]
    out["study_id"] = conn.execute("SELECT entip.study_of(%s, %s::jsonb) AS s",
                                   (object_type, Jsonb(out))).fetchone()["s"]
    return out


def _bbox(where: list, args: list, bbox: tuple[float, float, float, float] | None, col: str = "geom"):
    if bbox:
        where.append(f"t.{col} && ST_MakeEnvelope(%s, %s, %s, %s, 0)")
        args += list(bbox)


def cells(conn, scan_id: str, *, bbox=None, model_id: str | None = None, limit: int = 50000,
          after: str | None = None) -> list[dict]:
    where, args = ["t.scan_id = %s"], [scan_id]
    _bbox(where, args, bbox)
    if after:
        where.append("t.cell_id > %s"); args.append(after)
    sel = "t.cell_id, t.scan_id, t.frame, t.x_um, t.y_um, t.area_um2, t.source, t.model_id"
    join = ""
    if model_id:
        sel += ", p.score, p.label AS predicted_label"
        join = ("LEFT JOIN entip.prediction p ON p.model_id = %s AND p.target_type = 'cell' "
                "AND p.target_id = t.cell_id")
        args = [model_id, *args]
    q = f"SELECT {sel} FROM entip.cell t {join} WHERE {' AND '.join(where)} ORDER BY t.cell_id LIMIT %s"
    return _clean(conn.execute(q, [*args, limit]).fetchall())


def regions(conn, scan_id: str, *, bbox=None, kind: str | None = None) -> list[dict]:
    where, args = ["t.scan_id = %s"], [scan_id]
    _bbox(where, args, bbox)
    if kind:
        where.append("t.kind = %s"); args.append(kind)
    return _clean(conn.execute(f"SELECT {_cols(conn, 'region')}, ST_AsGeoJSON(t.geom)::jsonb AS geojson "
                               f"FROM entip.region t WHERE {' AND '.join(where)} ORDER BY t.region_id",
                               args).fetchall())


def annotations(conn, scan_id: str, *, feature: str | None = None, include_retracted: bool = False,
                bbox=None) -> list[dict]:
    where, args = ["t.scan_id = %s"], [scan_id]
    if feature:
        where.append("t.feature = %s"); args.append(feature)
    if not include_retracted:
        where.append("NOT t.retracted")
    _bbox(where, args, bbox)
    return _clean(conn.execute(f"SELECT {_cols(conn, 'annotation')} FROM entip.annotation t "
                               f"WHERE {' AND '.join(where)} ORDER BY t.created_at, t.annotation_id",
                               args).fetchall())


def predictions(conn, scan_id: str, model_id: str, *, order: str = "uncertain", limit: int = 200) -> list[dict]:
    order_sql = {"uncertain": "abs(score - 0.5)", "score": "score DESC", "id": "target_id"}.get(order)
    if order_sql is None:
        raise ValueError("order must be uncertain, score or id")
    return _clean(conn.execute(f"SELECT * FROM entip.prediction WHERE scan_id = %s AND model_id = %s "
                               f"ORDER BY {order_sql} LIMIT %s", (scan_id, model_id, limit)).fetchall())


def spatial_objects(conn, assay_id: str, *, bbox=None, frame: str | None = None, limit: int = 100000) -> list[dict]:
    where, args = ["t.assay_id = %s"], [assay_id]
    if frame:
        where.append("t.frame = %s"); args.append(frame)
    _bbox(where, args, bbox)
    return _clean(conn.execute(f"SELECT {_cols(conn, 'spatial_object')} FROM entip.spatial_object t "
                               f"WHERE {' AND '.join(where)} ORDER BY t.object_id LIMIT %s",
                               [*args, limit]).fetchall())


def transforms(conn, frame: str | None = None) -> list[dict]:
    if frame:
        return _clean(conn.execute(f"SELECT {_cols(conn, 'transform')} FROM entip.transform t "
                                   "WHERE src_frame = %s OR dst_frame = %s ORDER BY transform_id",
                                   (frame, frame)).fetchall())
    return _clean(conn.execute(f"SELECT {_cols(conn, 'transform')} FROM entip.transform t "
                               "ORDER BY transform_id").fetchall())


def simple_list(conn, table: str, where: dict[str, Any] | None = None, order: str = "created_at") -> list[dict]:
    w, args = [], []
    for k, v in (where or {}).items():
        if v is not None:
            w.append(f'"{k}" = %s'); args.append(v)
    q = f"SELECT {_cols(conn, table)} FROM entip.{table} t"
    if w:
        q += " WHERE " + " AND ".join(w)
    return _clean(conn.execute(q + f" ORDER BY {order}", args).fetchall())


def actions(conn, *, study_id: str | None = None, object_type: str | None = None, object_id: str | None = None,
            verb: str | None = None, since: str | None = None, limit: int = 200,
            with_changes: bool = True) -> list[dict]:
    where, args = [], []
    if study_id or object_type or object_id:
        sub, sargs = [], []
        if study_id:
            sub.append("c.study_id = %s"); sargs.append(study_id)
        if object_type:
            sub.append("c.object_type = %s"); sargs.append(object_type)
        if object_id:
            sub.append("c.object_id = %s"); sargs.append(object_id)
        where.append(f"a.action_id IN (SELECT c.action_id FROM entip.change c WHERE {' AND '.join(sub)})")
        args += sargs
    if verb:
        where.append("a.verb = %s"); args.append(verb)
    if since:
        where.append("a.at > %s"); args.append(since)
    q = "SELECT a.* FROM entip.action a"
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY a.at, a.action_id LIMIT %s"
    acts = _clean(conn.execute(q, [*args, limit]).fetchall())
    if with_changes and acts:
        ids = [a["action_id"] for a in acts]
        cq = "SELECT * FROM entip.change c WHERE c.action_id = ANY(%s)"
        cargs: list = [ids]
        if study_id:
            cq += " AND c.study_id = %s"; cargs.append(study_id)
        ch = _clean(conn.execute(cq + " ORDER BY c.change_id", cargs).fetchall())
        by: dict[str, list] = {}
        for c in ch:
            by.setdefault(c["action_id"], []).append(c)
        for a in acts:
            a["changes"] = by.get(a["action_id"], [])
    return acts
