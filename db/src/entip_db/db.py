"""Connections, migrations and the one write path (actions).

Every write goes through :func:`apply_action`: it opens an action with
``entip.begin_action`` and applies the records in the same transaction. The
database itself refuses writes that do not belong to an action, and its
triggers copy every change into ``entip.change``.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from importlib import resources
from typing import Any, Iterable

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import geometry
from .db_errors import EntipError

DEFAULT_DSN = "postgresql://entip:entip@localhost:5432/entip_dev"

# Object types the API can write, and the id column of each.
OBJECT_TYPES = {
    "patient": "study_id",
    "specimen": "specimen_id",
    "block": "block_id",
    "section": "section_id",
    "scan": "scan_id",
    "assay": "assay_id",
    "genomic_finding": "finding_id",
    "transform": "transform_id",
    "feature": "name",
    "model": "model_id",
    "cell": "cell_id",
    "region": "region_id",
    "spatial_object": "object_id",
    "annotation": "annotation_id",
    "prediction": None,  # composite key (model_id, target_type, target_id)
}
# Tables where whole sets may be replaced (a new segmentation run, a new registration).
DELETABLE = {"cell", "region", "spatial_object", "prediction"}
VERBS_WRITE = {"create", "upsert", "update", "retract", "delete"}

# Keys that would carry direct identifiers. Refused anywhere in a payload.
FORBIDDEN_KEYS = {
    "chi", "chi_number", "nhs_number", "nhs_no", "mrn", "hospital_number", "accession",
    "accession_number", "patient_name", "name_of_patient", "first_name", "last_name", "surname",
    "forename", "dob", "date_of_birth", "birth_date", "address", "postcode", "post_code",
    "phone", "email", "original_name", "clinical_uri",
}


def dsn() -> str:
    return os.environ.get("ENTIP_DB_DSN", DEFAULT_DSN)


def connect(conninfo: str | None = None) -> psycopg.Connection:
    # Autocommit: reads see committed data, and each action opens its own transaction.
    conn = psycopg.connect(conninfo or dsn(), row_factory=dict_row, autocommit=True)
    conn.execute("SET search_path = entip, public")
    return conn


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def migration_files() -> list[tuple[str, str]]:
    root = resources.files("entip_db") / "migrations"
    files = sorted(p for p in root.iterdir() if p.name.endswith(".sql"))
    return [(p.name.split("_", 1)[0], p.read_text(encoding="utf-8")) for p in files]


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply migrations not yet applied. Returns the versions applied."""
    conn.execute("CREATE TABLE IF NOT EXISTS public.entip_schema_migrations "
                 "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
    done = {r["version"] for r in conn.execute("SELECT version FROM public.entip_schema_migrations")}
    applied = []
    for version, text in migration_files():
        if version in done:
            continue
        with conn.transaction():
            conn.execute(text)
            conn.execute("INSERT INTO public.entip_schema_migrations (version) VALUES (%s)", (version,))
        applied.append(version)
    conn.execute("SET search_path = entip, public")
    tables(conn, refresh=True)
    return applied


def schema_version(conn: psycopg.Connection) -> str | None:
    r = conn.execute("SELECT max(version) AS v FROM public.entip_schema_migrations").fetchone()
    return r["v"] if r else None


# ---------------------------------------------------------------------------
# Table introspection
# ---------------------------------------------------------------------------

@dataclass
class Column:
    name: str
    type: str
    nullable: bool
    has_default: bool
    generated: bool
    comment: str | None


@dataclass
class Table:
    name: str
    columns: dict[str, Column]
    pk: list[str]
    comment: str | None

    @property
    def writable(self) -> set[str]:
        return {c.name for c in self.columns.values() if not c.generated}


_TABLES: dict[str, Table] | None = None


def tables(conn: psycopg.Connection, refresh: bool = False) -> dict[str, Table]:
    global _TABLES
    if _TABLES is not None and not refresh:
        return _TABLES
    rows = conn.execute("""
        SELECT c.table_name, c.column_name,
               CASE WHEN c.data_type = 'USER-DEFINED' THEN c.udt_name
                    WHEN c.domain_name IS NOT NULL THEN c.domain_name
                    WHEN c.data_type = 'ARRAY' THEN c.udt_name
                    ELSE c.data_type END AS type,
               c.is_nullable = 'YES' AS nullable,
               c.column_default IS NOT NULL AS has_default,
               c.is_generated = 'ALWAYS' AS generated,
               col_description(('entip.' || quote_ident(c.table_name))::regclass, c.ordinal_position) AS comment
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name AND t.table_type = 'BASE TABLE'
        WHERE c.table_schema = 'entip'
        ORDER BY c.table_name, c.ordinal_position""").fetchall()
    out: dict[str, Table] = {}
    for r in rows:
        t = out.setdefault(r["table_name"], Table(r["table_name"], {}, [], None))
        t.columns[r["column_name"]] = Column(r["column_name"], r["type"], r["nullable"], r["has_default"],
                                             r["generated"], r["comment"])
    for r in conn.execute("""
        SELECT tc.relname AS table_name, a.attname AS col, obj_description(tc.oid, 'pg_class') AS comment,
               array_position(i.indkey::int2[], a.attnum) AS pos
        FROM pg_index i
        JOIN pg_class tc ON tc.oid = i.indrelid
        JOIN pg_namespace n ON n.oid = tc.relnamespace AND n.nspname = 'entip'
        JOIN pg_attribute a ON a.attrelid = tc.oid AND a.attnum = ANY (i.indkey)
        WHERE i.indisprimary ORDER BY tc.relname, pos"""):
        if r["table_name"] in out:
            out[r["table_name"]].pk.append(r["col"])
            out[r["table_name"]].comment = r["comment"]
    _TABLES = out
    return out


# ---------------------------------------------------------------------------
# Payload checks and shaping
# ---------------------------------------------------------------------------

def check_no_identifiers(payload: Any, path: str = "") -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise EntipError(f"identifier field '{path}{k}' is not allowed in the research database")
            check_no_identifiers(v, f"{path}{k}.")
    elif isinstance(payload, (list, tuple)):
        for v in payload:
            check_no_identifiers(v, path)


_CHI_LIKE = re.compile(r"(?<!\d)\d{10}(?!\d)")


def check_no_chi_like(record: dict, table: str) -> None:
    """Refuse a bare 10-digit number in the id fields: that is the CHI format."""
    for k in ("study_id", "specimen_id", "block_id", "section_id", "scan_id", "link_code"):
        v = record.get(k)
        if isinstance(v, str) and _CHI_LIKE.search(v):
            raise EntipError(f"{table}.{k} looks like a CHI number (10 digits); use the study id")


def provenance_doc(p: dict) -> dict:
    if not isinstance(p, dict) or "actor" not in p:
        raise EntipError("provenance needs at least {'kind': 'person'|'model'|'program', 'actor': ...}")
    p = dict(p)
    p.setdefault("kind", "person")
    if p["kind"] not in ("person", "model", "program"):
        raise EntipError("provenance.kind must be person, model or program")
    return p


def shape_record(table: Table, rec: dict, prov: dict, for_update: bool = False) -> dict:
    """Map an API record onto table columns: frames, geometry, provenance, unknown keys."""
    rec = {k: v for k, v in rec.items() if k != "type"}  # tolerate {"type": "Patient", ...}
    name = table.name
    if name in ("cell", "region", "annotation") and "frame" not in rec and rec.get("scan_id") and not for_update:
        rec["frame"] = f"scan:{rec['scan_id']}"
    if name == "assay" and "frame" not in rec and not for_update and rec.get("section_id"):
        rec["frame"] = f"assay:{rec['assay_id']}"
    if name == "annotation" and "geometry" in rec:
        rec["geom"] = geometry.to_wkt(rec["geometry"])
    if name == "region":
        if "geometry" in rec:
            rec["geom"] = geometry.to_wkt(rec.pop("geometry"))
        elif {"x_um", "y_um", "w_um", "h_um"} <= set(rec):
            box = {k: rec.pop(k) for k in ("x_um", "y_um", "w_um", "h_um")}
            rec["geom"] = geometry.box_wkt(**box)
            rec.setdefault("kind", "tile")
    if name == "cell" and "outline_um" in rec:
        rec["outline"] = geometry.to_wkt({"type": "polygon", "points_um": rec.pop("outline_um")})
    if "provenance" in table.columns:
        if "provenance" in rec:
            rec["provenance"] = provenance_doc(rec["provenance"])
        elif not for_update:
            rec["provenance"] = prov
    unknown = set(rec) - table.writable
    if unknown:
        raise EntipError(f"unknown field(s) for {name}: {sorted(unknown)}; "
                         f"allowed: {sorted(table.writable - {'geom'})}")
    check_no_chi_like(rec, name)
    return rec


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------

def new_action_id() -> str:
    return f"act_{uuid.uuid4().hex}"


def _key_where(table: Table, rec: dict) -> tuple[sql.Composed, list]:
    missing = [k for k in table.pk if k not in rec]
    if missing:
        raise EntipError(f"{table.name}: key field(s) {missing} needed")
    parts = [sql.SQL("{} = %s").format(sql.Identifier(k)) for k in table.pk]
    return sql.SQL(" AND ").join(parts), [rec[k] for k in table.pk]


def _insert(conn, table: Table, recs: list[dict], upsert: bool) -> int:
    n = 0
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        groups.setdefault(tuple(sorted(r)), []).append(r)
    for cols, group in groups.items():
        ident = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
        q = sql.SQL("INSERT INTO entip.{t} ({c}) SELECT {c} FROM jsonb_populate_recordset(NULL::entip.{t}, %s)"
                    ).format(t=sql.Identifier(table.name), c=ident)
        if upsert:
            rest = [c for c in cols if c not in table.pk]
            pk = sql.SQL(", ").join(sql.Identifier(c) for c in table.pk)
            if rest:
                sets = sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in rest)
                q += sql.SQL(" ON CONFLICT ({pk}) DO UPDATE SET {s}").format(pk=pk, s=sets)
            else:
                q += sql.SQL(" ON CONFLICT ({pk}) DO NOTHING").format(pk=pk)
        cur = conn.execute(q, (Jsonb(group),))
        n += cur.rowcount
    return n


def _update(conn, table: Table, recs: list[dict]) -> int:
    n = 0
    for r in recs:
        where, args = _key_where(table, r)
        cols = [c for c in r if c not in table.pk]
        if not cols:
            continue
        ident = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
        q = sql.SQL("UPDATE entip.{t} SET ({c}) = (SELECT {c} FROM jsonb_populate_record(NULL::entip.{t}, %s)) "
                    "WHERE {w}").format(t=sql.Identifier(table.name), c=ident, w=where)
        cur = conn.execute(q, [Jsonb(r), *args])
        if cur.rowcount == 0:
            raise EntipError(f"{table.name} {[r[k] for k in table.pk]} not found")
        n += cur.rowcount
    return n


def _delete(conn, table: Table, recs: list[dict]) -> int:
    if table.name not in DELETABLE:
        raise EntipError(f"{table.name} rows are never deleted; use update or retract")
    n = 0
    for r in recs:
        if table.name in ("cell", "region") and set(r) == {"scan_id"}:
            q = sql.SQL("DELETE FROM entip.{t} WHERE scan_id = %s").format(t=sql.Identifier(table.name))
            n += conn.execute(q, (r["scan_id"],)).rowcount
            continue
        if table.name == "spatial_object" and set(r) == {"assay_id"}:
            n += conn.execute("DELETE FROM entip.spatial_object WHERE assay_id = %s", (r["assay_id"],)).rowcount
            continue
        if table.name == "prediction" and set(r) == {"model_id"}:
            n += conn.execute("DELETE FROM entip.prediction WHERE model_id = %s", (r["model_id"],)).rowcount
            continue
        where, args = _key_where(table, r)
        n += conn.execute(sql.SQL("DELETE FROM entip.{t} WHERE {w}").format(t=sql.Identifier(table.name), w=where),
                          args).rowcount
    return n


def type_name(t: str) -> str:
    """'SpatialObject' / 'spatial_object' / 'Genomic finding' -> 'spatial_object'."""
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(t).strip()).replace(" ", "_").lower()
    if snake not in OBJECT_TYPES:
        raise EntipError(f"unknown object type {t!r}; one of {sorted(OBJECT_TYPES)}")
    return snake


def _split_mixed(records: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group a mixed batch, keeping order, by each record's 'type' or 'object_type'."""
    out: list[tuple[str, list[dict]]] = []
    for r in records:
        t = r.get("object_type") or r.get("type")
        if not t:
            raise EntipError("with no object_type on the action, every record needs 'type' or 'object_type'")
        t = type_name(t)
        rec = {k: v for k, v in r.items() if k not in ("type", "object_type")}
        if out and out[-1][0] == t:
            out[-1][1].append(rec)
        else:
            out.append((t, [rec]))
    return out


def _apply_batch(conn, table: Table, verb: str, records: list[dict], prov: dict) -> int:
    if verb in ("create", "upsert"):
        shaped = [shape_record(table, r, prov) for r in records]
        return _insert(conn, table, shaped, upsert=verb == "upsert")
    if verb == "update":
        return _update(conn, table, [shape_record(table, r, prov, for_update=True) for r in records])
    if verb == "retract":
        if "retracted" not in table.columns:
            raise EntipError(f"{table.name} cannot be retracted")
        return _update(conn, table, [{**{k: r[k] for k in table.pk if k in r}, "retracted": True} for r in records])
    if verb == "delete":
        return _delete(conn, table, records)
    raise EntipError(f"unknown verb {verb}")


def apply_action(conn: psycopg.Connection, *, verb: str, object_type: str | None, records: Iterable[dict] = (),
                 provenance: dict, detail: dict | None = None, action_id: str | None = None) -> dict:
    """Run one logged action in one transaction. Returns {action_id, n, changes}."""
    records = list(records or [])
    prov = provenance_doc(provenance)
    detail = detail or {}
    check_no_identifiers({"records": records, "detail": detail, "provenance": prov})
    action_id = action_id or new_action_id()
    if verb in VERBS_WRITE:
        if object_type is None:
            batches = _split_mixed(records)  # each record names its own type
        elif object_type not in OBJECT_TYPES:
            raise EntipError(f"unknown object_type {object_type!r}; one of {sorted(OBJECT_TYPES)}")
        else:
            batches = [(object_type, records)]
    elif not re.fullmatch(r"[a-z][a-z0-9_.-]*", verb or ""):
        raise EntipError("verb must be lower-case letters, digits, '_', '.', '-'")
    elif records:
        raise EntipError(f"verb {verb!r} records an event only; use create/upsert/update/retract/delete "
                         "to change objects")
    tbls = tables(conn)
    try:
        with conn.transaction():
            conn.execute("SELECT entip.begin_action(%s, %s, %s, %s, %s)",
                         (action_id, verb, object_type, Jsonb(prov), Jsonb(detail)))
            n = 0
            if verb in VERBS_WRITE:
                for otype, recs in batches:
                    n += _apply_batch(conn, tbls[otype], verb, recs, prov)
            changes = conn.execute("SELECT count(*) AS c FROM entip.change WHERE action_id = %s",
                                   (action_id,)).fetchone()["c"]
    except psycopg.errors.UniqueViolation as e:
        raise EntipError(f"already exists: {e.diag.message_detail or e}") from e
    except psycopg.errors.ForeignKeyViolation as e:
        raise EntipError(f"missing parent: {e.diag.message_detail or e}") from e
    except (psycopg.errors.CheckViolation, psycopg.errors.NotNullViolation,
            psycopg.errors.InvalidTextRepresentation, psycopg.errors.InvalidParameterValue,
            psycopg.errors.DataException) as e:
        raise EntipError(f"{e.diag.message_primary}" + (f" ({e.diag.message_detail})" if e.diag.message_detail
                                                        else "")) from e
    except psycopg.errors.InternalError_ as e:
        raise EntipError(str(e.diag.message_primary)) from e
    return {"action_id": action_id, "verb": verb, "object_type": object_type, "n": n, "changes": changes}
