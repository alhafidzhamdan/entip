-- ENTIP central database, schema version 1.
--
-- Ontology: Patient > Specimen > Block > Section > Scan | Assay,
-- plus GenomicFinding, Transform, Feature, Model, Cell, Region,
-- SpatialObject, Annotation, Prediction.
--
-- Rules built in here (see db/README.md):
--   * No direct identifiers. The research side only knows study_id; the
--     CHI-to-study_id key stays in pillar 1's NHS-side key store.
--   * Positions are micrometres (SRID 0) in a named frame:
--     'scan:<scan_id>', 'assay:<assay_id>' or 'section:<section_id>'.
--   * Every object carries provenance (a person, or a model + version).
--   * Every write belongs to a logged action. Tables refuse writes unless the
--     transaction has set entip.action_id (entip.begin_action does this), and
--     each row change is copied to entip.change. action and change are
--     append-only.
--   * Synthetic or public data only in any repo or test.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS entip;
SET search_path = entip, public;

-- ---------------------------------------------------------------------------
-- Action log
-- ---------------------------------------------------------------------------

CREATE TABLE action (
    action_id   text PRIMARY KEY,
    verb        text NOT NULL CHECK (verb ~ '^[a-z][a-z0-9_.-]*$'),  -- create, update, retract, fork, register, train, ...
    object_type text,                                              -- main object type touched, if one
    at          timestamptz NOT NULL DEFAULT now(),
    provenance  jsonb NOT NULL CHECK (provenance ? 'kind' AND provenance ? 'actor'
                                      AND provenance->>'kind' IN ('person', 'model', 'program')),
    detail      jsonb NOT NULL DEFAULT '{}'::jsonb                 -- inputs, outputs, notes
);
COMMENT ON TABLE action IS 'One row per logged action (a request that changes data). Append-only.';

CREATE TABLE change (
    change_id   bigserial PRIMARY KEY,
    action_id   text NOT NULL REFERENCES action (action_id),
    at          timestamptz NOT NULL DEFAULT now(),
    object_type text NOT NULL,
    object_id   text,                  -- null for bulk statements (cells, spatial objects, predictions)
    op          text NOT NULL CHECK (op IN ('insert', 'update', 'delete')),
    study_id    text,                  -- the patient the change belongs to, when known
    n_rows      integer NOT NULL DEFAULT 1,
    before      jsonb,
    after       jsonb
);
CREATE INDEX change_action_idx ON change (action_id);
CREATE INDEX change_study_idx  ON change (study_id, at);
CREATE INDEX change_object_idx ON change (object_type, object_id);
COMMENT ON TABLE change IS 'One row per changed object (or per bulk statement), written by triggers. Append-only.';

CREATE FUNCTION forbid_rewrite() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'entip.%: the action log is append-only', TG_TABLE_NAME;
END $$;

CREATE TRIGGER action_append_only BEFORE UPDATE OR DELETE ON action
    FOR EACH ROW EXECUTE FUNCTION forbid_rewrite();
CREATE TRIGGER action_no_truncate BEFORE TRUNCATE ON action
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_rewrite();
CREATE TRIGGER change_append_only BEFORE UPDATE OR DELETE ON change
    FOR EACH ROW EXECUTE FUNCTION forbid_rewrite();
CREATE TRIGGER change_no_truncate BEFORE TRUNCATE ON change
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_rewrite();

-- Start an action in the current transaction. All writes until COMMIT are
-- recorded against it.
CREATE FUNCTION begin_action(p_action_id text, p_verb text, p_object_type text,
                             p_provenance jsonb, p_detail jsonb DEFAULT '{}'::jsonb)
RETURNS text LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO entip.action (action_id, verb, object_type, provenance, detail)
    VALUES (p_action_id, p_verb, p_object_type, p_provenance, coalesce(p_detail, '{}'::jsonb));
    PERFORM set_config('entip.action_id', p_action_id, true);  -- transaction-local
    RETURN p_action_id;
END $$;

CREATE FUNCTION current_action() RETURNS text LANGUAGE plpgsql STABLE AS $$
DECLARE a text := nullif(current_setting('entip.action_id', true), '');
BEGIN
    IF a IS NULL THEN
        RAISE EXCEPTION 'writes to the ENTIP ontology must belong to an action: call entip.begin_action first';
    END IF;
    RETURN a;
END $$;

-- ---------------------------------------------------------------------------
-- Frames and provenance helpers
-- ---------------------------------------------------------------------------

CREATE DOMAIN frame_name AS text CHECK (VALUE ~ '^(scan|assay|section):[A-Za-z0-9_.:-]+$');

CREATE DOMAIN provenance_doc AS jsonb CHECK (
    VALUE ? 'kind' AND VALUE ? 'actor' AND VALUE->>'kind' IN ('person', 'model', 'program')
);
COMMENT ON DOMAIN provenance_doc IS
  '{"kind": "person"|"model"|"program", "actor": "...", "version": "..."?, "at": "..."?}';

-- ---------------------------------------------------------------------------
-- Patient chain
-- ---------------------------------------------------------------------------

CREATE TABLE patient (
    study_id    text PRIMARY KEY CHECK (study_id ~ '^[A-Za-z0-9_-]{3,64}$'),
    cohort      text,
    sex         text CHECK (sex IN ('female', 'male', 'other', 'unknown')),
    age_band    text CHECK (age_band ~ '^([0-9]{1,3}-[0-9]{1,3}|[0-9]{1,3}\+|unknown)$'),
    diagnosis   text,                         -- integrated diagnosis (e.g. WHO CNS5 wording)
    who_grade   smallint CHECK (who_grade BETWEEN 1 AND 4),
    synthetic   boolean NOT NULL DEFAULT false,
    clinical    jsonb NOT NULL DEFAULT '{}'::jsonb,   -- other machine-readable clinical fields, no identifiers
    provenance  provenance_doc NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE patient IS
  'One row per pseudonymised patient. Never holds CHI, name, date of birth or accession numbers.';

CREATE TABLE specimen (
    specimen_id  text PRIMARY KEY,
    study_id     text NOT NULL REFERENCES patient (study_id),
    kind         text,                          -- resection, biopsy, ...
    site         text,                          -- anatomical site, free text or code
    day_offset   integer,                       -- days from the patient's first specimen; no calendar dates
    meta         jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance   provenance_doc NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX specimen_study_idx ON specimen (study_id);

CREATE TABLE block (
    block_id     text PRIMARY KEY,
    specimen_id  text NOT NULL REFERENCES specimen (specimen_id),
    label        text,
    meta         jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance   provenance_doc NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX block_specimen_idx ON block (specimen_id);

CREATE TABLE section (
    section_id    text PRIMARY KEY,
    block_id      text NOT NULL REFERENCES block (block_id),
    label         text,
    stain         text,                         -- H&E, Ki67, ...; null for unstained (e.g. spatial)
    thickness_um  double precision CHECK (thickness_um > 0),
    serial_index  integer,                      -- order within the block, for serial sections
    meta          jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance    provenance_doc NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX section_block_idx ON section (block_id);

CREATE TABLE scan (
    scan_id       text PRIMARY KEY,             -- pillar 1 uses the linkage code
    section_id    text NOT NULL REFERENCES section (section_id),
    link_code     text UNIQUE,                  -- code shared with the clinical copy (pillar 1)
    mpp           double precision NOT NULL CHECK (mpp > 0),   -- micrometres per pixel, level 0
    width_px      integer NOT NULL CHECK (width_px > 0),
    height_px     integer NOT NULL CHECK (height_px > 0),
    width_um      double precision GENERATED ALWAYS AS (width_px * mpp) STORED,
    height_um     double precision GENERATED ALWAYS AS (height_px * mpp) STORED,
    magnification double precision,
    scanner       text,
    format        text,                         -- tiff, svs, ndpi, dicom, ...
    uri           text,                         -- research copy location
    tile_source   text,                         -- DZI / IIIF URL the viewer opens
    sha256        text CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    meta          jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance    provenance_doc NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX scan_section_idx ON scan (section_id);

CREATE TABLE assay (
    assay_id       text PRIMARY KEY,
    section_id     text REFERENCES section (section_id),     -- spatial assays: the section measured
    specimen_id    text REFERENCES specimen (specimen_id),   -- bulk genomics: the specimen sampled
    kind           text NOT NULL,   -- visium, visium_hd, xenium, slide_tags, anndata, panel, wgs, wes, rnaseq, methylation_array, ...
    frame          frame_name,      -- spatial assays: 'assay:<assay_id>'
    n_obs          integer,
    n_vars         integer,
    uri            text,
    sha256         text CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    meta           jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance     provenance_doc NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    CHECK (section_id IS NOT NULL OR specimen_id IS NOT NULL)
);
CREATE INDEX assay_section_idx ON assay (section_id);
CREATE INDEX assay_specimen_idx ON assay (specimen_id);

CREATE TABLE genomic_finding (
    finding_id   text PRIMARY KEY,
    study_id     text NOT NULL REFERENCES patient (study_id),
    assay_id     text REFERENCES assay (assay_id),
    marker       text NOT NULL,       -- IDH, 1p19q, MGMT, CDKN2A, H3K27M, TERT, EGFR, BRAF, chr7_chr10, methylation_class, ...
    kind         text NOT NULL CHECK (kind IN ('mutation', 'codeletion', 'deletion', 'amplification',
                                               'copy_number', 'fusion', 'methylation', 'expression',
                                               'class', 'other')),
    result       text NOT NULL,       -- e.g. 'mutant', 'wildtype', 'codeleted', 'methylated', 'homozygous deletion'
    detail       text,                -- e.g. 'IDH1 R132H'
    value        double precision,    -- numeric value when there is one (VAF, score)
    unit         text,
    method       text,                -- IHC, NGS panel, MLPA, pyrosequencing, methylation array, ...
    reported_day integer,             -- day offset, as on specimen
    meta         jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance   provenance_doc NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX genomic_finding_study_idx ON genomic_finding (study_id, marker);

-- ---------------------------------------------------------------------------
-- Registration
-- ---------------------------------------------------------------------------

CREATE TABLE transform (
    transform_id  text PRIMARY KEY,
    src_frame     frame_name NOT NULL,
    dst_frame     frame_name NOT NULL,
    model         text NOT NULL CHECK (model IN ('identity', 'rigid', 'similarity', 'affine', 'nonrigid')),
    matrix        jsonb,                        -- 3x3 homogeneous, row-major, micrometres -> micrometres
    rmse_um       double precision NOT NULL CHECK (rmse_um >= 0),   -- fit error, required
    loo_rmse_um   double precision CHECK (loo_rmse_um >= 0),
    max_err_um    double precision CHECK (max_err_um >= 0),
    n_landmarks   integer,
    method        text,
    meta          jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance    provenance_doc NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    CHECK (model = 'nonrigid' OR jsonb_array_length(matrix) = 3)
);
CREATE INDEX transform_dst_idx ON transform (dst_frame);
CREATE INDEX transform_src_idx ON transform (src_frame);

-- ---------------------------------------------------------------------------
-- Features and models
-- ---------------------------------------------------------------------------

CREATE TABLE feature (
    name            text PRIMARY KEY,           -- e.g. 'mitotic_figure', 'necrosis'
    level           text NOT NULL CHECK (level IN ('cell', 'tile', 'region', 'spot')),
    description     text NOT NULL DEFAULT '',
    positives_only  boolean NOT NULL DEFAULT false,
    provenance      provenance_doc NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE model (
    model_id     text PRIMARY KEY,
    feature      text REFERENCES feature (name),
    kind         text NOT NULL,                 -- 'live-head:xgboost', 'retrain:mitoed', 'segmenter:cellpose-sam', ...
    encoder      text,                          -- 'h-optimus-0', ...
    status       text NOT NULL CHECK (status IN ('draft', 'challenger', 'champion', 'rejected', 'retired')),
    version      text,
    parent_id    text REFERENCES model (model_id),
    n_labels     integer,
    metrics      jsonb NOT NULL DEFAULT '{}'::jsonb,
    licence      text,                          -- licence of the weights / base model (NHS rule)
    uri          text,                          -- saved weights or head
    meta         jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance   provenance_doc NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX model_feature_idx ON model (feature, status);
-- At most one champion per feature.
CREATE UNIQUE INDEX model_one_champion ON model (feature) WHERE status = 'champion';

-- ---------------------------------------------------------------------------
-- Objects on slides and assays
-- ---------------------------------------------------------------------------

CREATE TABLE cell (
    cell_id     text PRIMARY KEY,
    scan_id     text NOT NULL REFERENCES scan (scan_id),
    frame       frame_name NOT NULL,
    x_um        double precision NOT NULL,
    y_um        double precision NOT NULL,
    geom        geometry(Point, 0) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(x_um, y_um), 0)) STORED,
    outline     geometry(Polygon, 0),           -- nucleus outline, micrometres, optional
    area_um2    double precision,
    source      text NOT NULL,                  -- 'model:cellpose-sam@4.0', 'synthetic', 'person:<name>'
    model_id    text REFERENCES model (model_id),
    meta        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CHECK (frame = 'scan:' || scan_id)
);
CREATE INDEX cell_scan_idx ON cell (scan_id);
CREATE INDEX cell_geom_idx ON cell USING gist (geom);

CREATE TABLE region (
    region_id   text PRIMARY KEY,
    scan_id     text NOT NULL REFERENCES scan (scan_id),
    frame       frame_name NOT NULL,
    kind        text NOT NULL DEFAULT 'polygon' CHECK (kind IN ('tile', 'polygon', 'hotspot', 'roi')),
    label       text,                           -- what the region is, e.g. 'necrosis', 'tumour', 'white_matter'
    geom        geometry(Polygon, 0) NOT NULL,
    source      text NOT NULL,
    model_id    text REFERENCES model (model_id),
    meta        jsonb NOT NULL DEFAULT '{}'::jsonb,
    provenance  provenance_doc,
    created_at  timestamptz NOT NULL DEFAULT now(),
    CHECK (frame = 'scan:' || scan_id)
);
CREATE INDEX region_scan_idx ON region (scan_id, label);
CREATE INDEX region_geom_idx ON region USING gist (geom);

CREATE TABLE spatial_object (
    object_id     text PRIMARY KEY,
    assay_id      text NOT NULL REFERENCES assay (assay_id),
    kind          text NOT NULL CHECK (kind IN ('spot', 'nucleus', 'bin', 'cell')),
    barcode       text NOT NULL,
    frame         frame_name NOT NULL,          -- frame of x_um, y_um (usually the H&E scan)
    x_um          double precision NOT NULL,
    y_um          double precision NOT NULL,
    geom          geometry(Point, 0) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(x_um, y_um), 0)) STORED,
    radius_um     double precision,
    transform_id  text REFERENCES transform (transform_id),  -- how it got into this frame
    label         text,
    label_score   double precision,
    meta          jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (assay_id, barcode),
    CHECK (frame = 'assay:' || assay_id OR transform_id IS NOT NULL)
);
CREATE INDEX spatial_object_assay_idx ON spatial_object (assay_id);
CREATE INDEX spatial_object_frame_idx ON spatial_object (frame);
CREATE INDEX spatial_object_geom_idx ON spatial_object USING gist (geom);

CREATE TABLE annotation (
    annotation_id  text PRIMARY KEY,
    scan_id        text NOT NULL REFERENCES scan (scan_id),
    frame          frame_name NOT NULL,
    feature        text NOT NULL REFERENCES feature (name),
    label          text NOT NULL CHECK (length(label) > 0),   -- 'positive' / 'negative' for live heads
    geometry       jsonb NOT NULL,     -- as sent: {"type":"point","x_um","y_um"} or {"type":"polygon","points_um":[[x,y],...]}
    geom           geometry(Geometry, 0) NOT NULL,
    target_type    text CHECK (target_type IN ('cell', 'region', 'spatial_object')),
    target_id      text,
    locked         boolean NOT NULL DEFAULT false,   -- held out for the retrain gate
    retracted      boolean NOT NULL DEFAULT false,   -- set by a 'retract' action, never deleted
    provenance     provenance_doc NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    CHECK (frame = 'scan:' || scan_id)
);
CREATE INDEX annotation_scan_idx ON annotation (scan_id, feature);
CREATE INDEX annotation_geom_idx ON annotation USING gist (geom);

CREATE TABLE prediction (
    model_id     text NOT NULL REFERENCES model (model_id),
    target_type  text NOT NULL CHECK (target_type IN ('cell', 'region', 'spatial_object')),
    target_id    text NOT NULL,
    scan_id      text REFERENCES scan (scan_id),
    score        double precision NOT NULL,
    label        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, target_type, target_id)
);
CREATE INDEX prediction_scan_idx ON prediction (scan_id, model_id);

-- ---------------------------------------------------------------------------
-- Which patient a row belongs to (for the per-patient action log)
-- ---------------------------------------------------------------------------

CREATE FUNCTION study_of_specimen(p text) RETURNS text LANGUAGE sql STABLE AS
  $$ SELECT study_id FROM entip.specimen WHERE specimen_id = p $$;
CREATE FUNCTION study_of_block(p text) RETURNS text LANGUAGE sql STABLE AS
  $$ SELECT entip.study_of_specimen(specimen_id) FROM entip.block WHERE block_id = p $$;
CREATE FUNCTION study_of_section(p text) RETURNS text LANGUAGE sql STABLE AS
  $$ SELECT entip.study_of_block(block_id) FROM entip.section WHERE section_id = p $$;
CREATE FUNCTION study_of_scan(p text) RETURNS text LANGUAGE sql STABLE AS
  $$ SELECT entip.study_of_section(section_id) FROM entip.scan WHERE scan_id = p $$;
CREATE FUNCTION study_of_assay(p text) RETURNS text LANGUAGE sql STABLE AS
  $$ SELECT coalesce(entip.study_of_section(section_id), entip.study_of_specimen(specimen_id))
     FROM entip.assay WHERE assay_id = p $$;

CREATE FUNCTION study_of_frame(f text) RETURNS text LANGUAGE sql STABLE AS $$
  SELECT CASE split_part(f, ':', 1)
           WHEN 'scan'    THEN entip.study_of_scan(substr(f, 6))
           WHEN 'assay'   THEN entip.study_of_assay(substr(f, 7))
           WHEN 'section' THEN entip.study_of_section(substr(f, 9))
         END $$;

CREATE FUNCTION study_of(tbl text, r jsonb) RETURNS text LANGUAGE sql STABLE AS $$
  SELECT CASE tbl
           WHEN 'patient'         THEN r->>'study_id'
           WHEN 'specimen'        THEN r->>'study_id'
           WHEN 'genomic_finding' THEN r->>'study_id'
           WHEN 'block'           THEN entip.study_of_specimen(r->>'specimen_id')
           WHEN 'section'         THEN entip.study_of_block(r->>'block_id')
           WHEN 'scan'            THEN entip.study_of_section(r->>'section_id')
           WHEN 'assay'           THEN coalesce(entip.study_of_section(r->>'section_id'),
                                                entip.study_of_specimen(r->>'specimen_id'))
           WHEN 'cell'            THEN entip.study_of_scan(r->>'scan_id')
           WHEN 'region'          THEN entip.study_of_scan(r->>'scan_id')
           WHEN 'annotation'      THEN entip.study_of_scan(r->>'scan_id')
           WHEN 'spatial_object'  THEN entip.study_of_assay(r->>'assay_id')
           WHEN 'prediction'      THEN entip.study_of_scan(r->>'scan_id')
           WHEN 'transform'       THEN coalesce(entip.study_of_frame(r->>'dst_frame'),
                                                entip.study_of_frame(r->>'src_frame'))
         END $$;

-- ---------------------------------------------------------------------------
-- Change triggers
-- ---------------------------------------------------------------------------

-- Row-level: one change row per object. TG_ARGV[0] is the id column.
CREATE FUNCTION log_row_change() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    a      text := entip.current_action();
    b      jsonb;
    n      jsonb;
    r      jsonb;
BEGIN
    IF TG_OP <> 'INSERT' THEN b := to_jsonb(OLD) - 'geom' - 'outline'; END IF;
    IF TG_OP <> 'DELETE' THEN n := to_jsonb(NEW) - 'geom' - 'outline'; END IF;
    r := coalesce(n, b);
    INSERT INTO entip.change (action_id, object_type, object_id, op, study_id, before, after)
    VALUES (a, TG_TABLE_NAME, r->>TG_ARGV[0], lower(TG_OP), entip.study_of(TG_TABLE_NAME, r), b, n);
    RETURN coalesce(NEW, OLD);
END $$;

-- Statement-level for bulk tables: one change row per statement and scan/assay.
CREATE FUNCTION log_bulk_insert() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE a text := entip.current_action();
BEGIN
    INSERT INTO entip.change (action_id, object_type, op, study_id, n_rows, after)
    SELECT a, TG_TABLE_NAME, 'insert', entip.study_of(TG_TABLE_NAME, g.r), g.n,
           jsonb_build_object('parent', g.parent)
    FROM (SELECT coalesce(to_jsonb(t)->>'scan_id', to_jsonb(t)->>'assay_id') AS parent,
                 min(to_jsonb(t)::text)::jsonb AS r, count(*)::int AS n
          FROM new_rows t GROUP BY 1) g;
    RETURN NULL;
END $$;

CREATE FUNCTION log_bulk_delete() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE a text := entip.current_action();
BEGIN
    INSERT INTO entip.change (action_id, object_type, op, study_id, n_rows, before)
    SELECT a, TG_TABLE_NAME, 'delete', entip.study_of(TG_TABLE_NAME, g.r), g.n,
           jsonb_build_object('parent', g.parent, 'ids', g.ids)
    FROM (SELECT coalesce(to_jsonb(t)->>'scan_id', to_jsonb(t)->>'assay_id') AS parent,
                 min(to_jsonb(t)::text)::jsonb AS r, count(*)::int AS n,
                 jsonb_agg(coalesce(to_jsonb(t)->>'cell_id', to_jsonb(t)->>'object_id',
                                    to_jsonb(t)->>'target_id')) AS ids
          FROM old_rows t GROUP BY 1) g;
    RETURN NULL;
END $$;

CREATE FUNCTION log_bulk_update() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE a text := entip.current_action();
BEGIN
    INSERT INTO entip.change (action_id, object_type, op, study_id, n_rows, after)
    SELECT a, TG_TABLE_NAME, 'update', entip.study_of(TG_TABLE_NAME, g.r), g.n,
           jsonb_build_object('parent', g.parent)
    FROM (SELECT coalesce(to_jsonb(t)->>'scan_id', to_jsonb(t)->>'assay_id') AS parent,
                 min(to_jsonb(t)::text)::jsonb AS r, count(*)::int AS n
          FROM new_rows t GROUP BY 1) g;
    RETURN NULL;
END $$;

-- Bulk tables still refuse writes outside an action, even when no rows change.
CREATE FUNCTION require_action() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM entip.current_action();
    RETURN NULL;
END $$;

DO $$
DECLARE
    t record;
BEGIN
    FOR t IN SELECT * FROM (VALUES
        ('patient', 'study_id'), ('specimen', 'specimen_id'), ('block', 'block_id'),
        ('section', 'section_id'), ('scan', 'scan_id'), ('assay', 'assay_id'),
        ('genomic_finding', 'finding_id'), ('transform', 'transform_id'),
        ('feature', 'name'), ('model', 'model_id'), ('region', 'region_id'),
        ('annotation', 'annotation_id')) AS v(tbl, idcol)
    LOOP
        EXECUTE format('CREATE TRIGGER %I AFTER INSERT OR UPDATE OR DELETE ON entip.%I
                        FOR EACH ROW EXECUTE FUNCTION entip.log_row_change(%L)',
                       t.tbl || '_log', t.tbl, t.idcol);
        EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON entip.%I
                        FOR EACH STATEMENT EXECUTE FUNCTION entip.forbid_rewrite()',
                       t.tbl || '_no_truncate', t.tbl);
    END LOOP;

    FOR t IN SELECT * FROM (VALUES ('cell'), ('spatial_object'), ('prediction')) AS v(tbl)
    LOOP
        EXECUTE format('CREATE TRIGGER %I BEFORE INSERT OR UPDATE OR DELETE ON entip.%I
                        FOR EACH STATEMENT EXECUTE FUNCTION entip.require_action()',
                       t.tbl || '_need_action', t.tbl);
        EXECUTE format('CREATE TRIGGER %I AFTER INSERT ON entip.%I REFERENCING NEW TABLE AS new_rows
                        FOR EACH STATEMENT EXECUTE FUNCTION entip.log_bulk_insert()',
                       t.tbl || '_log_ins', t.tbl);
        EXECUTE format('CREATE TRIGGER %I AFTER UPDATE ON entip.%I REFERENCING NEW TABLE AS new_rows
                        FOR EACH STATEMENT EXECUTE FUNCTION entip.log_bulk_update()',
                       t.tbl || '_log_upd', t.tbl);
        EXECUTE format('CREATE TRIGGER %I AFTER DELETE ON entip.%I REFERENCING OLD TABLE AS old_rows
                        FOR EACH STATEMENT EXECUTE FUNCTION entip.log_bulk_delete()',
                       t.tbl || '_log_del', t.tbl);
        EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON entip.%I
                        FOR EACH STATEMENT EXECUTE FUNCTION entip.forbid_rewrite()',
                       t.tbl || '_no_truncate', t.tbl);
    END LOOP;
END $$;

-- ---------------------------------------------------------------------------
-- One row per patient
-- ---------------------------------------------------------------------------

CREATE VIEW patient_row AS
SELECT
    p.study_id, p.cohort, p.sex, p.age_band, p.diagnosis, p.who_grade, p.synthetic, p.clinical,
    coalesce(m.molecular, '{}'::jsonb)                       AS molecular,
    coalesce(m.molecular_detail, '{}'::jsonb)                AS molecular_detail,
    (SELECT count(*) FROM specimen sp WHERE sp.study_id = p.study_id)::int AS n_specimens,
    coalesce(s.n_blocks, 0)                                  AS n_blocks,
    coalesce(s.n_sections, 0)                                AS n_sections,
    coalesce(s.n_scans, 0)                                   AS n_scans,
    coalesce(a.n_assays, 0)                                  AS n_assays,
    coalesce(s.scan_ids, ARRAY[]::text[])                    AS scan_ids,
    coalesce(s.stains, ARRAY[]::text[])                      AS stains,
    coalesce(a.assay_ids, ARRAY[]::text[])                   AS assay_ids,
    coalesce(a.assay_kinds, ARRAY[]::text[])                 AS assay_kinds,
    p.created_at
FROM patient p
LEFT JOIN LATERAL (
    -- latest finding per marker
    SELECT jsonb_object_agg(f.marker, f.result) AS molecular,
           jsonb_object_agg(f.marker, jsonb_build_object(
               'result', f.result, 'detail', f.detail, 'value', f.value, 'unit', f.unit,
               'method', f.method, 'finding_id', f.finding_id, 'assay_id', f.assay_id)) AS molecular_detail
    FROM (SELECT DISTINCT ON (g.marker) g.*
          FROM genomic_finding g WHERE g.study_id = p.study_id
          ORDER BY g.marker, g.reported_day DESC NULLS LAST, g.created_at DESC) f
) m ON true
LEFT JOIN LATERAL (
    SELECT count(DISTINCT b.block_id)::int   AS n_blocks,
           count(DISTINCT se.section_id)::int AS n_sections,
           count(DISTINCT sc.scan_id)::int    AS n_scans,
           array_remove(array_agg(DISTINCT sc.scan_id), NULL) AS scan_ids,
           array_remove(array_agg(DISTINCT se.stain), NULL)   AS stains
    FROM specimen sp
    JOIN block b ON b.specimen_id = sp.specimen_id
    LEFT JOIN section se ON se.block_id = b.block_id
    LEFT JOIN scan sc ON sc.section_id = se.section_id
    WHERE sp.study_id = p.study_id
) s ON true
LEFT JOIN LATERAL (
    SELECT count(*)::int AS n_assays,
           array_agg(x.assay_id ORDER BY x.assay_id) AS assay_ids,
           array_agg(DISTINCT x.kind) AS assay_kinds
    FROM (SELECT a1.assay_id, a1.kind FROM assay a1
          JOIN specimen sp ON sp.specimen_id = a1.specimen_id WHERE sp.study_id = p.study_id
          UNION
          SELECT a2.assay_id, a2.kind FROM assay a2
          JOIN section se ON se.section_id = a2.section_id
          JOIN block b ON b.block_id = se.block_id
          JOIN specimen sp ON sp.specimen_id = b.specimen_id WHERE sp.study_id = p.study_id) x
) a ON true;
COMMENT ON VIEW patient_row IS
  'One row per patient: clinical fields, latest result per molecular marker, and linked scans and assays.';
