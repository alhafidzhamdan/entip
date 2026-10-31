/* All data access for the GUI goes through this file.
 *
 *   mode "mock": synthetic data from ./mock and an in-browser live head (head.js).
 *   mode "http": the central database (P5) and the training backend (P2) over HTTP.
 *
 * Shapes follow the ontology in docs/pillar_specs.md: Patient > Specimen > Block > Section > Scan | Assay,
 * positions in µm in frame "scan:<scan_id>", every change an appended action with provenance.
 * The HTTP routes are the ones agreed with the database and training threads; see gui/README.md. */
(function (root) {
  "use strict";

  const STORE_KEY = "entip-gui-mock-actions-v1";

  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(`${opts && opts.method || "GET"} ${url} failed: ${r.status}`);
    return r.json();
  }

  function nowISO() { return new Date().toISOString(); }

  /* ---------- mock ---------- */
  class MockApi {
    constructor({ base = "mock" } = {}) {
      this.base = base;
      this.mode = "mock";
      this._patients = null;
      this._cells = new Map();
      this._heads = new Map(); // head_id -> LiveHead
      this._actions = this._loadActions();
    }

    _loadActions() {
      try { return JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); } catch (e) { return []; }
    }
    _saveActions() {
      try { localStorage.setItem(STORE_KEY, JSON.stringify(this._actions)); } catch (e) { /* private mode */ }
    }

    async _data() {
      if (!this._patients) this._patients = await getJSON(`${this.base}/patients.json`);
      return this._patients;
    }

    async listPatients() {
      const d = await this._data();
      return d.patients.map((p) => {
        const scans = [], assays = [];
        walk(p, (kind, o) => { if (kind === "scan") scans.push(o); if (kind === "assay") assays.push(o); });
        const reviewed = this._actions.filter((a) => a.study_id === p.study_id && a.verb === "label").length;
        return { ...rowOf(p), n_scans: scans.length, n_assays: assays.length, n_marks: reviewed,
                 thumb: scans.find((s) => s.tile_source) ? `${this.base}/slides/${scans.find((s) => s.tile_source).scan_id}/thumb.jpg` : null };
      });
    }

    async getPatient(studyId) {
      const d = await this._data();
      const p = d.patients.find((x) => x.study_id === studyId);
      if (!p) throw new Error(`No patient ${studyId}`);
      const copy = JSON.parse(JSON.stringify(p));
      walk(copy, (kind, o) => {
        if (kind === "scan" && o.tile_source) { o.thumb = `${this.base}/slides/${o.scan_id}/thumb.jpg`; o.tile_source = `${this.base}/${o.tile_source}`; }
      });
      return copy;
    }

    async getScan(scanId) {
      const d = await this._data();
      for (const p of d.patients) {
        let hit = null, path = [];
        walk(p, (kind, o, trail) => { if (kind === "scan" && o.scan_id === scanId) { hit = o; path = trail; } });
        if (hit) {
          return { ...hit, tile_source: hit.tile_source ? `${this.base}/${hit.tile_source}` : null,
                   study_id: p.study_id, diagnosis: p.diagnosis, path };
        }
      }
      throw new Error(`No scan ${scanId}`);
    }

    async getCells(scanId) {
      if (!this._cells.has(scanId)) this._cells.set(scanId, await getJSON(`${this.base}/slides/${scanId}/cells.json`));
      return this._cells.get(scanId);
    }

    async listActions({ study_id, scan_id } = {}) {
      return this._actions
        .filter((a) => (!study_id || a.study_id === study_id) && (!scan_id || a.scan_id === scan_id))
        .slice().reverse();
    }

    async logAction(a) {
      const rec = { action_id: `A${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, ts: nowISO(), ...a };
      this._actions.push(rec);
      this._saveActions();
      return rec;
    }

    async getOutputs(scanId) { // per-cell outputs of existing models (MitoEd mock) for one scan
      if (!this._outputs) this._outputs = new Map();
      if (!this._outputs.has(scanId)) {
        let o = [];
        try { o = [await getJSON(`${this.base}/slides/${scanId}/mitoed.json`)]; } catch (e) { o = []; }
        this._outputs.set(scanId, o);
      }
      return this._outputs.get(scanId);
    }

    async clearMockActions() { this._actions = []; this._saveActions(); this._heads.clear(); }

    async listFeatures() {
      const d = await this._data();
      const custom = new Set(this._actions.filter((a) => a.verb === "define_feature").map((a) => a.feature));
      return [...d.features, ...[...custom].filter((f) => !d.features.some((x) => x.name === f))
        .map((name) => ({ name, level: "cell", description: "Added by a user", source: "user" }))];
    }

    /* Training API (mirrors P2): heads are per feature and scan in the mock. */
    async createHead({ feature, level = "cell", scan_id }) {
      const head_id = `${feature}::${scan_id}`;
      if (!this._heads.has(head_id)) {
        const table = await this.getCells(scan_id);
        const head = new root.ENTIPHead.LiveHead({ feature, scanId: scan_id, cells: table.cells });
        // Rebuild from the action log: the log is the truth, the head is derived from it.
        for (const a of this._actions) {
          if (a.verb === "label" && a.feature === feature && a.scan_id === scan_id && head.index.has(a.object_id)) {
            head.labels.set(a.object_id, a.label);
          } else if (a.verb === "retract_label" && a.feature === feature && a.scan_id === scan_id) {
            head.labels.delete(a.object_id);
          }
        }
        if (head.labels.size) head.fit(600);
        this._heads.set(head_id, head);
      }
      return { head_id, level };
    }

    _head(id) { const h = this._heads.get(id); if (!h) throw new Error(`No head ${id}`); return h; }

    async addLabel(head_id, { scan_id, cell_id, label, actor, study_id, context }) {
      const h = this._head(head_id);
      await this.logAction({ verb: "label", object_type: "cell", object_id: cell_id, feature: h.feature,
        label: label ? 1 : 0, scan_id, study_id, actor: { kind: "person", name: actor }, ...(context ? { context } : {}) });
      const res = h.setLabel(cell_id, label);
      await this.logAction({ verb: "train_head", object_type: "model", object_id: head_id, feature: h.feature,
        scan_id, study_id, actor: { kind: "model", name: `live-head:${h.feature}`, version: `v${res.model_version}` },
        detail: `${res.n_pos} positive, ${res.n_neg} negative marks; ${res.changed} cells changed call` });
      return { ...res, queue: h.queue(20) };
    }

    async removeLabel(head_id, { scan_id, cell_id, actor, study_id }) {
      const h = this._head(head_id);
      await this.logAction({ verb: "retract_label", object_type: "cell", object_id: cell_id, feature: h.feature,
        scan_id, study_id, actor: { kind: "person", name: actor } });
      const res = h.setLabel(cell_id, null);
      return { ...res, queue: h.queue(20) };
    }

    async getQueue(head_id, { n = 20 } = {}) { return this._head(head_id).queue(n); }
    async getPredictions(head_id) { return this._head(head_id).predictions(); }
    async getLabels(head_id) { return [...this._head(head_id).labels].map(([cell_id, label]) => ({ cell_id, label })); }
    async headStatus(head_id) {
      const h = this._head(head_id);
      return { model_version: h.version, ...h.counts(), metrics: h.metrics(), history: h.history.slice(-12) };
    }
  }

  /* ---------- http (P5 database + P2 training) ----------
   * Database routes follow db/schema/README.md (schema 0001). Training routes follow training/README.md.
   * The database is the record: marks are written there first as annotations, then sent to the training
   * service for the live head. If the training service is down or does not know the scan yet, the mark is
   * still recorded and the GUI says so. */
  const SEX = { female: "F", male: "M", other: "Other", unknown: "Unknown" };
  const MARKER_LABEL = { "1p19q": "1p/19q", chr7gain_chr10loss: "+7/-10" };
  const humanise = (name) => { const s = String(name).replace(/_/g, " "); return s.charAt(0).toUpperCase() + s.slice(1); };
  const featureKey = (s) => String(s).toLowerCase().trim().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
  const sectionScans = (se) => se.scans || [];

  class HttpApi {
    constructor({ dbBase = "http://127.0.0.1:8705", trainBase = "http://127.0.0.1:8100" } = {}) {
      this.mode = "http";
      this.db = dbBase.replace(/\/$/, "");
      this.tr = trainBase.replace(/\/$/, "");
      this._ann = new Map();      // `${scan}|${feature}|${cell}` -> annotation_id, for retraction
      this._labelIds = new Map(); // `${head}|${cell}` -> training label_id
      this._heads = new Map();    // head_id -> {feature, scan_id, local labels}
      this._scans = new Map();
      this._cells = new Map();
      this._models = null;
    }
    _post(url, body, method = "POST") {
      return getJSON(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    }
    _act(verb, object_type, records, actor, detail) {
      const provenance = actor ? { kind: "person", actor: `person:${actor}` } : { kind: "program", actor: "gui" };
      return this._post(`${this.db}/actions`, { verb, object_type, provenance, records, detail: detail || {} });
    }
    _h(id) { return `${this.tr}/heads/${encodeURIComponent(id)}`; }

    _row(r) {
      const molecular = {};
      for (const [k, v] of Object.entries(r.molecular || {})) molecular[MARKER_LABEL[k] || k] = v;
      return { ...r, sex: SEX[r.sex] || r.sex, molecular, review_status: (r.clinical || {}).review_status || "New" };
    }

    async listPatients() {
      const rows = await getJSON(`${this.db}/patients?limit=1000`);
      return rows.map((r) => ({ ...this._row(r), model_outputs: [], thumb: null }));
    }

    async getPatient(id) {
      const p = this._row(await getJSON(`${this.db}/patients/${encodeURIComponent(id)}`));
      const outputs = [];
      for (const sp of p.specimens || []) {
        sp.procedure = humanise(sp.kind || "specimen");
        const specimenAssays = sp.assays || [];
        for (const bl of sp.blocks || []) {
          for (const se of bl.sections || []) {
            se.level = se.serial_index;
            se.assays = (se.assays || []).map(asy);
            for (const sc of sectionScans(se)) {
              sc.objective = sc.magnification ? `${sc.magnification}x` : "";
              sc.copy = "research";
              sc.linkage = "held in NHS key store";
              if (!sc.tile_source) sc.note = "No viewable image registered.";
              else if (/\.dzi$/.test(sc.tile_source) && sc.width_px) { // a small level of the Deep Zoom pyramid
                const top = Math.ceil(Math.log2(Math.max(sc.width_px, sc.height_px || 0)));
                sc.thumb = sc.tile_source.replace(/\.dzi$/, `_files/${top - Math.max(0, Math.ceil(Math.log2(Math.max(sc.width_px, sc.height_px || 0) / 512)))}/0_0.jpg`);
              }
              if (sc.tile_source) {
                for (const o of await this.getOutputs(sc.scan_id)) {
                  const n = o.predictions.filter((x) => x.p >= o.threshold).length;
                  const area = (sc.meta || {}).tissue_area_mm2;
                  outputs.push({ object_type: "scan", object_id: sc.scan_id, value: area ? (n / area).toFixed(1) : n,
                    measure: area ? `${humanise(o.feature)}s per mm²` : `${humanise(o.feature)}s`,
                    detail: `${n} called at ≥ ${o.threshold}${area ? ` in ${area} mm² of tissue` : ""}`,
                    source: { kind: "model", name: o.model, version: o.version } });
                }
              }
            }
          }
          // specimen-level (bulk) assays sit on the first section of the first block for display
          if (specimenAssays.length && bl === sp.blocks[0] && bl.sections && bl.sections[0]) {
            bl.sections[0].assays = [...bl.sections[0].assays, ...specimenAssays.map(asy)];
          }
        }
      }
      p.model_outputs = outputs;
      return p;
    }

    async getScan(id) {
      if (!this._scans.has(id)) {
        const sc = await getJSON(`${this.db}/scans/${encodeURIComponent(id)}`);
        let diagnosis = "";
        try { diagnosis = (await getJSON(`${this.db}/patients/${encodeURIComponent(sc.study_id)}`)).diagnosis; } catch (e) { /* optional */ }
        this._scans.set(id, { ...sc, diagnosis, objective: sc.magnification ? `${sc.magnification}x` : "",
          copy: "research", linkage: "held in NHS key store", tissue_area_mm2: (sc.meta || {}).tissue_area_mm2 });
      }
      return this._scans.get(id);
    }

    async getCells(id) {
      if (this._cells.has(id)) return this._cells.get(id);
      const scan = await this.getScan(id);
      const cells = [];
      let after = null;
      for (;;) { // page through by cell_id
        const q = new URLSearchParams({ limit: 50000 }); if (after) q.set("after", after);
        const page = await getJSON(`${this.db}/scans/${encodeURIComponent(id)}/cells?${q}`);
        for (const c of page) cells.push({ cell_id: c.cell_id, x_um: c.x_um, y_um: c.y_um,
          r_um: c.area_um2 ? Math.sqrt(c.area_um2 / Math.PI) : 4, source: c.source });
        if (page.length < 50000) break;
        after = page[page.length - 1].cell_id;
      }
      const regions = (await getJSON(`${this.db}/scans/${encodeURIComponent(id)}/regions`))
        .filter((r) => r.geojson && r.geojson.type === "Polygon")
        .map((r) => ({ region_id: r.region_id, kind: (r.meta || {}).label || r.kind, polygon_um: r.geojson.coordinates[0] }));
      const t = { scan_id: id, frame: `scan:${id}`, mpp: scan.mpp, cells, regions,
        tissue_area_mm2: scan.tissue_area_mm2, source: { kind: "model", name: cells[0] ? cells[0].source : "none" } };
      this._cells.set(id, t);
      return t;
    }

    async _modelList() {
      if (!this._models) this._models = await getJSON(`${this.db}/models`);
      return this._models;
    }

    async getOutputs(scanId) {
      const out = [];
      for (const m of await this._modelList()) {
        if (!m.feature || String(m.kind || "").startsWith("segmenter")) continue;
        const preds = await getJSON(`${this.db}/scans/${encodeURIComponent(scanId)}/predictions?${new URLSearchParams({ model_id: m.model_id, order: "score", limit: 100000 })}`);
        if (!preds.length) continue;
        const meta = m.meta || {};
        out.push({ model: meta.display_name || m.model_id, model_id: m.model_id, version: m.version || "", feature: m.feature,
          threshold: meta.threshold ?? 0.5, predictions: preds.map((x) => ({ cell_id: x.target_id, p: x.score })) });
      }
      return out;
    }

    async listActions({ study_id } = {}) {
      if (!study_id) return [];
      const rows = await getJSON(`${this.db}/patients/${encodeURIComponent(study_id)}/actions?limit=5000`);
      const out = [];
      for (const a of rows) {
        const prov = a.provenance || {};
        const actor = { kind: prov.kind === "person" ? "person" : "model", name: String(prov.actor || "").replace(/^person:/, ""), version: prov.version };
        if (a.object_type === "annotation") {
          for (const ch of a.changes || []) {
            const x = ch.after || ch.before || {};
            out.push({ ts: a.at, actor, verb: a.verb === "retract" ? "retract_label" : "label", object_id: x.target_id || ch.object_id || "",
              feature: humanise(x.feature || ""), label: x.label === "positive" ? 1 : 0, scan_id: x.scan_id });
          }
        } else {
          out.push({ ts: a.at, actor, verb: `${a.verb} ${a.object_type || ""}`.trim(), scan_id: null });
        }
      }
      return out.reverse();
    }

    logAction(a) {
      if (a.verb === "define_feature") {
        return this._act("upsert", "feature", [{ name: featureKey(a.feature), level: a.level || "cell", description: "Added in the GUI" }], a.actor && a.actor.name);
      }
      return this._post(`${this.db}/actions`, { verb: featureKey(a.verb), provenance: { kind: "person", actor: `person:${a.actor ? a.actor.name : "gui"}` }, detail: a });
    }

    async listFeatures() {
      const rows = await getJSON(`${this.db}/features`);
      return rows.map((f) => ({ name: humanise(f.name), key: f.name, level: f.level, description: f.description || "", source: "database" }));
    }

    /* Training: record in the database, then train in the training service if it can. */
    async createHead({ feature, level = "cell", scan_id }) {
      const key = featureKey(feature);
      let head_id = null, live = false;
      try {
        head_id = (await this._post(`${this.tr}/heads`, { feature: key, level, scan_id, actor: "gui" })).head_id; live = true;
      } catch (e) {
        try { const hs = await getJSON(`${this.tr}/heads`); const hit = hs.find((h) => h.feature && h.feature.name === key); if (hit) { head_id = hit.head_id; live = true; } } catch (e2) { /* training service down */ }
      }
      if (live) { // the training service must also know this scan's cells
        try { await getJSON(`${this._h(head_id)}/queue?${new URLSearchParams({ scan_id, n: 1 })}`); } catch (e) { live = false; }
      }
      head_id = head_id || `db:${key}:${scan_id}`;
      this._heads.set(head_id, { key, scan_id, live, labels: new Map() });
      return { head_id, level, live };
    }

    async getLabels(head_id) {
      const h = this._heads.get(head_id);
      const anns = await getJSON(`${this.db}/scans/${encodeURIComponent(h.scan_id)}/annotations?${new URLSearchParams({ feature: h.key })}`);
      h.labels.clear();
      for (const a of anns) {
        if (a.target_type !== "cell" || !a.target_id) continue;
        h.labels.set(a.target_id, a.label === "positive" ? 1 : 0);
        this._ann.set(`${h.scan_id}|${h.key}|${a.target_id}`, a.annotation_id);
      }
      return [...h.labels].map(([cell_id, label]) => ({ cell_id, label }));
    }

    async addLabel(head_id, { scan_id, cell_id, label, actor }) {
      const h = this._heads.get(head_id);
      const cell = (this._cells.get(scan_id) || { cells: [] }).cells.find((c) => c.cell_id === cell_id);
      const k = `${scan_id}|${h.key}|${cell_id}`;
      if (this._ann.has(k)) await this._act("retract", "annotation", [{ annotation_id: this._ann.get(k) }], actor);
      const annotation_id = `ann_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
      await this._act("create", "annotation", [{ annotation_id, scan_id, feature: h.key, label: label ? "positive" : "negative",
        geometry: { type: "point", x_um: cell ? cell.x_um : 0, y_um: cell ? cell.y_um : 0 }, target_type: "cell", target_id: cell_id }], actor);
      this._ann.set(k, annotation_id);
      h.labels.set(cell_id, label ? 1 : 0);
      return this._train(head_id, () => this._post(`${this._h(head_id)}/labels`, { scan_id, cell_id, label: label ? 1 : 0, actor }).then((r) => {
        if (r.label_id) this._labelIds.set(`${head_id}|${cell_id}`, r.label_id);
        return r;
      }));
    }

    async removeLabel(head_id, { scan_id, cell_id, actor }) {
      const h = this._heads.get(head_id);
      const k = `${scan_id}|${h.key}|${cell_id}`;
      if (this._ann.has(k)) { await this._act("retract", "annotation", [{ annotation_id: this._ann.get(k) }], actor); this._ann.delete(k); }
      h.labels.delete(cell_id);
      const id = this._labelIds.get(`${head_id}|${cell_id}`);
      return this._train(head_id, () => id
        ? getJSON(`${this._h(head_id)}/labels/${encodeURIComponent(id)}?${new URLSearchParams({ actor })}`, { method: "DELETE" })
        : Promise.reject(new Error("no training label")));
    }

    async _train(head_id, call) {
      const h = this._heads.get(head_id);
      const counts = () => { let n_pos = 0, n_neg = 0; for (const v of h.labels.values()) v ? n_pos++ : n_neg++; return { n_pos, n_neg }; };
      if (!h.live) return { model_version: null, ...counts(), recorded_only: true, warning: "Saved to the database. Live training is not connected for this slide yet." };
      try { return await call(); }
      catch (e) { return { model_version: null, ...counts(), recorded_only: true, warning: "Saved to the database. The training service could not use it." }; }
    }

    async getQueue(head_id, { scan_id, n = 20 } = {}) {
      const h = this._heads.get(head_id);
      if (h.live) {
        try { return await getJSON(`${this._h(head_id)}/queue?${new URLSearchParams({ scan_id, n })}`); } catch (e) { /* fall back */ }
      }
      // Fallback: the database's own uncertainty order for an existing model of this feature.
      const m = (await this._modelList()).find((x) => x.feature === h.key && !String(x.kind || "").startsWith("segmenter"));
      if (!m) return [];
      const preds = await getJSON(`${this.db}/scans/${encodeURIComponent(scan_id)}/predictions?${new URLSearchParams({ model_id: m.model_id, order: "uncertain", limit: n * 3 })}`);
      return preds.filter((x) => !h.labels.has(x.target_id)).slice(0, n).map((x) => ({ cell_id: x.target_id, p: x.score, reason: "uncertain" }));
    }

    async getPredictions(head_id, { scan_id } = {}) {
      const h = this._heads.get(head_id);
      if (!h.live) return [];
      try {
        const r = await getJSON(`${this._h(head_id)}/predictions?${new URLSearchParams({ scan_id })}`);
        return Array.isArray(r) ? r : r.predictions || [];
      } catch (e) { return []; }
    }

    async headStatus(head_id) {
      const h = this._heads.get(head_id);
      let n_pos = 0, n_neg = 0; for (const v of h.labels.values()) v ? n_pos++ : n_neg++;
      if (h.live) {
        try {
          const r = await getJSON(this._h(head_id));
          const m = r.draft || r.champion || {};
          return { model_version: m.version ?? m.model_id ?? 0, n_pos, n_neg, metrics: m.metrics || {}, history: [], live: true };
        } catch (e) { /* fall through */ }
      }
      return { model_version: null, n_pos, n_neg, metrics: {}, history: [], live: false };
    }
  }

  function asy(a) {
    const meta = a.meta || {};
    return { ...a, kind: humanise(a.kind), platform: meta.platform || meta.panel || "", status: meta.status || (a.uri ? "registered" : "recorded"),
      result: meta.result || null, source: meta.result ? { kind: "model", name: meta.platform || humanise(a.kind) } : null };
  }

  /* ---------- helpers ---------- */
  function rowOf(p) {
    const { specimens, ...row } = p;
    return row;
  }

  // Visit every object in a patient tree: specimen, block, section, scan, assay.
  function walk(p, fn) {
    for (const sp of p.specimens || []) {
      fn("specimen", sp, [sp.specimen_id]);
      for (const bl of sp.blocks || []) {
        fn("block", bl, [sp.specimen_id, bl.block_id]);
        for (const se of bl.sections || []) {
          const trail = [sp.specimen_id, bl.block_id, se.section_id];
          fn("section", se, trail);
          for (const sc of se.scans || []) fn("scan", sc, trail);
          for (const as of se.assays || []) fn("assay", as, trail);
        }
      }
    }
  }

  function createApi(cfg = {}) {
    return cfg.mode === "http" ? new HttpApi(cfg) : new MockApi(cfg);
  }

  root.ENTIPApi = { createApi, walk, MockApi, HttpApi, featureKey };
})(typeof self !== "undefined" ? self : this);
