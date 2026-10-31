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
   * Training routes follow training/README.md (entip_train). Database routes are provisional until
   * the db/ pillar lands; they are the shapes listed in gui/README.md. Not yet run end to end. */
  class HttpApi {
    constructor({ dbBase = "/api/db", trainBase = "/api/train" } = {}) {
      this.mode = "http";
      this.db = dbBase.replace(/\/$/, "");
      this.tr = trainBase.replace(/\/$/, "");
      this._labelIds = new Map(); // `${head}|${cell}` -> label_id, for retractions
    }
    _post(url, body, method = "POST") {
      return getJSON(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    }
    _h(id) { return `${this.tr}/heads/${encodeURIComponent(id)}`; }

    listPatients() { return getJSON(`${this.db}/patients`); }
    getPatient(id) { return getJSON(`${this.db}/patients/${encodeURIComponent(id)}`); }
    getScan(id) { return getJSON(`${this.db}/scans/${encodeURIComponent(id)}`); }
    getCells(id) { return getJSON(`${this.db}/scans/${encodeURIComponent(id)}/cells`); }
    getOutputs(id) { return getJSON(`${this.db}/scans/${encodeURIComponent(id)}/outputs`); }
    listActions(q = {}) { return getJSON(`${this.db}/actions?${new URLSearchParams(q)}`); }
    logAction(a) { return this._post(`${this.db}/actions`, a); }

    async listFeatures() {
      const heads = await getJSON(`${this.tr}/heads`);
      return heads.map((h) => ({ name: h.feature.name, level: h.feature.level, description: h.feature.description || "",
                                 head_id: h.head_id, source: "training" }));
    }
    async createHead({ feature, level = "cell", scan_id }) {
      try {
        return await this._post(`${this.tr}/heads`, { feature, level, scan_id, actor: "gui" });
      } catch (e) { // already exists: find it
        const hit = (await this.listFeatures()).find((f) => f.name === feature);
        if (!hit) throw e;
        return { head_id: hit.head_id, level: hit.level };
      }
    }
    async addLabel(h, { scan_id, cell_id, label, actor }) {
      const r = await this._post(`${this._h(h)}/labels`, { scan_id, cell_id, label, actor });
      if (r.label_id) this._labelIds.set(`${h}|${cell_id}`, r.label_id);
      return r;
    }
    async removeLabel(h, { cell_id, actor }) {
      const id = this._labelIds.get(`${h}|${cell_id}`);
      if (!id) throw new Error("This mark was made in another session; remove it from the training service.");
      const r = await getJSON(`${this._h(h)}/labels/${encodeURIComponent(id)}?${new URLSearchParams({ actor })}`, { method: "DELETE" });
      this._labelIds.delete(`${h}|${cell_id}`);
      return r;
    }
    getQueue(h, { scan_id, n = 20 } = {}) { return getJSON(`${this._h(h)}/queue?${new URLSearchParams({ scan_id, n })}`); }
    async getPredictions(h, { scan_id } = {}) {
      const r = await getJSON(`${this._h(h)}/predictions?${new URLSearchParams({ scan_id })}`);
      return Array.isArray(r) ? r : r.predictions || [];
    }
    async getLabels() { return []; } // no list route yet; marks show once made in this session
    async headStatus(h) {
      const r = await getJSON(this._h(h));
      const m = r.draft || r.champion || {};
      return { model_version: m.version ?? m.model_id ?? 0, n_pos: r.n_pos, n_neg: r.n_neg, metrics: m.metrics || {}, history: [] };
    }
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

  root.ENTIPApi = { createApi, walk, MockApi, HttpApi };
})(typeof self !== "undefined" ? self : this);
