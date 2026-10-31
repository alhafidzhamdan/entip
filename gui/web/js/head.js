/* Mock live head: what the training backend (P2, tier 1) does, small enough to run in the browser.
 * A warm-started logistic regression on per-cell embeddings, an uncertainty queue (|p - 0.5|) with a
 * random share, and background negatives when only positives have been marked.
 * Used only by the mock API. The real backend replaces it behind the same calls (see api.js). */
(function (root) {
  "use strict";

  function rng(seed) { // mulberry32, so queues are repeatable in tests
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  class LiveHead {
    constructor({ feature, scanId, cells, seed = 1, l2 = 0.05 }) {
      this.feature = feature;
      this.scanId = scanId;
      this.cells = cells; // [{cell_id, emb:[...]}]
      this.index = new Map(cells.map((c, i) => [c.cell_id, i]));
      this.dim = cells.length ? cells[0].emb.length : 0;
      this.w = new Float64Array(this.dim);
      this.b = 0;
      this.l2 = l2;
      this.labels = new Map(); // cell_id -> 0 | 1 (latest wins)
      this.version = 0;
      this.rand = rng(seed);
      this.p = new Float64Array(cells.length).fill(0.5);
      this.history = []; // [{version, n_pos, n_neg, changed}]
    }

    counts() {
      let pos = 0, neg = 0;
      for (const v of this.labels.values()) v ? pos++ : neg++;
      return { n_pos: pos, n_neg: neg };
    }

    setLabel(cellId, label) {
      if (!this.index.has(cellId)) throw new Error("unknown cell " + cellId);
      if (label === null) this.labels.delete(cellId);
      else this.labels.set(cellId, label ? 1 : 0);
      return this.fit();
    }

    /* Training rows: the marked cells, plus down-weighted random unmarked cells as background
     * negatives while there are no marked negatives (positives-only marking). */
    rows() {
      const rows = [];
      for (const [id, y] of this.labels) rows.push({ i: this.index.get(id), y, wt: 1 });
      const { n_pos, n_neg } = this.counts();
      if (n_pos > 0 && n_neg === 0) {
        const want = Math.min(this.cells.length - n_pos, Math.max(10, 3 * n_pos));
        const r = rng(1000 + this.version);
        let guard = 0;
        const used = new Set(rows.map((x) => x.i));
        while (rows.length < n_pos + want && guard++ < want * 20) {
          const i = Math.floor(r() * this.cells.length);
          if (!used.has(i)) { used.add(i); rows.push({ i, y: 0, wt: 0.3 }); }
        }
      }
      return rows;
    }

    fit(epochs = 300, lr = 0.3) {
      const rows = this.rows();
      const { n_pos, n_neg } = this.counts();
      const before = this.p.slice();
      if (rows.length && n_pos + n_neg > 0) {
        const w = this.w; // warm start from the last version
        for (let e = 0; e < epochs; e++) {
          const g = new Float64Array(this.dim);
          let gb = 0, tw = 0;
          for (const { i, y, wt } of rows) {
            const x = this.cells[i].emb;
            let z = this.b;
            for (let d = 0; d < this.dim; d++) z += w[d] * x[d];
            const err = (sigmoid(z) - y) * wt;
            for (let d = 0; d < this.dim; d++) g[d] += err * x[d];
            gb += err; tw += wt;
          }
          for (let d = 0; d < this.dim; d++) w[d] -= lr * (g[d] / tw + this.l2 * w[d]);
          this.b -= lr * gb / tw;
        }
        this.predictAll();
      }
      this.version++;
      let changed = 0;
      for (let i = 0; i < this.p.length; i++) if ((before[i] >= 0.5) !== (this.p[i] >= 0.5)) changed++;
      const entry = { version: this.version, n_pos, n_neg, changed };
      this.history.push(entry);
      return { model_version: this.version, n_pos, n_neg, metrics: this.metrics(), changed };
    }

    predictAll() {
      for (let i = 0; i < this.cells.length; i++) {
        const x = this.cells[i].emb;
        let z = this.b;
        for (let d = 0; d < this.dim; d++) z += this.w[d] * x[d];
        this.p[i] = sigmoid(z);
      }
    }

    metrics() {
      let flagged = 0;
      for (const v of this.p) if (v >= 0.5) flagged++;
      let agree = 0, n = 0;
      for (const [id, y] of this.labels) { n++; if ((this.p[this.index.get(id)] >= 0.5) === (y === 1)) agree++; }
      return {
        flagged,
        // Agreement with the marks the head was trained on. Optimistic; a held-out check is P2's job.
        train_agreement: n ? agree / n : null,
      };
    }

    predictions() {
      return this.cells.map((c, i) => ({ cell_id: c.cell_id, p: this.p[i] }));
    }

    /* Review queue: most uncertain unmarked cells, with a random share so the head also sees
     * cells it is confident about. Before any marks it is all random. */
    queue(n = 20, randomShare = 0.2) {
      const unl = [];
      for (let i = 0; i < this.cells.length; i++) if (!this.labels.has(this.cells[i].cell_id)) unl.push(i);
      const { n_pos, n_neg } = this.counts();
      const r = rng(7 + this.version);
      const out = [];
      const taken = new Set();
      const nRand = n_pos + n_neg === 0 ? n : Math.round(n * randomShare);
      if (n_pos + n_neg > 0) {
        unl.sort((a, b) => Math.abs(this.p[a] - 0.5) - Math.abs(this.p[b] - 0.5));
        for (const i of unl) {
          if (out.length >= n - nRand) break;
          out.push({ i, reason: "uncertain" }); taken.add(i);
        }
      }
      let guard = 0;
      while (out.length < n && guard++ < n * 50 && unl.length > taken.size) {
        const i = unl[Math.floor(r() * unl.length)];
        if (!taken.has(i)) { taken.add(i); out.push({ i, reason: "random" }); }
      }
      return out.map(({ i, reason }) => ({ cell_id: this.cells[i].cell_id, p: this.p[i], reason,
        x_um: this.cells[i].x_um, y_um: this.cells[i].y_um }));
    }
  }

  const api = { LiveHead, rng };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ENTIPHead = api;
})(typeof self !== "undefined" ? self : this);
