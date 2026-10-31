// Run: node --test gui/tests
const test = require("node:test");
const assert = require("node:assert");
const { LiveHead, rng } = require("../web/js/head.js");

function cells(n = 400, seed = 3) {
  const r = rng(seed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const pos = i % 10 === 0; // 10% positives, shifted on dim 0
    const emb = Array.from({ length: 6 }, (_, d) => (r() - 0.5) * 2 + (pos && d === 0 ? 3 : 0));
    out.push({ cell_id: `C${i}`, x_um: i, y_um: i, emb, _pos: pos });
  }
  return out;
}

test("queue is all random before any marks", () => {
  const h = new LiveHead({ feature: "f", scanId: "s", cells: cells() });
  const q = h.queue(20);
  assert.strictEqual(q.length, 20);
  assert.ok(q.every((x) => x.reason === "random"));
});

test("learns from a few marks and ranks unmarked cells", () => {
  const cs = cells();
  const h = new LiveHead({ feature: "f", scanId: "s", cells: cs });
  for (const c of cs.slice(0, 40)) h.setLabel(c.cell_id, c._pos ? 1 : 0);
  const p = h.predictions();
  const meanPos = avg(p.filter((x, i) => cs[i]._pos).map((x) => x.p));
  const meanNeg = avg(p.filter((x, i) => !cs[i]._pos).map((x) => x.p));
  assert.ok(meanPos > meanNeg + 0.3, `pos ${meanPos} vs neg ${meanNeg}`);
  const q = h.queue(20);
  assert.ok(q.every((x) => !h.labels.has(x.cell_id)), "queue skips marked cells");
  assert.strictEqual(q.filter((x) => x.reason === "random").length, 4, "20% random share");
});

test("positives only: background negatives keep the head from calling everything yes", () => {
  const cs = cells();
  const h = new LiveHead({ feature: "f", scanId: "s", cells: cs });
  for (const c of cs.filter((c) => c._pos).slice(0, 5)) h.setLabel(c.cell_id, 1);
  const m = h.metrics();
  assert.ok(m.flagged < cs.length / 2, `flagged ${m.flagged}`);
});

test("removing a mark retrains and bumps the version", () => {
  const cs = cells();
  const h = new LiveHead({ feature: "f", scanId: "s", cells: cs });
  h.setLabel("C0", 1);
  const r = h.setLabel("C0", null);
  assert.strictEqual(r.model_version, 2);
  assert.strictEqual(r.n_pos, 0);
});

const avg = (a) => a.reduce((s, x) => s + x, 0) / a.length;
