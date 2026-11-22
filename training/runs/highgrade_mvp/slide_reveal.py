"""Step 3a: score where the frozen MVP head points, and show it over the slides, after the calls are locked.

    python slide_reveal.py --pilot mvp-pilot1 --calls mvp-pilot1/slide-calls-AH.csv --bags bags/

This runs once, and only after ``slide_pilot.py compare`` has recorded the checksum of the same calls file. It re-scores
each slide with ``private/frozen-head.joblib`` and stops if a slide score differs from the private key.

**The location rules, fixed before any result was seen.** A model area is a connected group of tiles at or above the
head's own tile threshold. An outline is scored on the tiles it touches.

- *Outlines found*: outlines touched by any model area.
- *Model areas on MVP*: model areas touching any outline.
- *Top area on MVP*: slides whose highest-scoring model area touches an outline.
- *MVP tiles found* and *model tiles on MVP*: the same two questions per tile, so one huge area cannot score well by
  touching everything.
- Slides called absent report how many have any model area. Can't-tell slides are left out.

Outlines cluster within slides, so confidence intervals resample slides. A model area outside the outlines is only a
false alarm if the outlines are complete. This is one rater on a pilot set, so nothing here validates localisation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HEAT_FLOOR = 0.2     # display only: lighter tiles are left off the heatmap to keep the page small


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inside(poly: np.ndarray, x: float, y: float) -> bool:
    a, b = poly, np.roll(poly, -1, 0)
    cross = (a[:, 1] > y) != (b[:, 1] > y)
    if not cross.any():
        return False
    a, b = a[cross], b[cross]
    return bool((x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1]) + a[:, 0]).sum() % 2)


def edge_hits_box(poly: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> bool:
    """Whether any polygon edge crosses the box (Liang-Barsky, one pass over all edges)."""
    a, d = poly, np.roll(poly, -1, 0) - poly
    lo, hi, ok = np.zeros(len(a)), np.ones(len(a)), np.ones(len(a), bool)
    for p, q in ((-d[:, 0], a[:, 0] - x0), (d[:, 0], x1 - a[:, 0]), (-d[:, 1], a[:, 1] - y0), (d[:, 1], y1 - a[:, 1])):
        flat = p == 0
        ok &= ~(flat & (q < 0))
        r = np.divide(q, p, out=np.zeros_like(q, dtype=float), where=~flat)
        lo = np.where(p < 0, np.maximum(lo, r), lo)
        hi = np.where(p > 0, np.minimum(hi, r), hi)
    return bool((ok & (lo <= hi)).any())


def outline_tiles(coords: np.ndarray, tile_um: float, poly: np.ndarray) -> np.ndarray:
    """Indices of tiles that overlap or touch a freehand outline, including a tile that wholly contains it."""
    (ax, ay), (bx, by) = poly.min(0), poly.max(0)
    near = np.flatnonzero((coords[:, 0] + tile_um >= ax) & (coords[:, 0] <= bx) &
                          (coords[:, 1] + tile_um >= ay) & (coords[:, 1] <= by))
    out = []
    for i in near:
        x0, y0 = coords[i]
        x1, y1 = x0 + tile_um, y0 + tile_um
        if (((poly[:, 0] >= x0) & (poly[:, 0] <= x1) & (poly[:, 1] >= y0) & (poly[:, 1] <= y1)).any()
                or inside(poly, x0 + tile_um / 2, y0 + tile_um / 2) or edge_hits_box(poly, x0, y0, x1, y1)):
            out.append(i)
    return np.asarray(out, int)


def case_result(case: str, key: dict, call: dict, head: str, bags: str) -> tuple:
    """One slide's location counts, and the data its reveal page needs."""
    from entip_glioma.highgrade.bags import tile_ids
    from entip_glioma.highgrade.regions import RegionDetector
    from entip_glioma.wsi.bags import Bag
    det = RegionDetector.load(head)
    bag = Bag.load(Path(bags) / f"{key['slide']}.npz")
    raw, comps, per_scan, _, _ = det._score([bag])
    if abs(float(raw) - float(key["raw_score"])) > 1e-5:
        raise SystemExit(f"{case}: the re-scored slide differs from the private key")
    p, t = per_scan[bag.scan_id], float(key["tile_um"])
    ox, oy = float(key["origin_x_um"]), float(key["origin_y_um"])
    index = {name: i for i, name in enumerate(tile_ids(bag))}
    areas = [np.asarray([index[n] for n in c["tile_ids"]], int) for c in comps]
    model = np.zeros(len(p), bool)
    for idx in areas:
        model[idx] = True
    drawn = json.loads(call.get("regions_um") or "[]") if call["call"] == "present" else []
    touched = [outline_tiles(bag.coords_um, t, np.asarray(g, float) + [ox, oy]) for g in drawn]
    mvp = np.zeros(len(p), bool)
    for idx in touched:
        mvp[idx] = True
    found = [bool(len(idx) and model[idx].any()) for idx in touched]
    on_mvp = [bool(mvp[idx].any()) for idx in areas]
    row = {"case": case, "category": key["category"], "call": call["call"], "raw_score": float(key["raw_score"]),
           "outlines": sum(len(i) > 0 for i in touched), "outlines_off_tissue": sum(len(i) == 0 for i in touched),
           "outlines_found": int(sum(found)), "model_areas": len(areas), "model_areas_on_mvp": int(sum(on_mvp)),
           "top_area_on_mvp": int(on_mvp[0]) if on_mvp else 0, "mvp_tiles": int(mvp.sum()),
           "mvp_tiles_found": int((mvp & model).sum()), "model_tiles": int(model.sum()),
           "model_tiles_on_mvp": int((mvp & model).sum())}
    keep = np.flatnonzero(p >= HEAT_FLOOR)
    cells = np.round((bag.coords_um[keep] - [ox, oy]) / t).astype(int)
    heat = [int(v) for c, q in zip(cells, np.round(p[keep] * 100).astype(int)) for v in (c[0], c[1], q)]
    page = {"case": case, "category": key["category"], "call": call["call"], "score": round(float(raw), 3), "tile": t,
            "heat": heat, "outlines": drawn, "found": found,
            "areas": [{"x": round(c["x_um"] - ox, 1), "y": round(c["y_um"] - oy, 1), "w": round(c["w_um"], 1),
                       "h": round(c["h_um"], 1), "score": round(c["weight"], 3), "tiles": c["n_tiles"], "on": hit}
                      for c, hit in zip(comps, on_mvp)]}
    return row, page


def share(rows: list, top: str, bottom: str, seed: int) -> dict:
    """A pooled share with a 95% interval from resampling slides."""
    a, b = np.array([r[top] for r in rows], float), np.array([r[bottom] for r in rows], float)
    out = {"count": int(a.sum()), "of": int(b.sum()), "share": round(float(a.sum() / b.sum()), 3) if b.sum() else None,
           "ci95": None}
    if b.sum() and len(rows) > 1:
        rng, draws = np.random.default_rng(seed), []
        for _ in range(2000):
            i = rng.integers(len(rows), size=len(rows))
            if b[i].sum():
                draws.append(a[i].sum() / b[i].sum())
        out["ci95"] = [round(float(v), 3) for v in np.percentile(draws, [2.5, 97.5])]
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarise(rows: list, seed: int) -> dict:
    present = [r for r in rows if r["call"] == "present" and r["outlines"]]
    absent = [r for r in rows if r["call"] == "absent"]
    top, flagged = sum(r["top_area_on_mvp"] for r in present), sum(r["model_areas"] > 0 for r in absent)
    per_slide = [r["outlines_found"] / r["outlines"] for r in present]
    return {"present_slides_scored": len(present),
            "present_slides_without_usable_outline": sum(r["call"] == "present" and not r["outlines"] for r in rows),
            "outlines_off_tissue": sum(r["outlines_off_tissue"] for r in rows),
            "outlines_found": share(present, "outlines_found", "outlines", seed),
            "mean_share_of_outlines_found_per_slide": round(float(np.mean(per_slide)), 3) if per_slide else None,
            "model_areas_on_mvp": share(present, "model_areas_on_mvp", "model_areas", seed),
            "top_area_on_mvp": {"count": top, "of": len(present), "ci95": wilson(top, len(present))},
            "mvp_tiles_found": share(present, "mvp_tiles_found", "mvp_tiles", seed),
            "model_tiles_on_mvp": share(present, "model_tiles_on_mvp", "model_tiles", seed),
            "absent_slides_with_a_model_area": {"count": flagged, "of": len(absent), "ci95": wilson(flagged, len(absent))},
            "model_areas_on_absent_slides": int(sum(r["model_areas"] for r in absent))}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MVP model reveal</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7;--card:#fff;--on:#185fa5;--ontext:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441;--card:#2c2c2a;--on:#85b7eb;--ontext:#1d1d1b}}
html,body{height:100%;margin:0}body{display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line)}
h1{font-size:16px;font-weight:500;margin:0}button{font:inherit;font-size:13px;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
label{font-size:13px;display:inline-flex;gap:4px;align-items:center}kbd{font-size:11px;opacity:.7}.key{display:inline-block;width:12px;height:12px;border-radius:2px}
#facts{padding:4px 12px;border-bottom:1px solid var(--line);color:var(--muted);font-size:12px}
main{flex:1;min-height:0;display:flex}#wrap{flex:1;min-width:0;position:relative;background:#f4f2ee}canvas{position:absolute;inset:0;width:100%;height:100%;cursor:grab;touch-action:none}
aside{width:210px;overflow:auto;border-left:1px solid var(--line);padding:8px}aside h2{font-size:13px;font-weight:500;margin:0 0 6px}aside button{display:block;width:100%;text-align:left;margin-bottom:4px;font-size:12px}
@media (max-width:640px){main{flex-direction:column}aside{width:auto;max-height:130px;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header><h1>MVP model reveal</h1><button id="back">Back <kbd>b</kbd></button><strong id="where"></strong><button id="next">Next <kbd>n</kbd></button>
<label><input type="checkbox" id="heat" checked><span class="key" style="background:linear-gradient(90deg,#f5c542,#d62718)"></span>Heatmap <kbd>h</kbd></label>
<label><input type="checkbox" id="model" checked><span class="key" style="background:#d6168b"></span>Model areas <kbd>m</kbd></label>
<label><input type="checkbox" id="mine" checked><span class="key" style="background:#0b4f9e"></span>Your outlines <kbd>o</kbd></label>
<label>Opacity <input type="range" id="alpha" min="10" max="90" value="45"></label><button id="fit">Fit <kbd>f</kbd></button></header>
<div id="facts"></div><main><div id="wrap"><canvas id="c"></canvas></div><aside><h2>Model areas, strongest first</h2><div id="list"></div></aside></main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id), cv = $("c"), ctx = cv.getContext("2d"), cache = new Map();
let i = 0, v = {x: 0, y: 0, s: 1}, sets = [], queued = false, drag = null;
const cur = () => D.cases[i], NAME = {present: "MVP present", absent: "absent", cant: "can't tell"};
function redraw() { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw(); }); } }
function size() { const d = devicePixelRatio || 1; cv.width = cv.clientWidth * d; cv.height = cv.clientHeight * d; ctx.setTransform(d, 0, 0, d, 0, 0); if (!(v.s > 0)) fit(); redraw(); }
function fit() { const c = cur(), w = cv.clientWidth, h = cv.clientHeight; if (!w || !h) { v.s = 0; return; } v.s = .96 * Math.min(w / c.w, h / c.h); v.x = (c.w - w / v.s) / 2; v.y = (c.h - h / v.s) / 2; redraw(); }
function image(cs, l, k) { const u = `tiles/${cs}/L${l}/${k}.jpg`; let m = cache.get(u);
  if (!m) { if (cache.size > 900) cache.delete(cache.keys().next().value); m = new Image(); m.onload = redraw; m.src = u; cache.set(u, m); } return m; }
function layer(c, l, load) { const size = D.blockUm * 2 ** l, w = cv.clientWidth / v.s, h = cv.clientHeight / v.s;
  for (let col = Math.max(0, Math.floor(v.x / size)); col <= Math.floor((v.x + w) / size); col++)
    for (let row = Math.max(0, Math.floor(v.y / size)); row <= Math.floor((v.y + h) / size); row++) {
      const k = `${col}_${row}`; if (!sets[l].has(k)) continue;
      const m = load ? image(c.case, l, k) : cache.get(`tiles/${c.case}/L${l}/${k}.jpg`);
      if (m && m.complete && m.naturalWidth) ctx.drawImage(m, (col * size - v.x) * v.s, (row * size - v.y) * v.s, size * v.s + .5, size * v.s + .5); } }
const sx = x => (x - v.x) * v.s, sy = y => (y - v.y) * v.s;
function stroke(w, color) { ctx.lineWidth = w; ctx.strokeStyle = color; ctx.stroke(); }
function label(text, x, y, color) { ctx.font = "600 13px system-ui,sans-serif"; ctx.lineWidth = 3; ctx.strokeStyle = "#fff"; ctx.strokeText(text, x, y); ctx.fillStyle = color; ctx.fillText(text, x, y); }
function draw() { const c = cur(), W = cv.clientWidth, H = cv.clientHeight; ctx.setLineDash([]); ctx.fillStyle = "#f4f2ee"; ctx.fillRect(0, 0, W, H);
  const want = Math.max(0, Math.min(c.levels - 1, Math.floor(Math.log2(1 / (v.s * D.mpp))))); layer(c, c.levels - 1, true);
  for (let l = c.levels - 2; l > want; l--) layer(c, l, false); if (want < c.levels - 1) layer(c, want, true);
  if ($("heat").checked) { const t = c.tile, px = t * v.s + .5, a = +$("alpha").value / 100;
    for (let n = 0; n < c.heat.length; n += 3) { const x = sx(c.heat[n] * t), y = sy(c.heat[n + 1] * t); if (x > W || y > H || x + px < 0 || y + px < 0) continue;
      const p = c.heat[n + 2] / 100; ctx.fillStyle = `hsla(${45 - 42 * p},88%,50%,${a * (.25 + .75 * p)})`; ctx.fillRect(x, y, px, px); } }
  if ($("mine").checked) c.outlines.forEach((g, n) => { ctx.beginPath(); g.forEach((p, j) => j ? ctx.lineTo(sx(p[0]), sy(p[1])) : ctx.moveTo(sx(p[0]), sy(p[1]))); ctx.closePath(); ctx.lineJoin = "round"; stroke(4, "#fff"); stroke(2, "#0b4f9e"); });
  if ($("model").checked) c.areas.forEach((a, n) => { ctx.beginPath(); ctx.rect(sx(a.x), sy(a.y), a.w * v.s, a.h * v.s); ctx.setLineDash([]); stroke(4, "#fff"); ctx.setLineDash(a.on ? [] : [6, 4]); stroke(2, "#d6168b"); ctx.setLineDash([]); label(n + 1, sx(a.x) + 3, sy(a.y) - 4, "#a30f69"); }); }
function show(n) { i = Math.max(0, Math.min(D.cases.length - 1, n)); const c = cur(); sets = []; for (let l = 0; l < c.levels; l++) sets[l] = new Set(c.blocks[l].split(" "));
  const on = c.areas.filter(a => a.on).length, got = c.found.filter(Boolean).length; $("where").textContent = `${c.case} (${i + 1} of ${D.cases.length})`;
  $("facts").textContent = `Your call: ${NAME[c.call]} | ${c.category} | model score ${c.score.toFixed(3)} | ${c.areas.length} model area${c.areas.length === 1 ? "" : "s"}` + (c.outlines.length ? `, ${on} on your outlines | ${got} of ${c.outlines.length} outlines found` : "") + " | solid box touches an outline, dashed does not | research pilot, not calibrated";
  const list = $("list"); list.replaceChildren(); if (!c.areas.length) list.textContent = "No model areas on this slide.";
  c.areas.forEach((a, n) => { const b = document.createElement("button"); b.textContent = `${n + 1}. ${a.score.toFixed(2)}, ${a.tiles} tile${a.tiles === 1 ? "" : "s"}${c.outlines.length ? (a.on ? ", on outline" : ", off outline") : ""}`;
    b.onclick = () => { const m = Math.max(a.w, a.h) * 2.2, s = Math.min(cv.clientWidth, cv.clientHeight) / m; v.s = s; v.x = a.x + a.w / 2 - cv.clientWidth / s / 2; v.y = a.y + a.h / 2 - cv.clientHeight / s / 2; redraw(); }; list.append(b); }); fit(); }
["heat", "model", "mine", "alpha"].forEach(id => $(id).oninput = redraw); $("back").onclick = () => show(i - 1); $("next").onclick = () => show(i + 1); $("fit").onclick = fit;
cv.onpointerdown = e => { cv.setPointerCapture(e.pointerId); drag = {x: e.offsetX, y: e.offsetY, vx: v.x, vy: v.y}; };
cv.onpointermove = e => { if (drag) { v.x = drag.vx - (e.offsetX - drag.x) / v.s; v.y = drag.vy - (e.offsetY - drag.y) / v.s; redraw(); } }; cv.onpointerup = cv.onpointercancel = () => { drag = null; };
cv.addEventListener("wheel", e => { e.preventDefault(); const s = Math.max(.002, Math.min(8 / D.mpp, v.s * Math.exp(-e.deltaY * .0016))), x = v.x + e.offsetX / v.s, y = v.y + e.offsetY / v.s; v.s = s; v.x = x - e.offsetX / s; v.y = y - e.offsetY / s; redraw(); }, {passive: false});
const flip = id => { $(id).checked = !$(id).checked; redraw(); };
document.addEventListener("keydown", e => { if (e.metaKey || e.ctrlKey || e.altKey) return; const k = e.key.toLowerCase(); if (k === "n") show(i + 1); else if (k === "b") show(i - 1); else if (k === "f") fit(); else if (k === "h") flip("heat"); else if (k === "m") flip("model"); else if (k === "o") flip("mine"); });
new ResizeObserver(size).observe($("wrap")); v.s = 0; show(0); size();
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", required=True, help="the pilot folder holding private/, results/ and review-pack/")
    ap.add_argument("--calls", required=True, help="the locked slide-calls CSV used by compare")
    ap.add_argument("--bags", required=True)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pilot, out = Path(a.pilot), Path(a.pilot) / "reveal"
    done = pilot / "results" / "pilot-results.json"
    if not done.exists():
        raise SystemExit("run slide_pilot.py compare first: the model's locations stay hidden until the calls are locked")
    if (out / "location-results.json").exists():
        raise SystemExit(f"{out / 'location-results.json'} already exists: the location scoring is run once")
    res = json.loads(done.read_text())
    if res.get("calls_sha256") != sha256(a.calls):
        raise SystemExit("this calls file is not the one the comparison locked")
    head = pilot / "private" / "frozen-head.joblib"
    keys = {r["case"]: r for r in csv.DictReader(open(pilot / "private" / "key.csv", newline=""))}
    if sha256(head) != res["head_sha256"]:
        raise SystemExit("the frozen head has changed since the pilot was built")
    calls = {r["case"]: r for r in csv.DictReader(open(a.calls, newline=""))}
    if set(keys) != set(calls):
        raise SystemExit("the calls do not cover the same cases as the key")
    text = (pilot / "review-pack" / "review.html").read_text()
    found = re.search(r"^const D = (.*);$", text, re.M)
    if not found:
        raise SystemExit("could not read the slide layout from review.html")
    data = json.loads(found.group(1))
    layout = {c["case"]: c for c in data["cases"]}
    rows, pages = [], []
    with ProcessPoolExecutor(max(1, a.workers or int(os.environ.get("NSLOTS", 1)))) as pool:
        jobs = [pool.submit(case_result, c, k, calls[c], str(head), a.bags) for c, k in keys.items()]
        for n, job in enumerate(as_completed(jobs), start=1):
            row, page = job.result()
            rows.append(row)
            pages.append({**layout[row["case"]], **page})
            print(f"scored {n}/{len(jobs)} slides", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["case"])
    pages.sort(key=lambda p: p["case"])
    result = {"status": "pilot: one rater, localisation not validated", "head_id": res["head_id"],
              "head_sha256": res["head_sha256"], "calls_sha256": res["calls_sha256"],
              "tile_um": float(next(iter(keys.values()))["tile_um"]), "all": summarise(rows, a.seed),
              "by_category": {c: summarise([r for r in rows if r["category"] == c], a.seed)
                              for c in sorted({r["category"] for r in rows})}}
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    page = {"mpp": data["mpp"], "blockUm": data["blockUm"], "cases": pages}
    (out / "reveal.html").write_text(PAGE.replace("__DATA__", json.dumps(page, separators=(",", ":"))))
    with (out / "location-slides.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "location-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
