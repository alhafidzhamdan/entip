"""Step 3a: a blinded slide-level pilot of a frozen MVP head on reserved patients.

    python slide_pilot.py build --head head.joblib --reserved mvp-final-test/patients.csv --bags bags/ \
        --slides slides/ --out mvp-pilot1
    python slide_pilot.py compare --key mvp-pilot1/private/key.csv --calls calls-AH.csv --out mvp-pilot1/results

**build** freezes the head, gives each slide a random case code, and scores it: every tile, joined into foci, with the
strongest focus as the slide score. Scores go only to ``private/key.csv``. ``review-pack.tar`` holds ``review.html`` and
image tiles, with no slide name, patient, grade, metadata, label image or model output.

**The review.** Pan and zoom each whole slide; call MVP present, absent or can't tell; shift-click one example if
present. Calls stay in the browser and download as a CSV.

**compare** runs only when every case has a call. It reports how the score ranks present above absent slides, and
sensitivity and specificity at cutoffs fixed in advance (``CUTOFFS``). This is a pilot: the cutoffs are not calibrated,
and a clinical threshold needs separate calibration slides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
from common import find_slide  # noqa: E402

CUTOFFS = (0.3, 0.5, 0.7)
CALLS = ("present", "absent", "cant")


def block_keys(coords: np.ndarray, tile_um: float, ox: float, oy: float, size: float) -> set:
    """(column, row) of every block of ``size`` µm that any tile touches."""
    x0 = np.floor((coords[:, 0] - ox) / size).astype(int)
    y0 = np.floor((coords[:, 1] - oy) / size).astype(int)
    x1 = np.floor((coords[:, 0] + tile_um - ox - 1e-6) / size).astype(int)
    y1 = np.floor((coords[:, 1] + tile_um - oy - 1e-6) / size).astype(int)
    return {(c, r) for a, b, d, e in {(int(p), int(q), int(s), int(u)) for p, q, s, u in zip(x0, y0, x1, y1)}
            for c in range(a, d + 1) for r in range(b, e + 1)}


def pack_case(case: str, row: dict, head: str, bags: str, slides: str, tiles: str, mpp: float, block_px: int) -> tuple:
    """Writes one slide's image pyramid and scores it. Returns what the page may see, and what stays private."""
    from PIL import Image
    from entip_glioma.highgrade.regions import RegionDetector
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    bag = Bag.load(Path(bags) / f"{row['slide']}.npz")
    if len(bag) == 0:
        raise SystemExit(f"{case}: the slide has no tissue tiles")
    raw, foci, _, n_tiles, ood = RegionDetector.load(head)._score([bag])
    rd = open_slide(str(find_slide(Path(slides), row["slide"])))
    t, block_um = bag.tile_um, block_px * mpp
    ox, oy = (float(v) for v in bag.coords_um.min(0))
    w, h = (float(v) for v in bag.coords_um.max(0) + t - [ox, oy])
    levels = 1
    while block_um * 2 ** (levels - 1) < max(w, h):
        levels += 1
    blocks = {}
    for lv in range(levels):
        size = block_um * 2 ** lv
        keys = sorted(block_keys(bag.coords_um, t, ox, oy, size))
        folder = Path(tiles) / case / f"L{lv}"
        folder.mkdir(parents=True, exist_ok=True)
        for c, r in keys:
            im = rd.read_um(ox + (c + 0.5) * size, oy + (r + 0.5) * size, size, block_px)
            Image.fromarray(im).save(folder / f"{c}_{r}.jpg", quality=82)
        blocks[lv] = " ".join(f"{c}_{r}" for c, r in keys)
    public = {"case": case, "w": w, "h": h, "levels": levels, "blocks": blocks}
    private = {"case": case, "patient": row["patient"], "slide": row["slide"], "category": row["category"],
               "diagnosis": row.get("diagnosis", ""), "origin_x_um": ox, "origin_y_um": oy, "tile_um": t,
               "native_mpp": float(rd.mpp), "n_tiles": n_tiles, "raw_score": round(float(raw), 6), "n_foci": len(foci),
               "ood_fraction": round(float(ood), 4),
               "top_foci": json.dumps([{k: round(float(f[k]), 2) for k in ("x_um", "y_um", "w_um", "h_um", "weight")}
                                       for f in foci[:10]])}
    return public, private


VIEWER = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MVP slide review</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7;--card:#fff;--on:#185fa5;--ontext:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441;--card:#2c2c2a;--on:#85b7eb;--ontext:#1d1d1b}}
html,body{height:100%;margin:0}body{display:flex;flex-direction:column;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line)}
h1{font-size:16px;font-weight:500;margin:0}button,input{font:inherit;font-size:13px;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
button{cursor:pointer}button.on{background:var(--on);color:var(--ontext);border-color:var(--on)}kbd{font-size:11px;opacity:.7}
#hint,#status{color:var(--muted);font-size:12px}#hint{padding:4px 12px;border-bottom:1px solid var(--line)}
#wrap{flex:1;min-height:0;position:relative;background:#f4f2ee}canvas{position:absolute;inset:0;width:100%;height:100%;cursor:grab;touch-action:none}
</style></head><body>
<header><h1>MVP slide review</h1><label>Initials <input id="rater" size="5"></label>
<button id="back">Back <kbd>b</kbd></button><strong id="where"></strong><button id="next">Next <kbd>n</kbd></button>
<button class="call" data-v="present">MVP present <kbd>p</kbd></button><button class="call" data-v="absent">Absent <kbd>a</kbd></button>
<button class="call" data-v="cant">Can't tell <kbd>c</kbd></button><button id="fit">Fit <kbd>f</kbd></button>
<input id="note" placeholder="comment (optional)"><span id="status"></span><button id="dl">Download calls</button></header>
<div id="hint">Search the whole slide. Drag to pan and scroll to zoom. If MVP is present, shift-click one clear example. Use the working definition: at least two layers of plump proliferating wall cells around a lumen, or a glomeruloid tuft.</div>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id), cv = $("c"), ctx = cv.getContext("2d");
let st = {}; try { st = JSON.parse(localStorage.getItem(D.study) || "{}"); } catch (e) { st = {}; }
st.calls = st.calls || {}; st.i = Math.min(st.i || 0, D.cases.length - 1);
const save = () => { try { localStorage.setItem(D.study, JSON.stringify(st)); } catch (e) {} };
const cache = new Map(); let v = {x: 0, y: 0, s: 1}, sets = [], since = performance.now(), queued = false;
const cur = () => D.cases[st.i], rec = () => st.calls[cur().case] || (st.calls[cur().case] = {});
function tick() { const n = performance.now(); rec().seconds = (rec().seconds || 0) + (n - since) / 1000; since = n; save(); }
function redraw() { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw(); }); } }
function size() { const d = devicePixelRatio || 1; cv.width = cv.clientWidth * d; cv.height = cv.clientHeight * d; ctx.setTransform(d, 0, 0, d, 0, 0); redraw(); }
function fit() { const c = cur(), w = cv.clientWidth, h = cv.clientHeight; v.s = .96 * Math.min(w / c.w, h / c.h); v.x = (c.w - w / v.s) / 2; v.y = (c.h - h / v.s) / 2; redraw(); }
function image(cs, l, k) { const u = `tiles/${cs}/L${l}/${k}.jpg`; let m = cache.get(u);
  if (!m) { if (cache.size > 900) cache.delete(cache.keys().next().value); m = new Image(); m.onload = redraw; m.src = u; cache.set(u, m); } return m; }
function layer(c, l, load) { const size = D.blockUm * 2 ** l, w = cv.clientWidth / v.s, h = cv.clientHeight / v.s;
  for (let col = Math.max(0, Math.floor(v.x / size)); col <= Math.floor((v.x + w) / size); col++)
    for (let row = Math.max(0, Math.floor(v.y / size)); row <= Math.floor((v.y + h) / size); row++) {
      const k = `${col}_${row}`; if (!sets[l].has(k)) continue;
      const u = `tiles/${c.case}/L${l}/${k}.jpg`, m = load ? image(c.case, l, k) : cache.get(u);
      if (m && m.complete && m.naturalWidth) ctx.drawImage(m, (col * size - v.x) * v.s, (row * size - v.y) * v.s, size * v.s + .5, size * v.s + .5); } }
function draw() { const c = cur(); ctx.fillStyle = "#f4f2ee"; ctx.fillRect(0, 0, cv.clientWidth, cv.clientHeight);
  const want = Math.max(0, Math.min(c.levels - 1, Math.floor(Math.log2(1 / (v.s * D.mpp)))));
  layer(c, c.levels - 1, true); for (let l = c.levels - 2; l > want; l--) layer(c, l, false); if (want < c.levels - 1) layer(c, want, true);
  const r = st.calls[c.case]; if (r && r.x != null) { const x = (r.x - v.x) * v.s, y = (r.y - v.y) * v.s;
    ctx.lineWidth = 4; ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.arc(x, y, 15, 0, 7); ctx.stroke(); ctx.lineWidth = 2; ctx.strokeStyle = "#0b4f9e"; ctx.beginPath(); ctx.arc(x, y, 15, 0, 7); ctx.stroke(); } }
function ui() { const r = rec(), done = D.cases.filter(c => (st.calls[c.case] || {}).call).length;
  $("where").textContent = `Slide ${st.i + 1} of ${D.cases.length}`; $("note").value = r.comment || "";
  document.querySelectorAll(".call").forEach(b => b.classList.toggle("on", r.call === b.dataset.v));
  $("status").textContent = `${done} of ${D.cases.length} called` + (r.call === "present" && r.x == null ? " | shift-click an example" : ""); }
function show(i) { tick(); st.i = Math.max(0, Math.min(D.cases.length - 1, i)); save(); sets = []; const c = cur();
  for (let l = 0; l < c.levels; l++) sets[l] = new Set(c.blocks[l].split(" ")); ui(); fit(); }
function call(x) { const r = rec(); r.call = r.call === x ? undefined : x; if (r.call !== "present") { delete r.x; delete r.y; } save(); ui(); redraw(); }
document.querySelectorAll(".call").forEach(b => b.onclick = () => call(b.dataset.v));
$("back").onclick = () => show(st.i - 1); $("next").onclick = () => show(st.i + 1); $("fit").onclick = fit;
$("rater").value = st.rater || ""; $("rater").oninput = e => { st.rater = e.target.value.trim(); save(); };
$("note").oninput = e => { rec().comment = e.target.value; save(); };
let drag = null;
cv.onpointerdown = e => { cv.setPointerCapture(e.pointerId); drag = {x: e.offsetX, y: e.offsetY, vx: v.x, vy: v.y, moved: false, shift: e.shiftKey}; };
cv.onpointermove = e => { if (!drag) return; const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  if (drag.moved && !drag.shift) { v.x = drag.vx - dx / v.s; v.y = drag.vy - dy / v.s; redraw(); } };
cv.onpointerup = e => { if (drag && drag.shift && !drag.moved) { const r = rec(); r.call = "present"; r.x = v.x + e.offsetX / v.s; r.y = v.y + e.offsetY / v.s; save(); ui(); redraw(); } drag = null; };
function zoom(f, px, py) { const s = Math.max(.002, Math.min(8 / D.mpp, v.s * f)), x = v.x + px / v.s, y = v.y + py / v.s; v.s = s; v.x = x - px / s; v.y = y - py / s; redraw(); }
cv.addEventListener("wheel", e => { e.preventDefault(); zoom(Math.exp(-e.deltaY * .0016), e.offsetX, e.offsetY); }, {passive: false});
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return; const k = e.key.toLowerCase(), m = 90 / v.s;
  if (k === "p") call("present"); else if (k === "a") call("absent"); else if (k === "c") call("cant"); else if (k === "n") show(st.i + 1); else if (k === "b") show(st.i - 1); else if (k === "f") fit();
  else if (k === "+" || k === "=") zoom(1.4, cv.clientWidth / 2, cv.clientHeight / 2); else if (k === "-") zoom(1 / 1.4, cv.clientWidth / 2, cv.clientHeight / 2);
  else if (k === "arrowleft") { v.x -= m; redraw(); } else if (k === "arrowright") { v.x += m; redraw(); } else if (k === "arrowup") { v.y -= m; redraw(); } else if (k === "arrowdown") { v.y += m; redraw(); } });
$("dl").onclick = () => { tick(); if (!st.rater) { alert("Enter your initials first"); $("rater").focus(); return; }
  const q = s => '"' + String(s == null ? "" : s).replace(/"/g, '""') + '"', rows = [["rater", "case", "call", "example_x_um", "example_y_um", "seconds", "comment"]];
  D.cases.forEach(c => { const r = st.calls[c.case] || {}; rows.push([st.rater, c.case, r.call || "", r.x == null ? "" : r.x.toFixed(1), r.y == null ? "" : r.y.toFixed(1), r.seconds ? r.seconds.toFixed(1) : "", r.comment || ""]); });
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.map(r => r.map(q).join(",")).join("\n") + "\n"], {type: "text/csv"})); a.download = `slide-calls-${st.rater}.csv`; a.click(); };
addEventListener("resize", size); size(); sets = []; for (let l = 0; l < cur().levels; l++) sets[l] = new Set(cur().blocks[l].split(" ")); ui(); fit();
</script></body></html>
"""


def build(a) -> int:
    out = Path(a.out)
    private = out / "private"
    if (private / "key.csv").exists():
        raise SystemExit(f"{private / 'key.csv'} already exists: a built pilot is never overwritten")
    rows = list(csv.DictReader(open(a.reserved, newline="")))
    if not rows or len({r["patient"] for r in rows}) != len(rows):
        raise SystemExit("the reserved list must have one slide per patient")
    if a.trained:
        trained = set()
        for p in a.trained:
            trained |= {r["patient"] for r in csv.DictReader(open(p, newline=""))}
        clash = trained & {r["patient"] for r in rows}
        if clash:
            raise SystemExit(f"{len(clash)} reserved patients appear in the head's training data")
    private.mkdir(parents=True, exist_ok=True)
    frozen = private / "frozen-head.joblib"
    shutil.copy2(a.head, frozen)
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    from entip_glioma.highgrade.regions import RegionDetector
    det = RegionDetector.load(frozen)
    if det.cfg.feature != "microvascular_proliferation":
        raise SystemExit(f"{a.head} is a {det.cfg.feature} head")
    random.Random(f"{a.seed}-cases").shuffle(rows)
    pack = out / "review-pack"
    (pack / "tiles").mkdir(parents=True, exist_ok=True)
    public, secret = [], []
    with ProcessPoolExecutor(max(1, a.workers or int(os.environ.get("NSLOTS", 1)))) as pool:
        jobs = [pool.submit(pack_case, f"C{n:03d}", r, str(frozen), a.bags, a.slides, str(pack / "tiles"), a.mpp, a.block_px)
                for n, r in enumerate(rows, start=1)]
        for n, job in enumerate(as_completed(jobs), start=1):
            pub, sec = job.result()
            public.append(pub)
            secret.append({**sec, "head_id": det.model_id, "head_sha256": sha})
            print(f"packed {n}/{len(rows)} slides", file=sys.stderr, flush=True)
    public.sort(key=lambda c: c["case"])
    secret.sort(key=lambda c: c["case"])
    data = {"study": f"entip-mvp-pilot-{a.seed}-{len(public)}-{sha[:8]}", "mpp": a.mpp, "blockUm": a.block_px * a.mpp,
            "cases": public}
    (pack / "review.html").write_text(VIEWER.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    with (private / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(secret[0]))
        w.writeheader()
        w.writerows(secret)
    with tarfile.open(out / "review-pack.tar", "w") as tar:
        tar.add(pack, arcname="review-pack")
    print(json.dumps({"slides": len(public), "pack_gb": round((out / "review-pack.tar").stat().st_size / 1e9, 2),
                      "pack": str(out / "review-pack.tar")}, indent=2))
    return 0


def rank_auc(y, s) -> float | None:
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarise(rows: list, seed: int) -> dict:
    y, s = np.array([int(r["call"] == "present") for r in rows]), np.array([float(r["raw_score"]) for r in rows])
    out = {"slides": len(rows), "present": int(y.sum()), "absent": int((1 - y).sum()), "auroc": rank_auc(y, s),
           "auroc_95ci": None, "cutoffs": {}}
    if out["auroc"] is not None:
        rng, draws = np.random.default_rng(seed), []
        for _ in range(2000):
            i = rng.integers(len(y), size=len(y))
            got = rank_auc(y[i], s[i])
            if got is not None:
                draws.append(got)
        out["auroc"] = round(out["auroc"], 3)
        out["auroc_95ci"] = [round(float(v), 3) for v in np.percentile(draws, [2.5, 97.5])]
    for cut in CUTOFFS:
        tp, fn = int(((s >= cut) & (y == 1)).sum()), int(((s < cut) & (y == 1)).sum())
        tn, fp = int(((s < cut) & (y == 0)).sum()), int(((s >= cut) & (y == 0)).sum())
        out["cutoffs"][str(cut)] = {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
                                    "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
                                    "sensitivity_95ci": wilson(tp, tp + fn),
                                    "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
                                    "specificity_95ci": wilson(tn, tn + fp)}
    return out


def compare(a) -> int:
    out = Path(a.out)
    if (out / "pilot-results.json").exists():
        raise SystemExit(f"{out / 'pilot-results.json'} already exists: the comparison is run once")
    key = {r["case"]: r for r in csv.DictReader(open(a.key, newline=""))}
    calls = {r["case"]: r for r in csv.DictReader(open(a.calls, newline=""))}
    if set(key) != set(calls):
        raise SystemExit("the calls do not cover the same cases as the key")
    missing = [c for c, r in calls.items() if r["call"] not in CALLS]
    if missing:
        raise SystemExit(f"{len(missing)} slides have no call, e.g. {missing[:3]}")
    rows, hits, clicked = [], 0, 0
    for c, k in key.items():
        r = {**k, "call": calls[c]["call"], "example_x_um": calls[c].get("example_x_um", ""),
             "example_y_um": calls[c].get("example_y_um", "")}
        r["example_in_top_focus"] = ""
        if r["call"] == "present" and r["example_x_um"] != "":
            clicked += 1
            x, y = float(k["origin_x_um"]) + float(r["example_x_um"]), float(k["origin_y_um"]) + float(r["example_y_um"])
            t = float(k["tile_um"])
            near = any(f["x_um"] - t <= x <= f["x_um"] + f["w_um"] + t and f["y_um"] - t <= y <= f["y_um"] + f["h_um"] + t
                       for f in json.loads(k["top_foci"]))
            hits += near
            r["example_in_top_focus"] = int(near)
        rows.append(r)
    used = [r for r in rows if r["call"] != "cant"]
    res = {"status": "pilot: not validated, cutoffs not calibrated", "head_id": rows[0]["head_id"],
           "head_sha256": rows[0]["head_sha256"], "cant_tell": len(rows) - len(used), "all": summarise(used, a.seed),
           "by_category": {c: summarise([r for r in used if r["category"] == c], a.seed)
                           for c in sorted({r["category"] for r in used})},
           "clicked_examples_inside_a_top_10_focus": {"hits": hits, "clicked": clicked}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot-results.json").write_text(json.dumps(res, indent=2) + "\n")
    with (out / "slides.csv").open("w", newline="") as f:
        cols = ["case", "category", "call", "raw_score", "n_foci", "ood_fraction", "example_in_top_focus"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["case"]))
    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="freeze the head, score the slides, and write the blinded review pack")
    b.add_argument("--head", required=True)
    b.add_argument("--reserved", required=True, help="patients.csv from reserve_test.py")
    b.add_argument("--trained", action="append", default=[], help="key CSV of the head's training patients (repeat)")
    b.add_argument("--bags", required=True)
    b.add_argument("--slides", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--mpp", type=float, default=0.5, help="µm per pixel of the finest review level")
    b.add_argument("--block-px", type=int, default=896)
    b.add_argument("--workers", type=int)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=build)
    c = sub.add_parser("compare", help="compare locked calls with the frozen head's slide scores")
    c.add_argument("--key", required=True, help="private/key.csv")
    c.add_argument("--calls", required=True, help="slide-calls CSV from the review page")
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=compare)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
