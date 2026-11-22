"""A blinded, one-tile-at-a-time review page for pathologists, and the summary of their ratings.

``build`` turns a grade model view folder (``grade_mil.py --view``: ``view-key.csv`` and ``tiles/``) into one
self-contained file, ``tile-review.html``, that any pathologist can open offline in a browser. The tiles are
shuffled across slides and renamed, so the page shows no grade, slide, score, or whether a tile was a top tile or
a random one. The answer key stays in ``tile-review-key.csv``: keep it, and send only the HTML.

``summary`` joins the downloaded ``ratings-<initials>.csv`` files with the key: how often the grade model's top
tiles show MVP or necrosis, against random tiles and against grade 3 slides, and how well the raters agree.

    python rate_tiles.py build --view grade-mil --out grade-mil/review
    python rate_tiles.py summary --key grade-mil/review/tile-review-key.csv ratings-AH.csv ratings-XY.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

RATINGS = [  # value, key, name, definition shown to the rater
    ("mvp", "1", "MVP", "Microvascular proliferation: multilayered, mitotically active vessel walls, often glomeruloid; "
                        "not just plump endothelium or thick-walled, hyalinised or many vessels"),
    ("necrosis", "2", "Necrosis", "Tumour necrosis, palisading or not: eosinophilic granular debris with nuclear dust; "
                                  "not haemorrhage, cautery, crush or fibrin"),
    ("neither", "3", "Neither MVP/necrosis", "No MVP and no necrosis in the outlined tile"),
    ("both", "4", "Both MVP and necrosis", "MVP and necrosis in the outlined tile"),
    ("gemistocytic", "5", "Gemistocytic", "Mainly gemistocytes (plump cells with glassy pink cytoplasm), "
                                          "with no MVP and no necrosis"),
    ("haemorrhage", "6", "Haemorrhage", "Mainly blood or haemorrhage, with no MVP and no necrosis"),
    ("artefact", "7", "Artefact", "Blurry area, bubble, fold, pen, slide edge, crush, cautery or other artefact stops you judging the tile"),
    ("unassessable", "8", "Can't assess", "Too little tissue, or not judgeable for another reason"),
    ("unsure", "9", "Unsure", "You can judge the tile but cannot decide between the options"),
    ("other", "0", "Other / free text", "Something else: type what you see; the box opens when you choose it"),
]
FEATURE = {"mvp": (1, 0), "necrosis": (0, 1), "both": (1, 1), "neither": (0, 0), "gemistocytic": (0, 0), "haemorrhage": (0, 0)}
# The study's working rule, based on the WHO description. It says nothing about grade, to keep raters blinded.
MVP_DEFINITION = """<p class="deflead">Working definition for this review, based on the WHO description.</p>
<h3>Call MVP if the tile shows</h3>
<ul><li>A vessel with <b>at least two layers</b> of plump proliferating wall cells (endothelial cells, pericytes,
smooth muscle cells) around a lumen</li>
<li>A <b>glomeruloid</b> tuft, sometimes in garland-like chains</li></ul>
<h3>Not MVP on its own</h3>
<ul><li>Plump or hypertrophic endothelium in a single layer</li>
<li>Many thin-walled capillaries (&ldquo;chicken-wire&rdquo; pattern)</li>
<li>Thick or hyalinised vessels</li>
<li>Tangentially cut vessels, or adjacent vessels that look layered</li>
<li>Reactive vessels around an infarct, abscess or organising haemorrhage</li></ul>
<p class="deflead">If the field is too coarse to tell, press <kbd>9</kbd> Unsure.</p>"""


# ------------------------------------------------------------------------------------------------ build

def build(a) -> int:
    view = Path(a.view)
    rows = list(csv.DictReader(open(view / "view-key.csv")))
    if not rows:
        raise SystemExit(f"{view / 'view-key.csv'} has no slides")
    items = []
    for r in rows:
        for kind, n_max in (("top", a.top), ("random", a.random)):
            have = int(r[kind])
            for rank in range(1, min(have, n_max if n_max is not None else have) + 1):
                src = view / "tiles" / f"{r['code']}-{kind}{rank}.jpg"
                if not src.exists():
                    raise SystemExit(f"missing tile image {src}")
                items.append({"code": r["code"], "slide": r["slide"], "patient": r["patient"],
                              "category": r["category"], "kind": kind, "rank": rank, "p": r["p"], "src": src})
    random.Random(a.seed).shuffle(items)
    width = len(str(len(items)))
    for i, it in enumerate(items, start=1):
        it["item"] = f"T{i:0{width}d}"
    id_text = "|".join(f"{it['item']}={it['src'].name}" for it in items)
    if a.round > 1:  # a new round has a new page id, so browsers start empty and the summary ignores earlier rounds
        id_text = f"round{a.round}|{id_text}"
    page_id = hashlib.sha1(id_text.encode()).hexdigest()[:12]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tile-review-key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "code", "slide", "patient", "category", "kind", "rank", "p", "image", "page_id"])
        for it in items:
            w.writerow([it["item"], it["code"], it["slide"], it["patient"], it["category"], it["kind"], it["rank"],
                        it["p"], it["src"].name, page_id])
    data = [{"id": it["item"],
             "src": "data:image/jpeg;base64," + base64.b64encode(it["src"].read_bytes()).decode("ascii")}
            for it in items]
    (out / "tile-review.html").write_text(page(a.title, page_id, data, a.field_um, a.endpoint), encoding="utf-8")
    mb = (out / "tile-review.html").stat().st_size / 1e6
    print(json.dumps({"tiles": len(items), "slides": len(rows), "page_id": page_id, "html_mb": round(mb, 1),
                      "ratings_go_to": a.endpoint or "the rater's browser (download as CSV)",
                      "send": str(out / "tile-review.html"), "keep": str(out / "tile-review-key.csv")}, indent=2))
    return 0


def page(title: str, page_id: str, data: list, field_um: float, endpoint: str | None = None) -> str:
    keys = "".join(f'<button type="button" class="rate" data-v="{v}" title="{html.escape(d)}">'
                   f'<kbd>{k}</kbd><span>{html.escape(n)}</span></button>' for v, k, n, d in RATINGS)
    legend = "".join(f"<dt><kbd>{k}</kbd> {html.escape(n)}</dt>"
                     + (f'<dd class="mvpdef">Microvascular proliferation.{MVP_DEFINITION}</dd>' if v == "mvp"
                        else f"<dd>{html.escape(d)}</dd>") for v, k, n, d in RATINGS)
    cfg = json.dumps({"page": page_id, "endpoint": endpoint or "", "ratings": [[v, k, n] for v, k, n, _ in RATINGS]})
    if endpoint:
        keep = ("Each rating is sent to the study team as you make it, so you can close the page at any time and "
                "come back to it in the same browser.")
        privacy = ("The page shows no diagnosis, grade or model output. It sends only your initials and ratings. "
                   "Research use only.")
        thanks = "Thank you. Your ratings have been sent; you can close this page, or go back and change any tile."
    else:
        keep = ("Ratings are kept in this browser as you go. To finish on another computer, download them and load "
                "the file there.")
        privacy = "The page shows no diagnosis, grade or model output, and sends nothing anywhere. Research use only."
        thanks = "Thank you. Download your ratings and send the file back. You can still go back and change any tile."
    return TEMPLATE.replace("%KEEP%", keep).replace("%PRIVACY%", privacy).replace("%THANKS%", thanks) \
        .replace("%TITLE%", html.escape(title)).replace("%KEYS%", keys).replace("%LEGEND%", legend) \
        .replace("%MVPDEF%", MVP_DEFINITION) \
        .replace("%FIELD%", f"{field_um:.0f}").replace("%TILE%", f"{field_um / 3:.0f}") \
        .replace("%CFG%", cfg).replace("%DATA%", json.dumps(data, separators=(",", ":")))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>%TITLE%</title>
<style>
:root{--bg:#f6f5f1;--panel:#fff;--fg:#1d1d1b;--muted:#6b6a64;--line:#dddbd2;--accent:#3b5bdb;--accent-fg:#fff;
--mvp:#c2255c;--necrosis:#e8590c;--both:#7048e8;--neither:#2b8a3e;--gemistocytic:#5c7cfa;--haemorrhage:#c92a2a;--artefact:#9c6b00;--other:#0c8599;--unassessable:#868e96;--unsure:#5f3dc4;--focus:#3b5bdb}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#21211f;--fg:#ecebe6;--muted:#a9a79f;--line:#3a3a37;
--accent:#748ffc;--accent-fg:#111}}
*{box-sizing:border-box}
button,input,textarea{color:inherit}
textarea::placeholder{color:var(--muted);opacity:1}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
button,input{font:inherit;color:inherit}
kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;
padding:0 5px;background:var(--bg)}
.hidden{display:none!important}
/* start */
#start{max-width:760px;margin:0 auto;padding:40px 16px}
#start h1{font-size:26px;font-weight:600;margin:0 0 6px}
#start .lede{color:var(--muted);margin:0 0 24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:0 0 16px}
.card h2{font-size:15px;font-weight:600;margin:0 0 8px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;margin:0}
dt{font-weight:600;white-space:nowrap} dd{margin:0;color:var(--muted)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=text]{border:1px solid var(--line);background:var(--bg);border-radius:8px;padding:8px 10px;width:120px;
text-transform:uppercase}
.btn{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:8px 14px;cursor:pointer}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-fg);font-weight:600}
.btn:disabled{opacity:.5;cursor:default}
.muted{color:var(--muted);font-size:13px}
/* review */
#review{display:grid;grid-template-columns:minmax(0,1fr) 320px;height:100vh}
#stage{display:flex;flex-direction:column;min-width:0;padding:14px 16px}
#top{display:flex;align-items:center;gap:12px;margin-bottom:10px}
#top .who{font-weight:600}
#bar{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
#bar i{display:block;height:100%;background:var(--accent);width:0}
#count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px;white-space:nowrap}
#imgwrap{flex:1;display:flex;align-items:center;justify-content:center;min-height:0;position:relative;overflow:hidden}
#img{width:min(100%,calc(100vh - 80px));height:auto;aspect-ratio:1;object-fit:contain;border-radius:6px;cursor:zoom-in;
box-shadow:0 1px 3px rgba(0,0,0,.15);background:#fff}
#img.zoom{object-fit:none;object-position:center;transform:scale(2.2);cursor:zoom-out}
#side{background:var(--panel);border-left:1px solid var(--line);padding:16px;display:flex;flex-direction:column;
gap:14px;overflow:auto}
#itemid{font:600 13px ui-monospace,Menlo,monospace;color:var(--muted)}
.rates{display:grid;gap:8px}
.rate{display:flex;align-items:center;gap:10px;border:1px solid var(--line);background:var(--bg);border-radius:10px;
padding:10px 12px;cursor:pointer;text-align:left}
.rate:hover{border-color:var(--muted)}
.rate.on{color:#fff;border-color:transparent}
.rate.on kbd{background:rgba(255,255,255,.2);border-color:rgba(255,255,255,.4);color:#fff}
.rate[data-v=mvp].on{background:var(--mvp)} .rate[data-v=necrosis].on{background:var(--necrosis)}
.rate[data-v=both].on{background:var(--both)} .rate[data-v=neither].on{background:var(--neither)}
.rate[data-v=gemistocytic].on{background:var(--gemistocytic)} .rate[data-v=haemorrhage].on{background:var(--haemorrhage)} .rate[data-v=artefact].on{background:var(--artefact)} .rate[data-v=other].on{background:var(--other)}
.rate[data-v=unassessable].on{background:var(--unassessable)} .rate[data-v=unsure].on{background:var(--unsure)}
.warn{color:var(--mvp)!important;font-weight:600}
#note.need{border-color:var(--other);box-shadow:0 0 0 2px var(--other)}
.check{display:flex;align-items:center;gap:8px;cursor:pointer}
textarea{width:100%;min-height:60px;border:1px solid var(--line);background:var(--bg);border-radius:8px;padding:8px;
resize:vertical}
.nav{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.help{font-size:12px;color:var(--muted);line-height:1.7}
.spacer{flex:1}
.mvpdef h3{font-size:13px;font-weight:600;margin:10px 0 3px}
.mvpdef ul{margin:0;padding-left:18px;font-size:13px;line-height:1.5}
.mvpdef .deflead{font-size:12px;color:var(--muted);margin:6px 0 0}
#mvpbox{border:1px solid var(--line);border-radius:10px;padding:8px 12px;background:var(--bg)}
#mvpbox summary{cursor:pointer;font-size:13px;font-weight:600}
/* done */
#done{max-width:620px;margin:0 auto;padding:60px 16px;text-align:center}
#done h1{font-size:24px;font-weight:600}
.tally{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:18px 0 26px}
.tally button{border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:5px 13px;font-size:13px;
cursor:pointer}
.tally button:hover:not(:disabled){border-color:var(--accent)} .tally button:disabled{opacity:.45;cursor:default}
/* check by group */
#qc{padding:16px 20px 40px}
#qchead{position:sticky;top:0;z-index:2;background:var(--bg);padding:4px 0 10px}
#qctabs{justify-content:flex-start;margin:8px 0 0}
#qctabs button.on{background:var(--accent);border-color:var(--accent);color:var(--accent-fg)}
#qcgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.qtile{display:block;padding:0;border:2px solid var(--line);background:var(--panel);border-radius:8px;overflow:hidden;
cursor:pointer;text-align:left}
.qtile:hover,.qtile:focus-visible{border-color:var(--accent)} .qtile.moved{border-color:var(--necrosis)}
.qtile img{display:block;width:100%;aspect-ratio:1}
.qtile span{display:block;padding:3px 8px;font-size:12px;color:var(--muted)} .qtile.moved span{color:var(--fg);font-weight:600}
@media (max-width:820px){#review{grid-template-columns:1fr;height:auto}#imgwrap{min-height:60vh}
#side{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>

<section id="start">
  <h1>%TITLE%</h1>
  <p class="lede">You will see <b id="ntiles"></b> fields from gliomas, one at a time, in random order.
  Judge only the <b>outlined tile in the centre</b>; the ring around it is context.</p>
  <div class="card"><h2>What to call</h2><dl>%LEGEND%</dl>
  <p class="muted" style="margin:12px 0 0">Each field is %FIELD% µm across; the outlined tile is %TILE% µm. Click the image, or press
  <kbd>z</kbd>, to zoom.</p></div>
  <div class="card"><h2>Your initials</h2>
    <div class="row"><input type="text" id="initials" maxlength="6" autocomplete="off" placeholder="e.g. AH">
    <button class="btn primary" id="go" disabled>Start</button>
    <label class="btn" for="import" title="Continue from a ratings CSV saved earlier">Load saved ratings…</label>
    <input type="file" id="import" accept=".csv,text/csv" class="hidden"></div>
    <p class="muted" id="resume" style="margin:10px 0 0">%KEEP%</p></div>
  <p class="muted">%PRIVACY%</p>
</section>

<section id="review" class="hidden">
  <div id="stage">
    <div id="top"><span class="who" id="who"></span><div id="bar"><i></i></div><span id="count"></span></div>
    <div id="imgwrap"><img id="img" alt="tile field"></div>
    <p class="muted" style="margin:8px 0 0;text-align:center">%FIELD% µm field · judge the outlined %TILE% µm tile</p>
  </div>
  <aside id="side">
    <div class="row" style="justify-content:space-between"><span id="itemid"></span>
      <span class="muted" id="state"></span></div>
    <p class="muted hidden" id="sync" style="margin:0"></p>
    <div class="rates">%KEYS%</div>
    <details id="mvpbox" class="mvpdef"><summary>MVP definition <kbd>m</kbd></summary>%MVPDEF%</details>
    <textarea id="note" class="hidden" placeholder="Describe what you see (required)"></textarea>
    <p class="muted hidden" id="needtext" style="margin:-6px 0 0">Type what you see, then save it.</p>
    <button class="btn primary hidden" id="savenote">Save and next (⌘/Ctrl Enter)</button>
    <div class="nav"><button class="btn" id="prev">← Back</button><button class="btn" id="next">Next →</button></div>
    <button class="btn" id="unrated">Next unrated <kbd>u</kbd></button>
    <button class="btn hidden" id="togroup">Back to group <kbd>Esc</kbd></button>
    <div class="spacer"></div>
    <button class="btn primary" id="dl">Download ratings</button>
    <div class="help"><kbd>1</kbd> to <kbd>9</kbd> rate and move on ·
    <kbd>0</kbd> other / free text: type, then <kbd>⌘/Ctrl Enter</kbd> · <kbd>←</kbd> <kbd>→</kbd> move ·
    <kbd>z</kbd> zoom · <kbd>m</kbd> MVP definition · <kbd>Esc</kbd> leave the text box</div>
  </aside>
</section>

<section id="done" class="hidden">
  <h1>All tiles rated</h1>
  <p class="muted">%THANKS%</p>
  <p class="muted" id="sync2"></p>
  <h2 style="font-size:15px;margin:22px 0 0">Check your ratings by group</h2>
  <p class="muted" style="margin:2px 0 0">Choose a group to see all its tiles, and open any tile to keep or change its rating.</p>
  <div class="tally" id="tally"></div>
  <div class="row" style="justify-content:center">
    <button class="btn primary" id="dl2">Download ratings</button><button class="btn" id="back">Review again</button></div>
</section>

<section id="qc" class="hidden">
  <div id="qchead">
    <div class="row"><button class="btn" id="qcdone">← All groups</button><b id="qctitle"></b>
      <span class="muted">Click a tile to check it. Ratings you change are saved straight away.</span></div>
    <div class="tally" id="qctabs"></div>
  </div>
  <div id="qcgrid"></div>
</section>

<script>
const CFG = %CFG%;
const DATA = %DATA%;
const $ = s => document.querySelector(s);
const NAMES = Object.fromEntries(CFG.ratings.map(([v,k,n]) => [v,n]));
const KEYMAP = Object.fromEntries(CFG.ratings.map(([v,k]) => [k,v]));
let rater = "", R = {}, i = 0, shownAt = 0, noteFor = null;  // noteFor: the tile whose text is in the box
let qc = null;  // checking a group: {v, ids (tile positions, fixed when the group opens), pos}

const store = {
  key: () => `tile-review:${CFG.page}:${rater}`,
  load() { try { return JSON.parse(localStorage.getItem(this.key()) || "{}"); } catch (e) { return {}; } },
  save() { try { localStorage.setItem(this.key(), JSON.stringify(R)); } catch (e) {} }
};
const done = r => !!(r && r.rating && (r.rating !== "other" || (r.note || "").trim()));
const nRated = () => DATA.filter(d => done(R[d.id])).length;

$("#ntiles").textContent = DATA.length;
$("#initials").addEventListener("input", e => {
  rater = e.target.value.trim().toUpperCase().replace(/[^A-Z0-9]/g, "");
  $("#go").disabled = !rater;
  if (rater) { const n = Object.values(store.load()).filter(done).length;
    $("#go").textContent = n ? `Continue (${n} rated)` : "Start"; }
});
$("#initials").addEventListener("keydown", e => { if (e.key === "Enter" && rater) start(); });
$("#go").onclick = () => start();

function start() {
  R = Object.assign(store.load(), R);
  $("#start").classList.add("hidden"); $("#done").classList.add("hidden"); $("#review").classList.remove("hidden");
  $("#who").textContent = rater;
  const first = DATA.findIndex(d => !done(R[d.id]));
  show(first < 0 ? 0 : first);
  sync.resendUnconfirmed();
  if (first < 0) finish();
}

function show(k) {
  saveNote();
  i = Math.max(0, Math.min(DATA.length - 1, k));
  const d = DATA[i], r = R[d.id] || {};
  const img = $("#img"); img.classList.remove("zoom"); img.src = d.src;
  const after = DATA[qc ? qc.ids[qc.pos + 1] : i + 1]; if (after) new Image().src = after.src;
  $("#itemid").textContent = d.id;
  document.querySelectorAll(".rate").forEach(b => b.classList.toggle("on", b.dataset.v === r.rating));
  $("#note").value = r.note || ""; noteFor = d.id;
  noteState();
  const n = nRated();
  $("#count").textContent = qc ? `Checking ${NAMES[qc.v]} · ${qc.pos + 1} of ${qc.ids.length}`
    : `${i + 1} of ${DATA.length} · ${n} rated`;
  $("#bar i").style.width = (qc ? 100 * (qc.pos + 1) / qc.ids.length : 100 * n / DATA.length) + "%";
  $("#prev").disabled = qc ? qc.pos === 0 : i === 0; $("#next").disabled = !qc && i === DATA.length - 1;
  $("#unrated").classList.toggle("hidden", !!qc); $("#togroup").classList.toggle("hidden", !qc);
  shownAt = performance.now();
}

function entry() { const d = DATA[i]; return R[d.id] = R[d.id] || {}; }
function saveNote() { if (!DATA[i] || noteFor !== DATA[i].id || $("#review").classList.contains("hidden")) return;
  const v = $("#note").value.trim(), e = entry();
  if (e.rating !== "other") return;
  if (v !== (e.note || "")) { e.note = v; store.save(); sync.push(DATA[i].id); } }

// Online saving: each change joins a queue kept in this browser, sent in order, and kept until it has gone.
const sync = {
  q: [], busy: false, timer: null,
  qkey: () => `tile-review-queue:${CFG.page}`,
  client() { let c = ""; try { c = localStorage.getItem("tile-review-client") || ""; } catch (e) {}
    if (!c) { c = Math.random().toString(36).slice(2, 10); try { localStorage.setItem("tile-review-client", c); } catch (e) {} }
    return c; },
  init() { if (!CFG.endpoint) return; try { this.q = JSON.parse(localStorage.getItem(this.qkey()) || "[]"); } catch (e) {}
    $("#sync").classList.remove("hidden"); this.show(); this.flush(); },
  keep() { try { localStorage.setItem(this.qkey(), JSON.stringify(this.q)); } catch (e) {} },
  push(id, quiet) { if (!CFG.endpoint || !rater) return; const r = R[id] || {};
    this.q = this.q.filter(x => !(x.item === id && x.rater === rater));
    this.q.push({rater, page_id: CFG.page, item: id, rating: r.rating || "", note: r.note || "",
                 seconds: r.seconds ?? "", rated_at: r.at || new Date().toISOString(), client: this.client()});
    if (quiet) return; this.keep(); this.show(); this.flush(); },
  // A rating counts as sent only when the sheet replies that it saved every row in the batch.
  async flush() {
    if (!CFG.endpoint || this.busy || !this.q.length) return;
    this.busy = true; const batch = this.q.slice(0, 100); this.show();
    try {
      const res = await fetch(CFG.endpoint, {method: "POST", headers: {"Content-Type": "text/plain;charset=utf-8"},
                                             body: JSON.stringify({rows: batch})});
      let j = null; try { j = await res.json(); } catch (e) {}
      if (!res.ok || !j || !j.ok || j.saved !== batch.length)
        throw new Error(j && j.error ? j.error : `the sheet did not confirm (${res.status})`);
      batch.forEach(x => { const r = R[x.item]; if (r && x.rater === rater) r.sent = sig(r); });
      if (rater) store.save();
      this.q = this.q.filter(x => !batch.includes(x)); this.keep(); this.error = "";
    } catch (e) {
      this.error = navigator.onLine === false ? "offline" : String(e.message || e);
      clearTimeout(this.timer); this.timer = setTimeout(() => this.flush(), 30000);
    }
    this.busy = false; this.show();
    if (this.q.length && !this.error) this.flush();
  },
  // Every rated tile whose current rating the sheet has not confirmed goes (again) into the queue.
  resendUnconfirmed() { if (!CFG.endpoint || !rater) return;
    DATA.forEach(d => { const r = R[d.id]; if (done(r) && r.sent !== sig(r)) this.push(d.id, true); });
    this.keep(); this.show(); this.flush(); },
  show() { if (!CFG.endpoint) return; const n = this.q.length;
    const sent = DATA.filter(d => R[d.id] && R[d.id].rating && R[d.id].sent === sig(R[d.id])).length;
    const t = !n ? (sent ? `All ${sent} ratings saved to the study sheet.` : "Ratings are saved to the study sheet as you go.")
      : this.error === "offline" ? `Offline: ${n} not sent yet. They go when you are back online; keep this page open.`
      : this.error ? `Not sent yet: ${n} waiting (${this.error}). Retrying every 30 s; keep this page open.`
      : `Sending ${n}…`;
    for (const el of [$("#sync"), $("#sync2")]) { el.textContent = t; el.classList.toggle("warn", !!this.error); } }
};
const sig = r => (r.rating || "") + "\u0001" + (r.note || "");
window.addEventListener("online", () => sync.flush());

function needText(on) {
  $("#note").classList.toggle("need", on); $("#needtext").classList.toggle("hidden", !on);
}
// Typing changes nothing until the note is saved: with the button, by rating, or by leaving the tile.
function noteState() {
  const r = entry(), isOther = r.rating === "other", typed = $("#note").value.trim();
  const dirty = isOther && typed !== (r.note || "");
  $("#note").classList.toggle("hidden", !isOther);
  needText(isOther && !typed);
  $("#savenote").classList.toggle("hidden", !(isOther && typed));
  $("#state").textContent = dirty ? "not saved yet" : done(r) ? "rated" : (isOther ? "needs text" : "");
}
function commitNote() {
  saveNote(); const r = entry();
  if (done(r)) return advance();
  noteState(); $("#note").focus();
}
function rate(v) {
  const e = entry();
  if (qc && v === e.rating && v !== "other") return qcStep(1);  // kept as it was
  if (e.rating === "other" && v !== "other") { e.note = ""; $("#note").value = ""; }  // the text belonged to Other
  e.rating = v; e.at = new Date().toISOString();
  if (!qc || e.seconds == null) e.seconds = Math.round((performance.now() - shownAt) / 100) / 10;  // first-read time
  store.save();
  if (done(e)) sync.push(DATA[i].id);
  document.querySelectorAll(".rate").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  if (v === "other") { noteState(); $("#note").focus(); return; }
  advance();
}
function advance() {
  if (qc) return qcStep(1);
  if (nRated() === DATA.length) { show(i); return finish(); }
  const nxt = DATA.findIndex((d, k) => k > i && !done(R[d.id]));
  show(nxt >= 0 ? nxt : i + 1);
}
function nextUnrated() { let k = DATA.findIndex((d, j) => j > i && !done(R[d.id]));
  if (k < 0) k = DATA.findIndex(d => !done(R[d.id])); if (k >= 0) show(k); else finish(); }

function finish() {
  saveNote(); qc = null;
  $("#review").classList.add("hidden"); $("#qc").classList.add("hidden"); $("#done").classList.remove("hidden");
  const c = {}; DATA.forEach(d => { const v = (R[d.id] || {}).rating; if (v) c[v] = (c[v] || 0) + 1; });
  $("#tally").innerHTML = groupButtons(c);
}

// Check by group: a grid of one rating's tiles, then a walk through them. The list stays fixed while it is open, so a
// tile whose rating changes stays in place, marked with its new rating.
const groupCounts = () => { const c = {}; DATA.forEach(d => { const v = (R[d.id] || {}).rating; if (v) c[v] = (c[v] || 0) + 1; });
  return c; };
const groupButtons = (c, cur) => CFG.ratings.map(([v,,n]) =>
  `<button data-v="${v}" class="${v === cur ? "on" : ""}" ${c[v] ? "" : "disabled"}>${n}: ${c[v] || 0}</button>`).join("");
function openGroup(v) {
  qc = {v, pos: 0, ids: DATA.map((d, k) => (R[d.id] || {}).rating === v ? k : -1).filter(k => k >= 0)};
  qcGrid(); window.scrollTo(0, 0);
}
function qcGrid() {
  saveNote();
  if (!$("#review").classList.contains("hidden") && !done(entry())) { noteState(); $("#note").focus(); return; }
  $("#done").classList.add("hidden"); $("#review").classList.add("hidden"); $("#qc").classList.remove("hidden");
  $("#qctitle").textContent = `${NAMES[qc.v]} (${qc.ids.length})`;
  $("#qctabs").innerHTML = groupButtons(groupCounts(), qc.v);
  const grid = $("#qcgrid"); grid.textContent = "";
  let current = null;
  qc.ids.forEach((k, pos) => {
    const d = DATA[k], now = (R[d.id] || {}).rating, b = document.createElement("button"), im = new Image(),
      cap = document.createElement("span");
    b.className = "qtile" + (now !== qc.v ? " moved" : ""); b.onclick = () => qcOpen(pos);
    im.loading = "lazy"; im.alt = d.id; im.src = d.src;
    cap.textContent = now !== qc.v ? `${d.id} → ${NAMES[now]}` : d.id;
    b.append(im, cap); grid.append(b); if (pos === qc.pos) current = b;
  });
  if (current) current.scrollIntoView({block: "center"});
}
function qcOpen(pos) {
  qc.pos = pos; $("#qc").classList.add("hidden"); $("#review").classList.remove("hidden"); show(qc.ids[pos]);
}
function qcStep(d) {
  saveNote();
  if (!done(entry())) { noteState(); $("#note").focus(); return; }  // Other still needs its text
  const p = qc.pos + d;
  if (p < 0) return;
  if (p >= qc.ids.length) return qcGrid();
  qc.pos = p; show(qc.ids[p]);
}
const move = d => { if (qc) qcStep(d); else show(i + d); };

function csvCell(s) { s = String(s ?? ""); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
function download() {
  saveNote();
  const head = ["rater","page_id","item","rating","note","seconds","rated_at"];
  const lines = [head.join(",")];
  DATA.forEach(d => { const r = R[d.id]; if (!r || (!r.rating && !r.note)) return;
    lines.push([rater, CFG.page, d.id, r.rating || "", r.note || "", r.seconds ?? "", r.at || ""]
      .map(csvCell).join(",")); });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  a.download = `ratings-${rater}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function parseCSV(t) {
  const rows = []; let row = [], f = "", q = false;
  for (let k = 0; k < t.length; k++) { const ch = t[k];
    if (q) { if (ch === '"') { if (t[k + 1] === '"') { f += '"'; k++; } else q = false; } else f += ch; }
    else if (ch === '"') q = true; else if (ch === ",") { row.push(f); f = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && t[k + 1] === "\n") k++; row.push(f); rows.push(row); row = []; f = ""; }
    else f += ch; }
  if (f || row.length) { row.push(f); rows.push(row); }
  return rows;
}
$("#import").addEventListener("change", async e => {
  const file = e.target.files[0]; if (!file) return;
  const rows = parseCSV(await file.text()); const h = rows.shift() || [];
  const col = n => h.indexOf(n);
  if (col("item") < 0 || col("rating") < 0) { alert("This is not a ratings file from this page."); return; }
  const other = rows.find(r => r[col("page_id")] && r[col("page_id")] !== CFG.page);
  if (other) { alert("These ratings belong to a different set of tiles."); return; }
  const ids = new Set(DATA.map(d => d.id)); let n = 0;
  rows.forEach(r => { const id = r[col("item")]; if (!ids.has(id)) return;
    R[id] = {rating: r[col("rating")] || undefined, note: r[col("note")] || "",
             seconds: r[col("seconds")] ? +r[col("seconds")] : undefined, at: r[col("rated_at")] || undefined}; n++;
    if (!rater && r[col("rater")]) rater = r[col("rater")]; });
  if (rater) { $("#initials").value = rater; $("#go").disabled = false; }
  $("#resume").textContent = `Loaded ${n} ratings${rater ? " for " + rater : ""}. Press Continue.`;
  $("#go").textContent = "Continue";
  if (rater) store.save();
});

document.querySelectorAll(".rate").forEach(b => b.onclick = () => rate(b.dataset.v));
$("#prev").onclick = () => move(-1); $("#next").onclick = () => move(1);
$("#togroup").onclick = qcGrid; $("#qcdone").onclick = finish;
for (const el of [$("#tally"), $("#qctabs")]) el.addEventListener("click", e => {
  const b = e.target.closest("button[data-v]"); if (b && !b.disabled) openGroup(b.dataset.v); });
$("#unrated").onclick = nextUnrated;
$("#dl").onclick = download; $("#dl2").onclick = download;
$("#back").onclick = () => { $("#done").classList.add("hidden"); $("#review").classList.remove("hidden"); show(0); };
$("#img").onclick = () => $("#img").classList.toggle("zoom");
$("#note").addEventListener("input", noteState);
$("#savenote").onclick = commitNote;

document.addEventListener("keydown", e => {
  if (!$("#qc").classList.contains("hidden")) { if (e.key === "Escape") finish(); return; }
  if ($("#review").classList.contains("hidden")) return;
  if (e.target.tagName === "TEXTAREA") {
    if (e.key === "Escape") e.target.blur();
    else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); e.target.blur(); commitNote(); }
    return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (KEYMAP[k]) { e.preventDefault(); rate(KEYMAP[k]); }
  else if (k === "z") { e.preventDefault(); $("#img").classList.toggle("zoom"); }
  else if (k === "m") { e.preventDefault(); $("#mvpbox").open = !$("#mvpbox").open; }
  else if (k === "u" && !qc) { e.preventDefault(); nextUnrated(); }
  else if (e.key === "Escape" && qc) { e.preventDefault(); qcGrid(); }
  else if (e.key === "ArrowRight" || k === "j") { e.preventDefault(); move(1); }
  else if (e.key === "ArrowLeft" || k === "k") { e.preventDefault(); move(-1); }
});
window.addEventListener("beforeunload", saveNote);
sync.init();
</script></body></html>
"""


# ---------------------------------------------------------------------------------------------- summary

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def cohen(a: list, b: list):
    """Cohen's kappa for any number of categories."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def read_ratings(files, key: dict) -> dict:
    """``{rater: {item: row}}``; refuses ratings made on a different page."""
    pages = {r["page_id"] for r in key.values()}
    out = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if r.get("page_id") and r["page_id"] not in pages:
                continue  # another review page's ratings (a shared sheet)
            if r["item"] not in key or not r.get("rating"):
                continue
            rater = (r.get("rater") or Path(f).stem.replace("ratings-", "")).upper()
            out.setdefault(rater, {})[r["item"]] = r
    return out


def tally(rows: list) -> dict:
    """Proportions of MVP, necrosis and either among assessable tiles, with 95% Wilson intervals."""
    ok = [r for r in rows if r["rating"] in FEATURE]
    n = len(ok)
    mvp = sum(FEATURE[r["rating"]][0] for r in ok)
    nec = sum(FEATURE[r["rating"]][1] for r in ok)
    either = sum(max(FEATURE[r["rating"]]) for r in ok)
    c = Counter(r["rating"] for r in rows)
    f = lambda k: None if n == 0 else round(k / n, 3)  # noqa: E731
    return {"rated": len(rows), "assessable": n, "mvp": f(mvp), "necrosis": f(nec), "either": f(either),
            "either_95ci": wilson(either, n), "gemistocytic": c["gemistocytic"],
            "haemorrhage": c["haemorrhage"], "artefact": c["artefact"], "other": c["other"],
            "unassessable": c["unassessable"], "unsure": c["unsure"]}


def summary(a) -> int:
    key = {r["item"]: r for r in csv.DictReader(open(a.key))}
    files = list(a.ratings)
    if a.sheet:
        import tempfile
        import urllib.request
        with urllib.request.urlopen(a.sheet, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        if not text.startswith("received_at,"):
            raise SystemExit(f"the sheet URL did not return ratings (got {text[:60]!r}); check ?key=")
        tmp = Path(tempfile.mkdtemp()) / "sheet.csv"
        tmp.write_text(text)
        files.append(tmp)
    if not files:
        raise SystemExit("give ratings files or --sheet")
    raw = read_ratings(files, key)
    if not raw:
        raise SystemExit("no ratings matched the key")
    groups = [("grade4 top", "astro4", "top"), ("grade4 random", "astro4", "random"),
              ("grade3 top", "astro3", "top"), ("grade3 random", "astro3", "random")]
    bands = [("top 1-3", 1, 3), ("top 4-6", 4, 6), ("top 7-12", 7, 12)]
    out = {"tiles": len(key), "raters": {}, "agreement": {}}
    for rater, got in sorted(raw.items()):
        rows = [dict(key[i], **r) for i, r in got.items()]
        res = {"rated": len(rows), "of": len(key)}
        for name, cat, kind in groups:
            res[name] = tally([r for r in rows if r["category"] == cat and r["kind"] == kind])
        for cat in ("astro4", "astro3"):
            for name, lo, hi in bands:
                res[f"{'grade4' if cat == 'astro4' else 'grade3'} {name}"] = tally(
                    [r for r in rows if r["category"] == cat and r["kind"] == "top" and lo <= int(r["rank"]) <= hi])
        # Slide level: does any top tile on the slide show MVP or necrosis?
        per = defaultdict(list)
        for r in rows:
            if r["kind"] == "top" and r["rating"] in FEATURE:
                per[(r["category"], r["code"])].append(max(FEATURE[r["rating"]]))
        for cat, name in (("astro4", "grade4"), ("astro3", "grade3")):
            hits = [any(v) for (c, _), v in per.items() if c == cat]
            res[f"{name} slides with MVP or necrosis in a top tile"] = (
                {"slides": len(hits), "with": sum(hits), "share": round(sum(hits) / len(hits), 3)} if hits else None)
        secs = sorted(float(r["seconds"]) for r in rows if r.get("seconds"))
        res["median_seconds_per_tile"] = secs[len(secs) // 2] if secs else None
        res["texts"] = [{"item": r["item"], "rating": r["rating"], "text": r["note"], "category": r["category"],
                         "kind": r["kind"], "rank": int(r["rank"]), "code": r["code"]}
                        for r in sorted(rows, key=lambda r: r["item"]) if (r.get("note") or "").strip()]
        out["raters"][rater] = res
    for r1, r2 in combinations(sorted(raw), 2):
        both = [i for i in raw[r1] if i in raw[r2]]
        a5 = [raw[r1][i]["rating"] for i in both]
        b5 = [raw[r2][i]["rating"] for i in both]
        ok = [i for i in both if raw[r1][i]["rating"] in FEATURE and raw[r2][i]["rating"] in FEATURE]
        e1 = [max(FEATURE[raw[r1][i]["rating"]]) for i in ok]
        e2 = [max(FEATURE[raw[r2][i]["rating"]]) for i in ok]
        out["agreement"][f"{r1} vs {r2}"] = {
            "tiles_both_rated": len(both), "kappa_all_ratings": cohen(a5, b5),
            "kappa_mvp_or_necrosis": cohen(e1, e2),
            "kappa_mvp": cohen([FEATURE[raw[r1][i]["rating"]][0] for i in ok],
                               [FEATURE[raw[r2][i]["rating"]][0] for i in ok]),
            "kappa_necrosis": cohen([FEATURE[raw[r1][i]["rating"]][1] for i in ok],
                                    [FEATURE[raw[r2][i]["rating"]][1] for i in ok])}
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="make tile-review.html (send) and tile-review-key.csv (keep)")
    b.add_argument("--view", required=True, help="a grade_mil.py --view output folder (view-key.csv, tiles/)")
    b.add_argument("--out", required=True)
    b.add_argument("--top", type=int, default=None, help="top tiles per slide to include (default: all)")
    b.add_argument("--random", type=int, default=None, help="random tiles per slide to include (default: all)")
    b.add_argument("--seed", type=int, default=0, help="shuffle order; the same seed gives the same page")
    b.add_argument("--round", type=int, default=1, help="review round: same tiles and order, every rater starts empty")
    b.add_argument("--title", default="Neuropathology tile expert annotations")
    b.add_argument("--field-um", type=float, default=336.0, help="width of each field (3 tiles of 112 µm)")
    b.add_argument("--endpoint", help="Google Apps Script web app URL (tile_review_sheet.gs): ratings are sent there "
                                      "as they are made")
    b.set_defaults(fn=build)
    s = sub.add_parser("summary", help="join ratings-<initials>.csv files with the key")
    s.add_argument("--key", required=True)
    s.add_argument("ratings", nargs="*", help="ratings-<initials>.csv files, or a CSV downloaded from the Sheet")
    s.add_argument("--sheet", help="the Apps Script URL with ?key=<READ_KEY>: read every rating straight from the Sheet")
    s.add_argument("--out", help="also write the summary JSON here")
    s.set_defaults(fn=summary)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
