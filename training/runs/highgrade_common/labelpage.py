"""The labelling page every step 3 sub-step uses: plain HTML and JavaScript, no network, labels kept in the
browser (localStorage) and downloaded as ``labels-<initials>.csv`` (``rater, tile_code, label, comment``).

One label per item, from the sub-step's keys. A key sets the label and moves to the next item; the same key again
clears it. ``j`` / ``k`` move between items. The page shows no diagnosis, model output or reason for picking.
"""
from __future__ import annotations

import html
import json


def page(study: str, title: str, slides: list, choices: list, instructions: str, column_px: int | None = None) -> str:
    """``slides``: ``[{"slide": "S001", "items": [{"code": ..., "images": [src, ...]}]}]``;
    ``choices``: ``[(key, name, definition)]``; ``instructions``: HTML shown under the keys;
    ``column_px``: smallest width of an item (default 260 px per image)."""
    btns = "".join(f'<button type="button" data-v="{c}" title="{html.escape(t)}">{html.escape(n)} '
                   f'<kbd>{c}</kbd></button>' for c, n, t in choices)
    legend = "".join(f"<li><kbd>{c}</kbd> <b>{html.escape(n)}</b>: {html.escape(t)}</li>" for c, n, t in choices)
    blocks = []
    for s in slides:
        figs = "".join(
            f'<figure data-tile="{it["code"]}" tabindex="0"><div class="imgs">'
            + "".join(f'<img loading="lazy" src="{src}" alt="item {it["code"]}">' for src in it["images"])
            + f'</div><figcaption><span class="code">{it["code"]}</span><div class="btns">{btns}</div>'
              f'<input class="note" placeholder="comment (optional)"></figcaption></figure>'
            for it in s["items"])
        blocks.append(f'<section><h2>{s["slide"]}</h2><div class="grid">{figs}</div></section>')
    keys = json.dumps("".join(c for c, _, _ in choices))
    cols = column_px or 260 * max(1, max((len(it["images"]) for s in slides for it in s["items"]), default=1))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7;--card:#fff;--accent:#185fa5;--on:#185fa5;--ontext:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441;--card:#2c2c2a;--accent:#85b7eb;--on:#85b7eb;--ontext:#1d1d1b}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:1}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}} h2{{font-size:16px;font-weight:500;margin:24px 0 8px}}
.code{{font-family:ui-monospace,monospace;color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min({cols}px,100%),1fr));gap:12px}}
figure{{margin:0;background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px;outline:none}}
figure.done{{border-color:var(--muted)}} figure.cur{{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 30%,transparent)}}
.imgs{{display:flex;gap:6px}} img{{flex:1;min-width:0;height:auto;display:block;border-radius:4px}}
.btns{{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}}
button{{font:inherit;font-size:13px;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
button.on{{background:var(--on);color:var(--ontext);border-color:var(--on)}}
kbd{{font-family:ui-monospace,monospace;font-size:11px;opacity:.7}}
.note{{width:100%;box-sizing:border-box;margin-top:4px;font:inherit;font-size:13px}}
ul{{margin:4px 0 0;padding-left:20px;color:var(--muted);font-size:13px}}
summary{{cursor:pointer;color:var(--muted);font-size:13px;margin-top:4px}}
</style></head><body>
<header><h1>{html.escape(title)}</h1>
<label>Labeller initials <input id="rater" size="6"></label>
<span id="progress"></span> <button id="dl" type="button">Download labels</button>
<details{" open" if column_px is None else ""}><summary>Definitions and keys</summary><ul>{legend}<li>{instructions}
<kbd>j</kbd> / <kbd>k</kbd> move to the next / previous item.</li></ul></details>
</header>
{''.join(blocks)}
<script>
const KEY = {json.dumps(study)}, KEYS = {keys};
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ st = {{}}; }}
st.labels = st.labels || {{}}; st.notes = st.notes || {{}};
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} }};
const rater = document.getElementById("rater");
rater.value = st.rater || ""; rater.oninput = () => {{ st.rater = rater.value.trim(); save(); }};
const figs = [...document.querySelectorAll("figure")];
let cur = 0;
function set(f, v) {{
  const t = f.dataset.tile;
  if (st.labels[t] === v) delete st.labels[t]; else st.labels[t] = v;
  save(); draw(f);
}}
function draw(f) {{
  const s = st.labels[f.dataset.tile] || "";
  f.querySelectorAll("button").forEach(b => b.classList.toggle("on", s === b.dataset.v));
  f.classList.toggle("done", !!s);
  const n = figs.filter(g => st.labels[g.dataset.tile]).length;
  document.getElementById("progress").textContent = `${{n}} of ${{figs.length}} labelled`;
}}
function focus(i) {{
  figs[cur].classList.remove("cur"); cur = Math.max(0, Math.min(figs.length - 1, i));
  figs[cur].classList.add("cur");
  figs[cur].style.scrollMarginTop = (document.querySelector("header").offsetHeight + 8) + "px";
  figs[cur].scrollIntoView({{block: {json.dumps("start" if column_px else "nearest")}}});
}}
figs.forEach((f, i) => {{
  f.querySelectorAll("button").forEach(b => b.onclick = e => {{ e.stopPropagation(); focus(i); set(f, b.dataset.v); }});
  f.onclick = () => focus(i);
  const note = f.querySelector(".note");
  note.value = st.notes[f.dataset.tile] || "";
  note.oninput = () => {{ st.notes[f.dataset.tile] = note.value; save(); }};
  draw(f);
}});
document.addEventListener("keydown", e => {{
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "j") focus(cur + 1);
  else if (k === "k") focus(cur - 1);
  else if (k.length === 1 && KEYS.includes(k)) {{ const was = st.labels[figs[cur].dataset.tile];
    set(figs[cur], k); if (was !== k) focus(cur + 1); }}
}});
document.getElementById("dl").onclick = () => {{
  if (!st.rater) {{ alert("Enter your initials first"); rater.focus(); return; }}
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = [["rater", "tile_code", "label", "comment"]].concat(
    figs.map(f => [st.rater, f.dataset.tile, st.labels[f.dataset.tile] || "", st.notes[f.dataset.tile] || ""]));
  const blob = new Blob([rows.map(r => r.map(q).join(",")).join("\\n") + "\\n"], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `labels-${{st.rater}}.csv`; a.click();
}};
focus(0);
</script></body></html>
"""
