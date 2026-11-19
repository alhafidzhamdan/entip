"""Pick candidate tiles from glioma slides for a pathologist to label as necrosis, MVP, neither or not assessable.

    python make_labels.py --manifest trial.csv --bags bags/ --slides slides/ --out labels-r1 \
        [--model model/] [--head necrosis.joblib --head mvp.joblib] [--hints hints.csv] [--exclude r0/key.csv] \
        [--per-group 20] [--attended 6] [--diverse 8] [--uncertain 0] [--likely 0] [--random 4] [--round r1]

The labels are our own work on openly licensed slides (TCGA, CPTAC: CC BY), so a step 3 head trained on them can
be used commercially. Never pre-select tiles with a model trained on non-commercial labels (Ivy GAP, BraTS-Path).

Slides: one per patient, ``--per-group`` patients from each glioma group in the manifest (``label == glioma``;
``--groups`` to choose). ``--hints`` (CSV ``patient,necrosis,mvp`` with ``present`` / ``absent`` / blank, e.g. from
the TCGA pathology reports) puts patients with a hint first, so both present and absent cases are labelled.

Tiles per slide, without repeats, in this order:

* ``--attended``: the triage model's most attended tiles (``--model``); in a glioblastoma these are often necrosis
  or vessels;
* ``--diverse``: one tile from each of k clusters of the slide's embeddings, so the slide's range of tissue is
  covered without any model;
* ``--uncertain`` / ``--likely`` (round 2 on): per step 3 head (``--head``), the tiles nearest 0.5 and the tiles
  with the highest probability, which are the most useful to label next and check the head's positives;
* ``--random``: random tissue tiles, which keep the label set honest about how rare the features are.

``--exclude`` takes key files from earlier rounds so no tile is labelled twice. The page does not show why a tile
was picked, the diagnosis or any model output.

Writes, in ``--out``: ``labels.html`` (the labelling page; it reads ``tiles/``), ``tiles/<code>.jpg`` (a 3 x 3 tile
field with the tile outlined), and ``key.csv`` (where each tile is, and why it was picked). Turn the labels into
step 3 training files with ``export_labels.py``.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attention_review"))
from make_review import crop, find_slide  # noqa: E402  (same tile field as the attention review)

CHOICES = [
    ("n", "Necrosis", "tumour necrosis in the outlined tile, palisading or not (not haemorrhage, crush or "
                      "cautery artefact)"),
    ("m", "MVP", "microvascular proliferation: multilayered vessel walls of proliferating endothelial cells and "
                 "pericytes, glomeruloid tufts or garlands (not just many or thick-walled vessels)"),
    ("0", "Neither", "assessable tissue with neither feature"),
    ("x", "Not assessable", "background, blood, crush, folds, ink, out of focus or too little tissue"),
]


def read_hints(path) -> dict:
    if not path:
        return {}
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        out[r["patient"]] = {f: (r.get(f) or "").strip().lower() for f in ("necrosis", "mvp")}
    return out


def pick_patients(rows: list, groups, per_group: int, hints: dict, seed: int) -> list:
    """One slide row per patient: ``per_group`` patients from each group, hinted patients first."""
    rng = random.Random(seed)
    by = defaultdict(dict)
    for r in rows:
        if r["group"] in groups:
            by[r["group"]].setdefault(r["PatientID"], []).append(r)
    picked = []
    for g in sorted(by):
        pats = sorted(by[g])
        rng.shuffle(pats)
        pats.sort(key=lambda p: p not in hints)          # stable: hinted first, random within each part
        for p in pats[:per_group]:
            picked.append(rng.choice(sorted(by[g][p], key=lambda r: r["SeriesInstanceUID"])))
    return picked


def diverse(features: np.ndarray, k: int, seed: int) -> list:
    """The tile nearest each of k cluster centres of the slide's embeddings."""
    from sklearn.cluster import KMeans
    if k <= 0 or len(features) == 0:
        return []
    k = min(k, len(features))
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(features), size=min(4000, len(features)), replace=False)
    km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(features[sub])
    d = ((features[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1) if len(features) <= 20000 else None
    if d is None:                                         # large slide: nearest within the subsample
        d = np.full((len(features), k), np.inf)
        d[sub] = ((features[sub][:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    return [int(i) for i in np.argmin(d, axis=0)]


def choose_tiles(bag, a, model, heads, rng, seen: set) -> list:
    """(tile index, source) pairs for one slide, without repeats or tiles labelled in an earlier round."""
    out, used = [], {i for i, (x, y) in enumerate(bag.coords_um) if (round(float(x), 2), round(float(y), 2)) in seen}

    def add(idx, source, n):
        k = 0
        for i in idx:
            if k >= n:
                break
            i = int(i)
            if i not in used:
                used.add(i)
                out.append((i, source))
                k += 1

    if model is not None and a.attended:
        add(np.argsort(-np.asarray(model.predict(bag)["attention"])), "attended", a.attended)
    add(diverse(bag.features, a.diverse, a.seed), "diverse", a.diverse)
    for h in heads:
        p = h.tile_p(bag)
        add(np.argsort(np.abs(p - 0.5)), f"uncertain:{h.cfg.feature}", a.uncertain)
        add(np.argsort(-p), f"likely:{h.cfg.feature}", a.likely)
    add(rng.permutation(len(bag)), "random", a.random)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="trial.csv: PatientID, SeriesInstanceUID, label, group")
    ap.add_argument("--bags", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="triage model folder, for --attended")
    ap.add_argument("--head", action="append", default=[], help="step 3 head from an earlier round (repeat)")
    ap.add_argument("--hints", help="CSV patient,necrosis,mvp (present / absent / blank)")
    ap.add_argument("--exclude", action="append", default=[], help="key.csv of an earlier round (repeat)")
    ap.add_argument("--groups", nargs="*", help="manifest groups to sample (default: every glioma group)")
    ap.add_argument("--per-group", type=int, default=20, help="patients per group, one slide each")
    ap.add_argument("--attended", type=int, default=6)
    ap.add_argument("--diverse", type=int, default=8)
    ap.add_argument("--uncertain", type=int, default=0, help="per head, tiles nearest 0.5")
    ap.add_argument("--likely", type=int, default=0, help="per head, tiles with the highest probability")
    ap.add_argument("--random", type=int, default=4)
    ap.add_argument("--round", default="r1", help="prefix for tile codes, unique per round")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide

    rows = list(csv.DictReader(open(a.manifest, newline="")))
    groups = set(a.groups or {r["group"] for r in rows if r.get("label") == "glioma"})
    if not groups:
        raise SystemExit("no glioma groups in the manifest; name them with --groups")
    hints = read_hints(a.hints)
    model = None
    if a.model and a.attended:
        from entip_glioma.triage.model import TriageModel
        model = TriageModel.load(a.model)
    heads = []
    if a.head:
        from entip_glioma.highgrade.regions import RegionDetector
        heads = [RegionDetector.load(p) for p in a.head]
    seen = defaultdict(set)
    for k in a.exclude:
        for r in csv.DictReader(open(k, newline="")):
            seen[r["slide"]].add((round(float(r["x_um"]), 2), round(float(r["y_um"]), 2)))

    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    picked = pick_patients(rows, groups, a.per_group, hints, a.seed)
    random.Random(a.seed).shuffle(picked)               # slide order does not follow group or hint
    key, slides, n = [], [], 0
    for si, r in enumerate(picked, start=1):
        uid = r["SeriesInstanceUID"]
        bp = Path(a.bags) / f"{uid}.npz"
        if not bp.exists():
            print(f"skip {r['PatientID']}: no bag {bp.name}", file=sys.stderr)
            continue
        bag = Bag.load(bp)
        if len(bag) == 0:
            continue
        if heads and any(h.encoder_version != bag.encoder for h in heads):
            raise SystemExit(f"a head was trained on another encoder than {bag.encoder}")
        tiles = choose_tiles(bag, a, model, heads, rng, seen[uid])
        random.Random(f"{a.seed}-{uid}").shuffle(tiles)
        rd = open_slide(str(find_slide(Path(a.slides), uid)))
        code_s = f"S{si:03d}"
        codes = []
        h = hints.get(r["PatientID"], {})
        for i, source in tiles:
            n += 1
            code = f"{a.round}-T{n:05d}"
            x, y = (float(v) for v in bag.coords_um[i])
            crop(rd, x, y, bag.tile_um).save(out / "tiles" / f"{code}.jpg", quality=90)
            codes.append(code)
            key.append({"tile_code": code, "round": a.round, "slide_code": code_s, "patient": r["PatientID"],
                        "slide": uid, "label": r.get("label", ""), "group": r["group"],
                        "diagnosis": r.get("diagnosis", ""), "hint_necrosis": h.get("necrosis", ""),
                        "hint_mvp": h.get("mvp", ""), "source": source, "tile_index": i, "x_um": x, "y_um": y,
                        "tile_um": bag.tile_um, "encoder": bag.encoder})
        slides.append({"slide": code_s, "tiles": codes})
        print(f"{code_s}  {r['PatientID']:<16} {r['group']:<22} {len(codes)} tiles", file=sys.stderr)
    if not key:
        raise SystemExit("no tiles: check --bags and the manifest")

    with (out / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    (out / "labels.html").write_text(page(f"entip-highgrade-{a.round}-{a.seed}-{len(key)}", slides))
    src = defaultdict(int)
    for k in key:
        src[k["source"]] += 1
    print(json.dumps({"slides": len(slides), "tiles": len(key), "by_source": dict(src),
                      "groups": sorted(groups), "out": str(out)}, indent=2))
    return 0


def page(study: str, slides: list) -> str:
    """The labelling page: plain HTML and JavaScript, no network; labels kept in localStorage.

    Click a tile (or move with j / k) and press n (necrosis), m (MVP), 0 (neither) or x (not assessable);
    n and m can both be set. Labels download as labels-<initials>.csv."""
    btns = "".join(f'<button type="button" data-v="{c}" title="{html.escape(t)}">{html.escape(n)} '
                   f'<kbd>{c}</kbd></button>' for c, n, t in CHOICES)
    legend = "".join(f"<li><kbd>{c}</kbd> <b>{html.escape(n)}</b>: {html.escape(t)}</li>" for c, n, t in CHOICES)
    blocks = []
    for s in slides:
        tiles = "".join(f'<figure data-tile="{t}" tabindex="0"><img loading="lazy" src="tiles/{t}.jpg" '
                        f'alt="tile {t}"><figcaption><span class="code">{t}</span><div class="btns">{btns}</div>'
                        f'<input class="note" placeholder="comment (optional)"></figcaption></figure>'
                        for t in s["tiles"])
        blocks.append(f'<section><h2>{s["slide"]}</h2><div class="grid">{tiles}</div></section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>High-grade tile labels</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7;--card:#fff;--accent:#185fa5;--on:#185fa5;--ontext:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441;--card:#2c2c2a;--accent:#85b7eb;--on:#85b7eb;--ontext:#1d1d1b}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:1}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}} h2{{font-size:16px;font-weight:500;margin:24px 0 8px}}
.code{{font-family:ui-monospace,monospace;color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border:2px solid var(--line);border-radius:8px;padding:8px;outline:none}}
figure.done{{border-color:var(--muted)}} figure.cur{{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 30%,transparent)}}
img{{width:100%;height:auto;display:block;border-radius:4px}}
.btns{{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}}
button{{font:inherit;font-size:13px;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
button.on{{background:var(--on);color:var(--ontext);border-color:var(--on)}}
kbd{{font-family:ui-monospace,monospace;font-size:11px;opacity:.7}}
.note{{width:100%;box-sizing:border-box;margin-top:4px;font:inherit;font-size:13px}}
ul{{margin:4px 0 0;padding-left:20px;color:var(--muted);font-size:13px}}
summary{{cursor:pointer;color:var(--muted);font-size:13px;margin-top:4px}}
</style></head><body>
<header><h1>High-grade tile labels</h1>
<label>Labeller initials <input id="rater" size="6"></label>
<span id="progress"></span> <button id="dl" type="button">Download labels</button>
<details open><summary>Definitions and keys</summary><ul>{legend}<li>Label the <b>outlined tile</b> in the
centre (112 µm); a feature counts if any part of it is inside the outline. The surrounding field is context.
<kbd>j</kbd> / <kbd>k</kbd> move to the next / previous tile.</li></ul></details>
</header>
{''.join(blocks)}
<script>
const KEY = {json.dumps(study)};
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ st = {{}}; }}
st.labels = st.labels || {{}}; st.notes = st.notes || {{}};
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} }};
const rater = document.getElementById("rater");
rater.value = st.rater || ""; rater.oninput = () => {{ st.rater = rater.value.trim(); save(); }};
const figs = [...document.querySelectorAll("figure")];
let cur = 0;
function set(f, v) {{
  const t = f.dataset.tile; let s = st.labels[t] || "";
  if (v === "n" || v === "m") {{ s = s.replace(/[0x]/g, ""); s = s.includes(v) ? s.replace(v, "") : s + v;
    s = [...s].sort().reverse().join(""); }}         // "nm" order
  else s = s === v ? "" : v;
  if (s) st.labels[t] = s; else delete st.labels[t];
  save(); draw(f);
}}
function draw(f) {{
  const s = st.labels[f.dataset.tile] || "";
  f.querySelectorAll("button").forEach(b => b.classList.toggle("on", s.includes(b.dataset.v)));
  f.classList.toggle("done", !!s);
  const n = figs.filter(g => st.labels[g.dataset.tile]).length;
  document.getElementById("progress").textContent = `${{n}} of ${{figs.length}} labelled`;
}}
function focus(i) {{
  figs[cur].classList.remove("cur"); cur = Math.max(0, Math.min(figs.length - 1, i));
  figs[cur].classList.add("cur"); figs[cur].scrollIntoView({{block: "nearest"}});
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
  else if ("nm0x".includes(k) && k.length === 1) {{ set(figs[cur], k); if (k === "0" || k === "x") focus(cur + 1); }}
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


if __name__ == "__main__":
    sys.exit(main())
