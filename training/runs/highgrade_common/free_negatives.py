"""Steps 3a and 3b: pick the free-negative tiles and check them by eye before any head trains on them.

    python free_negatives.py --manifest trial.csv --bags bags/ --slides slides/ --out free-r1 [--per-slide 200]

Free-negative slides (``astro3``, ``grade2``; see ``common.py``) have no MVP and no necrosis by their grade, so their
tiles are negatives for both heads without labelling. This picks ``--per-slide`` random tiles from each one (never a
held-out patient) and writes ``review.html``: each slide's tiles as a grid of 3 x 3 fields, tile outlined.

On the page, click a tile to flag it: once for MVP, twice for necrosis, three times for both, four to clear. Tick
"Checked" when you have looked at the whole slide. Download ``review-<initials>.csv``.

``export.py --free-negatives free-r1 --free-review review-AH.csv`` then uses, for each feature, the tiles of every
slide someone ticked as checked, unless anyone flagged a tile on that slide for that feature: that slide is left
out of the feature's free negatives and listed in ``summary.json``. Slides nobody checked are not used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FREE_NEGATIVE, crop, find_slide, read_manifest, read_patients, xy  # noqa: E402

FLAGS = ("", "mvp", "necrosis", "both")


def review_page(study: str, slides: list) -> str:
    """``slides``: ``[{"slide": "F001", "tiles": [(code, src), ...]}]``."""
    blocks = []
    for s in slides:
        figs = "".join(f'<figure data-tile="{c}"><img loading="lazy" src="{src}" alt="tile {c}"><span></span></figure>'
                       for c, src in s["tiles"])
        blocks.append(f'<section data-slide="{s["slide"]}"><h2>{s["slide"]} <span class="n">{len(s["tiles"])} tiles'
                      f'</span> <label><input type="checkbox" class="chk"> Checked</label></h2>'
                      f'<div class="grid">{figs}</div></section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Free-negative check</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7;--mvp:#185fa5;--nec:#a32d2d;--both:#7f3c8d}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441;--mvp:#85b7eb;--nec:#f09595;--both:#d7a3e0}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:1}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}} h2{{font-size:16px;font-weight:500;margin:24px 0 8px}}
.n{{color:var(--muted);font-size:13px;font-weight:400}} h2 label{{font-size:14px;font-weight:400;margin-left:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px}}
figure{{margin:0;position:relative;border:3px solid transparent;border-radius:6px;cursor:pointer}}
figure img{{width:100%;height:auto;display:block;border-radius:3px}}
figure span{{position:absolute;left:4px;top:4px;font-size:12px;padding:0 6px;border-radius:4px;color:#fff}}
figure.mvp{{border-color:var(--mvp)}} figure.mvp span{{background:var(--mvp)}}
figure.necrosis{{border-color:var(--nec)}} figure.necrosis span{{background:var(--nec)}}
figure.both{{border-color:var(--both)}} figure.both span{{background:var(--both)}}
section.done h2{{color:var(--muted)}}
p{{margin:4px 0;color:var(--muted);font-size:13px}}
button{{font:inherit;font-size:13px;padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}}
</style></head><body>
<header><h1>Free-negative check (steps 3a and 3b)</h1>
<label>Initials <input id="rater" size="6"></label> <span id="progress"></span>
<button id="dl" type="button">Download review</button>
<p>These slides should have no MVP and no necrosis. Each image is a 3 &times; 3 field; the outlined tile is the one
used. Click a tile to flag it: once MVP, twice necrosis, three times both, four to clear. Tick <b>Checked</b> when you
have looked at every tile of a slide; only checked slides are used.</p></header>
{''.join(blocks)}
<script>
const KEY = {json.dumps(study)}, FLAGS = {json.dumps(list(FLAGS))};
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ st = {{}}; }}
st.flags = st.flags || {{}}; st.checked = st.checked || {{}};
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} }};
const rater = document.getElementById("rater");
rater.value = st.rater || ""; rater.oninput = () => {{ st.rater = rater.value.trim(); save(); }};
const secs = [...document.querySelectorAll("section")];
function progress() {{
  const n = secs.filter(s => st.checked[s.dataset.slide]).length;
  const f = Object.values(st.flags).filter(Boolean).length;
  document.getElementById("progress").textContent = `${{n}} of ${{secs.length}} slides checked, ${{f}} tiles flagged`;
}}
function draw(f) {{
  const v = st.flags[f.dataset.tile] || "";
  FLAGS.forEach(c => c && f.classList.toggle(c, v === c));
  f.querySelector("span").textContent = v;
}}
secs.forEach(s => {{
  const chk = s.querySelector(".chk");
  chk.checked = !!st.checked[s.dataset.slide]; s.classList.toggle("done", chk.checked);
  chk.onchange = () => {{ st.checked[s.dataset.slide] = chk.checked; s.classList.toggle("done", chk.checked); save(); progress(); }};
  s.querySelectorAll("figure").forEach(f => {{
    draw(f);
    f.onclick = () => {{ const t = f.dataset.tile;
      st.flags[t] = FLAGS[(FLAGS.indexOf(st.flags[t] || "") + 1) % FLAGS.length];
      if (!st.flags[t]) delete st.flags[t]; save(); draw(f); progress(); }};
  }});
}});
progress();
document.getElementById("dl").onclick = () => {{
  if (!st.rater) {{ alert("Enter your initials first"); rater.focus(); return; }}
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = [["rater", "slide_code", "tile_code", "flag", "slide_checked"]];
  secs.forEach(s => s.querySelectorAll("figure").forEach(f => {{
    const fl = st.flags[f.dataset.tile] || "", c = !!st.checked[s.dataset.slide];
    if (c || fl) rows.push([st.rater, s.dataset.slide, f.dataset.tile, fl, c ? "yes" : "no"]); }}));
  const blob = new Blob([rows.map(r => r.map(q).join(",")).join("\\n") + "\\n"], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `review-${{st.rater}}.csv`; a.click();
}};
</script></body></html>
"""


def read_review(folder, reviews, holdout: set = frozenset()) -> tuple[dict, dict]:
    """Free negatives from a checked ``free_negatives.py`` folder.

    Returns ``labels`` (feature -> slide -> (x, y) -> 0, for ``mvp`` and ``necrosis``), ``tile_um`` (slide -> µm),
    and a summary: slides checked, slides not checked, and per feature the slides left out for a flag (with the
    flagged tile codes, to label properly)."""
    key = {r["tile_code"]: r for r in csv.DictReader(open(Path(folder) / "key.csv", newline=""))}
    checked, flagged = set(), {"mvp": defaultdict(list), "necrosis": defaultdict(list)}
    for p in reviews:
        for r in csv.DictReader(open(p, newline="")):
            k = key.get(r["tile_code"])
            if k is None:
                raise SystemExit(f"{p}: {r['tile_code']} is not in {folder}/key.csv")
            if r.get("slide_checked", "").lower() == "yes":
                checked.add(k["slide"])
            fl = (r.get("flag") or "").strip().lower()
            for feat in ("mvp", "necrosis"):
                if fl in (feat, "both"):
                    flagged[feat][k["slide"]].append(r["tile_code"])
    labels, tile_um = {"mvp": defaultdict(dict), "necrosis": defaultdict(dict)}, {}
    slides = {k["slide"] for k in key.values() if k["patient"] not in holdout}
    for k in key.values():
        s = k["slide"]
        if s not in checked or s not in slides:
            continue
        tile_um[s] = float(k["tile_um"])
        for feat in labels:
            if s not in flagged[feat]:
                labels[feat][s][xy(k["x_um"], k["y_um"])] = 0
    info = {"slides": len(slides), "checked": len(checked & slides), "not_checked": len(slides - checked),
            "tiles_used": {f: sum(len(v) for v in labels[f].values()) for f in labels},
            "left_out_for_flags": {f: {s: sorted(c) for s, c in sorted(flagged[f].items())} for f in labels}}
    return labels, tile_um, info


def merge_free(labels: dict, tile_um: dict, free: dict, free_um: dict, exclude=()) -> int:
    """Add one feature's free negatives (slide -> (x, y) -> 0) to ``labels`` without overwriting any label, except
    on ``exclude`` slides. Returns the tiles added."""
    n = 0
    for s, tl in free.items():
        if s in exclude:
            continue
        tile_um[s] = free_um[s]
        for p, v in tl.items():
            if p not in labels[s]:
                labels[s][p] = v
                n += 1
    return n


def add_free_args(ap) -> None:
    ap.add_argument("--free-negatives", help="folder written by highgrade_common/free_negatives.py")
    ap.add_argument("--free-review", action="append", default=[], help="review-<initials>.csv from its page (repeat)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="append", default=[], help="CSV of held-out patients: never picked")
    ap.add_argument("--per-slide", type=int, default=200, help="random tiles per free-negative slide")
    ap.add_argument("--max-slides", type=int, help="at most this many slides (one per patient)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    holdout, rng = read_patients(a.holdout), np.random.default_rng(a.seed)
    rows, pats = [], set()
    for r in sorted(read_manifest(a.manifest), key=lambda r: r["SeriesInstanceUID"]):
        if r["category"] in FREE_NEGATIVE and r.get("label", "glioma") == "glioma" and r["PatientID"] not in holdout \
                and r["PatientID"] not in pats and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists():
            rows.append(r)
            pats.add(r["PatientID"])
    rng.shuffle(rows)
    rows = rows[:a.max_slides] if a.max_slides else rows
    key, slides, n = [], [], 0
    for si, r in enumerate(rows, start=1):
        uid = r["SeriesInstanceUID"]
        bag = Bag.load(Path(a.bags) / f"{uid}.npz")
        if len(bag) == 0:
            continue
        idx = sorted(int(i) for i in rng.choice(len(bag), size=min(a.per_slide, len(bag)), replace=False))
        rd = open_slide(str(find_slide(Path(a.slides), uid)))
        code_s, tiles = f"F{si:03d}", []
        for i in idx:
            n += 1
            code = f"free-T{n:05d}"
            x, y = (float(v) for v in bag.coords_um[i])
            crop(rd, x, y, bag.tile_um, 288).save(out / "tiles" / f"{code}.jpg", quality=85)
            tiles.append((code, f"tiles/{code}.jpg"))
            key.append({"tile_code": code, "slide_code": code_s, "patient": r["PatientID"], "slide": uid,
                        "category": r["category"], "tile_index": i, "x_um": x, "y_um": y, "tile_um": bag.tile_um,
                        "encoder": bag.encoder})
        slides.append({"slide": code_s, "tiles": tiles})
        print(f"{code_s}  {r['PatientID']:<16} {r['category']:<7} {len(tiles)} tiles", file=sys.stderr)
    if not key:
        raise SystemExit("no free-negative slides with bags: check --bags and the manifest's diagnosis columns")
    with (out / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    (out / "review.html").write_text(review_page(f"entip-free-{a.seed}-{len(key)}", slides))
    print(json.dumps({"slides": len(slides), "tiles": len(key), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
