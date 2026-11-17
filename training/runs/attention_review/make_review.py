"""Build a blinded attention review: do the tiles a triage model attends to show the diagnostic tissue?

    python make_review.py --report report.json --manifest manifest.csv --bags bags/ --model model/ \
        --slides slides/ --out review-dbta [--per-class 12] [--top 8] [--controls 8] [--seed 0]

For a sample of patients (stratified by class, plus every patient the model got wrong), it takes the slide the
model bases its call on, the model's ``--top`` most-attended tiles and ``--controls`` random tiles from the rest
of the tissue, and crops each as a 3 x 3 tile field with the tile in question outlined. Tiles are shuffled within
each case and given codes, so a rater sees the case's WHO CNS5 diagnosis but not the model's call, whether it was
right, or which tiles it attended to.

``--report``: a training report (``oof_patients``: calls by fold models that had not seen the patient) or an
external evaluation report (``patients``, from ``entip-train triage evaluate --out``). An external set is better
for the study itself: the saved model never saw those patients, so the tiles it attends to are the ones it would
use in practice. On a training set the saved model has seen every patient; use that for a pilot.

Writes, in ``--out``:

* ``review.html``: the rating page. Open it in a browser (it reads ``tiles/`` next to it), rate each tile, and
  press "Download scores": it saves ``scores-<rater>.csv``. Ratings are kept in the browser between sessions.
* ``tiles/<code>.jpg``: the images.
* ``key.csv``: which tiles were attended and which were controls, with the model's calls. **Keep it from raters.**

Analyse with ``analyse.py key.csv scores-*.csv``.
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

SCORES = [
    ("D", "Diagnostic", "shows the tissue that defines the stated diagnosis (for a non-neoplastic case: "
                        "representative parenchyma)"),
    ("T", "Tumour, not specific", "tumour, but this field alone would not support the stated diagnosis"),
    ("N", "Non-neoplastic", "normal or reactive tissue only, in a tumour case"),
    ("A", "Artefact / not assessable", "background, blood, crush, folds, ink, out of focus or too little tissue"),
]


def find_slide(slides: Path, uid: str) -> Path:
    p = slides / uid
    if p.exists():
        return p
    hits = sorted(slides.glob(f"{uid}.*"))
    if not hits:
        raise FileNotFoundError(f"no slide for {uid} in {slides}")
    return hits[0]


def report_rows(rep: dict) -> list:
    """One row per patient: patient, label, call, confidence, correct, and the reference call's origin."""
    rep = rep.get("report", rep)
    if rep.get("oof_patients"):
        return [{**r, "reference_call": "out-of-fold (fold model had not seen the patient)"}
                for r in rep["oof_patients"]]
    if rep.get("patients"):
        return [{**r, "reference_call": "external evaluation (model never saw the patient)"}
                for r in rep["patients"]]
    raise SystemExit("report has neither oof_patients nor patients (use the full evaluation report: --out)")


def sample(rows: list, per_class: int, seed: int) -> list:
    rng = random.Random(seed)
    wrong = [r for r in rows if not r["correct"]]
    chosen = {r["patient"] for r in wrong}
    by = defaultdict(list)
    for r in rows:
        if r["correct"]:
            by[r["label"]].append(r)
    for lab in sorted(by):
        rs = sorted(by[lab], key=lambda r: r["patient"])
        rng.shuffle(rs)
        chosen |= {r["patient"] for r in rs[:per_class]}
    return [r for r in rows if r["patient"] in chosen]


def crop(rd, x_um: float, y_um: float, tile_um: float, px: int = 576):
    """A 3 x 3 tile field centred on the tile, with the tile outlined."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(rd.read_um(x_um + tile_um / 2, y_um + tile_um / 2, 3 * tile_um, px))
    d, a = ImageDraw.Draw(im), px / 3
    d.rectangle([a, a, 2 * a, 2 * a], outline=(0, 0, 0), width=3)
    d.rectangle([a - 3, a - 3, 2 * a + 3, 2 * a + 3], outline=(255, 255, 255), width=2)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--slides", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=12, help="correctly called patients per class")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--controls", type=int, default=8, help="random tiles from outside the top 10%% of attention")
    ap.add_argument("--id-col", default="SeriesInstanceUID")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from entip_glioma.triage import who_cns5
    from entip_glioma.triage.model import CLASSES, TriageModel, p_tumour
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide

    rng = np.random.default_rng(a.seed)
    model = TriageModel.load(a.model)
    man = defaultdict(list)
    for r in csv.DictReader(open(a.manifest, newline="")):
        man[r["PatientID"]].append(r)
    picked = sample(report_rows(json.load(open(a.report))), a.per_class, a.seed)
    out = Path(a.out)
    (out / "tiles").mkdir(parents=True, exist_ok=True)

    cases, key, n = [], [], 0
    order = list(range(len(picked)))
    random.Random(a.seed).shuffle(order)             # case codes do not follow class or correctness
    for ci, pi in enumerate(order, start=1):
        r = picked[pi]
        slides = [s for s in man.get(r["patient"], []) if (Path(a.bags) / f"{s[a.id_col]}.npz").exists()]
        if not slides:
            print(f"skip {r['patient']}: no bag", file=sys.stderr)
            continue
        scored = []
        for s in slides:
            b = Bag.load(Path(a.bags) / f"{s[a.id_col]}.npz")
            if len(b):
                g = model.predict(b)
                scored.append((float(p_tumour(g["probs"][None])[0]), b, g, s))
        if not scored:
            continue
        _, bag, g, srow = max(scored, key=lambda t: t[0])   # the slide the model bases its call on
        att = np.asarray(g["attention"], np.float64)
        rank = np.argsort(-att)
        top = rank[:a.top]
        pool = rank[max(a.top, int(np.ceil(0.1 * len(att)))):]
        ctrl = rng.choice(pool, size=min(a.controls, len(pool)), replace=False) if len(pool) else []
        rd = open_slide(str(find_slide(Path(a.slides), srow[a.id_col])))
        case = f"C{ci:03d}"
        dx = srow.get("diagnosis") or who_cns5.class_name(r["label"])
        tiles = [("attended", int(i), k + 1) for k, i in enumerate(top)] + [("control", int(i), None) for i in ctrl]
        random.Random(f"{a.seed}-{case}").shuffle(tiles)
        codes = []
        for kind, i, k in tiles:
            n += 1
            code = f"T{n:05d}"
            x, y = (float(v) for v in bag.coords_um[i])
            crop(rd, x, y, bag.tile_um).save(out / "tiles" / f"{code}.jpg", quality=90)
            codes.append(code)
            key.append({"tile_code": code, "case_code": case, "patient": r["patient"], "slide": srow[a.id_col],
                        "label": r["label"], "diagnosis": dx, "set": kind, "attention_rank": k or "",
                        "weight": round(float(att[i]), 6), "x_um": x, "y_um": y,
                        "reference_call": r["call"], "reference_confidence": r["confidence"],
                        "reference_correct": r["correct"], "reference_origin": r["reference_call"],
                        "model_call": CLASSES[int(np.argmax(g["probs"]))],
                        "model_confidence": round(float(np.max(g["probs"])), 4)})
        cases.append({"case": case, "diagnosis": dx, "tiles": codes})
        print(f"{case}  {r['patient']:<16} {len(codes)} tiles", file=sys.stderr)

    with (out / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    study = f"entip-attention-{Path(a.report).stem}-{a.seed}"
    (out / "review.html").write_text(page(study, cases))
    print(json.dumps({"cases": len(cases), "tiles": len(key), "attended": sum(k["set"] == "attended" for k in key),
                      "controls": sum(k["set"] == "control" for k in key),
                      "wrong_cases": len({k["case_code"] for k in key if not k["reference_correct"]}),
                      "out": str(out)}, indent=2))
    return 0


def page(study: str, cases: list) -> str:
    """The rating page: plain HTML and a little JavaScript, no network; ratings kept in localStorage."""
    opts = "".join(f'<label title="{html.escape(t)}"><input type="radio" name="__" value="{c}"> {c} {html.escape(n)}'
                   f"</label>" for c, n, t in SCORES)
    legend = "".join(f"<li><b>{c} {html.escape(n)}</b>: {html.escape(t)}</li>" for c, n, t in SCORES)
    blocks = []
    for c in cases:
        tiles = "".join(
            f'<figure data-tile="{t}"><img loading="lazy" src="tiles/{t}.jpg" alt="tile {t}">'
            f'<figcaption><span class="code">{t}</span>{opts.replace("__", t)}'
            f'<input class="note" data-note="{t}" placeholder="comment (optional)"></figcaption></figure>'
            for t in c["tiles"])
        blocks.append(f'<section><h2>{c["case"]} <span class="dx">WHO CNS5: {html.escape(c["diagnosis"])}</span>'
                      f'</h2><div class="grid">{tiles}</div></section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attention review</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#5f5e5a;--line:#d3d1c7;--card:#fff;--accent:#185fa5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1d1d1b;--fg:#ecebe6;--muted:#b4b2a9;--line:#444441;--card:#2c2c2a;--accent:#85b7eb}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:1}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}} h2{{font-size:16px;font-weight:500;margin:24px 0 8px}}
.dx{{color:var(--muted);font-weight:400;margin-left:8px}} .code{{font-family:ui-monospace,monospace;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}}
figure.done{{border-color:var(--accent)}} img{{width:100%;height:auto;display:block;border-radius:4px}}
figcaption label{{display:block;font-size:13px}} .note{{width:100%;box-sizing:border-box;margin-top:4px;font:inherit;font-size:13px}}
button{{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
ul{{margin:4px 0 0;padding-left:20px;color:var(--muted);font-size:13px}}
</style></head><body>
<header><h1>Attention review</h1>
<label>Rater initials <input id="rater" size="6"></label>
<span id="progress"></span> <button id="dl">Download scores</button>
<ul>{legend}<li>Judge the <b>outlined tile</b> in the centre; the surrounding field is context. Each field is 3 tiles
(about 336 µm) across.</li></ul></header>
{''.join(blocks)}
<script>
const KEY = {json.dumps(study)};
let st = {{}};
try {{ st = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ st = {{}}; }}
st.scores = st.scores || {{}}; st.notes = st.notes || {{}};
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(st)); }} catch (e) {{}} }};
const rater = document.getElementById("rater");
rater.value = st.rater || ""; rater.oninput = () => {{ st.rater = rater.value.trim(); save(); }};
const figs = [...document.querySelectorAll("figure")];
function progress() {{
  const n = figs.filter(f => st.scores[f.dataset.tile]).length;
  document.getElementById("progress").textContent = `${{n}} of ${{figs.length}} rated`;
  figs.forEach(f => f.classList.toggle("done", !!st.scores[f.dataset.tile]));
}}
document.querySelectorAll("input[type=radio]").forEach(r => {{
  if (st.scores[r.name] === r.value) r.checked = true;
  r.onchange = () => {{ st.scores[r.name] = r.value; save(); progress(); }};
}});
document.querySelectorAll(".note").forEach(n => {{
  n.value = st.notes[n.dataset.note] || "";
  n.oninput = () => {{ st.notes[n.dataset.note] = n.value; save(); }};
}});
document.getElementById("dl").onclick = () => {{
  if (!st.rater) {{ alert("Enter your initials first"); rater.focus(); return; }}
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const rows = [["rater", "tile_code", "score", "comment"]].concat(
    figs.map(f => [st.rater, f.dataset.tile, st.scores[f.dataset.tile] || "", st.notes[f.dataset.tile] || ""]));
  const blob = new Blob([rows.map(r => r.map(q).join(",")).join("\\n") + "\\n"], {{type: "text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `scores-${{st.rater}}.csv`; a.click();
}};
progress();
</script></body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
