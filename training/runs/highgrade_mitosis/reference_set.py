"""Step 3c: a blinded, exhaustively counted mitosis reference set, with no model involved.

    python reference_set.py draw --manifest trial.csv --bags bags/ --slides slides/ --exclude used.csv --out mitosis-ref1
    python reference_set.py build --bags bags/ --slides slides/ --out mitosis-ref1
    python reference_set.py lock --key mitosis-ref1/private/key.csv --marks mitosis-marks-AH.csv --out mitosis-ref1/private
    python reference_set.py agree --locked mitosis-ref1/private --marks mitosis-marks-XY.csv

**draw** picks patients at random by category, one 40x slide each (native pixel size at most 0.30 µm), leaving out every
patient in an ``--exclude`` file, and four random non-overlapping 0.5 mm² tissue fields per slide. Where tissue lies
comes from the bag's tile coordinates; no embedding or model output is read. A category that runs short is topped up
from ``--top-up``, and the draw says so.

**build** gives each field a random code and writes ``review-pack.tar`` (all fields) and ``second-reader-pack.tar`` (one
random field per patient). They hold field images at 0.25 µm per pixel and a page, with no slide, patient, grade or
position. The link stays in ``private/key.csv``.

**The review.** Search all nine sections of each field and mark every mitosis, every uncertain figure, and any
look-alikes worth recording. A figure counts if its centre is inside the field. Give the share of the field that could
be assessed.

**lock** runs once, on a complete marks file. It fixes the marks as this run's reference and reports only totals.
**agree** compares a second reader's fields with the locked marks. Neither changes the other reader's marks.
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
from common import find_slide, quotas, read_manifest, read_patients  # noqa: E402

FIELD_MM2 = 0.5
FIELD_UM = math.sqrt(FIELD_MM2) * 1000
UM_PER_PX = 0.25
IMAGE_PX = 2988                     # the field plus a margin of about 20 µm, so a cell on the edge can be judged
IMAGE_UM = IMAGE_PX * UM_PER_PX
MARGIN_UM = (IMAGE_UM - FIELD_UM) / 2
MAX_NATIVE_MPP = 0.30
WINDOW_TILES = 7                    # 7 x 112 µm tiles hold one field
MIN_TISSUE = 0.9
MATCH_UM = 7.5
LABELS = ("mitosis", "uncertain", "pyknosis", "apoptosis", "karyorrhexis", "lymphocyte", "other_mimic")


def open_reader(slides: str, slide: str):
    from entip_train.slides import open_slide
    return open_slide(str(find_slide(Path(slides), slide)))


def tile_coords(bags: str, slide: str) -> tuple:
    """Tile positions only: the embeddings in the bag are never read."""
    with np.load(Path(bags) / f"{slide}.npz", allow_pickle=False) as z:
        return np.asarray(z["coords_um"], float), float(z["tile_um"])


def tissue_windows(coords: np.ndarray, tile_um: float, k: int = WINDOW_TILES, min_tissue: float = MIN_TISSUE) -> list:
    """(column, row) of every k x k tile window with at least ``min_tissue`` of its tiles in the bag."""
    if len(coords) == 0:
        return []
    ij = np.rint((coords - coords.min(0)) / tile_um).astype(int)
    grid = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), int)
    grid[ij[:, 1], ij[:, 0]] = 1
    if min(grid.shape) < k:
        return []
    s = np.zeros((grid.shape[0] + 1, grid.shape[1] + 1), int)
    s[1:, 1:] = grid.cumsum(0).cumsum(1)
    n = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    rows, cols = np.nonzero(n >= math.ceil(min_tissue * k * k))
    return [(int(c), int(r)) for c, r in zip(cols, rows)]


def pick_fields(coords: np.ndarray, tile_um: float, n: int, rng: random.Random, k: int = WINDOW_TILES) -> list:
    """Top-left (x, y) in µm of ``n`` random non-overlapping fields, or [] if the slide cannot hold them."""
    windows = tissue_windows(coords, tile_um, k)
    rng.shuffle(windows)
    chosen = []
    for c, r in windows:
        if all(abs(c - c2) >= k or abs(r - r2) >= k for c2, r2 in chosen):
            chosen.append((c, r))
            if len(chosen) == n:
                break
    if len(chosen) < n:
        return []
    ox, oy = coords.min(0)
    half = k * tile_um / 2
    return [(float(ox + c * tile_um + half - FIELD_UM / 2), float(oy + r * tile_um + half - FIELD_UM / 2))
            for c, r in chosen]


def write_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def draw(a) -> int:
    out = Path(a.out) / "private"
    if (out / "fields.csv").exists():
        raise SystemExit(f"{out / 'fields.csv'} already exists: a drawn set is never redrawn")
    used = read_patients(a.exclude)
    want = quotas(a.per_category)
    if a.top_up not in want:
        raise SystemExit(f"--top-up {a.top_up} is not in --per-category")
    by = defaultdict(lambda: defaultdict(list))
    for r in read_manifest(a.manifest):
        if r.get("label", "glioma") == "glioma" and r["PatientID"] not in used \
                and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists():
            by[r["category"]][r["PatientID"]].append(r)
    rng = random.Random(a.seed)
    queue = {}
    for cat in want:
        queue[cat] = sorted(by[cat])
        rng.shuffle(queue[cat])

    def take(cat: str, n: int) -> list:
        got = []
        while queue[cat] and len(got) < n:
            pat = queue[cat].pop(0)
            options = sorted(by[cat][pat], key=lambda r: r["SeriesInstanceUID"])
            random.Random(f"{a.seed}-{pat}").shuffle(options)
            for r in options:
                uid = r["SeriesInstanceUID"]
                coords, tile_um = tile_coords(a.bags, uid)
                fields = pick_fields(coords, tile_um, a.fields, random.Random(f"{a.seed}-{uid}"))
                if not fields:
                    continue
                mpp = float(open_reader(a.slides, uid).mpp)
                if mpp <= MAX_NATIVE_MPP:
                    got.append((r, mpp, fields))
                    break
        return got

    picked, short = [], {}
    for cat, n in want.items():
        got = take(cat, n)
        picked += got
        if len(got) < n:
            short[cat] = n - len(got)
    if short:
        extra = take(a.top_up, sum(short.values()))
        picked += extra
    total = sum(want.values())
    if len(picked) != total:
        raise SystemExit(f"only {len(picked)} of {total} eligible patients found: nothing written")
    out.mkdir(parents=True, exist_ok=True)
    patients, fields = [], []
    for r, mpp, fs in sorted(picked, key=lambda x: x[0]["PatientID"]):
        patients.append({"patient": r["PatientID"], "slide": r["SeriesInstanceUID"], "category": r["category"],
                         "diagnosis": r.get("diagnosis", ""), "native_mpp": mpp})
        for n, (x, y) in enumerate(fs, start=1):
            fields.append({"patient": r["PatientID"], "slide": r["SeriesInstanceUID"], "field_of_slide": n,
                           "x_um": round(x, 3), "y_um": round(y, 3), "side_um": round(FIELD_UM, 4)})
    write_csv(out / "patients.csv", patients)
    write_csv(out / "fields.csv", fields)
    print(json.dumps({"patients": len(patients), "fields": len(fields), "area_mm2": len(fields) * FIELD_MM2,
                      "by_category": dict(Counter(p["category"] for p in patients)), "wanted": want,
                      "topped_up_from": a.top_up if short else None, "short": short}, indent=2))
    return 0


def pack_slide(slide: str, rows: list, slides: str, folder: str) -> list:
    from PIL import Image
    rd = open_reader(slides, slide)
    for r in rows:
        cx, cy = float(r["x_um"]) + FIELD_UM / 2, float(r["y_um"]) + FIELD_UM / 2
        im = Image.fromarray(rd.read_um(cx, cy, IMAGE_UM, IMAGE_PX))
        im.save(Path(folder) / f"{r['field']}.jpg", quality=92, subsampling=0)
    return [r["field"] for r in rows]


def page(study: str, codes: list) -> str:
    data = {"study": study, "fieldUm": FIELD_UM, "marginUm": MARGIN_UM, "imageUm": IMAGE_UM, "labels": list(LABELS),
            "fields": codes}
    return VIEWER.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def write_pack(out: Path, name: str, study: str, codes: list, source: Path) -> Path:
    pack = out / name
    (pack / "fields").mkdir(parents=True, exist_ok=True)
    if source != pack / "fields":
        for c in codes:
            shutil.copyfile(source / f"{c}.jpg", pack / "fields" / f"{c}.jpg")
    (pack / "review.html").write_text(page(study, codes), encoding="utf-8")
    path = out / f"{name}.tar"
    with tarfile.open(path, "w") as tar:
        tar.add(pack, arcname=name)
    return path


def build(a) -> int:
    out = Path(a.out)
    private = out / "private"
    if (private / "key.csv").exists():
        raise SystemExit(f"{private / 'key.csv'} already exists: a built set is never overwritten")
    rows = list(csv.DictReader(open(private / "fields.csv", newline="")))
    if not rows:
        raise SystemExit("no drawn fields")
    clash = read_patients(a.exclude) & {r["patient"] for r in rows}
    if clash:
        raise SystemExit(f"{len(clash)} drawn patients were used in earlier work")
    rng = random.Random(f"{a.seed}-codes")
    rng.shuffle(rows)
    for n, r in enumerate(rows, start=1):
        r["field"] = f"F{n:03d}"
    by_patient = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["field"]):
        by_patient[r["patient"]].append(r)
    second = {rng.choice(v)["field"] for _, v in sorted(by_patient.items())}
    folder = out / "review-pack" / "fields"
    folder.mkdir(parents=True, exist_ok=True)
    by_slide = defaultdict(list)
    for r in rows:
        by_slide[r["slide"]].append(r)
    workers = max(1, a.workers or int(os.environ.get("NSLOTS", 1)))
    jobs = [(s, v, a.slides, str(folder)) for s, v in sorted(by_slide.items())]
    if workers == 1:
        for n, j in enumerate(jobs, start=1):
            pack_slide(*j)
            print(f"packed {n}/{len(jobs)} slides", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(workers) as pool:
            for n, job in enumerate(as_completed([pool.submit(pack_slide, *j) for j in jobs]), start=1):
                job.result()
                print(f"packed {n}/{len(jobs)} slides", file=sys.stderr, flush=True)
    codes = sorted(r["field"] for r in rows)
    sha = hashlib.sha256(f"{a.seed}|{'|'.join(r['slide'] + r['x_um'] for r in sorted(rows, key=lambda r: r['field']))}"
                         .encode()).hexdigest()[:12]
    main_tar = write_pack(out, "review-pack", f"entip-mitosis-ref-{sha}", codes, folder)
    second_codes = sorted(second)
    random.Random(f"{a.seed}-second").shuffle(second_codes)
    second_tar = write_pack(out, "second-reader-pack", f"entip-mitosis-ref-{sha}-second", second_codes, folder)
    key = [{"field": r["field"], "patient": r["patient"], "slide": r["slide"], "x_um": r["x_um"], "y_um": r["y_um"],
            "side_um": r["side_um"], "second_reader": int(r["field"] in second)}
           for r in sorted(rows, key=lambda r: r["field"])]
    write_csv(private / "key.csv", key)
    print(json.dumps({"fields": len(codes), "second_reader_fields": len(second_codes),
                      "review_pack_gb": round(main_tar.stat().st_size / 1e9, 2),
                      "second_pack_gb": round(second_tar.stat().st_size / 1e9, 2), "pack": str(main_tar),
                      "second_pack": str(second_tar)}, indent=2))
    return 0


def read_marks(path: str, want: set) -> tuple:
    """One reader's complete marks: (rater, field -> row, field -> [(x, y, label)]). Refuses anything incomplete."""
    fields, points = {}, defaultdict(list)
    raters = set()
    for r in csv.DictReader(open(path, newline="")):
        raters.add((r.get("rater") or "").strip().upper())
        if r["kind"] == "field":
            fields[r["field"]] = r
        elif r["kind"] == "point":
            x, y, lab = float(r["x_um"]), float(r["y_um"]), r["label"]
            if lab not in LABELS:
                raise SystemExit(f"{r['field']}: unknown label {lab!r}")
            if not (0 <= x <= FIELD_UM and 0 <= y <= FIELD_UM):
                raise SystemExit(f"{r['field']}: a mark lies outside the field")
            points[r["field"]].append((x, y, lab))
    if len(raters) != 1 or "" in raters:
        raise SystemExit("one marks file must come from one named reader")
    if set(fields) != want or set(points) - want:
        raise SystemExit("the marks do not cover the same fields as the key")
    open_fields = sorted(f for f, r in fields.items() if r["complete"] != "1")
    if open_fields:
        raise SystemExit(f"{len(open_fields)} fields are not complete: {', '.join(open_fields)}")
    for f, r in fields.items():
        if not 0 <= float(r["assessable_pct"]) <= 100:
            raise SystemExit(f"{f}: assessable share must be 0 to 100")
    return raters.pop(), fields, points


def lock(a) -> int:
    out = Path(a.out)
    if (out / "locked-points.csv").exists():
        raise SystemExit(f"{out / 'locked-points.csv'} already exists: marks are locked once")
    key = {r["field"]: r for r in csv.DictReader(open(a.key, newline=""))}
    rater, fields, points = read_marks(a.marks, set(key))
    frows, prows = [], []
    for f in sorted(key):
        k, r = key[f], fields[f]
        pts = points.get(f, [])
        frows.append({"field": f, "patient": k["patient"], "slide": k["slide"], "rater": rater,
                      "assessable_pct": float(r["assessable_pct"]),
                      "assessable_mm2": round(FIELD_MM2 * float(r["assessable_pct"]) / 100, 4),
                      "mitoses": sum(p[2] == "mitosis" for p in pts), "uncertain": sum(p[2] == "uncertain" for p in pts),
                      "comment": r.get("comment", "")})
        for x, y, lab in pts:
            prows.append({"field": f, "patient": k["patient"], "slide": k["slide"], "rater": rater,
                          "field_x_um": round(x, 2), "field_y_um": round(y, 2),
                          "slide_x_um": round(float(k["x_um"]) + x, 2), "slide_y_um": round(float(k["y_um"]) + y, 2),
                          "label": lab})
    write_csv(out / "locked-fields.csv", frows)
    with (out / "locked-points.csv").open("w", newline="") as fh:
        names = ["field", "patient", "slide", "rater", "field_x_um", "field_y_um", "slide_x_um", "slide_y_um", "label"]
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(prows)
    labels = Counter(p["label"] for p in prows)
    info = {"rater": rater, "fields": len(frows), "assessable_mm2": round(sum(r["assessable_mm2"] for r in frows), 3),
            "marks": {k: labels.get(k, 0) for k in LABELS}, "target_mitoses": 200,
            "reached_target": labels.get("mitosis", 0) >= 200,
            "sha256_points": hashlib.sha256((out / "locked-points.csv").read_bytes()).hexdigest(),
            "sha256_fields": hashlib.sha256((out / "locked-fields.csv").read_bytes()).hexdigest()}
    (out / "lock.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


def match(a_pts: list, b_pts: list, radius: float = MATCH_UM) -> int:
    """Closest pairs first, each point used once: how many points of ``a_pts`` have a partner in ``b_pts``."""
    pairs = sorted((math.dist(p[:2], q[:2]), i, j) for i, p in enumerate(a_pts) for j, q in enumerate(b_pts)
                   if math.dist(p[:2], q[:2]) <= radius)
    ua, ub, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in ua and j not in ub:
            ua.add(i)
            ub.add(j)
            n += 1
    return n


def agree(a) -> int:
    locked = Path(a.locked)
    path = locked / "agreement.json"
    if path.exists():
        raise SystemExit(f"{path} already exists")
    if not (locked / "locked-points.csv").exists():
        raise SystemExit("lock the first reader's marks before comparing")
    key = list(csv.DictReader(open(locked / "key.csv", newline="")))
    want = {r["field"] for r in key if r["second_reader"] == "1"}
    rater, _, second = read_marks(a.marks, want)
    first = defaultdict(list)
    first_rater = None
    for r in csv.DictReader(open(locked / "locked-points.csv", newline="")):
        first_rater = r["rater"]
        if r["field"] in want:
            first[r["field"]].append((float(r["field_x_um"]), float(r["field_y_um"]), r["label"]))
    if rater == first_rater:
        raise SystemExit("the second reader must be a different person")

    def compare(keep: set) -> dict:
        both = only1 = only2 = 0
        diffs = []
        for f in sorted(want):
            p = [x for x in first.get(f, []) if x[2] in keep]
            q = [x for x in second.get(f, []) if x[2] in keep]
            m = match(p, q)
            both, only1, only2 = both + m, only1 + len(p) - m, only2 + len(q) - m
            diffs.append(abs(len(p) - len(q)))
        denom = 2 * both + only1 + only2
        return {"matched": both, "first_only": only1, "second_only": only2,
                "positive_agreement": round(2 * both / denom, 3) if denom else None,
                "mean_abs_count_difference_per_field": round(float(np.mean(diffs)), 3)}

    info = {"first_reader": first_rater, "second_reader": rater, "fields": len(want), "match_radius_um": MATCH_UM,
            "definite_mitoses": compare({"mitosis"}), "mitoses_or_uncertain": compare({"mitosis", "uncertain"})}
    path.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


VIEWER = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Mitosis reference count</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:light;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202124}
*{box-sizing:border-box}body{margin:0;height:100vh;display:flex;background:#f4f4f2}
#view{flex:1;min-width:0;position:relative;background:#30302e}canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
aside{width:310px;flex:none;overflow:auto;background:#fff;border-left:1px solid #d8d8d3;padding:14px}
h1{font-size:17px;font-weight:600;margin:0 0 3px}.muted{color:#68685f;font-size:12px}section{border-top:1px solid #e6e6e1;margin-top:11px;padding-top:10px}
button{font:inherit;border:1px solid #b9b9b1;background:#fff;border-radius:5px;padding:5px 8px;cursor:pointer}button:hover{background:#f0f0ec}button.go{background:#1f6f54;border-color:#1f6f54;color:#fff}button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.lab{display:flex;width:100%;gap:8px;text-align:left;margin-bottom:3px;padding:4px 7px}.lab.on{outline:2px solid #1f6f54}.dot{width:17px;height:17px;border-radius:50%;color:#fff;font-size:11px;font-weight:700;display:inline-flex;align-items:center;justify-content:center}
kbd{font:11px ui-monospace,monospace;background:#efefea;border:1px solid #cfcfc8;border-radius:3px;padding:0 4px;margin-left:auto}
#map{display:grid;grid-template-columns:repeat(3,38px);gap:3px}#map button{height:30px;padding:0}#map .seen{background:#dcefe6;border-color:#65a88c}#map .cur{outline:2px solid #1f6f54}
input,textarea{font:inherit;border:1px solid #b9b9b1;border-radius:4px;padding:4px}textarea{width:100%;height:44px;resize:vertical}
#msg{min-height:18px;color:#a3341f;font-size:12px;margin-top:6px}#start{position:absolute;inset:0;z-index:5;background:#f4f4f2;display:flex;align-items:center;justify-content:center;padding:24px}#start div{max-width:560px;background:#fff;border:1px solid #d8d8d3;border-radius:8px;padding:22px}
</style></head><body><div id="start"><div><h1>Mitosis reference count</h1><p>Mark <b>every</b> mitotic figure whose centre lies inside the white square. Search all nine sections of each field. The page shows no diagnosis, patient or model output.</p>
<p><b>Mitosis:</b> a definite mitotic figure (prometaphase, metaphase, anaphase, telophase, atypical or granular) with no nuclear membrane. Count a telophase pair once. <b>Do not mark prophase as a mitosis.</b> <b>Uncertain:</b> you cannot decide. Look-alikes are optional, but useful.</p>
<p class="row"><label>Your initials <input id="who" maxlength="6" size="8" autocomplete="off"></label><button class="go" id="begin">Start</button></p><p class="muted">Marks save in this browser. Download them often, and load the file to continue elsewhere.</p><div id="startmsg" style="border:0;padding:0;color:#a3341f"></div></div></div>
<main id="view"><canvas id="cv"></canvas></main><aside><h1 id="title"></h1><div class="muted" id="progress"></div>
<section><div class="row"><button id="prevF">Previous field</button><button id="nextF">Next field</button></div></section>
<section><b>Mark</b> <span class="muted">click to add, click a mark to select</span><div id="labels" style="margin-top:6px"></div><div class="row"><button id="del">Delete selected</button><span class="muted"><kbd>⌫</kbd></span></div></section>
<section><b>Sections</b> <span class="muted">view all nine</span><div class="row" style="margin-top:6px;align-items:flex-start"><div id="map"></div><div><button id="nextS">Next <kbd>N</kbd></button><br><button id="whole" style="margin-top:4px">Whole field <kbd>F</kbd></button></div></div></section>
<section><label><b>Assessable share</b> <input id="pct" type="number" min="0" max="100" step="5" style="width:68px"> %</label><div class="muted">Part of the square lost to folds, blur, glass or similar.</div><div style="margin-top:6px"><textarea id="comment" placeholder="Comment (optional)"></textarea></div>
<div class="row" style="margin-top:6px"><button class="go" id="done">Field complete <kbd style="color:#222">↵</kbd></button></div><div id="msg"></div></section>
<section><div class="row"><button id="save">Download marks</button><label><button id="loadB" type="button">Load marks</button><input id="load" type="file" accept=".csv" hidden></label></div></section></aside>
<script>
const D=__DATA__,KEY="entip-"+D.study,cv=document.getElementById("cv"),ctx=cv.getContext("2d"),$=id=>document.getElementById(id);
const COL=["#16833f","#c77700","#6a4fb3","#b5356f","#176fa5","#50606b","#8a5a2b"],LET=["M","U","P","A","K","L","O"],ORDER=[0,1,2,5,4,3,6,7,8];
const blank=()=>({marks:[],seen:Array(9).fill(false),pct:null,comment:"",done:false});
let st={rater:"",i:0,active:0,data:{}};try{const o=JSON.parse(localStorage.getItem(KEY)||"null");if(o&&o.data)st=o}catch(e){}
D.fields.forEach(f=>{if(!st.data[f])st.data[f]=blank()});
let v={s:1,x:0,y:0},img=new Image(),selected=null,section=-1,drag=null;
const cur=()=>st.data[D.fields[st.i]],store=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){say("This browser could not save: download your marks now.")}},say=t=>{$("msg").textContent=t||""};
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=Math.round(r.width*d);cv.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0)}
function fitRect(x,y,w,h){const r=cv.getBoundingClientRect();if(!r.width||!r.height)return;v.s=Math.min(r.width/w,r.height/h);v.x=x+w/2-r.width/2/v.s;v.y=y+h/2-r.height/2/v.s;draw()}
function whole(){section=-1;fitRect(0,0,D.imageUm,D.imageUm);paint()}
function go(k){section=k;const q=D.fieldUm/3,c=k%3,r=Math.floor(k/3),o=8;cur().seen[k]=true;say("");store();fitRect(D.marginUm+c*q-o,D.marginUm+r*q-o,q+2*o,q+2*o);paint()}
function reopen(){cur().done=false}
function nextSection(){const a=section<0?-1:ORDER.indexOf(section);go(ORDER[(a+1)%9])}
function draw(){const r=cv.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);ctx.fillStyle="#30302e";ctx.fillRect(0,0,r.width,r.height);if(img.complete&&img.naturalWidth){ctx.imageSmoothingEnabled=v.s<4;ctx.drawImage(img,-v.x*v.s,-v.y*v.s,D.imageUm*v.s,D.imageUm*v.s)}
const X=u=>(u+D.marginUm-v.x)*v.s,Y=u=>(u+D.marginUm-v.y)*v.s,F=D.fieldUm*v.s;ctx.fillStyle="rgba(20,20,20,.48)";const x0=X(0),y0=Y(0);ctx.fillRect(0,0,r.width,Math.max(0,y0));ctx.fillRect(0,y0+F,r.width,Math.max(0,r.height-y0-F));ctx.fillRect(0,y0,Math.max(0,x0),F);ctx.fillRect(x0+F,y0,Math.max(0,r.width-x0-F),F);ctx.lineWidth=2;ctx.strokeStyle="#fff";ctx.strokeRect(x0,y0,F,F);ctx.lineWidth=1;ctx.strokeStyle="rgba(255,255,255,.35)";for(let i=1;i<3;i++){ctx.beginPath();ctx.moveTo(x0+i*F/3,y0);ctx.lineTo(x0+i*F/3,y0+F);ctx.moveTo(x0,y0+i*F/3);ctx.lineTo(x0+F,y0+i*F/3);ctx.stroke()}
cur().marks.forEach((m,i)=>{const px=X(m.x),py=Y(m.y),c=COL[D.labels.indexOf(m.l)],R=Math.max(8,5*v.s);ctx.beginPath();ctx.arc(px,py,R,0,7);ctx.lineWidth=i===selected?4:2;ctx.strokeStyle=c;ctx.stroke();ctx.fillStyle=c;ctx.font="600 11px sans-serif";ctx.fillText(LET[D.labels.indexOf(m.l)],px+R+2,py-R+3)})}
function paint(){const c=cur(),n=D.fields.filter(f=>st.data[f].done).length;$("title").textContent=`Field ${st.i+1} of ${D.fields.length}`;$("progress").textContent=`${n} complete · reader ${st.rater||"-"}`;$("pct").value=c.pct??"";$("comment").value=c.comment||"";$("done").textContent=c.done?"Complete":"Field complete";
$("labels").innerHTML=D.labels.map((l,i)=>`<button class="lab ${i===st.active?"on":""}" data-l="${i}"><span class="dot" style="background:${COL[i]}">${LET[i]}</span>${l.replace("_"," ")}<kbd>${i+1}</kbd></button>`).join("");
$("map").innerHTML=Array.from({length:9},(_,k)=>`<button class="${c.seen[k]?"seen":""} ${k===section?"cur":""}" data-s="${k}">${ORDER.indexOf(k)+1}</button>`).join("")}
function showField(i){st.i=Math.max(0,Math.min(D.fields.length-1,i));selected=null;section=-1;say("");store();paint();img=new Image();img.onload=()=>whole();img.onerror=()=>say("The field image did not load: keep review.html beside the fields folder.");img.src=`fields/${D.fields[st.i]}.jpg`;whole()}
function point(e){const r=cv.getBoundingClientRect();return{sx:e.clientX-r.left,sy:e.clientY-r.top}}
function clickAt(p){const c=cur(),fx=p.sx/v.s+v.x-D.marginUm,fy=p.sy/v.s+v.y-D.marginUm;let best=-1,bd=1e9;c.marks.forEach((m,i)=>{const d=Math.hypot((m.x-fx)*v.s,(m.y-fy)*v.s);if(d<Math.max(10,5*v.s)&&d<bd){bd=d;best=i}});if(best>=0){selected=best;draw();return}if(fx<0||fy<0||fx>D.fieldUm||fy>D.fieldUm){say("Marks go inside the white square.");return}say("");c.marks.push({x:+fx.toFixed(2),y:+fy.toFixed(2),l:D.labels[st.active]});selected=null;reopen();store();paint();draw()}
cv.addEventListener("pointerdown",e=>{cv.setPointerCapture(e.pointerId);const p=point(e);drag={...p,x:v.x,y:v.y,moved:false}});cv.addEventListener("pointermove",e=>{if(!drag)return;const p=point(e),dx=p.sx-drag.sx,dy=p.sy-drag.sy;if(Math.hypot(dx,dy)>4)drag.moved=true;if(drag.moved){v.x=drag.x-dx/v.s;v.y=drag.y-dy/v.s;draw()}});cv.addEventListener("pointerup",e=>{if(drag&&!drag.moved)clickAt(point(e));drag=null});
cv.addEventListener("wheel",e=>{e.preventDefault();const p=point(e),ux=p.sx/v.s+v.x,uy=p.sy/v.s+v.y,f=Math.exp(-e.deltaY*.0016),r=cv.getBoundingClientRect(),min=Math.min(r.width,r.height)/D.imageUm*.7;v.s=Math.max(min,Math.min(12,v.s*f));v.x=ux-p.sx/v.s;v.y=uy-p.sy/v.s;draw()},{passive:false});
function setLabel(i){const c=cur();if(selected!==null&&c.marks[selected]){c.marks[selected].l=D.labels[i];reopen();store()}else st.active=i;store();paint();draw()}
function removeMark(){if(selected===null)return;cur().marks.splice(selected,1);selected=null;reopen();store();paint();draw()}
function complete(){const c=cur(),left=c.seen.filter(x=>!x).length,p=$("pct").value===""?NaN:+$("pct").value;if(left){say(`View the remaining ${left} section${left>1?"s":""} first.`);return}if(!(p>=0&&p<=100)){say("Enter the assessable share, 0 to 100.");return}c.pct=p;c.comment=$("comment").value;c.done=true;store();say("");const a=D.fields.findIndex((f,i)=>i>st.i&&!st.data[f].done),b=D.fields.findIndex(f=>!st.data[f].done),to=a>=0?a:b;if(to>=0)showField(to);else{paint();say("All fields complete. Download your marks.")}}
const q=s=>{s=String(s??"");return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s};
function toCsv(){const L=["rater,field,kind,assessable_pct,complete,x_um,y_um,label,comment"];D.fields.forEach(f=>{const c=st.data[f];L.push([st.rater,f,"field",c.pct??"",c.done?1:0,"","","",q(c.comment)].join(","));c.marks.forEach(m=>L.push([st.rater,f,"point","","",m.x,m.y,m.l,""].join(",")))});return L.join("\n")+"\n"}
function parse(t){const R=[];let r=[],s="",Q=false;for(let i=0;i<t.length;i++){const c=t[i];if(Q){if(c==='"'&&t[i+1]==='"'){s+='"';i++}else if(c==='"')Q=false;else s+=c}else if(c==='"')Q=true;else if(c===","){r.push(s);s=""}else if(c==="\n"){r.push(s);R.push(r);r=[];s=""}else if(c!=="\r")s+=c}if(s||r.length){r.push(s);R.push(r)}return R}
function loadCsv(t){const R=parse(t),H=R.shift()||[],ix=n=>H.indexOf(n);if(["rater","field","kind","x_um","label"].some(n=>ix(n)<0))throw new Error("not a marks file");const data={};D.fields.forEach(f=>data[f]=blank());let who="";R.forEach(r=>{const f=r[ix("field")];if(!data[f])throw new Error("this file is from a different pack");who=r[ix("rater")]||who;if(r[ix("kind")]==="field"){data[f].pct=r[ix("assessable_pct")]===""?null:+r[ix("assessable_pct")];data[f].done=r[ix("complete")]==="1";data[f].comment=r[ix("comment")]||"";if(data[f].done)data[f].seen=Array(9).fill(true)}else data[f].marks.push({x:+r[ix("x_um")],y:+r[ix("y_um")],l:r[ix("label")]})});st.data=data;st.rater=who||st.rater;store();showField(st.i)}
$("begin").onclick=()=>{const w=$("who").value.trim().toUpperCase();if(!w){$("startmsg").textContent="Enter your initials first.";return}st.rater=w;store();$("start").style.display="none";size();showField(st.i)};
$("labels").onclick=e=>{const b=e.target.closest("[data-l]");if(b)setLabel(+b.dataset.l)};$("map").onclick=e=>{const b=e.target.closest("[data-s]");if(b)go(+b.dataset.s)};$("nextS").onclick=nextSection;$("whole").onclick=whole;$("del").onclick=removeMark;$("done").onclick=complete;$("prevF").onclick=()=>showField(st.i-1);$("nextF").onclick=()=>showField(st.i+1);
$("pct").onchange=()=>{const c=cur();c.pct=$("pct").value===""?null:+$("pct").value;reopen();store();paint()};$("comment").onchange=()=>{cur().comment=$("comment").value;store()};$("pct").addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();$("pct").blur();complete()}});document.addEventListener("click",e=>{const b=e.target.closest("button");if(b)b.blur()});
$("save").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([toCsv()],{type:"text/csv"}));a.download=`mitosis-marks-${st.rater||"reader"}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};$("loadB").onclick=()=>$("load").click();$("load").onchange=async e=>{const f=e.target.files[0];if(!f)return;try{loadCsv(await f.text());say("Marks loaded.")}catch(x){say("Could not load: "+x.message)}e.target.value=""};
addEventListener("keydown",e=>{if(/INPUT|TEXTAREA/.test(document.activeElement.tagName)||$("start").style.display!=="none")return;const k=e.key;if(k>="1"&&k<="7")setLabel(+k-1);else if(k==="n"||k==="N"||k===" "){e.preventDefault();nextSection()}else if(k==="f"||k==="F")whole();else if(k==="Backspace"||k==="Delete"){e.preventDefault();removeMark()}else if(k==="Enter")complete();else if(k==="Escape"){selected=null;draw()}else if(k==="ArrowRight")showField(st.i+1);else if(k==="ArrowLeft")showField(st.i-1)});
addEventListener("resize",()=>{size();if(section>=0)go(section);else whole()});if(st.rater){$("who").value=st.rater}paint();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="draw patients and random tissue fields, with no model")
    d.add_argument("--manifest", required=True)
    d.add_argument("--bags", required=True, help="used only for where the tissue is")
    d.add_argument("--slides", required=True)
    d.add_argument("--exclude", action="append", default=[], help="CSV of patients already used or reserved (repeat)")
    d.add_argument("--out", required=True)
    d.add_argument("--per-category", nargs="*", default=["gbm=6", "astro4=3", "oligo3=4", "astro3=4", "grade2=3"])
    d.add_argument("--top-up", default="gbm", help="category that fills any shortfall")
    d.add_argument("--fields", type=int, default=4)
    d.add_argument("--seed", type=int, default=20261125)
    d.set_defaults(fn=draw)
    b = sub.add_parser("build", help="write the blinded review packs")
    b.add_argument("--bags", required=True)
    b.add_argument("--slides", required=True)
    b.add_argument("--exclude", action="append", default=[], help="the same files given to draw (repeat)")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int)
    b.add_argument("--seed", type=int, default=20261125)
    b.set_defaults(fn=build)
    k = sub.add_parser("lock", help="fix the first reader's complete marks as the reference")
    k.add_argument("--key", required=True, help="private/key.csv")
    k.add_argument("--marks", required=True, help="mitosis-marks CSV from the review page")
    k.add_argument("--out", required=True, help="the private folder")
    k.set_defaults(fn=lock)
    g = sub.add_parser("agree", help="compare a second reader with the locked marks")
    g.add_argument("--locked", required=True, help="the private folder")
    g.add_argument("--marks", required=True, help="the second reader's marks CSV")
    g.set_defaults(fn=agree)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
