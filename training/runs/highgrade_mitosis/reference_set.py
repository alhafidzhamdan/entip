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
look-alikes worth recording, including prophase, which is not counted. Each mark can carry ticked details: a phase,
atypical and granular for a mitosis, one or more types for a look-alike, and image flags for any mark. A field with no
mitosis must be ticked as having none. A figure counts if its centre is inside the field. Give the share of the field
that could be assessed.

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
LABELS = ("mitosis", "uncertain", "look_alike")
PHASES = ("prometaphase", "metaphase", "anaphase", "telophase")
FEATURES = {"mitosis": PHASES + ("atypical", "granular"),
            "look_alike": ("prophase", "pyknosis", "apoptosis", "karyorrhexis", "lymphocyte", "endothelial", "other"),
            "any": ("blurred", "at_edge")}
NAMES = {"look_alike": "Look-alike", "other": "Other mimic", "at_edge": "Hard to judge at the edge",
         "endothelial": "Endothelial or reactive"}


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
            "phases": list(PHASES), "features": {k: list(v) for k, v in FEATURES.items()}, "names": NAMES,
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


def read_features(r: dict, label: str) -> tuple:
    """A mark's ticked details. Each must suit the mark type, with one phase at most."""
    got = tuple(sorted({x for x in (r.get("features") or "").split(";") if x}))
    bad = [x for x in got if x not in FEATURES.get(label, ()) + FEATURES["any"]]
    if bad:
        raise SystemExit(f"{r['field']}: {', '.join(bad)} cannot be ticked on a {label} mark")
    if sum(x in PHASES for x in got) > 1:
        raise SystemExit(f"{r['field']}: a mitosis has one phase at most")
    return got


def read_marks(path: str, want: set) -> tuple:
    """One reader's complete marks: (rater, field -> row, field -> [(x, y, label, features)]). Refuses anything
    incomplete, and a field with no mitosis unless the reader ticked that it has none."""
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
            points[r["field"]].append((x, y, lab, read_features(r, lab)))
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
        none, mitoses = r.get("no_mitoses") == "1", sum(p[2] == "mitosis" for p in points.get(f, []))
        if mitoses and none:
            raise SystemExit(f"{f}: marked mitoses but also ticked that it has none")
        if not mitoses and not none:
            raise SystemExit(f"{f}: no mitosis marked and no-mitoses not ticked")
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
                      "no_mitoses": int(r.get("no_mitoses") == "1"), "comment": r.get("comment", "")})
        for x, y, lab, feats in pts:
            prows.append({"field": f, "patient": k["patient"], "slide": k["slide"], "rater": rater,
                          "field_x_um": round(x, 2), "field_y_um": round(y, 2),
                          "slide_x_um": round(float(k["x_um"]) + x, 2), "slide_y_um": round(float(k["y_um"]) + y, 2),
                          "label": lab, "features": ";".join(feats)})
    write_csv(out / "locked-fields.csv", frows)
    with (out / "locked-points.csv").open("w", newline="") as fh:
        names = ["field", "patient", "slide", "rater", "field_x_um", "field_y_um", "slide_x_um", "slide_y_um", "label",
                 "features"]
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(prows)
    labels = Counter(p["label"] for p in prows)
    info = {"rater": rater, "fields": len(frows), "assessable_mm2": round(sum(r["assessable_mm2"] for r in frows), 3),
            "marks": {k: labels.get(k, 0) for k in LABELS},
            "mitosis_details": dict(Counter(x for p in prows if p["label"] == "mitosis" for x in p["features"].split(";") if x)),
            "fields_with_no_mitoses": sum(r["no_mitoses"] for r in frows), "target_mitoses": 200,
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


VIEWER = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Mitosis field count</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:light;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202124;--go:#1f6f54;--line:#e3e3de;--muted:#66665e}
*{box-sizing:border-box}body{margin:0;height:100vh;display:flex;background:#f4f4f2;overflow:hidden}
#view{flex:1;min-width:0;position:relative;background:#30302e}canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
#hud{position:absolute;left:12px;top:12px;display:flex;gap:6px;flex-wrap:wrap;pointer-events:none}#hud span{background:rgba(20,20,20,.72);color:#fff;border-radius:5px;padding:3px 8px;font-size:12px}
#zoom{position:absolute;right:12px;bottom:12px;display:flex;gap:4px}#zoom button{background:rgba(255,255,255,.92);min-width:34px}
aside{width:340px;flex:none;overflow:auto;background:#fff;border-left:1px solid #d8d8d3;padding:14px 14px 28px}
h1{font-size:17px;font-weight:600;margin:0}h2{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 6px}
.muted{color:var(--muted);font-size:12px}section{border-top:1px solid var(--line);margin-top:12px;padding-top:11px}
button{font:inherit;border:1px solid #b9b9b1;background:#fff;border-radius:5px;padding:5px 8px;cursor:pointer;color:inherit}button:hover{background:#f0f0ec}button.go{background:var(--go);border-color:var(--go);color:#fff}button.go:hover{background:#185a44}button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.lab{display:flex;align-items:center;width:100%;gap:8px;text-align:left;margin-bottom:4px;padding:6px 8px}.lab.on{outline:2px solid var(--go);background:#e9f4ee}
.dot{width:18px;height:18px;border-radius:50%;color:#fff;font-size:11px;font-weight:700;display:inline-flex;align-items:center;justify-content:center;flex:none}
kbd{font:11px ui-monospace,monospace;background:#efefea;border:1px solid #cfcfc8;border-radius:3px;padding:0 4px;margin-left:auto;color:#333}
.grid{display:grid;gap:3px}#map{grid-template-columns:repeat(3,38px)}#map button,#fields button{height:30px;padding:0}#fields{grid-template-columns:repeat(8,1fr)}
.seen{background:#dcefe6!important;border-color:#65a88c!important}.cur{outline:2px solid var(--go)}
.chk{display:flex;align-items:center;gap:7px;padding:3px 2px;cursor:pointer}.chk input{margin:0;width:15px;height:15px;accent-color:var(--go)}.grp{margin:6px 0 4px;font-size:12px;color:var(--muted)}
.mk{display:flex;align-items:center;gap:7px;width:100%;text-align:left;padding:4px 6px;margin-bottom:3px;font-size:13px}.mk.on{outline:2px solid var(--go);background:#e9f4ee}
#count{display:flex;gap:10px;font-size:13px;margin-bottom:6px}#count b{font-size:16px}
.none{display:flex;gap:8px;align-items:flex-start;background:#f6f6f3;border:1px solid var(--line);border-radius:6px;padding:8px;cursor:pointer}.none input{margin-top:3px;width:16px;height:16px;accent-color:var(--go)}
input,textarea{font:inherit;border:1px solid #b9b9b1;border-radius:4px;padding:4px}textarea{width:100%;height:44px;resize:vertical}
#msg{min-height:18px;color:#a3341f;font-size:12px;margin-top:6px}#msg.ok{color:var(--go)}
#start{position:absolute;inset:0;z-index:5;background:#f4f4f2;display:flex;align-items:center;justify-content:center;padding:24px;overflow:auto}#start>div{max-width:620px;background:#fff;border:1px solid #d8d8d3;border-radius:8px;padding:22px}#start p{margin:8px 0}#start ul{margin:6px 0;padding-left:20px}
</style></head><body><div id="start"><div><h1>Mitosis field count</h1>
<p>Mark <b>every</b> mitotic figure whose centre lies inside the white square, and search all nine sections of each field. No diagnosis, patient or model output is shown.</p>
<ul><li><b>Mitosis</b> <kbd>1</kbd>: a definite figure with no nuclear membrane: prometaphase, metaphase, anaphase, telophase, atypical or granular. Count a telophase pair once.</li>
<li><b>Uncertain</b> <kbd>2</kbd>: you cannot decide.</li>
<li><b>Look-alike</b> <kbd>3</kbd>: not a mitosis but could be mistaken for one, including <b>prophase</b>, which is not counted. Look-alikes are optional, but useful.</li></ul>
<p>After placing a mark, tick its details on the right. You can tick several: a phase, atypical and granular; several look-alike types; and image flags.</p>
<p>If a field has no mitoses, tick <b>No mitoses in this field</b> before completing it, so a zero is recorded on purpose.</p>
<p class="row"><label>Your initials <input id="who" maxlength="6" size="8" autocomplete="off"></label><button class="go" id="begin">Start</button></p><p class="muted">Marks save in this browser. Download them often, and load the file to continue elsewhere.</p><div id="startmsg" style="color:#a3341f"></div></div></div>
<main id="view"><canvas id="cv"></canvas><div id="hud"></div><div id="zoom"><button id="zin" title="Zoom in (+)">+</button><button id="zout" title="Zoom out (−)">−</button><button id="hide" title="Hide marks (H)">Hide marks</button></div></main>
<aside><div class="row" style="justify-content:space-between"><h1 id="title"></h1><span class="muted" id="progress"></span></div>
<section style="margin-top:8px"><h2>Fields</h2><div id="fields" class="grid"></div></section>
<section><h2>Sections <span class="muted" style="text-transform:none;letter-spacing:0">view all nine</span></h2><div class="row" style="align-items:flex-start"><div id="map" class="grid"></div><div><button id="nextS">Next section <kbd>N</kbd></button><br><button id="whole" style="margin-top:4px">Whole field <kbd>F</kbd></button></div></div></section>
<section><h2>New mark</h2><div id="labels"></div><div class="muted">Click the image to add. Click a mark to select it, then press a number to change its type.</div></section>
<section><h2>Selected mark</h2><div id="feats"></div><div class="row" style="margin-top:6px"><button id="del">Delete <kbd>⌫</kbd></button><button id="undo">Undo <kbd>⌘Z</kbd></button></div></section>
<section><h2>This field</h2><div id="count"></div><div id="marks"></div>
<label class="none" style="margin-top:6px"><input type="checkbox" id="none"><span><b>No mitoses in this field</b><br><span class="muted">I searched all nine sections and found none.</span></span></label></section>
<section><h2>Finish field</h2><label><b>Assessable share</b> <input id="pct" type="number" min="0" max="100" step="5" style="width:68px"> %</label><div class="muted">Part of the square that can be assessed, after folds, blur or glass.</div><div style="margin-top:6px"><textarea id="comment" placeholder="Comment (optional)"></textarea></div>
<div class="row" style="margin-top:6px"><button class="go" id="done">Field complete <kbd>↵</kbd></button></div><div id="msg"></div></section>
<section><div class="row"><button id="save">Download marks</button><button id="loadB" type="button">Load marks</button><input id="load" type="file" accept=".csv" hidden></div></section></aside>
<script>
const D=__DATA__,KEY="entip-"+D.study+"-v2",cv=document.getElementById("cv"),ctx=cv.getContext("2d"),$=id=>document.getElementById(id);
const COL={mitosis:"#14873d",uncertain:"#d97a00",look_alike:"#7c4dcc"},LET={mitosis:"M",uncertain:"U",look_alike:"L"},PH=D.phases,ORDER=[0,1,2,5,4,3,6,7,8];
const SHORT={granular:"G",atypical:"A"},nice=s=>D.names[s]||s.replace(/_/g," ").replace(/^./,c=>c.toUpperCase());
const allowed=l=>(D.features[l]||[]).concat(D.features.any);
const blank=()=>({marks:[],seen:Array(9).fill(false),pct:null,comment:"",done:false,none:false});
let st={rater:"",i:0,active:0,data:{}};try{const o=JSON.parse(localStorage.getItem(KEY)||"null");if(o&&o.data)st=o}catch(e){}
D.fields.forEach(f=>{if(!st.data[f])st.data[f]=blank()});
let v={s:1,x:0,y:0},img=new Image(),selected=null,section=-1,drag=null,hidden=false,undo=[];
const cur=()=>st.data[D.fields[st.i]],store=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){say("This browser could not save: download your marks now.")}};
const say=(t,ok)=>{$("msg").textContent=t||"";$("msg").className=ok?"ok":""};
const tally=c=>{const n={mitosis:0,uncertain:0,look_alike:0};c.marks.forEach(m=>n[m.l]++);return n};
function size(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=Math.round(r.width*d);cv.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0)}
function fitRect(x,y,w,h){const r=cv.getBoundingClientRect();if(!r.width||!r.height)return;v.s=Math.min(r.width/w,r.height/h);v.x=x+w/2-r.width/2/v.s;v.y=y+h/2-r.height/2/v.s;draw()}
function whole(){section=-1;fitRect(0,0,D.imageUm,D.imageUm);paint()}
function go(k){section=k;const q=D.fieldUm/3,c=k%3,r=Math.floor(k/3),o=8;cur().seen[k]=true;store();fitRect(D.marginUm+c*q-o,D.marginUm+r*q-o,q+2*o,q+2*o);paint()}
function nextSection(){const a=section<0?-1:ORDER.indexOf(section);go(ORDER[(a+1)%9])}
function zoom(f,p){const r=cv.getBoundingClientRect();p=p||{sx:r.width/2,sy:r.height/2};const ux=p.sx/v.s+v.x,uy=p.sy/v.s+v.y,min=Math.min(r.width,r.height)/D.imageUm*.7;v.s=Math.max(min,Math.min(14,v.s*f));v.x=ux-p.sx/v.s;v.y=uy-p.sy/v.s;draw()}
function snap(){undo.push(JSON.stringify(cur()));if(undo.length>80)undo.shift()}
function changed(){cur().done=false;store();paint();draw()}
function draw(){const r=cv.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);ctx.fillStyle="#30302e";ctx.fillRect(0,0,r.width,r.height);if(img.complete&&img.naturalWidth){ctx.imageSmoothingEnabled=v.s<4;ctx.drawImage(img,-v.x*v.s,-v.y*v.s,D.imageUm*v.s,D.imageUm*v.s)}
const X=u=>(u+D.marginUm-v.x)*v.s,Y=u=>(u+D.marginUm-v.y)*v.s,F=D.fieldUm*v.s,x0=X(0),y0=Y(0);ctx.fillStyle="rgba(20,20,20,.48)";ctx.fillRect(0,0,r.width,Math.max(0,y0));ctx.fillRect(0,y0+F,r.width,Math.max(0,r.height-y0-F));ctx.fillRect(0,y0,Math.max(0,x0),F);ctx.fillRect(x0+F,y0,Math.max(0,r.width-x0-F),F);
ctx.lineWidth=2;ctx.strokeStyle="#fff";ctx.strokeRect(x0,y0,F,F);ctx.lineWidth=1;ctx.strokeStyle="rgba(255,255,255,.35)";for(let i=1;i<3;i++){ctx.beginPath();ctx.moveTo(x0+i*F/3,y0);ctx.lineTo(x0+i*F/3,y0+F);ctx.moveTo(x0,y0+i*F/3);ctx.lineTo(x0+F,y0+i*F/3);ctx.stroke()}
if(section>=0){const q=F/3;ctx.lineWidth=2;ctx.strokeStyle="rgba(255,255,255,.8)";ctx.setLineDash([6,5]);ctx.strokeRect(x0+(section%3)*q,y0+Math.floor(section/3)*q,q,q);ctx.setLineDash([])}
if(hidden)return;cur().marks.forEach((m,i)=>{const px=X(m.x),py=Y(m.y),c=COL[m.l],R=Math.max(9,5.5*v.s),on=i===selected;ctx.beginPath();ctx.arc(px,py,R,0,7);if(on){ctx.lineWidth=7;ctx.strokeStyle="rgba(255,255,255,.9)";ctx.stroke()}ctx.lineWidth=on?3.5:2.2;ctx.strokeStyle=c;ctx.stroke();
const t=LET[m.l]+m.f.filter(x=>SHORT[x]).map(x=>SHORT[x]).join("");ctx.font="700 12px sans-serif";const w=ctx.measureText(t).width+8;ctx.fillStyle=c;ctx.fillRect(px+R*.7,py-R-14,w,16);ctx.fillStyle="#fff";ctx.fillText(t,px+R*.7+4,py-R-2)})}
function paint(){const c=cur(),n=D.fields.filter(f=>st.data[f].done).length,t=tally(c);
$("title").textContent=`Field ${st.i+1} of ${D.fields.length}`;$("progress").textContent=`${n} of ${D.fields.length} complete · ${st.rater||"-"}`;
$("hud").innerHTML=`<span>${section<0?"Whole field":"Section "+(ORDER.indexOf(section)+1)+" of 9"}</span><span>${c.seen.filter(Boolean).length}/9 sections viewed</span>${c.done?"<span>Complete</span>":""}${hidden?"<span>Marks hidden</span>":""}`;
$("fields").innerHTML=D.fields.map((f,i)=>`<button class="${st.data[f].done?"seen":""} ${i===st.i?"cur":""}" data-i="${i}" title="${f}">${i+1}</button>`).join("");
$("map").innerHTML=Array.from({length:9},(_,k)=>`<button class="${c.seen[k]?"seen":""} ${k===section?"cur":""}" data-s="${k}">${ORDER.indexOf(k)+1}</button>`).join("");
$("labels").innerHTML=D.labels.map((l,i)=>`<button class="lab ${i===st.active?"on":""}" data-l="${i}"><span class="dot" style="background:${COL[l]}">${LET[l]}</span>${nice(l)}<kbd>${i+1}</kbd></button>`).join("");
const m=selected!==null?c.marks[selected]:null;
if(!m)$("feats").innerHTML='<div class="muted">Select a mark to tick its details.</div>';else{const box=(x,k)=>`<label class="chk"><input type="checkbox" data-f="${x}" ${m.f.includes(x)?"checked":""}>${nice(x)}${k?`<kbd>${k}</kbd>`:""}</label>`;
let h=`<div class="row" style="margin-bottom:4px"><span class="dot" style="background:${COL[m.l]}">${LET[m.l]}</span><b>${nice(m.l)}</b><span class="muted">mark ${selected+1}</span></div>`;
if(m.l==="mitosis")h+=`<div class="grp">Phase, one at most</div>${PH.map(x=>box(x)).join("")}<div class="grp">Features</div>${box("atypical","A")}${box("granular","G")}`;
else if(m.l==="look_alike")h+=`<div class="grp">Type, tick all that apply</div>${D.features.look_alike.map(x=>box(x)).join("")}`;
h+=`<div class="grp">Image</div>${D.features.any.map(x=>box(x)).join("")}`;$("feats").innerHTML=h}
$("del").disabled=!m;
$("count").innerHTML=`<span><b>${t.mitosis}</b> mitoses</span><span><b>${t.uncertain}</b> uncertain</span><span><b>${t.look_alike}</b> look-alikes</span>`;
$("marks").innerHTML=c.marks.map((m,i)=>`<button class="mk ${i===selected?"on":""}" data-m="${i}"><span class="dot" style="background:${COL[m.l]}">${LET[m.l]}</span><span>${nice(m.l)}${m.f.length?` <span class="muted">· ${m.f.map(nice).join(", ")}</span>`:""}</span></button>`).join("");
$("none").checked=!!c.none;$("none").disabled=t.mitosis>0;$("pct").value=c.pct??"";$("comment").value=c.comment||"";$("done").innerHTML=c.done?"Complete ✓":"Field complete <kbd>↵</kbd>";$("hide").textContent=hidden?"Show marks":"Hide marks"}
function showField(i){st.i=Math.max(0,Math.min(D.fields.length-1,i));selected=null;section=-1;undo=[];say("");store();paint();img=new Image();img.onload=()=>whole();img.onerror=()=>say("The field image did not load: keep review.html beside the fields folder.");img.src=`fields/${D.fields[st.i]}.jpg`;whole()}
function point(e){const r=cv.getBoundingClientRect();return{sx:e.clientX-r.left,sy:e.clientY-r.top}}
function clickAt(p){const c=cur(),fx=p.sx/v.s+v.x-D.marginUm,fy=p.sy/v.s+v.y-D.marginUm;let best=-1,bd=1e9;if(!hidden)c.marks.forEach((m,i)=>{const d=Math.hypot((m.x-fx)*v.s,(m.y-fy)*v.s);if(d<Math.max(11,5.5*v.s)&&d<bd){bd=d;best=i}});
if(best>=0){selected=best;paint();draw();return}if(fx<0||fy<0||fx>D.fieldUm||fy>D.fieldUm){say("Marks go inside the white square.");return}
snap();const l=D.labels[st.active];c.marks.push({x:+fx.toFixed(2),y:+fy.toFixed(2),l,f:[]});if(l==="mitosis")c.none=false;selected=c.marks.length-1;say("");changed()}
cv.addEventListener("pointerdown",e=>{cv.setPointerCapture(e.pointerId);const p=point(e);drag={...p,x:v.x,y:v.y,moved:false}});cv.addEventListener("pointermove",e=>{if(!drag)return;const p=point(e),dx=p.sx-drag.sx,dy=p.sy-drag.sy;if(Math.hypot(dx,dy)>4)drag.moved=true;if(drag.moved){v.x=drag.x-dx/v.s;v.y=drag.y-dy/v.s;draw()}});cv.addEventListener("pointerup",e=>{if(drag&&!drag.moved)clickAt(point(e));drag=null});
cv.addEventListener("wheel",e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0016),point(e))},{passive:false});
function setLabel(i){const c=cur(),m=selected!==null?c.marks[selected]:null;if(m){snap();m.l=D.labels[i];m.f=m.f.filter(x=>allowed(m.l).includes(x));if(m.l==="mitosis")c.none=false;st.active=i;changed()}else{st.active=i;store();paint()}}
function toggle(x){const m=selected!==null?cur().marks[selected]:null;if(!m){say("Select a mark first.");return}if(!allowed(m.l).includes(x)){say(`${nice(x)} does not apply to a ${nice(m.l).toLowerCase()}.`);return}snap();if(m.f.includes(x))m.f=m.f.filter(y=>y!==x);else{if(PH.includes(x))m.f=m.f.filter(y=>!PH.includes(y));m.f.push(x)}say("");changed()}
function removeMark(){if(selected===null)return;snap();cur().marks.splice(selected,1);selected=null;changed()}
function doUndo(){if(!undo.length){say("Nothing to undo in this field.");return}st.data[D.fields[st.i]]=JSON.parse(undo.pop());if(selected!==null&&!cur().marks[selected])selected=null;say("Undone.",1);store();paint();draw()}
function complete(){const c=cur(),left=c.seen.filter(x=>!x).length,p=$("pct").value===""?NaN:+$("pct").value,t=tally(c);if(left){say(`View the remaining ${left} section${left>1?"s":""} first.`);return}
if(!(p>=0&&p<=100)){say("Enter the assessable share, 0 to 100.");return}if(!t.mitosis&&!c.none){say("No mitoses marked: tick “No mitoses in this field” to confirm a zero.");return}
c.pct=p;c.comment=$("comment").value;c.done=true;store();const a=D.fields.findIndex((f,i)=>i>st.i&&!st.data[f].done),b=D.fields.findIndex(f=>!st.data[f].done),to=a>=0?a:b;if(to>=0){showField(to);say(`Field ${D.fields.indexOf(D.fields[to])+1} opened.`,1)}else{paint();say("All fields complete. Download your marks.",1)}}
const q=s=>{s=String(s??"");return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s};
function toCsv(){const L=["rater,field,kind,assessable_pct,complete,no_mitoses,x_um,y_um,label,features,comment"];D.fields.forEach(f=>{const c=st.data[f];L.push([st.rater,f,"field",c.pct??"",c.done?1:0,c.none?1:0,"","","","",q(c.comment)].join(","));c.marks.forEach(m=>L.push([st.rater,f,"point","","","",m.x,m.y,m.l,m.f.join(";"),""].join(",")))});return L.join("\n")+"\n"}
function parse(t){const R=[];let r=[],s="",Q=false;for(let i=0;i<t.length;i++){const c=t[i];if(Q){if(c==='"'&&t[i+1]==='"'){s+='"';i++}else if(c==='"')Q=false;else s+=c}else if(c==='"')Q=true;else if(c===","){r.push(s);s=""}else if(c==="\n"){r.push(s);R.push(r);r=[];s=""}else if(c!=="\r")s+=c}if(s||r.length){r.push(s);R.push(r)}return R}
function loadCsv(t){const R=parse(t),H=R.shift()||[],ix=n=>H.indexOf(n),get=(r,n)=>ix(n)<0?"":(r[ix(n)]||"");if(["rater","field","kind","x_um","label"].some(n=>ix(n)<0))throw new Error("not a marks file");const data={};D.fields.forEach(f=>data[f]=blank());let who="";
R.forEach(r=>{if(r.length<2)return;const f=get(r,"field");if(!data[f])throw new Error("this file is from a different pack");who=get(r,"rater")||who;if(get(r,"kind")==="field"){data[f].pct=get(r,"assessable_pct")===""?null:+get(r,"assessable_pct");data[f].done=get(r,"complete")==="1";data[f].none=get(r,"no_mitoses")==="1";data[f].comment=get(r,"comment");if(data[f].done)data[f].seen=Array(9).fill(true)}
else{const l=get(r,"label");if(!D.labels.includes(l))throw new Error(`unknown mark type ${l}`);data[f].marks.push({x:+get(r,"x_um"),y:+get(r,"y_um"),l,f:get(r,"features").split(";").filter(x=>x&&allowed(l).includes(x))})}});st.data=data;st.rater=who||st.rater;store();showField(st.i)}
$("begin").onclick=()=>{const w=$("who").value.trim().toUpperCase();if(!w){$("startmsg").textContent="Enter your initials first.";return}st.rater=w;store();$("start").style.display="none";size();showField(st.i)};
$("labels").onclick=e=>{const b=e.target.closest("[data-l]");if(b)setLabel(+b.dataset.l)};$("map").onclick=e=>{const b=e.target.closest("[data-s]");if(b)go(+b.dataset.s)};$("fields").onclick=e=>{const b=e.target.closest("[data-i]");if(b)showField(+b.dataset.i)};
$("marks").onclick=e=>{const b=e.target.closest("[data-m]");if(!b)return;selected=+b.dataset.m;const m=cur().marks[selected],r=cv.getBoundingClientRect();v.x=m.x+D.marginUm-r.width/2/v.s;v.y=m.y+D.marginUm-r.height/2/v.s;paint();draw()};
$("feats").addEventListener("change",e=>{const b=e.target.closest("[data-f]");if(b)toggle(b.dataset.f)});
$("none").onchange=()=>{const c=cur();if(tally(c).mitosis){$("none").checked=false;return}c.none=$("none").checked;changed()};
$("nextS").onclick=nextSection;$("whole").onclick=whole;$("del").onclick=removeMark;$("undo").onclick=doUndo;$("done").onclick=complete;$("zin").onclick=()=>zoom(1.4);$("zout").onclick=()=>zoom(1/1.4);$("hide").onclick=()=>{hidden=!hidden;paint();draw()};
$("pct").onchange=()=>{cur().pct=$("pct").value===""?null:+$("pct").value;changed()};$("comment").onchange=()=>{cur().comment=$("comment").value;store()};$("pct").addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();$("pct").blur();complete()}});document.addEventListener("click",e=>{const b=e.target.closest("button");if(b)b.blur()});
$("save").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([toCsv()],{type:"text/csv"}));a.download=`mitosis-marks-${st.rater||"reader"}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);say("Marks downloaded.",1)};$("loadB").onclick=()=>$("load").click();$("load").onchange=async e=>{const f=e.target.files[0];if(!f)return;try{loadCsv(await f.text());say("Marks loaded.",1)}catch(x){say("Could not load: "+x.message)}e.target.value=""};
addEventListener("keydown",e=>{const el=document.activeElement;if((el.tagName==="INPUT"&&el.type!=="checkbox")||el.tagName==="TEXTAREA"||$("start").style.display!=="none")return;if(el.type==="checkbox")el.blur();const k=e.key;
if((e.metaKey||e.ctrlKey)&&(k==="z"||k==="Z")){e.preventDefault();doUndo();return}if(e.metaKey||e.ctrlKey||e.altKey)return;
if(k>="1"&&k<=String(D.labels.length))setLabel(+k-1);else if(k==="g"||k==="G")toggle("granular");else if(k==="a"||k==="A")toggle("atypical");else if(k==="h"||k==="H"){hidden=!hidden;paint();draw()}
else if(k==="+"||k==="=")zoom(1.4);else if(k==="-"||k==="_")zoom(1/1.4);else if(k==="n"||k==="N"||k===" "){e.preventDefault();nextSection()}else if(k==="f"||k==="F")whole();
else if(k==="Backspace"||k==="Delete"){e.preventDefault();removeMark()}else if(k==="Enter")complete();else if(k==="Escape"){selected=null;paint();draw()}else if(k==="ArrowRight")showField(st.i+1);else if(k==="ArrowLeft")showField(st.i-1)});
addEventListener("resize",()=>{size();if(section>=0)go(section);else whole()});if(st.rater)$("who").value=st.rater;paint();
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
