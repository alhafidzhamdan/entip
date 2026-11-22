"""Step 3b: a fresh, blinded whole-slide necrosis review, with no model involved.

    python slide_review.py draw --manifest trial.csv --bags bags/ --out necrosis-fresh2/private
    python slide_review.py build --patients necrosis-fresh2/private/patients.csv --bags bags/ --slides slides/ \
        --expect 24 --out necrosis-fresh2
    python slide_review.py lock --key necrosis-fresh2/private/key.csv --calls necrosis-calls-AH.csv \
        --out necrosis-fresh2/private

**draw** picks one slide per patient at random, with no model, and saves nothing unless every category is filled.

**build** gives each slide a random case code and writes ``review-pack.tar``: ``review.html`` and image tiles of the
tissue, with no slide name, patient, grade, metadata, label image or model output. The link from case to slide stays
in ``private/key.csv``. It refuses a patient named in any ``--exclude`` file.

**The review.** Pan and zoom each whole slide; call necrosis present, absent or can't tell; shift-click one clear
example if present. Calls stay in the browser and download as a CSV.

**lock** runs once, when every case has a call and every present call has an example. It fixes the calls as the slide
truth for this run, before any outlining or training, and reports only how many slides fall under each call.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
sys.path.insert(0, str(HERE.parent / "highgrade_mvp"))
from common import find_slide, pick_slides, quotas, read_manifest, read_patients  # noqa: E402
from slide_pilot import block_keys  # noqa: E402

CALLS = ("present", "absent", "cant")


def viewer(data: dict) -> str:
    """The necrosis page is its own file, so changes to the MVP viewer cannot alter this review."""
    page = (HERE / "review_viewer.html").read_text(encoding="utf-8")
    if page.count("__DATA__") != 1 or "MVP" in page:
        raise SystemExit("review_viewer.html is not a valid necrosis viewer")
    return page.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def pack_case(case: str, row: dict, bags: str, slides: str, tiles: str, mpp: float, block_px: int) -> tuple:
    """Writes one slide's image pyramid. Returns what the page may see, and what stays private."""
    from PIL import Image
    from entip_glioma.wsi.bags import Bag
    from entip_train.slides import open_slide
    bag = Bag.load(Path(bags) / f"{row['slide']}.npz")
    if len(bag) == 0:
        raise SystemExit(f"{case}: the slide has no tissue tiles")
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
               "native_mpp": float(rd.mpp), "n_tiles": len(bag)}
    return public, private


def draw(a) -> int:
    path = Path(a.out) / "patients.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists: a draw is never redone")
    want = quotas(a.per_category)
    picked = pick_slides(read_manifest(a.manifest), want, a.bags, a.seed, read_patients(a.exclude))
    got = Counter(r["category"] for r in picked)
    short = {c: f"{got[c]}/{n}" for c, n in want.items() if got[c] < n}
    if short:
        raise SystemExit(f"not enough patients, nothing saved: {short}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "slide", "category", "diagnosis"])
        for r in sorted(picked, key=lambda r: r["PatientID"]):
            w.writerow([r["PatientID"], r["SeriesInstanceUID"], r["category"], r.get("diagnosis", "")])
    print(json.dumps({"drawn": len(picked), "by_category": dict(got)}, indent=2))
    return 0


def build(a) -> int:
    out = Path(a.out)
    private = out / "private"
    if (private / "key.csv").exists():
        raise SystemExit(f"{private / 'key.csv'} already exists: a built review is never overwritten")
    rows = list(csv.DictReader(open(a.patients, newline="")))
    if not rows or len({r["patient"] for r in rows}) != len(rows):
        raise SystemExit("the patient list must have one slide per patient")
    if a.expect is not None and len(rows) != a.expect:
        raise SystemExit(f"{len(rows)} patients drawn, {a.expect} expected: check the draw before reviewing")
    clash = read_patients(a.exclude) & {r["patient"] for r in rows}
    if clash:
        raise SystemExit(f"{len(clash)} patients in the list were used in earlier work")
    random.Random(f"{a.seed}-cases").shuffle(rows)
    private.mkdir(parents=True, exist_ok=True)
    pack = out / "review-pack"
    (pack / "tiles").mkdir(parents=True, exist_ok=True)
    cases = [(f"C{n:03d}", r, a.bags, a.slides, str(pack / "tiles"), a.mpp, a.block_px)
             for n, r in enumerate(rows, start=1)]
    workers = max(1, a.workers or int(os.environ.get("NSLOTS", 1)))
    done = []
    if workers == 1:
        for n, c in enumerate(cases, start=1):
            done.append(pack_case(*c))
            print(f"packed {n}/{len(cases)} slides", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(workers) as pool:
            for n, job in enumerate(as_completed([pool.submit(pack_case, *c) for c in cases]), start=1):
                done.append(job.result())
                print(f"packed {n}/{len(cases)} slides", file=sys.stderr, flush=True)
    public = sorted((d[0] for d in done), key=lambda c: c["case"])
    secret = sorted((d[1] for d in done), key=lambda c: c["case"])
    sha = hashlib.sha256(f"{a.seed}|{'|'.join(s['slide'] for s in secret)}".encode()).hexdigest()
    data = {"study": f"entip-necrosis-review-{len(public)}-{sha[:12]}", "mpp": a.mpp, "blockUm": a.block_px * a.mpp,
            "cases": public}
    (pack / "review.html").write_text(viewer(data), encoding="utf-8")
    with (private / "key.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(secret[0]))
        w.writeheader()
        w.writerows(secret)
    with tarfile.open(out / "review-pack.tar", "w") as tar:
        tar.add(pack, arcname="review-pack")
    print(json.dumps({"slides": len(public), "pack_gb": round((out / "review-pack.tar").stat().st_size / 1e9, 2),
                      "pack": str(out / "review-pack.tar")}, indent=2))
    return 0


def lock(a) -> int:
    out = Path(a.out)
    path = out / "locked-calls.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists: calls are locked once")
    key = {r["case"]: r for r in csv.DictReader(open(a.key, newline=""))}
    calls = {r["case"]: r for r in csv.DictReader(open(a.calls, newline=""))}
    if set(key) != set(calls):
        raise SystemExit("the calls do not cover the same cases as the key")
    raters = {r["rater"].strip().upper() for r in calls.values()}
    if len(raters) != 1 or "" in raters:
        raise SystemExit("one calls file must come from one named rater")
    missing = sorted(c for c, r in calls.items() if r["call"] not in CALLS)
    if missing:
        raise SystemExit(f"{len(missing)} slides have no call: {', '.join(missing)}")
    no_example = sorted(c for c, r in calls.items() if r["call"] == "present" and not (
        r.get("example_x_um") and r.get("example_y_um")))
    if no_example:
        raise SystemExit(f"present without a clicked example: {', '.join(no_example)}")
    rows = []
    for c in sorted(key):
        r, k = calls[c], key[c]
        x, y = "", ""
        if r["call"] == "present":
            x = round(float(k["origin_x_um"]) + float(r["example_x_um"]), 2)
            y = round(float(k["origin_y_um"]) + float(r["example_y_um"]), 2)
        rows.append({"case": c, "patient": k["patient"], "slide": k["slide"], "rater": r["rater"].strip().upper(),
                     "call": r["call"], "example_slide_x_um": x, "example_slide_y_um": y,
                     "comment": r.get("comment", "")})
    out.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    info = {"locked": len(rows), "calls": dict(Counter(r["call"] for r in rows)), "rater": rows[0]["rater"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "lock.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="pick the patients at random, with no model")
    d.add_argument("--manifest", required=True)
    d.add_argument("--bags", required=True)
    d.add_argument("--exclude", action="append", default=[], help="CSV of patients to leave out (repeat)")
    d.add_argument("--per-category", nargs="*", default=["gbm=8", "astro4=6", "oligo3=4", "astro3=3", "grade2=3"])
    d.add_argument("--seed", type=int, default=20261123)
    d.add_argument("--out", required=True, help="the private folder")
    d.set_defaults(fn=draw)
    b = sub.add_parser("build", help="write the blinded whole-slide review pack")
    b.add_argument("--patients", required=True, help="patients.csv of the fresh draw")
    b.add_argument("--exclude", action="append", default=[], help="CSV of patients used in earlier work (repeat)")
    b.add_argument("--bags", required=True, help="used only for where the tissue is")
    b.add_argument("--slides", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--expect", type=int, help="stop unless the draw has this many patients")
    b.add_argument("--mpp", type=float, default=0.5, help="µm per pixel of the finest review level")
    b.add_argument("--block-px", type=int, default=896)
    b.add_argument("--workers", type=int)
    b.add_argument("--seed", type=int, default=20261124)
    b.set_defaults(fn=build)
    k = sub.add_parser("lock", help="fix the completed calls as this run's slide truth")
    k.add_argument("--key", required=True, help="private/key.csv")
    k.add_argument("--calls", required=True, help="necrosis-calls CSV from the review page")
    k.add_argument("--out", required=True, help="the private folder")
    k.set_defaults(fn=lock)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
