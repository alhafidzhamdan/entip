"""Step 3c: a small TCGA glioma review round, to teach the mitosis finder glioma.

    python glioma_review.py draw --manifest trial.csv --bags bags/ --slides slides/ --reference ref/private/patients.csv --out round1
    python glioma_review.py calls --finder mitosis-finder1 --slides slides/ --reference ref/private/patients.csv --out round1
    python glioma_review.py build --slides slides/ --reference ref/private/patients.csv --out round1
    python glioma_review.py page --out round1

**draw** picks 12 TCGA patients who are not in the sealed glioma reference set, one 40x slide each. Nine are
development patients, with one field to mark in full and three fields for the finder. Three are held out, with two
fields to mark in full, and are never used to train. They measure whether a retrained model has improved.

**calls** runs the frozen finder only on the development patients' finder fields. It never sees a field marked in full.

**build** writes two blinded packs. ``full-fields-pack.tar`` holds every field to mark in full, with no model output.
``calls-pack.tar`` holds 300 finder candidates in random order: confident, borderline and low scoring. Scores, patients
and slides stay in ``private/``.

**page** rewrites only the call review page and ``calls-pack.tar`` of a built round, for example after a label is added.
The keys, crops and order do not change, and labels already saved in the browser are kept.

Mitosis means prometaphase, metaphase, anaphase, telophase, atypical or granular. Prophase is not a mitosis.
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "highgrade_common"))
import reference_set as R  # noqa: E402
from common import quotas, read_manifest, read_patients  # noqa: E402

CALL_LABELS = ("mitosis", "granular", "prophase", "look_alike", "not_mitosis", "uncertain", "unusable")
CROP_UM, CROP_PX = 64.0, 256
HOLD_FROM = ("gbm", "oligo3", "astro3")


def reserved_patients(path: str) -> set:
    found = read_patients([path])
    if not found:
        raise SystemExit(f"{path} lists no patients: seal the reference set first")
    return found


def check_separate(rows: list, reserved: set) -> None:
    clash = {r["patient"] for r in rows} & reserved
    if clash:
        raise SystemExit(f"{len(clash)} patients are in the sealed reference set")


def draw(a) -> int:
    out = Path(a.out) / "private"
    if (out / "fields.csv").exists():
        raise SystemExit(f"{out / 'fields.csv'} already exists: a drawn round is never redrawn")
    reserved = reserved_patients(a.reference)
    want = quotas(a.per_category)
    by = defaultdict(lambda: defaultdict(list))
    for r in read_manifest(a.manifest):
        if r.get("label", "glioma") == "glioma" and r["PatientID"].startswith("TCGA-") \
                and r["PatientID"] not in reserved and (Path(a.bags) / f"{r['SeriesInstanceUID']}.npz").exists():
            by[r["category"]][r["PatientID"]].append(r)
    rng = random.Random(a.seed)
    queue = {}
    for cat in sorted(set(want) | {"gbm"}):
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
                coords, tile_um = R.tile_coords(a.bags, uid)
                fields = R.pick_fields(coords, tile_um, 4, random.Random(f"{a.seed}-{uid}"))
                if fields:
                    mpp = float(R.open_reader(a.slides, uid).mpp)
                    if mpp <= R.MAX_NATIVE_MPP:
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
        picked += take("gbm", sum(short.values()))
    total = sum(want.values())
    if len(picked) != total:
        raise SystemExit(f"only {len(picked)} of {total} eligible TCGA patients found: nothing written")
    if not 0 < a.hold < total:
        raise SystemExit("--hold must leave both development and held-out patients")
    held = []
    for cat in HOLD_FROM:
        match = next((p for p in picked if p[0]["category"] == cat and p not in held), None)
        if match and len(held) < a.hold:
            held.append(match)
    for p in picked:
        if len(held) < a.hold and p not in held:
            held.append(p)
    held_ids = {p[0]["PatientID"] for p in held}
    patients, rows = [], []
    for r, mpp, fs in sorted(picked, key=lambda p: p[0]["PatientID"]):
        group = "held_out" if r["PatientID"] in held_ids else "development"
        patients.append({"patient": r["PatientID"], "slide": r["SeriesInstanceUID"], "category": r["category"],
                         "group": group, "native_mpp": mpp})
        roles = ("full", "full") if group == "held_out" else ("full", "calls", "calls", "calls")
        for n, ((x, y), role) in enumerate(zip(fs, roles), start=1):
            rows.append({"patient": r["PatientID"], "slide": r["SeriesInstanceUID"], "group": group, "role": role,
                         "field_of_slide": n, "x_um": round(x, 3), "y_um": round(y, 3), "side_um": round(R.FIELD_UM, 4)})
    out.mkdir(parents=True, exist_ok=True)
    R.write_csv(out / "patients.csv", patients)
    R.write_csv(out / "fields.csv", rows)
    print(json.dumps({"patients": len(patients), "groups": dict(Counter(p["group"] for p in patients)),
                      "by_category": dict(Counter(p["category"] for p in patients)), "wanted": want, "short": short,
                      "full_fields": sum(r["role"] == "full" for r in rows),
                      "finder_fields": sum(r["role"] == "calls" for r in rows)}, indent=2))
    return 0


def load_finder(run: Path):
    import torch
    import finder as F
    info = json.load(open(run / "frozen.json"))
    if F.sha256(run / "frozen.pt") != info["sha256"]:
        raise SystemExit("the frozen finder has changed")
    c = torch.load(run / "frozen.pt", map_location="cpu", weights_only=False)
    dev = F.device()
    model = F.build_model(c["base"]).to(dev)
    model.load_state_dict(c["model"])
    return model, dev, float(info["cutoff"]), info["sha256"]


def calls(a) -> int:
    import numpy as np
    import finder as F
    out = Path(a.out) / "private"
    if (out / "candidates.csv").exists():
        raise SystemExit(f"{out / 'candidates.csv'} already exists")
    rows = list(csv.DictReader(open(out / "fields.csv", newline="")))
    check_separate(rows, reserved_patients(a.reference))
    todo = [r for r in rows if r["role"] == "calls" and r["group"] == "development"]
    model, dev, cutoff, sha = load_finder(Path(a.finder))
    found = []
    for n, r in enumerate(todo, start=1):
        cx, cy = float(r["x_um"]) + R.FIELD_UM / 2, float(r["y_um"]) + R.FIELD_UM / 2
        arr = np.asarray(R.open_reader(a.slides, r["slide"]).read_um(cx, cy, R.IMAGE_UM, R.IMAGE_PX))
        for x, y, score in F.peaks(F.predict_map(model, arr, dev)):
            sx, sy = cx + (float(x) - R.IMAGE_PX / 2) * R.UM_PER_PX, cy + (float(y) - R.IMAGE_PX / 2) * R.UM_PER_PX
            if abs(sx - cx) <= R.FIELD_UM / 2 and abs(sy - cy) <= R.FIELD_UM / 2:
                found.append({"patient": r["patient"], "slide": r["slide"], "field_of_slide": r["field_of_slide"],
                              "x_um": round(sx, 2), "y_um": round(sy, 2), "score": round(float(score), 5)})
        print(f"scored {n}/{len(todo)} fields, {len(found)} candidates", file=sys.stderr, flush=True)
    if not found:
        raise SystemExit("the finder proposed no candidates")
    tmp = out / "candidates.csv.tmp"
    R.write_csv(tmp, found)
    tmp.replace(out / "candidates.csv")
    (out / "finder.json").write_text(json.dumps({"finder_sha256": sha, "cutoff": cutoff}, indent=2) + "\n")
    print(json.dumps({"fields": len(todo), "candidates": len(found), "at_or_above_cutoff":
                      sum(c["score"] >= cutoff for c in found), "cutoff": cutoff}, indent=2))
    return 0


def choose_calls(rows: list, cutoff: float, n: int, seed: int) -> list:
    """About n/3 confident, borderline and low-scoring candidates, spread evenly over patients. Spare places are filled."""
    rng = random.Random(seed)
    strata = {"confident": [], "borderline": [], "low": []}
    below = sorted(float(r["score"]) for r in rows if float(r["score"]) < cutoff)
    middle = below[len(below) // 2] if below else cutoff      # the upper half of the calls below the cut-off
    for r in rows:
        s = float(r["score"])
        name = "confident" if s >= cutoff else "borderline" if s >= middle else "low"
        strata[name].append({**r, "stratum": name})
    ranked = {}
    for name, part in strata.items():
        by = defaultdict(list)
        for r in part:
            by[r["patient"]].append(r)
        order = []
        for p in sorted(by):
            rng.shuffle(by[p])
            order += [(i, rng.random(), r) for i, r in enumerate(by[p])]
        ranked[name] = [x[2] for x in sorted(order, key=lambda x: x[:2])]
    chosen, share = [], n // 3
    for name in ranked:
        chosen += ranked[name][:share]
        ranked[name] = ranked[name][share:]
    while len(chosen) < n and any(ranked.values()):
        for name in ranked:
            if ranked[name] and len(chosen) < n:
                chosen.append(ranked[name].pop(0))
    rng.shuffle(chosen)
    return chosen


def crop_slide(slide: str, rows: list, slides: str, folder: str) -> int:
    from PIL import Image
    rd = R.open_reader(slides, slide)
    for r in rows:
        im = Image.fromarray(rd.read_um(float(r["x_um"]), float(r["y_um"]), CROP_UM, CROP_PX))
        im.save(Path(folder) / f"{r['call']}.jpg", quality=92, subsampling=0)
    return len(rows)


def run_jobs(fn, jobs: list, workers: int) -> None:
    if workers <= 1:
        for j in jobs:
            fn(*j)
        return
    with ProcessPoolExecutor(workers) as pool:
        for job in [pool.submit(fn, *j) for j in jobs]:
            job.result()


def build(a) -> int:
    out = Path(a.out)
    private = out / "private"
    if (private / "calls-key.csv").exists():
        raise SystemExit(f"{private / 'calls-key.csv'} already exists: a built round is never overwritten")
    fields = list(csv.DictReader(open(private / "fields.csv", newline="")))
    candidates = list(csv.DictReader(open(private / "candidates.csv", newline="")))
    reserved = reserved_patients(a.reference)
    check_separate(fields, reserved)
    check_separate(candidates, reserved)
    cutoff = float(json.load(open(private / "finder.json"))["cutoff"])
    workers = max(1, a.workers or int(os.environ.get("NSLOTS", 1)))

    full = [dict(r) for r in fields if r["role"] == "full"]
    random.Random(f"{a.seed}-full").shuffle(full)
    for n, r in enumerate(full, start=1):
        r["field"] = f"F{n:03d}"
    folder = out / "full-fields-pack" / "fields"
    folder.mkdir(parents=True, exist_ok=True)
    by_slide = defaultdict(list)
    for r in full:
        by_slide[r["slide"]].append(r)
    run_jobs(R.pack_slide, [(s, v, a.slides, str(folder)) for s, v in sorted(by_slide.items())], workers)
    codes = sorted(r["field"] for r in full)
    sha = full_sha(full)
    full_tar = R.write_pack(out, "full-fields-pack", f"entip-mitosis-dev1-{sha}", codes, folder)

    chosen = choose_calls(candidates, cutoff, a.calls, a.seed)
    for n, r in enumerate(chosen, start=1):
        r["call"] = f"C{n:04d}"
    pack = out / "calls-pack"
    crops = pack / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    by_slide = defaultdict(list)
    for r in chosen:
        by_slide[r["slide"]].append(r)
    run_jobs(crop_slide, [(s, v, a.slides, str(crops)) for s, v in sorted(by_slide.items())], workers)
    calls_tar = write_call_page(out, sha, [r["call"] for r in chosen])
    R.write_csv(private / "full-key.csv", [{k: r[k] for k in ("field", "patient", "slide", "group", "x_um", "y_um",
                                                              "side_um")} for r in sorted(full, key=lambda r: r["field"])])
    R.write_csv(private / "calls-key.csv", [{k: r[k] for k in ("call", "patient", "slide", "field_of_slide", "x_um",
                                                               "y_um", "score", "stratum")}
                                            for r in sorted(chosen, key=lambda r: r["call"])])
    print(json.dumps({"full_fields": len(codes), "calls": len(chosen),
                      "call_strata": dict(Counter(r["stratum"] for r in chosen)), "full_pack": str(full_tar),
                      "calls_pack": str(calls_tar)}, indent=2))
    return 0


def full_sha(full: list) -> str:
    """Pack id from the full fields in their pack order, so a rebuilt page keeps the same saved-label key."""
    return hashlib.sha256("|".join(r["slide"] + r["x_um"] for r in full).encode()).hexdigest()[:12]


def write_call_page(out: Path, sha: str, calls: list) -> Path:
    pack = out / "calls-pack"
    data = {"study": f"entip-mitosis-calls1-{sha}", "labels": list(CALL_LABELS), "calls": calls}
    (pack / "review.html").write_text(CALL_PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))),
                                      encoding="utf-8")
    calls_tar = out / "calls-pack.tar"
    tmp = out / "calls-pack.tar.tmp"
    with tarfile.open(tmp, "w") as tar:
        tar.add(pack, arcname="calls-pack")
    tmp.replace(calls_tar)
    return calls_tar


def page(a) -> int:
    out = Path(a.out)
    private = out / "private"
    full = list(csv.DictReader(open(private / "full-key.csv", newline="")))
    keyed = list(csv.DictReader(open(private / "calls-key.csv", newline="")))
    calls = [r["call"] for r in keyed]
    missing = [c for c in calls if not (out / "calls-pack" / "crops" / f"{c}.jpg").exists()]
    if not full or not calls or missing:
        raise SystemExit("this round has no complete calls pack to rewrite")
    tar = write_call_page(out, full_sha(full), calls)
    print(json.dumps({"calls": len(calls), "labels": list(CALL_LABELS), "calls_pack": str(tar)}, indent=2))
    return 0


CALL_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Mitosis call review</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:light;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#202124}*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;background:#30302e}main{flex:1;display:flex;align-items:center;justify-content:center;padding:16px;min-width:0}
#box{position:relative;width:min(82vh,100%);aspect-ratio:1}#box img{width:100%;height:100%;image-rendering:auto;display:block}.t{position:absolute;background:#fff}.h{height:2px;width:7%;top:50%}.v{width:2px;height:7%;left:50%}
aside{width:300px;flex:none;background:#fff;padding:14px;overflow:auto}h1{font-size:17px;font-weight:600;margin:0 0 3px}.muted{color:#68685f;font-size:12px}section{border-top:1px solid #e6e6e1;margin-top:11px;padding-top:10px}
button{font:inherit;border:1px solid #b9b9b1;background:#fff;border-radius:5px;padding:5px 8px;cursor:pointer}button:hover{background:#f0f0ec}.lab{display:flex;width:100%;margin-bottom:4px;text-align:left}.lab.on{outline:2px solid #1f6f54;background:#e4f2eb}kbd{font:11px ui-monospace,monospace;background:#efefea;border:1px solid #cfcfc8;border-radius:3px;padding:0 4px;margin-left:auto}.row{display:flex;gap:6px;flex-wrap:wrap}input{font:inherit;border:1px solid #b9b9b1;border-radius:4px;padding:4px}
#start{position:fixed;inset:0;z-index:5;background:#f4f4f2;display:flex;align-items:center;justify-content:center;padding:24px}#start div{max-width:540px;background:#fff;border:1px solid #d8d8d3;border-radius:8px;padding:22px}#msg{min-height:18px;color:#a3341f;font-size:12px;margin-top:6px}
</style></head><body><div id="start"><div><h1>Mitosis call review</h1><p>Judge only the cell at the centre, between the four white ticks. Each crop is 64 µm wide. No score, diagnosis or patient is shown.</p>
<p><b>Mitosis:</b> prometaphase, metaphase, anaphase, telophase or atypical. <b>Granular:</b> a granular mitosis, which also counts. <b>Prophase</b> does not count. <b>Look-alike:</b> a cell that mimics a mitosis, such as pyknosis, apoptosis, karyorrhexis or a lymphocyte. <b>Not mitosis:</b> anything else, such as an ordinary nucleus or no cell. <b>Uncertain:</b> you cannot decide.</p>
<p class="row"><label>Your initials <input id="who" maxlength="6" size="8" autocomplete="off"></label><button id="begin">Start</button></p><p id="startmsg" class="muted">Marks save in this browser. Download them often.</p></div></div>
<main><div id="box"><img id="im" alt="Candidate cell at high power"><i class="t h" style="left:34%"></i><i class="t h" style="right:34%"></i><i class="t v" style="top:34%"></i><i class="t v" style="bottom:34%"></i></div></main>
<aside><h1 id="title"></h1><div class="muted" id="progress"></div><section id="labels"></section><section><div class="row"><button id="prev">Previous <kbd>←</kbd></button><button id="next">Next <kbd>→</kbd></button><button id="open">Next unlabelled</button></div></section>
<section><div class="row"><button id="save">Download labels</button><button id="loadB">Load labels</button><input id="load" type="file" accept=".csv" hidden></div><div id="msg"></div></section></aside>
<script>
const D=__DATA__,KEY="entip-"+D.study,$=id=>document.getElementById(id);let st={rater:"",i:0,labels:{}};try{const o=JSON.parse(localStorage.getItem(KEY)||"null");if(o&&o.labels)st=o}catch(e){}
const store=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){$("msg").textContent="This browser could not save: download your labels now."}};
const label=s=>({look_alike:"Look-alike",not_mitosis:"Not mitosis"})[s]||s.replace(/^./,c=>c.toUpperCase());
function paint(){const c=D.calls[st.i],n=D.calls.filter(x=>st.labels[x]).length;$("title").textContent=`Call ${st.i+1} of ${D.calls.length}`;$("progress").textContent=`${n} labelled · reader ${st.rater||"-"}`;$("im").src=`crops/${c}.jpg`;$("labels").innerHTML=D.labels.map((l,i)=>`<button class="lab ${st.labels[c]===l?"on":""}" data-l="${l}">${label(l)}<kbd>${i+1}</kbd></button>`).join("")}
function go(i){st.i=Math.max(0,Math.min(D.calls.length-1,i));$("msg").textContent="";store();paint()}
function nextOpen(){const a=D.calls.findIndex((c,i)=>i>st.i&&!st.labels[c]),b=D.calls.findIndex(c=>!st.labels[c]),to=a>=0?a:b;if(to>=0)go(to);else{paint();$("msg").textContent="All calls labelled. Download your labels."}}
function setLabel(l){st.labels[D.calls[st.i]]=l;store();nextOpen()}
const csv=()=>["rater,call,label"].concat(D.calls.filter(c=>st.labels[c]).map(c=>[st.rater,c,st.labels[c]].join(","))).join("\n")+"\n";
function loadCsv(t){const R=t.trim().split(/\r?\n/).map(l=>l.split(","));const H=R.shift()||[],a=H.indexOf("rater"),b=H.indexOf("call"),c=H.indexOf("label");if(a<0||b<0||c<0)throw new Error("not a labels file");const L={};let who="";R.forEach(r=>{if(!D.calls.includes(r[b]))throw new Error("this file is from a different pack");if(!D.labels.includes(r[c]))throw new Error("unknown label");L[r[b]]=r[c];who=r[a]||who});st.labels=L;st.rater=who||st.rater;store();paint();$("msg").textContent="Labels loaded."}
$("begin").onclick=()=>{const w=$("who").value.trim().toUpperCase();if(!w){$("startmsg").textContent="Enter your initials first.";return}st.rater=w;store();$("start").style.display="none";paint()};$("labels").onclick=e=>{const b=e.target.closest("[data-l]");if(b)setLabel(b.dataset.l)};$("prev").onclick=()=>go(st.i-1);$("next").onclick=()=>go(st.i+1);$("open").onclick=nextOpen;
$("save").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv()],{type:"text/csv"}));a.download=`mitosis-call-labels-${st.rater||"reader"}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};$("loadB").onclick=()=>$("load").click();$("load").onchange=async e=>{const f=e.target.files[0];if(!f)return;try{loadCsv(await f.text())}catch(x){$("msg").textContent="Could not load: "+x.message}e.target.value=""};
addEventListener("keydown",e=>{if(document.activeElement.tagName==="INPUT"||$("start").style.display!=="none")return;if(e.key>="1"&&e.key<=String(D.labels.length))setLabel(D.labels[+e.key-1]);else if(e.key==="ArrowRight")go(st.i+1);else if(e.key==="ArrowLeft"||e.key==="Backspace"){e.preventDefault();go(st.i-1)}});if(st.rater)$("who").value=st.rater;paint();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="draw a small set of TCGA patients outside the reference set")
    d.add_argument("--manifest", required=True)
    d.add_argument("--bags", required=True, help="used only for where the tissue is")
    d.add_argument("--slides", required=True)
    d.add_argument("--reference", required=True, help="the sealed reference set's private/patients.csv")
    d.add_argument("--out", required=True)
    d.add_argument("--per-category", nargs="*", default=["gbm=3", "astro4=3", "oligo3=2", "astro3=2", "grade2=2"])
    d.add_argument("--hold", type=int, default=3)
    d.add_argument("--seed", type=int, default=20261201)
    d.set_defaults(fn=draw)
    c = sub.add_parser("calls", help="run the frozen finder on the development finder fields")
    c.add_argument("--finder", required=True)
    c.add_argument("--slides", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=calls)
    b = sub.add_parser("build", help="write the blinded review packs")
    b.add_argument("--slides", required=True)
    b.add_argument("--reference", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--calls", type=int, default=300)
    b.add_argument("--workers", type=int)
    b.add_argument("--seed", type=int, default=20261201)
    b.set_defaults(fn=build)
    p = sub.add_parser("page", help="rewrite a built round's call review page with the current labels")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=page)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
