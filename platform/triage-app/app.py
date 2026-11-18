"""ENTIP triage app: triage a new whole-slide image with the distilled student, on your own machine.

    python platform/triage-app/app.py --student student.pt --model model-student [--slides ~/slides] [--port 8765]

Then open http://127.0.0.1:8765. Pick a slide (a file such as .svs/.ndpi/.tiff, or a DICOM series folder); the
app finds the tissue, embeds every tissue tile with the student (``student.pt``, about 22M parameters; Apple
GPU, CUDA or CPU), runs the triage model trained on student bags (``model-student/``), and shows:

* the call as a WHO CNS5 category, whether it is a **confident call** (calibrated confidence at or above the
  gate's threshold, slide not novel) or **for the pathologist to classify**, and the reasons;
* every class probability, tissue adequacy, and the novelty check (is this slide unlike the training slides?);
* where the model looked: an attention heat map over the slide and its most attended tiles, with how
  concentrated that attention is.

Results are kept in ``--out`` (default ``~/entip-triage``), one folder per run, and listed under Recent.
It listens on 127.0.0.1 only. Research use only: not for diagnosis.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
SLIDE_EXT = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".svslide", ".isyntax"}
TOP = 16
THUMB = 1200


class Counting:
    """The encoder, counting tiles as they are embedded (for the progress bar)."""

    def __init__(self, enc, job: dict):
        self._enc, self._job = enc, job

    def __getattr__(self, k):
        return getattr(self._enc, k)

    def embed(self, crops):
        out = self._enc.embed(crops)
        self._job["done"] += len(crops)
        return out


def concentration(att: np.ndarray) -> dict:
    """Share of attention on the top 1% of tiles (1% if even) and the effective number of tiles, 1/sum(w^2)."""
    w = att / att.sum()
    n, k = len(w), max(1, int(np.ceil(0.01 * len(w))))
    eff = 1.0 / float(np.sum(w ** 2))
    return {"tiles": n, "top1pct_share": round(float(np.sort(w)[::-1][:k].sum()), 4),
            "effective_tiles": round(eff, 1), "effective_fraction": round(eff / n, 4)}


def draw(rd, bag, att: np.ndarray, out: Path) -> list:
    """thumb.jpg (the tissue, framed), heat.png (attention, same size, transparent) and the top tiles."""
    from PIL import Image, ImageDraw
    lo, hi = bag.coords_um.min(0), bag.coords_um.max(0) + bag.tile_um
    cx, cy = (lo + hi) / 2
    side = float((hi - lo).max()) * 1.04
    Image.fromarray(rd.read_um(cx, cy, side, THUMB)).convert("RGB").save(out / "thumb.jpg", quality=88)
    s, x0, y0 = THUMB / side, cx - side / 2, cy - side / 2
    heat = Image.new("RGBA", (THUMB, THUMB), (0, 0, 0, 0))
    d = ImageDraw.Draw(heat)
    v = np.sqrt(att / att.max())                      # square root so mid weights stay visible
    for (x, y), a in zip(bag.coords_um, v):
        px, py = (x - x0) * s, (y - y0) * s
        d.rectangle([px, py, px + bag.tile_um * s, py + bag.tile_um * s], fill=(235, 90, 20, int(20 + 200 * a)))
    heat.save(out / "heat.png")
    (out / "tiles").mkdir(exist_ok=True)
    even = 1.0 / len(att)
    top = []
    for r, i in enumerate(np.argsort(-att)[:TOP], start=1):
        x, y = (float(t) for t in bag.coords_um[i])
        Image.fromarray(rd.read_um(x + bag.tile_um / 2, y + bag.tile_um / 2, bag.tile_um, 256)).convert("RGB") \
            .save(out / "tiles" / f"{r:02d}.jpg", quality=90)
        top.append({"rank": r, "weight": float(att[i]), "x_even": round(float(att[i]) / even, 1),
                    "x_um": x, "y_um": y, "img": f"tiles/{r:02d}.jpg",
                    "box": [round((x - x0) / side, 5), round((y - y0) / side, 5), round(bag.tile_um / side, 5)]})
    return top


class Engine:
    """Loads the student and the triage model once; runs one slide at a time in a worker thread."""

    def __init__(self, a):
        from entip_glioma.triage.model import TriageModel
        from entip_train.encoders import get_encoder
        self.a = a
        self.model = TriageModel.load(a.model)
        self.encoder = get_encoder(f"student:{a.student}")
        if self.encoder.version != self.model.encoder:
            raise SystemExit(f"the triage model was trained on bags from {self.model.encoder!r}, but {a.student} "
                             f"is {self.encoder.version!r}: use the model-student made from this student")
        self.out = Path(a.out).expanduser()
        self.out.mkdir(parents=True, exist_ok=True)
        self.jobs: dict = {}
        self.q: queue.Queue = queue.Queue()
        for p in sorted(self.out.glob("*/result.json")):
            try:
                r = json.loads(p.read_text())
                self.jobs[r["id"]] = {"id": r["id"], "state": "done", "slide": r["slide"]["name"],
                                      "path": r["slide"]["path"], "started": r.get("finished"), "result": r}
            except (OSError, ValueError, KeyError):
                pass
        threading.Thread(target=self._worker, daemon=True).start()

    def info(self) -> dict:
        m = self.model
        from entip_glioma.triage import who_cns5
        from entip_glioma.triage.model import CLASSES
        gate = m.report.get("cv", {}).get("gate") or {}
        return {"model_id": m.model_id, "version": m.version, "encoder": m.encoder,
                "device": getattr(self.encoder, "device", "cpu"),
                "threshold": round(min(m.threshold, 1.0), 4), "never_accepts": m.threshold > 1.0,
                "max_error": gate.get("max_error"), "coverage": gate.get("coverage"),
                "n_accepted": gate.get("n_accepted"), "n_validation": gate.get("n"),
                "classes": [{"label": c, "name": who_cns5.class_name(c)} for c, t in zip(CLASSES, m.trained) if t],
                "slides_dir": str(Path(self.a.slides).expanduser()) if self.a.slides else str(Path.home())}

    def submit(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.exists():
            raise ValueError(f"no such file or folder: {p}")
        jid = datetime.now().strftime("%Y%m%d-%H%M%S-") + re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)[:60]
        job = {"id": jid, "state": "queued", "stage": "waiting", "slide": p.name, "path": str(p),
               "started": time.time(), "done": 0, "total": None}
        self.jobs[jid] = job
        self.q.put(jid)
        return job

    def _worker(self):
        while True:
            job = self.jobs[self.q.get()]
            try:
                job["state"] = "running"
                job["result"] = self._run(job)
                job["state"] = "done"
            except Exception as e:               # report it on the page; the server keeps going
                job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc()
                shutil.rmtree(self.out / job["id"], ignore_errors=True)

    def _run(self, job: dict) -> dict:
        from entip_glioma.triage import who_cns5
        from entip_glioma.triage.model import CLASSES
        from entip_glioma.wsi.bags import extract_bag, tile_grid
        import entip_train.slides as slides
        a, t0, secs = self.a, time.perf_counter(), {}
        out = self.out / job["id"]
        out.mkdir(parents=True, exist_ok=True)

        job["stage"] = "opening the slide"
        rd = slides.open_slide(job["path"])
        job["stage"] = "finding tissue"
        xy, _ = tile_grid(rd, rd.width_um, rd.height_um, a.tile_um, a.min_tissue)
        job["total"] = len(xy)
        secs["tissue"] = round(time.perf_counter() - t0, 1)
        if not len(xy):
            raise ValueError("no tissue tiles found: blank, tiny or unreadable slide")

        job["stage"] = "embedding tissue tiles with the student"
        t = time.perf_counter()
        opener = functools.partial(slides.open_slide, job["path"])
        bag = extract_bag(rd, job["id"], rd.width_um, rd.height_um, Counting(self.encoder, job), tile_um=a.tile_um,
                          min_tissue=a.min_tissue, workers=a.workers, opener=opener if a.workers > 1 else None)
        secs["embed"] = round(time.perf_counter() - t, 1)
        bag.save(out / "bag.npz")

        job["stage"] = "classifying"
        recs = {r["finding"]: r for r in self.model.records([bag], study_id=job["id"])}
        g = self.model.predict(bag)
        att = np.asarray(g["attention"], np.float64)

        job["stage"] = "drawing the attention map"
        top = draw(rd, bag, att, out)
        tc, ad = recs.get("tissue_class"), recs["adequacy"]
        res = {"id": job["id"], "finished": time.time(),
               "slide": {"name": job["slide"], "path": job["path"], "reader": type(rd).__name__,
                         "mpp": round(float(rd.mpp), 4), "width_mm": round(rd.width_um / 1000, 1),
                         "height_mm": round(rd.height_um / 1000, 1)},
               "tiles": len(bag), "tile_um": bag.tile_um, "seconds": {**secs, "total": round(time.perf_counter() - t0, 1)},
               "model": {"model_id": self.model.model_id, "encoder": self.model.encoder},
               "adequacy": {"call": ad["call"], "status": ad["status"], "reasons": ad["reasons"],
                            "area_mm2": round(ad["value"]["tissue_area_mm2"], 2)},
               "attention": concentration(att), "top": top}
        if tc:
            res["decision"] = {
                "call": tc["call"], "name": who_cns5.class_name(tc["call"]), "status": tc["status"],
                "confidence": tc["confidence"], "threshold": tc["threshold"], "order_test": tc["order_test"],
                "reasons": tc["reasons"], "novel": g["novel"],
                "distance": None if g["distance"] is None else round(g["distance"], 3),
                "novelty_threshold": None if self.model.novelty is None else round(float(self.model.novelty.threshold), 3),
                "probabilities": [{"label": c, "name": who_cns5.class_name(c), "p": p}
                                  for c, p in sorted(tc["probabilities"].items(), key=lambda kv: -kv[1])],
                "p_tumour": tc["value"].get("p_tumour"), "p_glial": tc["value"].get("p_glial")}
        (out / "result.json").write_text(json.dumps(res, indent=2, default=float))
        return res


def handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, default=float).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/info":
                return self._json(engine.info())
            if u.path == "/api/jobs":
                rows = sorted(engine.jobs.values(), key=lambda j: -(j.get("started") or 0))
                return self._json([{k: j.get(k) for k in ("id", "state", "slide", "started")} |
                                   ({"call": j["result"].get("decision", {}).get("name"),
                                     "status": j["result"].get("decision", {}).get("status")}
                                    if j.get("result") else {}) for j in rows])
            m = re.fullmatch(r"/api/jobs/([^/]+)", u.path)
            if m and m.group(1) in engine.jobs:
                j = engine.jobs[m.group(1)]
                return self._json({k: v for k, v in j.items()} | {"elapsed": round(time.time() - j["started"], 1)
                                                                  if j["state"] in ("queued", "running") else None})
            m = re.fullmatch(r"/runs/([^/]+)/((?:tiles/)?[A-Za-z0-9_.-]+\.(?:jpg|png))", u.path)
            if m and m.group(1) in engine.jobs:
                f = engine.out / m.group(1) / m.group(2)
                if f.is_file():
                    return self._send(200, f.read_bytes(), "image/png" if f.suffix == ".png" else "image/jpeg")
            if u.path == "/api/browse":
                return self._browse(q.get("dir") or engine.info()["slides_dir"])
            return self._send(404, b"not found", "text/plain")

        def _browse(self, d: str):
            p = Path(d).expanduser()
            if not p.is_dir():
                return self._json({"error": f"not a folder: {p}"}, 400)
            entries = []
            try:
                for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                    if c.name.startswith("."):
                        continue
                    if c.is_dir():
                        dicom = any(x.suffix.lower() == ".dcm" for _, x in zip(range(40), c.iterdir()))
                        entries.append({"name": c.name, "path": str(c), "kind": "dicom" if dicom else "dir"})
                    elif c.suffix.lower() in SLIDE_EXT or c.suffix.lower() == ".dcm":
                        entries.append({"name": c.name, "path": str(c), "kind": "slide",
                                        "mb": round(c.stat().st_size / 1e6, 1)})
            except PermissionError:
                return self._json({"error": f"cannot read {p}"}, 400)
            return self._json({"dir": str(p), "parent": str(p.parent) if p.parent != p else None, "entries": entries})

        def do_POST(self):
            if urlparse(self.path).path != "/api/triage":
                return self._send(404, b"not found", "text/plain")
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                return self._json(engine.submit(str(body.get("path") or "")))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)

    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="the distilled student, student.pt")
    ap.add_argument("--model", required=True, help="the triage model trained on that student's bags (model-student/)")
    ap.add_argument("--slides", default=None, help="folder the slide browser opens in (default: your home folder)")
    ap.add_argument("--out", default="~/entip-triage", help="results, one folder per slide")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--tile-um", type=float, default=112.0, help="must match the training bags (default 112)")
    ap.add_argument("--min-tissue", type=float, default=0.5, help="must match the training bags (default 0.5)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="processes reading tiles")
    a = ap.parse_args(argv)
    engine = Engine(a)
    i = engine.info()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(engine))
    print(f"ENTIP triage: {i['model_id']} on {i['encoder']} ({i['device']}); threshold {i['threshold']}\n"
          f"open http://127.0.0.1:{a.port}   (Ctrl-C to stop; results in {engine.out})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
