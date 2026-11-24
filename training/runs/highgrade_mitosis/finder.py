"""Step 3c: train, freeze and test the first mitosis finder on public data.

    python finder.py train --run mitosis-finder1
    python finder.py freeze --run mitosis-finder1
    python finder.py test --run mitosis-finder1

**train** fits a compact U-Net from scratch on image patches at about 0.25 µm per pixel, with no pretrained weights.
Its target is the expert mask: mitoses should score high, annotated look-alikes low, other tissue low with a lighter
penalty because a few unmarked mitoses may be there, and prophase is ignored. Every few epochs it scores the ``tune``
cases and keeps the best model.

**freeze** fixes that model and its cut-off, chosen on ``tune`` only. **test** runs once on the unseen tumour types
with the frozen model and cannot be repeated.

A detection is one peak in the predicted map. It matches a true mitosis within 7.5 µm, each used once. An unmatched
detection on a prophase is neither right nor wrong.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

UM_PER_PX = 0.25
MATCH_PX = 30                       # 7.5 µm
PATCH = 512
STRIDE = 448
PEAK_FLOOR = 0.10
MAX_PEAKS = 3000
THRESHOLDS = np.round(np.arange(0.10, 0.991, 0.01), 2)
WEIGHT = {"background": 0.25, "pos": 5.0, "neg": 5.0}
GLIOMA = "tcga-glioma"              # images made from reviewed glioma fields and calls


def load_index(run: Path) -> dict:
    return json.load(open(run / "index.json"))


def open_image(path: str):
    """An H x W x 3 uint8 array. A TIFF is decoded once to ``<name>.rgb.npy`` and then memory-mapped."""
    if path.lower().endswith((".tif", ".tiff")):
        cached = Path(path + ".rgb.npy")
        if not cached.exists():
            from PIL import Image
            with Image.open(path) as im:
                decoded = np.asarray(im.convert("RGB"))
            tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
            with tmp.open("wb") as f:
                np.save(f, decoded)
            os.replace(tmp, cached)             # several loader processes may do this at once: the last one wins
        arr = np.load(cached, mmap_mode="r")
    else:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{path}: expected an RGB image, got shape {arr.shape}")
    return arr[..., :3]


def crop(arr, x0: int, y0: int, size: int) -> np.ndarray:
    """A square crop, padded with white outside the image."""
    out = np.full((size, size, 3), 255, np.uint8)
    H, W = arr.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + size), min(H, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(arr[sy0:sy1, sx0:sx1])
    return out


def targets(objects: list, x0: int, y0: int, size: int, out_px: int, background: float = WEIGHT["background"]) -> tuple:
    """Target and per-pixel weight maps for a crop, drawn from the expert masks. ``background`` is the weight of
    unmarked tissue: 0 for an image labelled only at its marks, such as a reviewed call."""
    from PIL import Image, ImageDraw
    scale = out_px / size
    y = Image.new("L", (out_px, out_px), 0)
    w = Image.new("L", (out_px, out_px), 1)        # 0 ignore, 1 background, 2 look-alike, 3 mitosis
    dy, dw = ImageDraw.Draw(y), ImageDraw.Draw(w)
    order = {"neg": 0, "pos": 1, "ignore": 2}      # where masks overlap, ignore wins, then mitosis
    near = [o for o in objects if x0 - 96 <= o["x"] <= x0 + size + 96 and y0 - 96 <= o["y"] <= y0 + size + 96]
    for o in sorted(near, key=lambda o: order[o["role"]]):
        for ring in o["rings"]:
            pts = [((ring[i] - x0) * scale, (ring[i + 1] - y0) * scale) for i in range(0, len(ring) - 1, 2)]
            if len(pts) < 3:
                continue
            dy.polygon(pts, fill=1 if o["role"] == "pos" else 0)
            dw.polygon(pts, fill={"neg": 2, "pos": 3, "ignore": 0}[o["role"]])
    code = np.asarray(w)
    weight = np.zeros(code.shape, np.float32)
    weight[code == 1] = background
    weight[code == 2] = WEIGHT["neg"]
    weight[code == 3] = WEIGHT["pos"]
    return np.asarray(y, np.float32), weight


def augment(img: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> tuple:
    k = int(rng.integers(4))
    img, y, w = np.rot90(img, k), np.rot90(y, k), np.rot90(w, k)
    if rng.random() < 0.5:
        img, y, w = img[:, ::-1], y[:, ::-1], w[:, ::-1]
    x = img.astype(np.float32) / 255
    x = np.clip(x * rng.uniform(0.85, 1.15, 3) + rng.uniform(-0.05, 0.05, 3), 0, 1)     # stain and scanner variation
    grey = x.mean(2, keepdims=True)
    x = np.clip(grey + (x - grey) * rng.uniform(0.8, 1.2), 0, 1)
    return x, y.copy(), w.copy()


def to_input(x: np.ndarray):
    import torch
    return torch.from_numpy(np.transpose(x * 2 - 1, (2, 0, 1)).astype(np.float32).copy())


class Patches:
    """Random training patches: 40% on a mitosis, 40% on a look-alike, 20% anywhere fully marked.

    With ``glioma_share`` above 0, that share of patches comes from the reviewed glioma images, which are otherwise
    too few to be drawn often. Patches anywhere come only from images marked in full."""

    def __init__(self, images: list, root: str, length: int, seed: int = 0, glioma_share: float = 0.0):
        self.images, self.root, self.length, self.seed, self.share = images, Path(root), length, seed, glioma_share
        glioma = [i for i, im in enumerate(images) if im["source"] == GLIOMA]
        if glioma_share > 0:
            parts = [[i for i, im in enumerate(images) if im["source"] != GLIOMA], glioma]
        else:
            parts = [list(range(len(images)))]
        self.groups = [self.group(p) for p in parts]
        if any(not g["pos"] for g in self.groups):
            raise SystemExit("no mitoses in the training split" if len(parts) == 1 else
                             "--glioma-share needs reviewed glioma mitoses and public mitoses in training")
        self.cache = {}
        self.rng = None

    def group(self, members: list) -> dict:
        full = [i for i in members if self.images[i].get("background", WEIGHT["background"]) > 0]
        area = np.array([self.images[i]["w"] * self.images[i]["h"] for i in full], float)
        return {"pos": [(i, n) for i in members for n, o in enumerate(self.images[i]["objects"]) if o["role"] == "pos"],
                "neg": [(i, n) for i in members for n, o in enumerate(self.images[i]["objects"]) if o["role"] == "neg"],
                "full": full, "area": area / area.sum() if len(full) else area}

    def __len__(self):
        return self.length

    def array(self, i: int):
        if i not in self.cache:
            if len(self.cache) >= 32:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = open_image(str(self.root / self.images[i]["path"]))
        return self.cache[i]

    def __getitem__(self, idx: int):
        import torch
        if self.rng is None:           # one stream per loader process, so every epoch draws new patches
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, int(torch.initial_seed()) % (2 ** 32) if info else 0])
        rng = self.rng
        g = self.groups[1] if self.share > 0 and rng.random() < self.share else self.groups[0]
        size = int(round(PATCH * rng.uniform(0.9, 1.1)))
        u = rng.random()
        pool = g["pos"] if u < 0.4 or not g["neg"] and u < 0.8 else g["neg"] if u < 0.8 else None
        if pool is None and not g["full"]:
            pool = g["pos"]
        if pool is None:
            i = int(g["full"][int(rng.choice(len(g["full"]), p=g["area"]))])
            cx, cy = rng.uniform(0, self.images[i]["w"]), rng.uniform(0, self.images[i]["h"])
        else:
            i, n = pool[int(rng.integers(len(pool)))]
            o = self.images[i]["objects"][n]
            jitter = size * 0.35
            cx, cy = o["x"] + rng.uniform(-jitter, jitter), o["y"] + rng.uniform(-jitter, jitter)
        x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
        patch = crop(self.array(i), x0, y0, size)
        if size != PATCH:
            from PIL import Image
            patch = np.asarray(Image.fromarray(patch).resize((PATCH, PATCH), Image.BILINEAR))
        y, w = targets(self.images[i]["objects"], x0, y0, size, PATCH,
                       self.images[i].get("background", WEIGHT["background"]))
        x, y, w = augment(patch, y, w, rng)
        return to_input(x), torch.from_numpy(y)[None], torch.from_numpy(w)[None]


def build_model(base: int = 24):
    import torch
    from torch import nn
    if base % 8:
        raise SystemExit("--base must be a multiple of 8")

    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.SiLU(),
                                     nn.Conv2d(b, b, 3, padding=1, bias=False), nn.GroupNorm(8, b))
            self.skip = nn.Conv2d(a, b, 1, bias=False) if a != b else nn.Identity()
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [base, base * 2, base * 4, base * 8, base * 12]
            self.first = Block(3, ch[0])
            self.down = nn.ModuleList(nn.Sequential(nn.Conv2d(ch[i], ch[i], 3, 2, 1), Block(ch[i], ch[i + 1]))
                                      for i in range(4))
            self.up = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(4)))
            self.dec = nn.ModuleList(Block(ch[i] * 2, ch[i]) for i in reversed(range(4)))
            self.out = nn.Conv2d(ch[0], 1, 1)
            nn.init.constant_(self.out.bias, -3.0)     # mitoses are rare: start from a low probability

        def forward(self, x):
            skips = [self.first(x)]
            for d in self.down:
                skips.append(d(skips[-1]))
            h = skips.pop()
            for u, d in zip(self.up, self.dec):
                h = d(torch.cat([u(h), skips.pop()], 1))
            return self.out(h)

    return UNet()


def loss_fn(logits, y, w):
    import torch
    import torch.nn.functional as F
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).sum() / w.sum().clamp_min(1)
    p = torch.sigmoid(logits) * (w > 0)
    t = y * (w > 0)
    dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_map(model, arr, dev, batch: int = 12) -> np.ndarray:
    """Probability map of a whole image, averaged over overlapping windows with more weight at their centres."""
    import torch
    H, W = arr.shape[:2]
    xs = list(range(0, max(W - PATCH, 0) + 1, STRIDE))
    ys = list(range(0, max(H - PATCH, 0) + 1, STRIDE))
    if xs[-1] != max(W - PATCH, 0):
        xs.append(max(W - PATCH, 0))
    if ys[-1] != max(H - PATCH, 0):
        ys.append(max(H - PATCH, 0))
    ramp = np.minimum(np.arange(1, PATCH + 1), np.arange(PATCH, 0, -1)).astype(np.float32)
    win = np.minimum(1, np.minimum.outer(ramp, ramp) / 32)
    prob, total = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    coords = [(x, y) for y in ys for x in xs]
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(coords), batch):
            part = coords[s:s + batch]
            x = torch.stack([to_input(crop(arr, cx, cy, PATCH).astype(np.float32) / 255) for cx, cy in part]).to(dev)
            p = torch.sigmoid(model(x))[:, 0].float().cpu().numpy()
            for (cx, cy), q in zip(part, p):
                h, w = min(PATCH, H - cy), min(PATCH, W - cx)
                prob[cy:cy + h, cx:cx + w] += q[:h, :w] * win[:h, :w]
                total[cy:cy + h, cx:cx + w] += win[:h, :w]
    return prob / np.maximum(total, 1e-6)


def peaks(prob: np.ndarray, floor: float = PEAK_FLOOR, radius: float = MATCH_PX) -> np.ndarray:
    """Detections (x, y, score): one per connected blob, with nearby blobs merged so one cell counts once."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    lab, n = ndi.label(prob >= floor, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 3), np.float32)
    idx = np.arange(1, n + 1)
    score = np.asarray(ndi.maximum(prob, lab, idx))
    idx = idx[np.argsort(-score)[:MAX_PEAKS]]     # an untrained model can produce very many blobs
    score = np.asarray(ndi.maximum(prob, lab, idx))
    com = np.asarray(ndi.center_of_mass(prob, lab, idx)).reshape(-1, 2)
    d = np.column_stack([com[:, 1], com[:, 0], score]).astype(np.float32)
    d = d[np.argsort(-d[:, 2], kind="stable")]
    close = cKDTree(d[:, :2]).query_ball_point(d[:, :2], radius)
    gone = np.zeros(len(d), bool)
    for i, others in enumerate(close):
        if not gone[i]:
            for j in others:
                if j > i:
                    gone[j] = True
    return d[~gone]


def match(det: np.ndarray, truth: np.ndarray, radius: float = MATCH_PX) -> tuple:
    """Highest score first, each truth used once: a boolean for each detection, and how many truths were found."""
    hit = np.zeros(len(det), bool)
    if len(det) == 0 or len(truth) == 0:
        return hit, 0
    free = np.ones(len(truth), bool)
    for i in np.argsort(-det[:, 2]):
        dist = np.hypot(truth[:, 0] - det[i, 0], truth[:, 1] - det[i, 1])
        dist[~free] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= radius:
            hit[i], free[j] = True, False
    return hit, int((~free).sum())


def near(points: np.ndarray, others: np.ndarray, radius: float = MATCH_PX) -> np.ndarray:
    if len(points) == 0 or len(others) == 0:
        return np.zeros(len(points), bool)
    return (np.hypot(points[:, 0, None] - others[:, 0], points[:, 1, None] - others[:, 1]) <= radius).any(1)


def role_xy(im: dict, role: str) -> np.ndarray:
    pts = [(o["x"], o["y"]) for o in im["objects"] if o["role"] == role]
    return np.asarray(pts, np.float32).reshape(-1, 2)


def score_image(det: np.ndarray, im: dict, threshold: float) -> dict:
    d = det[det[:, 2] >= threshold]
    pos, ign, neg = role_xy(im, "pos"), role_xy(im, "ignore"), role_xy(im, "neg")
    hit, tp = match(d, pos)
    other = d[~hit]
    fp = int((~near(other, ign)).sum())
    return {"tp": tp, "fp": fp, "fn": len(pos) - tp, "n_true": len(pos), "n_pred": tp + fp,
            "neg_total": len(neg), "neg_hit": int(near(neg, other[~near(other, ign)]).sum()),
            "mm2": im.get("mm2", im["w"] * im["h"] * (UM_PER_PX / 1000) ** 2)}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def summarise(rows: list) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    mm2 = sum(r["mm2"] for r in rows)
    negs, neg_hit = sum(r["neg_total"] for r in rows), sum(r["neg_hit"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"images": len(rows), "true_mitoses": tp + fn, "predicted": tp + fp, "matched": tp, "precision": round(prec, 4),
            "recall": round(rec, 4), "recall_95ci": wilson(tp, tp + fn),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0, "area_mm2": round(mm2, 2),
            "false_counts_per_mm2": round(fp / mm2, 4) if mm2 else None,
            "look_alikes_called_mitosis": round(neg_hit / negs, 4) if negs else None,
            "mean_abs_count_error_per_image": round(float(np.mean([abs(r["n_pred"] - r["n_true"]) for r in rows])), 3)}


def detect_all(model, images: list, root: str, dev) -> list:
    out = []
    for n, im in enumerate(images, start=1):
        out.append(peaks(predict_map(model, open_image(str(Path(root) / im["path"])), dev)))
        if n % 10 == 0 or n == len(images):
            print(f"  scored {n}/{len(images)} images", file=sys.stderr, flush=True)
    return out


def best_cutoff(dets: list, images: list) -> tuple:
    """The cut-off with the best pooled F1. Ties go to the higher cut-off, which gives fewer false counts."""
    scores, hits, counted, n_true = [], [], [], 0
    for d, im in zip(dets, images):
        pos, ign = role_xy(im, "pos"), role_xy(im, "ignore")
        hit, _ = match(d, pos)                     # highest score first, so a higher cut-off keeps the same matches
        scores.append(d[:, 2])
        hits.append(hit)
        counted.append(hit | ~near(d, ign))
        n_true += len(pos)
    s, h, c = np.concatenate(scores), np.concatenate(hits), np.concatenate(counted)
    best = (-1.0, float(THRESHOLDS[0]))
    for t in THRESHOLDS:
        m = s >= t
        tp, fp = int((m & h).sum()), int((m & c & ~h).sum())
        f1 = round(2 * tp / (2 * tp + fp + n_true - tp), 4) if tp else 0.0
        if f1 >= best[0]:
            best = (f1, float(t))
    return best


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(a) -> int:
    import torch
    run = Path(a.run)
    if (run / "frozen.json").exists():
        raise SystemExit("this run is frozen: it cannot be trained further")
    index = load_index(run)
    root = a.data or index["data"]
    tr = [im for im in index["images"] if im["split"] == "train"]
    tune = [im for im in index["images"] if im["split"] == "tune"]
    if not tr or not tune:
        raise SystemExit("the index needs train and tune images")
    if any(im["source"] == "lung" for im in tune):
        raise SystemExit("LUNG-MITO has no patient identifiers and must not be used for tuning")
    if any(im["source"] == GLIOMA for im in tune):
        raise SystemExit("reviewed glioma images are for training only")
    seed_all(a.seed)
    dev = device()
    model = build_model(a.base).to(dev)
    init = None
    if a.init:
        start_from = Path(a.init)
        info = json.load(open(start_from / "frozen.json"))
        if sha256(start_from / "frozen.pt") != info["sha256"]:
            raise SystemExit("the starting model has changed since it was frozen")
        c = torch.load(start_from / "frozen.pt", map_location=dev, weights_only=False)
        if c["base"] != a.base:
            raise SystemExit(f"the starting model has base {c['base']}, not {a.base}")
        model.load_state_dict(c["model"])
        init = {"run": str(start_from.resolve()), "sha256": info["sha256"]}
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best_f1 = 0, -1.0
    config = {"epochs": a.epochs, "steps": a.steps, "batch": a.batch, "lr": a.lr, "base": a.base, "seed": a.seed,
              "init": init, "glioma_share": a.glioma_share}
    last = run / "last.pt"
    if last.exists():
        c = torch.load(last, map_location=dev, weights_only=False)
        if c.get("config") != config:
            raise SystemExit(f"last.pt was trained with {c.get('config')}, not {config}: keep the same settings")
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        start, best_f1 = c["epoch"], c["best_f1"]
        print(f"resumed after epoch {start}", flush=True)
    log = run / "train-log.jsonl"
    data = Patches(tr, root, a.steps * a.batch, a.seed + start, a.glioma_share)
    gen = torch.Generator().manual_seed(a.seed + start)
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                                         pin_memory=amp, persistent_workers=a.workers > 0, generator=gen)
    for epoch in range(start + 1, a.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y, w in loader:
            x, y, w = x.to(dev), y.to(dev), w.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = loss_fn(model(x), y, w)
            if not torch.isfinite(loss):
                raise SystemExit(f"loss became {loss.item()} at epoch {epoch}: stopping")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
        row = {"epoch": epoch, "loss": round(running / a.steps, 5), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            dets = detect_all(model, tune, root, dev)
            f1, cut = best_cutoff(dets, tune)
            row.update({"tune_f1": f1, "tune_cutoff": cut,
                        "tune": summarise([score_image(d, im, cut) for d, im in zip(dets, tune)])})
            if f1 > best_f1:
                best_f1 = f1
                torch.save({"model": model.state_dict(), "base": a.base, "epoch": epoch, "cutoff": cut, "tune_f1": f1,
                            "tune": row["tune"], "config": config}, run / "best.pt")
        tmp = run / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1, "config": config}, tmp)
        tmp.replace(last)
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "tune"}), flush=True)
    print(json.dumps({"best_tune_f1": best_f1, "model": str(run / "best.pt")}, indent=2))
    return 0


def freeze(a) -> int:
    import torch
    run = Path(a.run)
    path = run / "frozen.json"
    if path.exists():
        raise SystemExit(f"{path} already exists: a model is frozen once")
    if not (run / "best.pt").exists():
        raise SystemExit("no best.pt: train first")
    c = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    shutil.copyfile(run / "best.pt", run / "frozen.pt")
    data = "stMIDOG++ masks (CC BY 4.0) on MIDOG++ images (CC0); LUNG-MITO (CC BY 4.0)"
    if any(im["source"] == GLIOMA for im in load_index(run)["images"] if im["split"] == "train"):
        data += "; TCGA glioma fields and calls reviewed in this project (development patients only)"
    config = c.get("config") or {}
    info = {"model": "frozen.pt", "sha256": sha256(run / "frozen.pt"), "index_sha256": sha256(run / "index.json"),
            "epoch": c["epoch"], "cutoff": c["cutoff"], "cutoff_chosen_on": "tune", "tune": c["tune"],
            "training_data": data, "pretrained_weights": None, "started_from": config.get("init"),
            "glioma_share": config.get("glioma_share", 0.0)}
    path.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


def test(a) -> int:
    import torch
    run = Path(a.run)
    out = run / "unseen-test.json"
    if out.exists():
        raise SystemExit(f"{out} already exists: the unseen test runs once")
    info = json.load(open(run / "frozen.json"))
    if sha256(run / "frozen.pt") != info["sha256"] or sha256(run / "index.json") != info["index_sha256"]:
        raise SystemExit("the frozen model or the split has changed")
    index = load_index(run)
    root = a.data or index["data"]
    images = [im for im in index["images"] if im["split"] == "unseen"]
    if not images:
        raise SystemExit("no unseen images")
    c = torch.load(run / "frozen.pt", map_location="cpu", weights_only=False)
    dev = device()
    model = build_model(c["base"]).to(dev)
    model.load_state_dict(c["model"])
    dets = detect_all(model, images, root, dev)
    rows, by = [], defaultdict(list)
    for d, im in zip(dets, images):
        r = score_image(d, im, info["cutoff"])
        rows.append(r)
        by[im["tumour"]].append(r)
    result = {"cutoff": info["cutoff"], "overall": summarise(rows),
              "by_tumour": {k: summarise(v) for k, v in sorted(by.items())},
              "note": "An internal check on unseen tumour types from the same collection, not a glioma result."}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the finder and keep the best model on the tune cases")
    t.add_argument("--run", required=True)
    t.add_argument("--data", help="default: the folder recorded by prepare")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--steps", type=int, default=500, help="batches per epoch")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--base", type=int, default=24, help="channels in the first layer")
    t.add_argument("--eval-every", type=int, default=5)
    t.add_argument("--workers", type=int, default=int(os.environ.get("NSLOTS", 4)))
    t.add_argument("--seed", type=int, default=20261126)
    t.add_argument("--init", help="a frozen finder run to start from, instead of random weights")
    t.add_argument("--glioma-share", type=float, default=0.0, help="share of patches from reviewed glioma images")
    t.set_defaults(fn=train)
    f = sub.add_parser("freeze", help="fix the best model and its cut-off")
    f.add_argument("--run", required=True)
    f.set_defaults(fn=freeze)
    e = sub.add_parser("test", help="score the frozen model once on the unseen tumour types")
    e.add_argument("--run", required=True)
    e.add_argument("--data", help="default: the folder recorded by prepare")
    e.set_defaults(fn=test)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
