"""Train a student to reproduce the teacher's tile embeddings, and measure how well it does on held-out patients.

Targets are standardised per dimension with the training patients' mean and std; the loss is mean squared error
plus (1 - cosine similarity) on them. The checkpoint with the best held-out cosine similarity (in the teacher's
own scale) is saved as ``student.pt``; ``last.pt`` holds the latest epoch with its optimiser so a run cut short
by a time limit can carry on (``--resume``).

``student.pt`` is a plain dict of tensors and strings (loaded with ``weights_only=True``): architecture, output
size, input size, image normalisation, the target mean/std that turn the student's output back into the
teacher's scale, the version string, the teacher's version, and the train / held-out patients.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .tiles import read_index, split_patients


def _device(name: Optional[str] = None) -> str:
    import torch
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dataset(tile_dir: Path, entries: list, mean, std, img_mean, img_std, augment: bool):
    import torch
    from torch.utils.data import Dataset

    class TileSet(Dataset):
        """All tiles of the given slides, read from memory-mapped arrays (opened lazily in each worker)."""

        def __init__(self):
            self.ids = [e["scan_id"] for e in entries]
            self.ends = np.cumsum([e["tiles"] for e in entries])
            self._maps = {}
            self.mean, self.std = torch.as_tensor(mean), torch.as_tensor(std)
            self.img_mean = torch.tensor(img_mean).view(3, 1, 1)
            self.img_std = torch.tensor(img_std).view(3, 1, 1)

        def __len__(self):
            return int(self.ends[-1]) if len(self.ends) else 0

        def _arrays(self, sid):
            if sid not in self._maps:
                self._maps[sid] = (np.load(tile_dir / f"{sid}.tiles.npy", mmap_mode="r"),
                                   np.load(tile_dir / f"{sid}.teacher.npy", mmap_mode="r"))
            return self._maps[sid]

        def __getitem__(self, i):
            s = int(np.searchsorted(self.ends, i, side="right"))
            row = i - (int(self.ends[s - 1]) if s else 0)
            tiles, teacher = self._arrays(self.ids[s])
            x = torch.from_numpy(np.array(tiles[row])).permute(2, 0, 1).float() / 255.0
            if augment:     # orientation means nothing in a section; mild stain variation between labs
                if torch.rand(()) < 0.5:
                    x = x.flip(2)
                x = torch.rot90(x, int(torch.randint(0, 4, ())), dims=(1, 2))
                x = ((x - 0.5) * (1 + 0.1 * (torch.rand(()) - 0.5)) + 0.5
                     + 0.03 * (torch.rand(3, 1, 1) - 0.5)).clamp(0, 1)
            x = (x - self.img_mean) / self.img_std
            y = torch.from_numpy(np.array(teacher[row], np.float32))
            return x, (y - self.mean) / self.std, y

    return TileSet()


def _target_stats(tile_dir: Path, entries: list) -> tuple[np.ndarray, np.ndarray]:
    s = s2 = None
    n = 0
    for e in entries:
        t = np.load(tile_dir / f"{e['scan_id']}.teacher.npy", mmap_mode="r").astype(np.float64)
        s = t.sum(0) if s is None else s + t.sum(0)
        s2 = (t * t).sum(0) if s2 is None else s2 + (t * t).sum(0)
        n += len(t)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def train_student(tile_dir: str | Path, out_dir: str | Path, arch: str = "vit_small_patch16_224",
                  pretrained: bool = True, epochs: int = 10, batch_size: int = 256, lr: float = 5e-4,
                  weight_decay: float = 0.05, holdout: float = 0.2, seed: int = 0, workers: Optional[int] = None,
                  device: Optional[str] = None, resume: bool = False, max_steps: Optional[int] = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from entip_train.encoders import HOPTIMUS_MEAN, HOPTIMUS_STD, build_student

    torch.manual_seed(seed)
    tile_dir, out_dir = Path(tile_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [e for e in read_index(tile_dir) if e["tiles"] > 0]
    teachers = {e["teacher"] for e in entries}
    if len(teachers) != 1:
        raise SystemExit(f"tile cache mixes teachers {sorted(teachers)}; distil from one teacher at a time")
    train_p, held_p = split_patients(entries, holdout, seed)
    tr = [e for e in entries if e["patient"] in train_p]
    va = [e for e in entries if e["patient"] in held_p]
    if not tr or not va:
        raise SystemExit("need tiles from both training and held-out patients")
    mean, std = _target_stats(tile_dir, tr)
    dim, px = int(entries[0]["dim"]), int(entries[0]["input_px"])
    dev = _device(device)
    workers = workers if workers is not None else max(0, min(8, int(os.environ.get("NSLOTS", os.cpu_count() or 1)) - 1))
    ds_tr = _dataset(tile_dir, tr, mean, std, HOPTIMUS_MEAN, HOPTIMUS_STD, augment=True)
    ds_va = _dataset(tile_dir, va, mean, std, HOPTIMUS_MEAN, HOPTIMUS_STD, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=len(ds_tr) > batch_size,
                       pin_memory=dev == "cuda", persistent_workers=workers > 0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=dev == "cuda")

    model = build_student(arch, dim, pretrained=pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(dl_tr))
    total = epochs * steps_per_epoch if max_steps is None else min(max_steps, epochs * steps_per_epoch)
    warm = max(1, min(500, total // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    start, best, history = 0, -2.0, []
    if resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
    amp = dict(device_type="cuda", dtype=torch.bfloat16) if dev == "cuda" else None
    t_mean, t_std = torch.as_tensor(mean, device=dev), torch.as_tensor(std, device=dev)
    step = start * steps_per_epoch
    meta = {"arch": arch, "teacher": next(iter(teachers)), "input_px": px, "out_dim": dim,
            "train_patients": sorted(train_p), "heldout_patients": sorted(held_p),
            "train_tiles": len(ds_tr), "heldout_tiles": len(ds_va), "device": dev}
    for epoch in range(start, epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, z, _ in dl_tr:
            if max_steps is not None and step >= max_steps:
                break
            x, z = x.to(dev, non_blocking=True), z.to(dev, non_blocking=True)
            with (torch.autocast(**amp) if amp else _null()):
                p = model(x).float()
            loss = torch.nn.functional.mse_loss(p, z) + (1 - torch.nn.functional.cosine_similarity(p, z, dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot, nb, step = tot + float(loss.detach()), nb + 1, step + 1
        val = _validate(model, dl_va, t_mean, t_std, dev, amp)
        rec = {"epoch": epoch, "train_loss": tot / max(1, nb), **val, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if val["heldout_cosine"] > best:
            best = val["heldout_cosine"]
            _save_student(out_dir / "student.pt", model, meta, mean, std, rec)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "history": history}, out_dir / "last.pt")
        if max_steps is not None and step >= max_steps:
            break
    report = {**meta, "best_heldout_cosine": best, "history": history}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _validate(model, dl, t_mean, t_std, dev, amp) -> dict:
    import torch
    model.eval()
    cos, sse, sst, n = 0.0, 0.0, 0.0, 0
    with torch.inference_mode():
        for x, z, y in dl:
            x, z, y = x.to(dev), z.to(dev), y.to(dev)
            with (torch.autocast(**amp) if amp else _null()):
                p = model(x).float()
            raw = p * t_std + t_mean
            cos += float(torch.nn.functional.cosine_similarity(raw, y, dim=1).sum())
            sse += float(((p - z) ** 2).sum())
            sst += float((z ** 2).sum())                       # z is standardised: its training mean is ~0
            n += len(x)
    return {"heldout_cosine": cos / max(1, n), "heldout_r2": 1 - sse / max(sst, 1e-12)}


def _save_student(path: Path, model, meta: dict, mean, std, rec: dict) -> None:
    import torch
    from entip_train.encoders import HOPTIMUS_MEAN, HOPTIMUS_STD
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].numpy().tobytes()[:4096])
    teacher = meta["teacher"].split("@")[0]
    version = f"student-{meta['arch']}-from-{teacher}-{h.hexdigest()[:10]}"
    torch.save({"version": version, "arch": meta["arch"], "out_dim": meta["out_dim"], "input_px": meta["input_px"],
                "img_mean": list(HOPTIMUS_MEAN), "img_std": list(HOPTIMUS_STD),
                "target_mean": torch.as_tensor(mean), "target_std": torch.as_tensor(std),
                "teacher": meta["teacher"], "epoch": rec["epoch"], "heldout_cosine": rec["heldout_cosine"],
                "train_patients": meta["train_patients"], "heldout_patients": meta["heldout_patients"],
                "state_dict": sd}, path)


def evaluate_student(tile_dir: str | Path, student: str | Path, patients: Optional[set] = None) -> dict:
    """Agreement with the teacher on the held-out patients stored in ``student.pt`` (or ``patients``):
    per-tile cosine similarity, and cosine similarity of each slide's mean embedding (what a slide-level model
    mostly sees)."""
    import torch
    from entip_train.encoders import StudentEncoder
    tile_dir = Path(tile_dir)
    enc = StudentEncoder(str(student))
    held = patients or set(torch.load(student, map_location="cpu", weights_only=True)["heldout_patients"])
    per_slide, tile_cos = [], []
    for e in read_index(tile_dir):
        if e["patient"] not in held or e["tiles"] == 0:
            continue
        tiles = np.load(tile_dir / f"{e['scan_id']}.tiles.npy", mmap_mode="r")
        teacher = np.load(tile_dir / f"{e['scan_id']}.teacher.npy").astype(np.float32)
        s = enc.embed(np.asarray(tiles))
        c = (s * teacher).sum(1) / (np.linalg.norm(s, axis=1) * np.linalg.norm(teacher, axis=1) + 1e-12)
        ms, mt = s.mean(0), teacher.mean(0)
        per_slide.append({"scan_id": e["scan_id"], "label": e.get("label"), "tiles": int(len(c)),
                          "tile_cosine": float(c.mean()),
                          "slide_mean_cosine": float(ms @ mt / (np.linalg.norm(ms) * np.linalg.norm(mt) + 1e-12))})
        tile_cos.append(c)
    allc = np.concatenate(tile_cos) if tile_cos else np.zeros(0)
    by_label = {}
    for r in per_slide:
        by_label.setdefault(r["label"], []).append(r["tile_cosine"])
    return {"student": enc.version, "heldout_slides": len(per_slide), "heldout_tiles": int(len(allc)),
            "tile_cosine_mean": float(allc.mean()) if len(allc) else None,
            "tile_cosine_p05": float(np.quantile(allc, 0.05)) if len(allc) else None,
            "slide_mean_cosine_mean": float(np.mean([r["slide_mean_cosine"] for r in per_slide])) if per_slide else None,
            "tile_cosine_by_label": {k: float(np.mean(v)) for k, v in by_label.items()}, "slides": per_slide}
