"""Frozen encoders that turn a crop into an embedding.

* ``h-optimus-0`` - bioptimus/H-optimus-0 (ViT-g/14, 1536-d), Apache-2.0, loaded with timm exactly as in
  MitoEd (``init_values=1e-5, dynamic_img_size=False``) and normalised with the mean/std MitoEd uses for the
  H-optimus models. Pretrained on 224 px tiles; needs ``pip install entip-train[foundation]`` and the weights
  from Hugging Face. The default for real slides.
* ``handcrafted`` - colour and texture statistics, about 40-d, no weights, CPU only. For tests, the synthetic
  demo and machines without a GPU. Not a substitute for a foundation model on real tissue.
* ``student:<file>`` - a small model distilled from H-optimus-0 (``entip-glioma distill``), saved as one
  ``student.pt``. It outputs H-optimus-0-sized embeddings and is fast enough for machines without a big GPU.
  Its version string is its own, so student and teacher bags are never mixed.

Each encoder declares its version string; the embedding cache is keyed by it, so changing weights never
mixes old and new vectors.
"""
from __future__ import annotations

from typing import List, Protocol

import numpy as np

ENCODER_LICENCES = {
    "h-optimus-0": ("Apache-2.0", "https://huggingface.co/bioptimus/H-optimus-0"),
    "handcrafted": ("MIT (this repository)", ""),
    "student": ("PolyForm Noncommercial (this repository); distilled from H-optimus-0 (Apache-2.0)", ""),
}

HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)   # as used for H-optimus models in MitoEd (models/encoders.py)
HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)


class Encoder(Protocol):
    name: str
    version: str
    input_px: int
    dim: int

    def embed(self, crops: np.ndarray) -> np.ndarray:
        """crops: (N, input_px, input_px, 3) uint8 RGB -> (N, dim) float32."""


class HandcraftedEncoder:
    name, version, input_px = "handcrafted", "handcrafted@1", 64

    def __init__(self):
        self.dim = len(self._one(np.full((self.input_px, self.input_px, 3), 200, np.uint8)))

    def _one(self, c: np.ndarray) -> np.ndarray:
        x = c.astype(np.float32) / 255.0
        n = x.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) / (n / 2)
        grey = x.mean(-1)
        feats: List[float] = []
        for ring in (r < 0.25, (r >= 0.25) & (r < 0.5), r >= 0.5):
            v = x[ring]
            g = grey[ring]
            feats += list(v.mean(0)) + list(v.std(0))
            feats += [float(np.quantile(g, q)) for q in (0.05, 0.25, 0.5)]
            feats += [float((g < t).mean()) for t in (0.35, 0.5)]
        gy, gx = np.gradient(grey)
        feats += [float(np.hypot(gx, gy).mean()), float(grey.mean()), float(grey.std())]
        return np.asarray(feats, np.float32)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        return np.stack([self._one(c) for c in crops]) if len(crops) else np.zeros((0, self.dim), np.float32)


class HOptimus0Encoder:
    name, input_px, dim = "h-optimus-0", 224, 1536

    def __init__(self, device: str | None = None, batch_size: int = 32):
        import timm
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                 else "cpu")
        self.model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5,
                                       dynamic_img_size=False, num_classes=0).eval().to(self.device)
        self.version = f"h-optimus-0@timm{timm.__version__}"
        self.batch_size = batch_size
        self.mean = torch.tensor(HOPTIMUS_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(HOPTIMUS_STD).view(1, 3, 1, 1)

    def embed(self, crops: np.ndarray) -> np.ndarray:
        t = self.torch
        out = []
        with t.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                x = t.from_numpy(crops[i:i + self.batch_size]).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - self.mean) / self.std).to(self.device)
                out.append(self.model(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)


def build_student(arch: str, out_dim: int, pretrained: bool = False):
    """A student network: a timm backbone (or ``tiny``, a small CNN for tests and CPU smoke runs) plus a linear
    layer to the teacher's embedding size. Shared by distillation (training) and ``StudentEncoder`` (use)."""
    import torch.nn as nn
    if arch == "tiny":
        body = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.GELU(), nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                             nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        width = 64
    else:
        import timm
        body = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        width = body.num_features
    return nn.Sequential(body, nn.Linear(width, out_dim))


class StudentEncoder:
    """A distilled student saved by ``entip-glioma distill train`` (one ``student.pt``: architecture, weights,
    input size, image normalisation and the teacher-space mean/std its outputs are rescaled with). Loaded with
    ``weights_only=True``, so the file cannot run code."""
    name = "student"

    def __init__(self, path: str, device: str | None = None, batch_size: int = 128):
        import torch
        self.torch = torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.version, self.input_px, self.dim = str(ck["version"]), int(ck["input_px"]), int(ck["out_dim"])
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                 else "cpu")
        self.model = build_student(ck["arch"], self.dim).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device)
        self.mean = torch.tensor(ck["img_mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(ck["img_std"]).view(1, 3, 1, 1)
        self.t_mean, self.t_std = ck["target_mean"].float(), ck["target_std"].float()
        self.batch_size = batch_size

    def embed(self, crops: np.ndarray) -> np.ndarray:
        t = self.torch
        out = []
        with t.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                x = t.from_numpy(np.array(crops[i:i + self.batch_size])).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - self.mean) / self.std).to(self.device)
                z = self.model(x).float().cpu()
                out.append((z * self.t_std + self.t_mean).numpy())   # back to the teacher's scale
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)


def get_encoder(name: str, **kw) -> Encoder:
    if name.lower().startswith("student:"):
        return StudentEncoder(name.split(":", 1)[1], **kw)
    n = name.lower().replace("_", "-")
    if n in ("h-optimus-0", "hoptimus0", "h-optimus0"):
        return HOptimus0Encoder(**kw)
    if n == "handcrafted":
        return HandcraftedEncoder()
    raise ValueError(f"unknown encoder {name!r}; open-licence options: {sorted(ENCODER_LICENCES)}")
