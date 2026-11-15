"""Per-scan embedding cache.

Embeddings are computed once per (scan, encoder version, level, field of view) and kept on disk as
``<root>/<scan_id>/<encoder version>/<level>-<field>um.npz`` (ids + float16 vectors), plus an in-memory
copy. Live heads then train in milliseconds because they never touch pixels again. New objects added
to a scan later (more cells) are embedded and appended on the next ``get``.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .encoders import Encoder
from .slides import Reader, centres

# Field of view per level, in micrometres. A cell crop of 56 µm at 224 px is 0.25 µm/px (MitoEd's
# working resolution); tiles use their own size.
DEFAULT_FIELD_UM = {"cell": 56.0}


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@+-]", "_", s)


class EmbeddingCache:
    def __init__(self, root: Optional[str | Path] = None, batch: int = 256):
        self.root = Path(root) if root else None
        self.mem: Dict[Tuple, Tuple[List[str], np.ndarray]] = {}
        self.batch = batch
        self.lock = threading.RLock()

    def _path(self, key: Tuple) -> Optional[Path]:
        if self.root is None:
            return None
        scan_id, ver, level, field = key
        return self.root / _safe(scan_id) / _safe(ver) / f"{level}-{field:g}um.npz"

    def _load(self, key) -> Tuple[List[str], np.ndarray]:
        if key in self.mem:
            return self.mem[key]
        p = self._path(key)
        if p is not None and p.exists():
            z = np.load(p, allow_pickle=False)
            self.mem[key] = (list(z["ids"]), z["emb"].astype(np.float32))
        else:
            self.mem[key] = ([], np.zeros((0, 0), np.float32))
        return self.mem[key]

    def _save(self, key) -> None:
        p = self._path(key)
        if p is None:
            return
        ids, emb = self.mem[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")         # np.savez keeps names ending in .npz
        np.savez(tmp, ids=np.asarray(ids), emb=emb.astype(np.float16))
        tmp.replace(p)

    def get(self, scan_id: str, level: str, items: Sequence, reader: Reader, encoder: Encoder,
            field_um: Optional[float] = None) -> np.ndarray:
        """Embeddings for ``items`` (Cells or Regions, in order), computing any that are missing."""
        if field_um is None:
            field_um = DEFAULT_FIELD_UM.get(level) or float(items[0].w_um)
        key = (scan_id, encoder.version, level, float(field_um))
        with self.lock:
            ids, emb = self._load(key)
            index = {k: i for i, k in enumerate(ids)}
            want = [self._id(it) for it in items]
            missing = [it for it, k in zip(items, want) if k not in index]
            if missing:
                new = self._embed(missing, reader, encoder, field_um)
                ids = ids + [self._id(it) for it in missing]
                emb = new if emb.size == 0 else np.concatenate([emb, new])
                self.mem[key] = (ids, emb)
                self._save(key)
                index = {k: i for i, k in enumerate(ids)}
            return emb[[index[k] for k in want]]

    @staticmethod
    def _id(it) -> str:
        return getattr(it, "cell_id", None) or it.region_id

    def _embed(self, items, reader, encoder, field_um) -> np.ndarray:
        out = []
        pts = centres(items)
        for i in range(0, len(pts), self.batch):
            crops = np.stack([reader.read_um(x, y, field_um, encoder.input_px) for x, y in pts[i:i + self.batch]])
            out.append(encoder.embed(crops).astype(np.float32))
        return np.concatenate(out)
