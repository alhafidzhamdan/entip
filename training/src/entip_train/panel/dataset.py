"""Patients as bags of tiles: every diagnostic slide of a patient pooled into one bag.

A patient's panel result belongs to the tumour, not to one slide, so the model sees all of the patient's tiles
at once and its attention can come from any slide. Each tile keeps the scan it came from, for evidence.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..wsi.bags import Bag


@dataclass
class PatientBag:
    study_id: str
    features: np.ndarray        # (N, D) float32, all tiles of all scans
    scan_index: np.ndarray      # (N,) int, index into scan_ids
    coords_um: np.ndarray       # (N, 2) tile top-left in its own scan's frame
    tile_um: float
    scan_ids: List[str]
    encoder: str
    group: str = ""             # cross-validation group (TCGA tissue source site; hospital for local data)

    def __len__(self) -> int:
        return len(self.features)

    @classmethod
    def from_bags(cls, study_id: str, bags: Sequence[Bag], group: str = "") -> "PatientBag":
        if not bags:
            raise ValueError(f"{study_id}: no slides")
        encs = {b.encoder for b in bags}
        if len(encs) != 1:
            raise ValueError(f"{study_id}: slides embedded with different encoders {sorted(encs)}")
        tiles = {b.tile_um for b in bags}
        if len(tiles) != 1:
            raise ValueError(f"{study_id}: slides tiled at different sizes {sorted(tiles)}")
        return cls(study_id,
                   np.concatenate([b.features for b in bags]).astype(np.float32),
                   np.concatenate([np.full(len(b), i) for i, b in enumerate(bags)]).astype(int),
                   np.concatenate([b.coords_um for b in bags]),
                   float(bags[0].tile_um), [b.scan_id for b in bags], encs.pop(), group)


def read_scan_map(path: str | Path) -> Dict[str, str]:
    """CSV ``scan_id, study_id`` (and optional ``group``) -> {scan_id: study_id}."""
    with open(path, newline="") as fh:
        return {r["scan_id"]: r["study_id"] for r in csv.DictReader(fh)}


def load_patients(bag_dir: str | Path, patient_of: Callable[[str], str],
                  group_of: Optional[Callable[[str], str]] = None) -> List[PatientBag]:
    """Read every ``*.npz`` bag under ``bag_dir`` and pool them by patient. ``patient_of(scan_id)`` names the
    patient (``labels.tcga_patient`` for TCGA, or a scan map's ``dict.get`` for local data); scans it maps to
    ``None`` are skipped."""
    by_patient: Dict[str, List[Bag]] = {}
    for p in sorted(Path(bag_dir).rglob("*.npz")):
        b = Bag.load(p)
        pid = patient_of(b.scan_id)
        if pid:
            by_patient.setdefault(pid, []).append(b)
    return [PatientBag.from_bags(pid, bags, group_of(bags[0].scan_id) if group_of else "")
            for pid, bags in sorted(by_patient.items())]
