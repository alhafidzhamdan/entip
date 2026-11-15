"""Synthetic patients for tests and the demo. No patient data, and not a model of real glioma biology.

Each synthetic patient gets a made-up subtype and marker calls loosely shaped like adult diffuse glioma groups
(so that 1p/19q only varies among IDH-mutant cases). Each altered marker shifts a share of the patient's tiles
along its own direction in embedding space. Markers in ``no_signal`` get no shift at all, so a working gate
must refuse to call them. Each patient belongs to one of a few "sites", with a small site-wide offset, so
site-grouped cross-validation is exercised.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .dataset import PatientBag
from .labels import Labels
from .markers import DEFAULT_MARKERS

ENCODER = "synthetic@1"

# subtype -> {finding: probability of the altered call}; 1p19q absent where it does not apply
SUBTYPES: Dict[str, Dict[str, float]] = {
    "codel": {"IDH": 1.0, "1p19q": 1.0, "ATRX": 0.05, "TP53": 0.1, "TERT": 0.9, "CDKN2A": 0.05, "EGFR": 0.0,
              "chr7_chr10": 0.0, "H3K27": 0.0, "H3G34": 0.0, "MGMT": 0.9},
    "astro": {"IDH": 1.0, "1p19q": 0.0, "ATRX": 0.8, "TP53": 0.9, "TERT": 0.05, "CDKN2A": 0.1, "EGFR": 0.0,
              "chr7_chr10": 0.0, "H3K27": 0.0, "H3G34": 0.0, "MGMT": 0.8},
    "gbm": {"IDH": 0.0, "ATRX": 0.05, "TP53": 0.3, "TERT": 0.8, "CDKN2A": 0.6, "EGFR": 0.45,
            "chr7_chr10": 0.7, "H3K27": 0.0, "H3G34": 0.0, "MGMT": 0.45},
    "midline": {"IDH": 0.0, "ATRX": 0.3, "TP53": 0.6, "TERT": 0.0, "CDKN2A": 0.0, "EGFR": 0.0,
                "chr7_chr10": 0.0, "H3K27": 1.0, "H3G34": 0.0, "MGMT": 0.2},
}
MIX = {"codel": 0.2, "astro": 0.25, "gbm": 0.45, "midline": 0.1}


def synthetic_cohort(n: int = 160, dim: int = 24, tiles: Tuple[int, int] = (40, 90), n_sites: int = 8,
                     strength: float = 2.5, share: float = 0.35, no_signal: Sequence[str] = ("MGMT", "H3G34"),
                     seed: int = 0, prefix: str = "SYN", world_seed: int = 0) -> Tuple[List[PatientBag], Labels, Dict[str, str]]:
    """Returns (patients, labels, subtype per patient). ``world_seed`` fixes the marker directions and site
    offsets, so cohorts drawn with different ``seed`` values come from the same synthetic world."""
    world = np.random.default_rng(world_seed)
    dirs = {f: v / np.linalg.norm(v) for f, v in
            ((f, world.normal(size=dim)) for f in DEFAULT_MARKERS)}
    site_off = world.normal(0, 0.3, (n_sites, dim))
    rng = np.random.default_rng(seed + 1_000_003 * (world_seed + 1))
    names, probs = list(MIX), np.array(list(MIX.values()))
    patients, labels, subtype = [], {}, {}
    for i in range(n):
        sid = f"{prefix}-{i:04d}"
        st = names[rng.choice(len(names), p=probs / probs.sum())]
        lab = {}
        for f, pa in SUBTYPES[st].items():
            lab[f] = DEFAULT_MARKERS[f].calls[int(rng.random() < pa)]
        site = int(rng.integers(n_sites))
        k = int(rng.integers(*tiles))
        x = rng.normal(size=(k, dim)) + site_off[site]
        for f, call in lab.items():
            if call == DEFAULT_MARKERS[f].altered and f not in no_signal:
                hit = rng.random(k) < share
                x[hit] += strength * dirs[f]
        n_scans = 1 + int(rng.random() < 0.3)
        scan_index = (np.arange(k) % n_scans).astype(int)
        coords = np.stack([rng.integers(0, 200, k) * 112.0, rng.integers(0, 200, k) * 112.0], 1)
        patients.append(PatientBag(sid, x.astype(np.float32), scan_index, coords, 112.0,
                                   [f"{sid}-S{j + 1}" for j in range(n_scans)], ENCODER, f"site{site}"))
        labels[sid] = lab
        subtype[sid] = st
    return patients, labels, subtype
