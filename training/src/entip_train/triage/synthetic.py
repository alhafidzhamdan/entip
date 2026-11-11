"""Synthetic tile bags for tests and the demo. No images, no patient data.

Each slide is a bag of random vectors: background tiles plus a share of class-specific "signature" tiles.
Each source adds its own constant offset to every tile, standing in for stain and scanner differences, so
the shortcut checks have something to find when ``source_shift`` is large.
"""
from __future__ import annotations

import numpy as np

from ..wsi.bags import Bag
from .labels import CLASSES


def synthetic_cohort(n_per_class=None, dim: int = 32, tiles=(40, 120), signal: float = 0.5,
                     signal_share: float = 0.1, sources_per_class: int = 2, source_shift: float = 0.3,
                     seed: int = 0):
    """Returns bags, labels, groups (source), patients. Default: four classes the public data supports."""
    n_per_class = n_per_class or {"normal_brain": 24, "glioma": 40, "other_primary_cns": 24, "metastasis": 16}
    rng = np.random.default_rng(seed)
    proto = {c: rng.normal(0, 1, dim) for c in CLASSES}
    bags, labels, groups, patients = [], [], [], []
    k = 0
    for c, n in n_per_class.items():
        shifts = [rng.normal(0, source_shift, dim) for _ in range(sources_per_class)]
        for i in range(n):
            s = i % sources_per_class
            nt = int(rng.integers(*tiles))
            x = rng.normal(0, 1, (nt, dim)) + shifts[s]
            m = max(1, int(signal_share * nt))
            x[:m] += signal * proto[c]
            pid = f"SYN-{k // 2:04d}"            # two slides per patient
            scan = f"SCN-{k:04d}"
            xy = np.column_stack([(np.arange(nt) % 10) * 112.0, (np.arange(nt) // 10) * 112.0])
            bags.append(Bag(scan, x.astype(np.float32), xy, 112.0, "synthetic@1", np.ones(nt, np.float32)))
            labels.append(c)
            groups.append(f"{c}-src{s}")
            patients.append(f"{c}-{pid}")
            k += 1
    return bags, labels, groups, patients
