"""Synthetic bags and beta values for tests and the demo. No patient data.

Each class has a few "signature" tile types; a slide mixes background tiles with tiles of its class's
signature, so attention-MIL has to find the informative tiles. Classes in one family share most of their
signature, so subclass calls are harder than family calls, as on real slides. Case ids look like TCGA
barcodes so site-grouped CV can be exercised.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..wsi.bags import Bag

ENCODER = "synthetic@1"


def synthetic_cases(classes: Dict[str, str], n_per_class: int = 30, dim: int = 24, n_sites: int = 10,
                    tiles: Tuple[int, int] = (60, 140), signal: float = 0.2, seed: int = 0):
    """Returns [(case_id, Bag, class)]."""
    rng = np.random.default_rng(seed)
    fams = sorted(set(classes.values()))
    fam_proto = {f: rng.normal(0, 0.8, dim) for f in fams}
    cls_proto = {c: fam_proto[f] + rng.normal(0, 0.3, dim) for c, f in classes.items()}
    site_shift = [rng.normal(0, 0.3, dim) for _ in range(n_sites)]       # stain/scanner batch effect
    out = []
    n = 0
    for c in classes:
        for _ in range(n_per_class):
            site = n % n_sites
            case = f"TCGA-S{site}-{n:04d}"
            nt = int(rng.integers(*tiles))
            x = rng.normal(0, 1, (nt, dim)) + site_shift[site]
            k = rng.random(nt) < signal
            x[k] += cls_proto[c]
            coords = np.stack([np.arange(nt) % 20 * 112.0, np.arange(nt) // 20 * 112.0], 1)
            out.append((case, Bag(f"{case}-01Z-00-DX1", x.astype(np.float32), coords, 112.0, ENCODER,
                                  np.ones(nt, np.float32)), c))
            n += 1
    return out


def synthetic_betas(classes: List[str], n_per_class: int = 20, n_cpgs: int = 400, seed: int = 0):
    """(sample_ids, cpg_ids, betas, labels): each class methylates its own block of CpGs."""
    rng = np.random.default_rng(seed)
    cpgs = [f"cg{i:08d}" for i in range(n_cpgs)]
    block = n_cpgs // (len(classes) + 1)
    ids, rows, labels = [], [], []
    for k, c in enumerate(classes):
        for j in range(n_per_class):
            b = rng.beta(2, 8, n_cpgs)
            b[k * block:(k + 1) * block] = rng.beta(8, 2, block)
            ids.append(f"{c}-{j}")
            rows.append(b)
            labels.append(c)
    return ids, cpgs, np.asarray(rows, np.float32), labels
