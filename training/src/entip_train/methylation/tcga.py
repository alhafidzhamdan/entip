"""Pair TCGA diagnostic H&E slides with 450K methylation from the same tumour.

TCGA barcodes: ``TCGA-TSS-PART-STV-...``. The first 12 characters are the case (patient), characters 14-15
are the sample type (01 = primary solid tumour, 02 = recurrent), and ``TSS`` is the tissue source site.
Diagnostic slides are the FFPE ones with ``DX`` in the slide part (e.g. ``...-01Z-00-DX1``).

Cross-validation groups by tissue source site so a model is never tested on slides from a site it trained on
(slides from one site share staining and scanning).

Downloads (GDC) run on your Mac or Eddie; this file works on lists of file names only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

BARCODE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})[A-Z]?")


@dataclass
class Pair:
    case_id: str
    slide: str
    methylation: str
    sample_type: str
    site: str


def parse(name: str):
    m = BARCODE.search(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def site_of(case_id: str) -> str:
    return case_id.split("-")[1]


def is_diagnostic(slide_name: str) -> bool:
    return bool(re.search(r"-DX\d", slide_name))


def pair_slides(slides: Iterable[str], methylation: Iterable[str], sample_types: Sequence[str] = ("01",)
                ) -> List[Pair]:
    """One pair per case: the first diagnostic slide and the first methylation sample of the same sample type."""
    meth: Dict[tuple, str] = {}
    for f in methylation:
        p = parse(f)
        if p and p[1] in sample_types:
            meth.setdefault(p, f)
    out, seen = [], set()
    for s in sorted(slides):
        p = parse(s)
        if not p or p[1] not in sample_types or not is_diagnostic(s) or p[0] in seen:
            continue
        m = meth.get(p)
        if m:
            seen.add(p[0])
            out.append(Pair(p[0], s, m, p[1], site_of(p[0])))
    return out


def site_folds(case_ids: Sequence[str], k: int = 5, seed: int = 0) -> List[int]:
    """Fold per case, with every case from one tissue source site in the same fold (greedy balance)."""
    import numpy as np
    sites: Dict[str, List[int]] = {}
    for i, c in enumerate(case_ids):
        sites.setdefault(site_of(c), []).append(i)
    rng = np.random.default_rng(seed)
    order = sorted(sites, key=lambda s: (-len(sites[s]), rng.random()))
    load = [0] * k
    fold = [0] * len(case_ids)
    for s in order:
        f = int(np.argmin(load))
        for i in sites[s]:
            fold[i] = f
        load[f] += len(sites[s])
    return fold
