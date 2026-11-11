"""Pair TCGA diagnostic H&E slides with 450K methylation from the same tumour.

TCGA barcodes: ``TCGA-TSS-PART-STV-...``. The first 12 characters are the case (patient), characters 14-15
are the sample type (01 = primary solid tumour, 02 = recurrent), and ``TSS`` is the tissue source site.
Diagnostic slides are the FFPE ones with ``DX`` in the slide part (e.g. ``...-01Z-00-DX1``).

Cross-validation groups by tissue source site so a model is never tested on slides from a site it trained on
(slides from one site share staining and scanning).

Slide files from the GDC keep the barcode in their name (``TCGA-06-0125-01Z-00-DX1.<uuid>.svs``), but
methylation beta files do not (they are named by file UUID). Download the GDC **sample sheet** with the
methylation files and pass it to ``read_sample_sheet``; it maps each file name to its sample barcode. List
only 450K (or EPIC) files: the old 27K array shares too few CpGs (the label step refuses low coverage).

Downloads (GDC) run on your Mac or Eddie; this file works on file names and sample sheets only.
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
    """Tissue source site of a TCGA case id; any other id is its own group."""
    parts = case_id.split("-")
    return parts[1] if len(parts) > 2 and parts[0] == "TCGA" else case_id


def is_diagnostic(slide_name: str) -> bool:
    return bool(re.search(r"-DX\d", slide_name))


def read_sample_sheet(path: str) -> Dict[str, str]:
    """GDC sample sheet (TSV with ``File Name`` and ``Sample ID`` columns) -> {file name: sample barcode}.
    Column names are matched ignoring case, spaces and underscores."""
    import csv
    norm = lambda h: h.lower().replace(" ", "").replace("_", "")
    with open(path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    head = [norm(h) for h in rows[0]]
    try:
        fi, si = head.index("filename"), head.index("sampleid")
    except ValueError:
        raise ValueError(f"{path}: needs 'File Name' and 'Sample ID' columns, got {rows[0]}")
    out = {}
    for r in rows[1:]:
        if len(r) > max(fi, si):
            out[r[fi]] = r[si].split(",")[0].strip()    # pooled files list several samples; take the first
    return out


def pair_slides(slides: Iterable[str], methylation: Iterable[str], sample_types: Sequence[str] = ("01",),
                sample_of: Optional[Dict[str, str]] = None) -> List[Pair]:
    """One pair per case: the first diagnostic slide and the first methylation sample of the same sample type.
    ``sample_of`` maps methylation file names to barcodes (from ``read_sample_sheet``) when the names lack them."""
    meth: Dict[tuple, str] = {}
    for f in methylation:
        p = parse(sample_of.get(f, f) if sample_of else f)
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


def group_folds(groups: Sequence[str], k: int = 5, seed: int = 0) -> List[int]:
    """Fold per item, with every item of one group in the same fold (largest groups first, greedy balance)."""
    import numpy as np
    members: Dict[str, List[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(str(g), []).append(i)
    rng = np.random.default_rng(seed)
    ties = {g: rng.random() for g in sorted(members)}
    order = sorted(members, key=lambda g: (-len(members[g]), ties[g]))
    load = [0] * k
    fold = [0] * len(groups)
    for g in order:
        f = int(np.argmin(load))
        for i in members[g]:
            fold[i] = f
        load[f] += len(members[g])
    return fold


def site_folds(case_ids: Sequence[str], k: int = 5, seed: int = 0) -> List[int]:
    """Fold per case, with every case from one tissue source site in the same fold."""
    return group_folds([site_of(c) for c in case_ids], k, seed)
