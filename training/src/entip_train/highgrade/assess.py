"""Run the three step-2 detectors on one specimen and build its database actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .actions import build_actions, cell_records, region_records
from .bags import TileBag
from .contract import FeatureResult
from .mitoses import MitosisCounter
from .regions import RegionDetector


@dataclass
class Assessment:
    study_id: str
    results: List[FeatureResult]
    actions: List[dict] = field(default_factory=list)

    def records(self) -> List[dict]:
        return [r.to_record(self.study_id) for r in self.results]


def assess_specimen(study_id: str, bags: Sequence[TileBag], region_detectors: Sequence[RegionDetector] = (),
                    mitoses: Optional[MitosisCounter] = None, candidates: Optional[Sequence[dict]] = None
                    ) -> Assessment:
    """``bags``: one per scan of the specimen. ``candidates``: MitoEd rows over the same scans."""
    results, models, regions, cells, preds = [], [], [], [], []
    by_scan = {b.scan_id: b for b in bags}
    for det in region_detectors:
        res, tile_p = det.assess(bags)
        results.append(res)
        models.append(det.model_record())
        for sid, p in tile_p.items():
            r, pr = region_records(res, by_scan[sid], p, det.cfg.tile_threshold)
            regions += r
            preds += pr
    if mitoses is not None and candidates is not None:
        tissue = sum(b.tissue_area_mm2 for b in bags) if bags else None
        res = mitoses.assess(candidates, [b.scan_id for b in bags] or sorted({c["scan_id"] for c in candidates}),
                             tissue)
        results.append(res)
        models.append(mitoses.model_record())
        c, pr = cell_records(res, candidates)
        cells += c
        preds += pr
    return Assessment(study_id, results, build_actions(study_id, results, models, regions, cells, preds))
