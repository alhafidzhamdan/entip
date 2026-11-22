"""Run the step-3 region detectors on one specimen and build its database actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .actions import build_actions, region_records
from ..wsi.bags import Bag
from .contract import FeatureResult
from .regions import RegionDetector


@dataclass
class Assessment:
    study_id: str
    results: List[FeatureResult]
    actions: List[dict] = field(default_factory=list)

    def records(self) -> List[dict]:
        return [r.to_record(self.study_id) for r in self.results]


def assess_specimen(study_id: str, bags: Sequence[Bag], region_detectors: Sequence[RegionDetector] = ()
                    ) -> Assessment:
    """``bags``: one per scan of the specimen. No ``mitotic_activity`` record is made until a new counter exists."""
    results, models, regions, preds = [], [], [], []
    by_scan = {b.scan_id: b for b in bags}
    for det in region_detectors:
        res, tile_p = det.assess(bags)
        results.append(res)
        models.append(det.model_record())
        for sid, p in tile_p.items():
            r, pr = region_records(res, by_scan[sid], p, det.cfg.tile_threshold)
            regions += r
            preds += pr
    return Assessment(study_id, results, build_actions(study_id, results, models, regions, (), preds))
