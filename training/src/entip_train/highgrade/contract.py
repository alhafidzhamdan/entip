"""The output of each step-2 detector, in the step-5 contract ``entip.glioma_step/1``.

The contract lives in ``dx/contract/README.md`` (step-5 rules engine). One record per finding per patient:

``call``        present | absent | indeterminate (mitotic_activity "present" = elevated for a diffuse glioma)
``confidence``  calibrated probability that ``call`` is right
``threshold``   the accept threshold for this finding and model version
``status``      confident (use it) | uncertain (order the real test, here ``pathologist_review``)
``calibration`` how ``confidence`` was calibrated and whether the threshold met its error target on held-out
                specimens (``validated``); this is what the per-step confidence endpoint shows
``evidence``    pointers into the db: regions, tiles or cells to look at first

Calls: with p = probability the feature is present and t = threshold, the call is ``present`` when p >= t,
``absent`` when 1 - p >= t, otherwise ``indeterminate``. A guard (too little tissue, unfamiliar tissue,
uncertain candidate list) keeps the best-guess call but sets ``status`` to ``uncertain``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

CONTRACT = "entip.glioma_step/1"
STEP = "high_grade_features"
CALLS = ("present", "absent", "indeterminate")
FINDINGS = ("necrosis", "microvascular_proliferation", "mitotic_activity")
REVIEW = "pathologist_review"


@dataclass
class FeatureResult:
    finding: str
    scan_ids: List[str]
    call: str
    p_present: float
    confidence: float
    threshold: float
    status: str
    model_id: str
    model_version: str = ""
    licence: str = ""
    value: dict = field(default_factory=dict)          # area_mm2, mitoses_per_mm2, ...
    evidence: List[dict] = field(default_factory=list)  # [{kind, id, x_um, y_um, w_um?, h_um?, weight}]
    order_test: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    calibration: Optional[dict] = None                  # contract block: method, calibrated, validated, ...

    def __post_init__(self):
        if self.call not in CALLS:
            raise ValueError(f"call must be one of {CALLS}")
        if self.finding not in FINDINGS:
            raise ValueError(f"finding must be one of {FINDINGS}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_record(self, study_id: str) -> dict:
        """The ``entip.glioma_step/1`` record the rules engine reads."""
        rec = {
            "contract": CONTRACT, "step": STEP, "finding": self.finding, "study_id": study_id,
            "scan_ids": list(self.scan_ids), "call": self.call, "source": "ai",
            "confidence": round(float(self.confidence), 4), "threshold": float(self.threshold),
            "status": self.status,
            "probabilities": {"present": round(float(self.p_present), 4),
                              "absent": round(float(1 - self.p_present), 4)},
            "value": self.value,
            "evidence": [{k: v for k, v in e.items() if k in ("kind", "id", "x_um", "y_um", "w_um", "h_um", "weight")}
                         for e in self.evidence],
            "order_test": self.order_test, "reasons": list(self.reasons),
            "model": {"model_id": self.model_id, "version": self.model_version, "licence": self.licence},
            "provenance": {"kind": "model", "actor": f"model:{self.model_id}", "version": self.model_version},
        }
        if self.calibration:
            rec["calibration"] = self.calibration
        return rec


def decide(p: float, threshold: float, guards: Optional[List[str]] = None):
    """(call, confidence, status, order_test, reasons) from p(present). Guards force ``uncertain``."""
    reasons = list(guards or [])
    if p >= threshold:
        call, conf = "present", p
    elif 1 - p >= threshold:
        call, conf = "absent", 1 - p
    else:
        call, conf = "indeterminate", max(p, 1 - p)
        reasons.append(f"p(present) {p:.2f} is inside the uncertain band ({1 - threshold:.2f}, {threshold:.2f})")
    status = "uncertain" if reasons else "confident"
    return call, float(conf), status, (REVIEW if reasons else None), reasons
