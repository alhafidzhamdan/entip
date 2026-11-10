"""The step-4 agent: H&E bag(s) -> methylation family and class, with a calibrated confidence and a fallback.

Decision, per patient (all of a patient's H&E bags are pooled into one bag):

* Too little tissue (fewer than ``min_tiles`` tiles), a different encoder from training, or a triage call
  that says the tissue is not glioma: ``status = not_assessed``, ``call = indeterminate``, order the array.
* Otherwise the call is the **family** with the highest summed probability (the contract's vocabulary).
  ``confident`` if that family's calibrated probability reaches the family threshold, else ``uncertain`` and
  ``order_test = methylation_array``.
* The classifier **subclass** (e.g. "GBM, RTK II") is given in ``value.class`` only when it reaches its own
  threshold; otherwise ``value.class`` is null and ``value.candidates`` lists the top three.

Thresholds come from out-of-fold predictions on slides from held-out tissue source sites
(``train.py``): the lowest confidence at which the one-sided 95% upper bound on the error rate stays under
``max_error``. If no threshold reaches that, the threshold is above 1 and the agent always orders the array.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..wsi.abmil import ABMIL
from ..wsi.bags import Bag
from .classes import Taxonomy

CONTRACT = "entip.glioma_step/1"
STEP = FINDING = "methylation_class"
ORDER_TEST = "methylation_array"
GLIOMA_TRIAGE_CALLS = ("glioma",)


def pool_bags(bags: Sequence[Bag]) -> tuple[np.ndarray, list[tuple[str, float, float, float]]]:
    """Concatenate a patient's bags; also return (scan_id, x_um, y_um, tile_um) per tile for evidence."""
    if not bags:
        return np.zeros((0, 0), np.float32), []
    enc = {b.encoder for b in bags}
    if len(enc) > 1:
        raise ValueError(f"bags come from different encoders: {sorted(enc)}")
    feats = np.concatenate([b.features for b in bags])
    where = [(b.scan_id, float(x), float(y), b.tile_um) for b in bags for x, y in b.coords_um]
    return feats, where


@dataclass
class MethylationModel:
    model_id: str
    version: str
    classes: List[str]
    taxonomy: Dict[str, str]                 # class -> family
    encoder: str                             # encoder version the bags must come from
    temperature: float = 1.0
    class_threshold: float = 1.01
    family_threshold: float = 1.01
    min_tiles: int = 50
    label_source: str = ""
    metrics: dict = field(default_factory=dict)
    licence: str = "MIT (code); trained on openly licensed data only"
    mil: Optional[ABMIL] = None

    # ------------------------------------------------------------------ probabilities
    @property
    def families(self) -> List[str]:
        return sorted(set(self.taxonomy[c] for c in self.classes))

    def family_matrix(self) -> np.ndarray:
        fam = self.families
        M = np.zeros((len(self.classes), len(fam)), np.float32)
        for i, c in enumerate(self.classes):
            M[i, fam.index(self.taxonomy[c])] = 1
        return M

    def proba(self, feats: Sequence[np.ndarray]) -> np.ndarray:
        return self.mil.predict_proba(list(feats), temperature=self.temperature)

    # ------------------------------------------------------------------ the agent
    def predict(self, bags: Sequence[Bag], study_id: str, triage_call: Optional[str] = None,
                n_evidence: int = 8) -> dict:
        """One finding record (``entip.glioma_step/1``) for one patient."""
        scan_ids = sorted({b.scan_id for b in bags})
        rec = {"contract": CONTRACT, "step": STEP, "finding": FINDING, "study_id": study_id,
               "scan_ids": scan_ids, "source": "ai",
               "model": {"model_id": self.model_id, "version": self.version, "licence": self.licence},
               "provenance": {"kind": "model", "actor": self.model_id, "version": self.version}}
        reasons: List[str] = []
        feats, where = pool_bags(bags) if bags else (np.zeros((0, 0)), [])
        bad_enc = sorted({b.encoder for b in bags} - {self.encoder})
        if triage_call is not None and triage_call not in GLIOMA_TRIAGE_CALLS:
            reasons.append(f"triage called the tissue {triage_call!r}, not glioma: methylation class not assessed")
        if bad_enc:
            reasons.append(f"bags were embedded with {bad_enc}, model expects {self.encoder}")
        if len(feats) < self.min_tiles:
            reasons.append(f"only {len(feats)} tissue tiles (< {self.min_tiles})")
        if reasons:
            return {**rec, "call": "indeterminate", "confidence": 0.0, "threshold": _thr(self.family_threshold),
                    "status": "not_assessed", "probabilities": {}, "value": {"class": None},
                    "order_test": ORDER_TEST, "reasons": reasons, "evidence": []}

        p = self.proba([feats])[0]
        pf = p @ self.family_matrix()
        fam = self.families
        kf, kc = int(pf.argmax()), int(p.argmax())
        family, fconf = fam[kf], float(pf[kf])
        cls, cconf = self.classes[kc], float(p[kc])
        confident = fconf >= self.family_threshold
        top = np.argsort(-p)[:3]
        value = {"class": cls if (cconf >= self.class_threshold and self.taxonomy[cls] == family) else None,
                 "class_confidence": round(cconf, 4),
                 "candidates": [{"class": self.classes[i], "p": round(float(p[i]), 4)} for i in top],
                 "classifier": self.label_source or None, "n_tiles": int(len(feats))}
        reasons.append(f"family {family} p={fconf:.2f} vs threshold {_fmt(self.family_threshold)}")
        if value["class"] is None:
            reasons.append(f"subclass not called: top class {cls} p={cconf:.2f} vs threshold "
                           f"{_fmt(self.class_threshold)}")
        if self.family_threshold > 1:
            reasons.append("no threshold met the agreed error rate on held-out sites: always order the array")

        att = self.mil.attention(feats)
        ev = []
        for i in np.argsort(-att)[:n_evidence]:
            sid, x, y, t = where[i]
            ev.append({"kind": "tile", "id": tile_id(sid, x, y, t), "x_um": x, "y_um": y, "w_um": t, "h_um": t,
                       "weight": round(float(att[i]), 5)})
        return {**rec, "call": family, "confidence": round(fconf, 4), "threshold": _thr(self.family_threshold),
                "status": "confident" if confident else "uncertain",
                "probabilities": {f: round(float(v), 4) for f, v in zip(fam, pf)}, "value": value,
                "order_test": None if confident else ORDER_TEST, "reasons": reasons, "evidence": ev}

    # ------------------------------------------------------------------ files
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.mil.save(folder / "abmil.npz")
        meta = {k: v for k, v in asdict(self).items() if k != "mil"}
        (folder / "model.json").write_text(json.dumps(meta, indent=2, default=float))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "MethylationModel":
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        return cls(**meta, mil=ABMIL.load(folder / "abmil.npz"))


def tile_id(scan_id: str, x: float, y: float, t: float) -> str:
    return f"{scan_id}:tile@{x:g},{y:g},{t:g}"


def _thr(t: float) -> float:
    return float(min(t, 1.0))


def _fmt(t: float) -> str:
    return "none reached" if t > 1 else f"{t:.2f}"
