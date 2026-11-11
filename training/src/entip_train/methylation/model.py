"""The step-4 agent: H&E bag(s) -> methylation family and class, with a calibrated confidence and a fallback.

Decision, per patient (all of a patient's H&E bags are pooled into one bag):

* Too little tissue (fewer than ``min_tiles`` tiles), a different encoder from training, or a triage call
  that says the tissue is not glioma: ``status = not_assessed``, ``call = indeterminate``, order the array.
* The slide looks unlike anything in training (``novelty.py``): ``call = no_match``, ``status = uncertain``,
  order the array.
* Otherwise the call is the **family** with the highest summed probability (the contract's vocabulary).
  ``confident`` if that family's calibrated probability reaches the family threshold, else ``uncertain`` and
  ``order_test = methylation_array``.
* The classifier **subclass** (e.g. "GBM, RTK II") is given in ``value.class`` only when the family call is
  confident and the subclass reaches its own threshold; otherwise ``value.class`` is null and
  ``value.candidates`` lists the top three.

The model is the ensemble of the cross-validation members (``train.py``): every member was tested on tissue
source sites it never saw, and the deployed probability is the mean of the members' calibrated
probabilities. Thresholds come from those held-out predictions: the lowest confidence at which the
exact 95% upper bound on the error rate stays under ``max_error``. If no threshold reaches that, the
threshold is above 1 and the agent always orders the array. The thresholds were measured on single members,
not on the ensemble, and on TCGA only: confirm them on an external set before relying on them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..wsi.abmil import ABMIL
from ..wsi.bags import Bag
from .novelty import Novelty

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
    temperatures: List[float] = field(default_factory=list)   # one per ensemble member
    class_threshold: float = 1.01
    family_threshold: float = 1.01
    min_tiles: int = 50
    label_source: str = ""
    metrics: dict = field(default_factory=dict)
    licence: str = "MIT (code); trained on openly licensed data only"
    members: List[ABMIL] = field(default_factory=list)
    novelty: Optional[Novelty] = None

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
        """Mean of the members' temperature-scaled probabilities."""
        feats = list(feats)
        return np.mean([m.predict_proba(feats, temperature=t) for m, t in zip(self.members, self.temperatures)],
                       axis=0)

    def attention(self, feats: np.ndarray) -> np.ndarray:
        return np.mean([m.attention(feats) for m in self.members], axis=0)

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
                    "calibration": {"method": "none", "calibrated": False},
                    "order_test": ORDER_TEST, "reasons": reasons, "evidence": []}

        if self.novelty is not None:
            novel, score = self.novelty.is_novel(feats)
            if novel:
                return {**rec, "call": "no_match", "confidence": 0.0, "threshold": _thr(self.family_threshold),
                        "status": "uncertain", "probabilities": {},
                        "calibration": {"method": "none", "calibrated": False},
                        "value": {"class": None, "novelty_score": round(score, 4),
                                  "novelty_threshold": round(self.novelty.threshold, 4), "n_tiles": int(len(feats))},
                        "order_test": ORDER_TEST, "evidence": [],
                        "reasons": [f"slide is unlike the training slides (novelty {score:.2f} > "
                                    f"{self.novelty.threshold:.2f}): no class offered"]}

        p = self.proba([feats])[0]
        pf = p @ self.family_matrix()
        fam = self.families
        kf, kc = int(pf.argmax()), int(p.argmax())
        family, fconf = fam[kf], float(pf[kf])
        cls, cconf = self.classes[kc], float(p[kc])
        confident = fconf >= self.family_threshold
        top = np.argsort(-p)[:3]
        call_class = confident and cconf >= self.class_threshold and self.taxonomy[cls] == family
        value = {"class": cls if call_class else None,
                 "class_confidence": round(cconf, 4),
                 "candidates": [{"class": self.classes[i], "p": round(float(p[i]), 4)} for i in top],
                 "classifier": self.label_source or None, "n_tiles": int(len(feats))}
        reasons.append(f"family {family} p={fconf:.2f} vs threshold {_fmt(self.family_threshold)}")
        if value["class"] is None:
            reasons.append(f"subclass not called: top class {cls} p={cconf:.2f} vs threshold "
                           f"{_fmt(self.class_threshold)}")
        if self.family_threshold > 1:
            reasons.append("no threshold met the agreed error rate on held-out sites: always order the array")

        att = self.attention(feats)
        ev = []
        for i in np.argsort(-att)[:n_evidence]:
            sid, x, y, t = where[i]
            ev.append({"kind": "tile", "id": tile_id(sid, x, y, t), "x_um": x, "y_um": y, "w_um": t, "h_um": t,
                       "weight": round(float(att[i]), 5)})
        return {**rec, "call": family, "confidence": round(fconf, 4), "threshold": _thr(self.family_threshold),
                "calibration": self.calibration(),
                "status": "confident" if confident else "uncertain",
                "probabilities": {f: round(float(v), 4) for f, v in zip(fam, pf)}, "value": value,
                "order_test": None if confident else ORDER_TEST, "reasons": reasons, "evidence": ev}

    def calibration(self) -> dict:
        """How the family confidence was calibrated and whether its threshold met the error target on
        held-out sites (the contract's optional ``calibration`` block)."""
        sel = (self.metrics.get("family") or {}).get("selective") or {}
        out = {"method": "temperature", "calibrated": bool(self.temperatures),
               "validated": bool(self.family_threshold <= 1.0)}
        if sel.get("max_error") is not None:
            out["max_error"] = float(sel["max_error"])
        if sel.get("error_upper") is not None:
            out["error_upper"] = float(sel["error_upper"])
        if sel.get("coverage") is not None:
            out["coverage"] = float(sel["coverage"])
        if self.metrics.get("n_cases") is not None:
            out["n_heldout"] = int(self.metrics["n_cases"])
        return out

    # ------------------------------------------------------------------ files
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            m.save(folder / f"abmil_{i}.npz")
        if self.novelty is not None:
            self.novelty.save(folder / "novelty.npz")
        meta = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ("members", "novelty")}
        meta["n_members"] = len(self.members)
        (folder / "model.json").write_text(json.dumps(meta, indent=2, default=float))
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "MethylationModel":
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        n = meta.pop("n_members")
        nov = Novelty.load(folder / "novelty.npz") if (folder / "novelty.npz").exists() else None
        return cls(**meta, members=[ABMIL.load(folder / f"abmil_{i}.npz") for i in range(n)], novelty=nov)


def tile_id(scan_id: str, x: float, y: float, t: float) -> str:
    return f"{scan_id}:tile@{x:g},{y:g},{t:g}"


def _thr(t: float) -> float:
    return float(min(t, 1.0))


def _fmt(t: float) -> str:
    return "none reached" if t > 1 else f"{t:.2f}"
