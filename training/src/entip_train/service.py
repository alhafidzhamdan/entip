"""The training service: ties store, embedding cache, live heads, review queue and gated retrain together.

The HTTP API (``api.py``) is a thin layer over this class, so the same calls work from Python, tests,
or a notebook.

A *head* is one feature of interest (its ``head_id`` is the feature name). Each click adds an
Annotation, retrains a draft head (warm start), stores it as a Model row with status ``draft`` and
returns the new review queue. ``retrain`` runs a gated round that may promote a ``champion``.
"""
from __future__ import annotations

import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from . import __version__
from .cache import EmbeddingCache
from .encoders import Encoder
from .heads import (HeadConfig, LiveHead, background_negatives, cv_balanced_accuracy, labels_from_annotations,
                    review_queue)
from .ontology import NEGATIVE, POSITIVE, Annotation, Cell, Feature, Model, Region, Scan
from .retrain import GateConfig, run_round
from .slides import Reader, default_reader_factory, grid_regions
from .store import Store, new_id, now


def slug(name: str) -> str:
    """Feature key in the database's style: lower case, words joined by underscores ("mitotic_figure")."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not s:
        raise ValueError("feature name is empty")
    return s


def person(actor: str) -> str:
    return actor if ":" in actor else f"person:{actor}"


class TrainingService:
    def __init__(self, store: Store, encoder: Encoder, artifacts: str | Path, cache: Optional[EmbeddingCache] = None,
                 head_cfg: Optional[HeadConfig] = None, gate_cfg: Optional[GateConfig] = None,
                 reader_factory: Optional[Callable[[Scan], Reader]] = None, tile_um: float = 112.0,
                 write_predictions: bool = False):
        self.store, self.encoder = store, encoder
        self.artifacts = Path(artifacts)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.cache = cache or EmbeddingCache(self.artifacts / "embeddings")
        self.head_cfg = head_cfg or HeadConfig()
        self.gate_cfg = gate_cfg or GateConfig()
        self.reader_factory = reader_factory or default_reader_factory()
        self.write_predictions = write_predictions        # after each click, write the scan's scores to the store
        self.tile_um = tile_um
        self.readers: Dict[str, Reader] = {}
        self.heads: Dict[str, LiveHead] = {}
        self.locked_scans: Dict[str, set] = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs: Dict[str, dict] = {}
        self.skipped_scans: List[str] = []

    # ------------------------------------------------------------ scans
    def register_scan(self, scan: Scan, reader: Optional[Reader] = None, cells: Sequence[Cell] = (),
                      regions: Sequence[Region] = (), actor: str = "system") -> Scan:
        self.store.add_scan(scan, actor)
        if reader is not None:
            self.readers[scan.scan_id] = reader
        if cells:
            self.store.add_cells(cells, actor)
        if regions:
            self.store.add_regions(regions, actor)
        return scan

    def reader(self, scan_id: str) -> Reader:
        if scan_id not in self.readers:
            if self.reader_factory is None:
                raise KeyError(f"no reader for scan {scan_id}")
            self.readers[scan_id] = self.reader_factory(self.store.scan(scan_id))
        return self.readers[scan_id]

    def items(self, scan_id: str, level: str) -> List:
        if level == "cell":
            return self.store.cells(scan_id)
        regs = self.store.regions(scan_id)
        if not regs:
            regs = grid_regions(self.store.scan(scan_id), self.tile_um)
            self.store.add_regions(regs)
        return regs

    def embeddings(self, scan_id: str, level: str, items: Sequence) -> np.ndarray:
        return self.cache.get(scan_id, level, items, self.reader(scan_id), self.encoder)

    def warm_cache(self, scan_id: str, level: str = "cell") -> int:
        """Embed every object of a scan now, so the first click is fast."""
        it = self.items(scan_id, level)
        if it:
            self.embeddings(scan_id, level, it)
        return len(it)

    # ------------------------------------------------------------ heads
    def create_head(self, feature: str, level: str = "cell", actor: str = "system", description: str = "",
                    positives_only: bool = False, locked_scans: Sequence[str] = ()) -> str:
        try:                                              # an existing key (e.g. the database's) is used as is
            self.store.feature(feature)
            head_id = feature
        except KeyError:
            head_id = slug(feature)
        try:
            self.store.feature(head_id)
        except KeyError:
            self.store.add_feature(Feature(head_id, level, description or feature, person(actor), positives_only))
        if locked_scans:
            self.locked_scans.setdefault(head_id, set()).update(locked_scans)
        return head_id

    def head_info(self, head_id: str) -> dict:
        f = self.store.feature(head_id)
        anns = self.store.annotations(head_id)
        draft = self.store.latest_model(head_id, "draft")
        champ = self.store.latest_model(head_id, "champion")
        return {"head_id": head_id, "feature": asdict(f),
                "n_pos": sum(a.label == POSITIVE for a in anns if not a.locked),
                "n_neg": sum(a.label == NEGATIVE for a in anns if not a.locked),
                "n_locked": sum(a.locked for a in anns),
                "draft": draft.to_dict() if draft else None, "champion": champ.to_dict() if champ else None}

    # ------------------------------------------------------------ marks
    def add_label(self, head_id: str, scan_id: str, label, actor: str, cell_id: Optional[str] = None,
                  object_id: Optional[str] = None, x_um: Optional[float] = None, y_um: Optional[float] = None,
                  polygon_um: Optional[List[List[float]]] = None, locked: Optional[bool] = None,
                  train: bool = True, queue_n: int = 20, annotation_id: Optional[str] = None) -> dict:
        """One mark. ``label``: 1/0, True/False or "positive"/"negative". Returns the new head state.

        ``annotation_id``: the mark is already saved in the store (e.g. the GUI wrote it to the database
        first); it is not written again, only trained on."""
        lab = POSITIVE if label in (1, True, "1", POSITIVE, "pos") else NEGATIVE if label in (
            0, False, "0", NEGATIVE, "neg") else None
        if lab is None:
            raise ValueError("label must be 1/0 or positive/negative")
        self.store.scan(scan_id)
        a = self.store.get_annotation(scan_id, annotation_id) if annotation_id else None
        if a is not None and a.feature != head_id:
            raise ValueError(f"annotation {annotation_id} is for feature {a.feature}, not {head_id}")
        if a is None:
            oid = object_id or cell_id
            if oid:
                geom = {"type": "object", "object_id": oid}
            elif polygon_um:
                geom = {"type": "polygon", "points_um": polygon_um}
            elif x_um is not None and y_um is not None:
                geom = {"type": "point", "x_um": float(x_um), "y_um": float(y_um)}
            else:
                raise ValueError("give annotation_id, cell_id/object_id, x_um and y_um, or polygon_um")
            if locked is None:
                locked = scan_id in self.locked_scans.get(head_id, set())
            a = self.store.add_annotation(Annotation(annotation_id or new_id("ann"), scan_id, head_id, lab, geom,
                                                     person(actor), locked=bool(locked)))
        out = {"label_id": a.annotation_id, "locked": a.locked}
        if train and not a.locked:
            try:
                m = self.train(head_id, actor)
                out.update(model_version=m.model_id, metrics=m.metrics)
                if self.write_predictions:
                    self.predict(head_id, scan_id, m.model_id, store_scores=True)
                out["queue"] = self.queue(head_id, scan_id, queue_n)
            except ValueError as e:                       # e.g. only one class so far
                out.update(model_version=None, metrics={}, note=str(e), queue=[])
        info = self.head_info(head_id)
        out.update(n_pos=info["n_pos"], n_neg=info["n_neg"])
        return out

    def retract_label(self, head_id: str, label_id: str, actor: str, train: bool = True) -> dict:
        self.store.retract_annotation(label_id, person(actor))
        out = {"label_id": label_id, "retracted": True}
        if train:
            try:
                out["model_version"] = self.train(head_id, actor).model_id
            except ValueError as e:
                out["note"] = str(e)
        return out

    # ------------------------------------------------------------ training data
    def _dataset(self, head_id: str, locked: bool, with_background: bool = True):
        """Stack (X, y, w, groups, n_marked) across every scan with marks for this head. Scans with no pixels
        (no reader, nothing cached) are skipped and listed in ``self.skipped_scans``: their marks stay saved."""
        f = self.store.feature(head_id)
        anns = self.store.annotations(head_id, locked=locked)
        by_scan: Dict[str, List[Annotation]] = {}
        for a in anns:
            by_scan.setdefault(a.scan_id, []).append(a)
        rng = np.random.default_rng(self.head_cfg.seed + len(anns))
        Xs, ys, ws, gs, marked = [], [], [], [], []
        self.skipped_scans = []
        for sid, a in by_scan.items():
            items = self.items(sid, f.level)
            lab = labels_from_annotations(items, a, f.level, self.head_cfg.snap_um)
            if not lab:
                continue
            bg = []
            has_neg = any(y == 0 for y, _ in lab.values())
            if with_background and (f.positives_only or not has_neg):
                bg = background_negatives(items, lab, self.head_cfg, rng)
            idx = sorted(lab) + bg
            try:
                emb = self.embeddings(sid, f.level, [items[i] for i in idx])
            except KeyError:
                self.skipped_scans.append(sid)
                continue
            Xs.append(emb)
            ys += [lab[i][0] for i in sorted(lab)] + [0] * len(bg)
            ws += [1.0] * len(lab) + [self.head_cfg.bg_weight] * len(bg)
            gs += [sid] * len(idx)
            marked += [True] * len(lab) + [False] * len(bg)
        if not Xs:
            return np.zeros((0, 1)), np.zeros(0, int), np.zeros(0), [], np.zeros(0, bool)
        return np.concatenate(Xs), np.asarray(ys), np.asarray(ws, float), gs, np.asarray(marked)

    # ------------------------------------------------------------ tier 1
    def _save(self, head: LiveHead, model_id: str) -> str:
        p = self.artifacts / "heads" / f"{model_id}.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(head, p)
        return str(p)

    def load_head(self, model_id: str) -> LiveHead:
        if model_id not in self.heads:
            self.heads[model_id] = joblib.load(self.store.model(model_id).artifact)   # our own artifacts only
        return self.heads[model_id]

    def train(self, head_id: str, actor: str = "system", kind: Optional[str] = None) -> Model:
        with self.lock:
            X, y, w, _, marked = self._dataset(head_id, locked=False)
            cfg = replace(self.head_cfg, kind=kind or self.head_cfg.kind)
            parent = self.store.latest_model(head_id, "draft")
            head = LiveHead(cfg, self.load_head(parent.model_id) if parent else None)
            head.fit(X, y, w)                              # raises ValueError if one class only
            n_pos, n_neg = int((y[marked] == 1).sum()), int((y[marked] == 0).sum())
            metrics = {"n_pos": n_pos, "n_neg": n_neg, "n_background": int((~marked).sum()),
                       "warm_start": head.warm, "skipped_scans": list(self.skipped_scans),
                       "cv_balanced_accuracy": cv_balanced_accuracy(X, y, cfg, w, marked)}
            mid = new_id("head")
            self.heads[mid] = head
            m = Model(mid, head_id, f"live-head:{cfg.kind}", self.encoder.version, "draft", person(actor),
                      n_pos + n_neg, parent=parent.model_id if parent else None, metrics=metrics,
                      artifact=self._save(head, mid))
            return self.store.add_model(m)

    def _model_for(self, head_id: str, model: Optional[str]) -> Model:
        if model in (None, "draft", "latest"):
            m = self.store.latest_model(head_id, "draft")
        elif model == "champion":
            m = self.store.latest_model(head_id, "champion")
        else:
            m = self.store.model(model)
        if m is None:
            raise LookupError(f"no {model or 'draft'} model for {head_id} yet")
        return m

    def predict(self, head_id: str, scan_id: str, model: Optional[str] = None, store_scores: bool = False):
        f = self.store.feature(head_id)
        m = self._model_for(head_id, model)
        items = self.items(scan_id, f.level)
        ids = [getattr(it, "cell_id", None) or it.region_id for it in items]
        if not items:
            return m, ids, items, np.zeros(0)
        p = self.load_head(m.model_id).predict(self.embeddings(scan_id, f.level, items))
        if store_scores:
            self.store.put_scores(head_id, m.model_id, ids, p.tolist(), scan_id=scan_id,
                                  target_type="cell" if f.level == "cell" else "region")
        return m, ids, items, p

    def predictions(self, head_id: str, scan_id: str, model: Optional[str] = None) -> dict:
        m, ids, items, p = self.predict(head_id, scan_id, model)
        rows = [{"object_id": i, "p": float(v), **_xy(it)} for i, it, v in zip(ids, items, p)]
        if self.store.feature(head_id).level == "cell":
            for r in rows:
                r["cell_id"] = r["object_id"]
        return {"model_version": m.model_id, "predictions": rows}

    def queue(self, head_id: str, scan_id: str, n: int = 20, model: Optional[str] = None) -> List[dict]:
        f = self.store.feature(head_id)
        m, ids, items, p = self.predict(head_id, scan_id, model)
        anns = self.store.annotations(head_id, scan_id=scan_id)
        marked = labels_from_annotations(items, anns, f.level, self.head_cfg.snap_um)
        rows = review_queue(ids, p, {ids[i] for i in marked}, n, self.head_cfg.random_share,
                            seed=len(anns))
        pos = {i: it for i, it in zip(ids, items)}
        for r in rows:
            r.update(_xy(pos[r["object_id"]]))
            if f.level == "cell":
                r["cell_id"] = r["object_id"]
        return rows

    # ------------------------------------------------------------ tier 2
    def retrain(self, head_id: str, actor: str = "system", kind: Optional[str] = None, force: bool = False,
                extra: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
        """One gated round. ``extra`` = (X, y) from an external labelled set (same encoder)."""
        with self.lock:
            X, y, w, _, marked = self._dataset(head_id, locked=False)
            if extra is not None:
                X = np.concatenate([X, extra[0]]) if len(X) else extra[0]
                y = np.concatenate([y, extra[1]])
                w = np.concatenate([w, np.ones(len(extra[1]))])
            skipped = list(self.skipped_scans)
            Xl, yl, _, gl, _ = self._dataset(head_id, locked=True, with_background=False)
            skipped += [s for s in self.skipped_scans if s not in skipped]
            champ = self.store.latest_model(head_id, "champion")
            n_marks = int(marked.sum())
            new = n_marks - (champ.n_labels if champ else 0)
            cfg = replace(self.head_cfg, kind=kind or self.head_cfg.kind)
            chall = LiveHead(cfg)
            res = run_round(chall, self.load_head(champ.model_id) if champ else None, (X, y, w),
                            (Xl, yl, gl), new, self.gate_cfg, force)
            entry = {"head_id": head_id, **res.to_dict(), "skipped_scans": skipped, "gate": asdict(self.gate_cfg), "entip_train": __version__}
            if res.challenger is None:                    # not trained (too few labels / locked set too small)
                self.store.log(person(actor), "retrain_skipped", "Feature", head_id, entry)
                return entry
            mid = new_id("model")
            self.heads[mid] = chall
            status = "champion" if res.promoted else "rejected"
            if res.promoted and champ:                    # retire first: the database allows one champion
                self.store.set_model_status(champ.model_id, "retired", person(actor), {"replaced_by": mid})
            self.store.add_model(Model(mid, head_id, f"retrain:{cfg.kind}", self.encoder.version, status,
                                       person(actor), n_marks, parent=champ.model_id if champ else None,
                                       metrics={"locked": res.challenger, "reasons": res.reasons},
                                       artifact=self._save(chall, mid)))
            entry["model_id"], entry["status"] = mid, status
            return entry

    def submit_retrain(self, head_id: str, actor: str, **kw) -> str:
        jid = f"job_{uuid.uuid4().hex[:10]}"
        self.jobs[jid] = {"job_id": jid, "kind": "retrain", "head_id": head_id, "state": "queued",
                          "submitted_at": now()}

        def run():
            self.jobs[jid]["state"] = "running"
            try:
                self.jobs[jid].update(state="done", result=self.retrain(head_id, actor, **kw), finished_at=now())
            except Exception as e:                        # surfaced through GET /jobs/{id}
                self.jobs[jid].update(state="failed", error=f"{type(e).__name__}: {e}", finished_at=now())

        self.pool.submit(run)
        return jid


def _xy(it) -> dict:
    if isinstance(it, Cell):
        return {"x_um": it.x_um, "y_um": it.y_um}
    cx, cy = it.centre
    return {"x_um": cx, "y_um": cy, "w_um": it.w_um, "h_um": it.h_um}
