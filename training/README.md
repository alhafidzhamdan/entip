# training/ — live training backend (pillar 2)

Teach the platform any feature of interest from a few clicks, then retrain it safely. Research use only.

**Tier 1, live heads.** Each scan's cells (or tiles) are embedded once with a frozen encoder and cached.
A head is a small classifier on those embeddings. Every click adds an Annotation, retrains the head in
milliseconds (warm start from the previous version), and returns a review queue: the most uncertain
objects first (smallest |p − 0.5|) plus a 10% random share to catch confident errors. Features marked
mostly with positives get down-weighted background negatives sampled away from the positives.

**Tier 2, gated retrain.** A champion/challenger round after MitoEd's `src/mitoed/train/retrain.py`: train on
every non-locked mark across scans, score both models on a locked set that is never trained on, and promote
only if balanced accuracy does not drop, no scan drops by more than `tol_group`, and calibration (ECE) does not
worsen by more than `tol_ece`. Every round is recorded; the old champion is retired, not deleted. Models that are
not live heads (e.g. MitoEd's classifier) can use the same gate through the `Trainable` interface in `retrain.py`.

## Quick start

```bash
cd training
pip install -e ".[api,dev]"
pytest -q
entip-train demo                 # 20 clicks on a synthetic slide -> head + ranked queue (CPU, a few seconds)
entip-train serve --encoder handcrafted --cors http://localhost:8080   # API on :8100
```

**With the central database (db/ pillar):**

```bash
entip-train serve --db http://localhost:8000 --slides-dir ../gui/web/mock/slides --encoder handcrafted
```

Scans, cells, regions, features and marks are then read from the database, with its ids and feature keys
(`mitotic_figure`) used as they are. Marks, models (draft, champion, rejected, retired) and each click's scores
for the scan are written back through `POST /actions`, so they are in its action log. If the GUI has already
saved a mark, it sends `annotation_id` and the mark is trained on without being written again. Pixels come from
a DZI pyramid (`--slides-dir/<scan_id>/*.dzi`, or a `.dzi` tile_source) or OpenSlide; marks on scans with no
pixels stay saved and are listed under `skipped_scans`.

For real slides use `pip install -e ".[api,foundation]"` and `--encoder h-optimus-0` (Apache-2.0 weights from
Hugging Face). The `handcrafted` encoder is colour/texture statistics for tests and the synthetic demo only.

## API (used by the GUI, pillar 4)

Positions are micrometres in the frame `scan:<scan_id>`; `actor` is stored as provenance.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/heads` | `{feature, level: cell\|tile\|region, scan_id?, positives_only?, actor}` | `{head_id}` |
| POST | `/heads/{id}/labels` | `{scan_id, annotation_id \| cell_id \| object_id \| x_um+y_um \| polygon_um, label: 1\|0, actor}` | `{label_id, model_version, n_pos, n_neg, metrics, queue}` |
| DELETE | `/heads/{id}/labels/{label_id}` | `?actor=` | retraction (logged) |
| GET | `/heads/{id}/queue` | `?scan_id=&n=20&model=` | `[{cell_id, p, reason: uncertain\|random, x_um, y_um}]` |
| GET | `/heads/{id}/predictions` | `?scan_id=&model=draft\|champion\|<id>` | `{model_version, predictions: [{cell_id, p}]}` |
| POST | `/heads/{id}/retrain` | `{actor, kind?, force?}` | `{job_id}` |
| GET | `/jobs/{job_id}`, `/heads/{id}`, `/models`, `/actions` | | status, counts, model history, action log |
| POST | `/scans/synthetic` | `{scan_id, seed}` | registers a synthetic demo slide |

## Layout

| File | What |
|---|---|
| `ontology.py` | Scan, Feature, Cell, Region, Annotation, Model (from the shared rules) |
| `store.py`, `dbstore.py` | SQLite stand-in (no server needed), and the client for the db/ pillar's API |
| `slides.py` | readers (in-memory, DZI, OpenSlide) and the synthetic slide |
| `encoders.py`, `cache.py` | H-optimus-0 / handcrafted encoders; per-scan embedding cache on disk |
| `heads.py` | marks → labels, background negatives, live head, review queue |
| `retrain.py` | gate, metrics (balanced accuracy, ECE, per scan), champion/challenger round |
| `service.py`, `api.py`, `cli.py` | orchestration, HTTP API, `entip-train` command |

## Open items

- Database mode makes several HTTP calls per click (about 0.6 s per click in the tests, which include writing
  600 scores each time). A cross-scan `GET /annotations?feature=` in db/ would remove the walk over every
  patient that listing a feature's marks needs now.
- `tests/test_db.py` runs against a real PostgreSQL + PostGIS (as in db/tests) and is skipped without one.
- Cells must be supplied (from a segmenter upstream); this pillar does not segment.
- Probabilities are not calibrated beyond what the classifier gives; the gate checks ECE on the locked set.
- Joblib artifacts are only loaded from this service's own directory; do not load heads from elsewhere.
- Results so far are on synthetic data only. Nothing has been run with H-optimus-0 on real slides.
