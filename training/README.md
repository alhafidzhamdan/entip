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

For real slides use `pip install -e ".[api,foundation]"` and `--encoder h-optimus-0` (Apache-2.0 weights from
Hugging Face). The `handcrafted` encoder is colour/texture statistics for tests and the synthetic demo only.

## API (used by the GUI, pillar 4)

Positions are micrometres in the frame `scan:<scan_id>`; `actor` is stored as provenance.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/heads` | `{feature, level: cell\|tile\|region, scan_id?, positives_only?, actor}` | `{head_id}` |
| POST | `/heads/{id}/labels` | `{scan_id, cell_id \| object_id \| x_um+y_um \| polygon_um, label: 1\|0, actor}` | `{label_id, model_version, n_pos, n_neg, metrics, queue}` |
| DELETE | `/heads/{id}/labels/{label_id}` | `?actor=` | retraction (logged) |
| GET | `/heads/{id}/queue` | `?scan_id=&n=20&model=` | `[{cell_id, p, reason: uncertain\|random, x_um, y_um}]` |
| GET | `/heads/{id}/predictions` | `?scan_id=&model=draft\|champion\|<id>` | `{model_version, predictions: [{cell_id, p}]}` |
| POST | `/heads/{id}/retrain` | `{actor, kind?, force?}` | `{job_id}` |
| GET | `/jobs/{job_id}`, `/heads/{id}`, `/models`, `/actions` | | status, counts, model history, action log |
| POST | `/scans/synthetic` | `{scan_id, seed}` | registers a synthetic demo slide |

## High-grade features (glioma workflow step 2)

`entip_train.highgrade`: one detector per feature, each returning an `entip.glioma_step/1` record (the step-5
contract in `dx/contract/README.md`) with a call, a calibrated confidence, a status and, when uncertain, a
pathologist review with the regions or cells to look at first.

| Finding | Detector | Input |
|---|---|---|
| `necrosis` | tile head on frozen embeddings → 8-connected foci → best focus score | tile bag (`.npz`, step-1 format) |
| `microvascular_proliferation` | same machinery, its own head | same bag |
| `mitotic_activity` | MitoEd candidates → P(at least *k* true mitoses in the whole specimen) | `mitoed predict` CSV |

- **Calls.** With p = P(present) and threshold t: `present` if p ≥ t, `absent` if 1 − p ≥ t, otherwise
  `indeterminate` and a pathologist review.
- **Guards** set `status: uncertain` and order a review: model not validated (the default, so nothing is used
  unreviewed until it has been checked on pathologist-labelled slides), too few tissue tiles, too many tiles
  unlike the training tiles, an encoder or tile size different from training (refused).
- **Mitotic cut-off.** cIMPACT-NOW update 5 keeps "significant" mitotic activity for IDH-mutant astrocytoma
  grade 3, says no validated cut-off exists, and notes most neuropathologists use ≥ 2 mitoses in the whole
  specimen, or 1 in a very small biopsy ([Brat et al. 2020](https://doi.org/10.1007/s00401-020-02127-9)). Here
  `--min-mitoses` (default 2) and `--small-biopsy-mm2` (default off) are settings, not validated values. The same
  paper makes necrosis or microvascular proliferation (or CDKN2A/B homozygous deletion) grade 4 criteria for
  IDH-mutant astrocytoma; the step-5 engine applies those rules, not this step.
- **Database.** `actions.py` writes features, models (`draft`), tile and focus regions, mitosis candidate cells,
  predictions, then one `glioma_step` event with `detail = {study_id, records}`. Checked against a live
  PostgreSQL 16 + PostGIS db with the seeded synthetic patients.

```bash
entip-train highgrade-demo                                    # synthetic specimen, prints the three records
entip-train highgrade-demo --db http://127.0.0.1:8705          # also posts the actions (scan SCN-0001-A1-01)
entip-train highgrade-train --feature necrosis --bag s1.npz --labels s1.json --out necrosis.joblib
entip-train highgrade --study-id SYN-0001 --bag s1.npz --head necrosis.joblib --head mvp.joblib \
    --mitoed-cells SCN-0001-A1-01=cells.csv --db http://127.0.0.1:8705
```

Labels for `highgrade-train` are `{"positives": [polygon, ...], "negatives": [...]}` in µm in the scan frame.
For real slides build bags with the step-1 module (`entip_train.wsi.bags`: tissue mask, 224 px tiles at
0.5 µm/px, H-optimus-0); `highgrade.bags.embed_bag` is a stand-in for tests.

Not done yet: no head has been trained on real tissue. Candidate region labels are BraTS-Path 2024 and the Ivy
Glioblastoma Atlas ([Puchalski et al. 2018](https://doi.org/10.1126/science.aaf2666)); their licences have not
been checked against the NHS rule. MitoEd has not been run on CNS tumours (its training data, MIDOG++, has none
released). Thresholds must be set on held-out pathologist-labelled slides before `validated` is turned on.

## Layout

| File | What |
|---|---|
| `ontology.py` | Scan, Feature, Cell, Region, Annotation, Model (from the shared rules) |
| `store.py` | SQLite stand-in for the db/ pillar, with an append-only action log |
| `slides.py` | readers (in-memory, OpenSlide) and the synthetic slide |
| `encoders.py`, `cache.py` | H-optimus-0 / handcrafted encoders; per-scan embedding cache on disk |
| `heads.py` | marks → labels, background negatives, live head, review queue |
| `retrain.py` | gate, metrics (balanced accuracy, ECE, per scan), champion/challenger round |
| `service.py`, `api.py`, `cli.py` | orchestration, HTTP API, `entip-train` command |
| `highgrade/` | step 2 of the glioma workflow: necrosis, microvascular proliferation, mitotic activity |

## Open items

- Swap `store.py` for the db/ pillar's API once its schema lands (same method names); then Cell and Region rows
  come from db/ and scores are written back there.
- Cells must be supplied (from a segmenter upstream); this pillar does not segment.
- Probabilities are not calibrated beyond what the classifier gives; the gate checks ECE on the locked set.
- Joblib artifacts are only loaded from this service's own directory; do not load heads from elsewhere.
- Results so far are on synthetic data only. Nothing has been run with H-optimus-0 on real slides.
