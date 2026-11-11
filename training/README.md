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

## Step 1 of the AI glioma workflow: tumour and glial triage

`entip-train triage` answers two questions per H&E slide: is it tumour, and is the tumour glial? One
attention-MIL model gives six classes (`normal_brain`, `reactive`, `glioma`, `other_primary_cns`,
`metastasis`, `lymphoma`); tumour and glial probabilities are sums of those. Each slide either gets a
confident call or is sent to a pathologist. Anything other than a confident glioma call also goes to a
pathologist, because steps 2 to 4 only handle gliomas. Research use only.

```bash
pip install -e ".[idc,slides,foundation]"
entip-train triage demo                                   # synthetic cohort end to end, CPU, seconds
entip-train triage manifest --prep ffpe paxgene_paraffin --out triage_manifest.csv   # IDC, needs internet
# download the listed slides: IDCClient().download_from_selection(downloadDir="slides", seriesInstanceUID=[...])
entip-train triage extract <one .dcm of the series> --scan-id <SeriesInstanceUID> --out bags/   # per slide;
#   H-optimus-0 on 112 µm tiles, GPU or Apple MPS. Bags are named <scan-id>.npz, which train looks up.
entip-train triage train --manifest triage_manifest.csv --bags bags/ --out models/triage
entip-train triage predict bags/<scan>.npz --model models/triage --study-id SYN-0001 --db http://localhost:8000
```

**What it outputs.** Two records per patient in the step 5 contract (`entip.glioma_step/1`, defined on the
step 5 branch in `dx/contract/`): `adequacy` (tissue area rule; 1 mm² is a placeholder to set) and
`tissue_class` (call, calibrated confidence, threshold, status, probabilities, p_tumour / p_glial, top
attention tiles as evidence, `order_test`). `predict --db` writes them to the database as a `glioma_step`
event, plus a whole-tissue `roi` region with the call, the top tiles as `tile` regions, predictions on those
regions, and the model row (status `draft`).

**How confidence is set.** Five-fold cross-validation with patients kept together gives out-of-fold
probabilities. Temperature scaling and the accept threshold are fitted on those. The threshold is the lowest
confidence at which the one-sided 95% upper bound on the error rate of accepted slides stays under
`--max-error` (default 5%). The report gives coverage (share of slides accepted) at that threshold.

**Public data (IDC v25, counted 2026-11-10).** Glioma 5,433 slides, other primary CNS 856, normal brain 423
(GTEx cerebral cortex), metastasis 18, lymphoma 2, reactive 0. FFPE only: glioma 2,499, other primary CNS
550, no normal brain (GTEx is PAXgene-fixed). The manifests are in the project files under
`etip/datasets/triage/`. Classes with fewer than 5 patients are left out of training and listed in the
report, so the first public model covers glioma, other primary CNS and normal brain only.

**Shortcut checks.** Every public source here holds one class, so a model can learn the source (stain,
scanner, fixative) instead of the tissue. The training report includes `source_check` (how well the source can
be predicted within one class from the features) and `leave_one_source_out` (accuracy on a source the model
never saw). Read both before trusting a public-data result. Edinburgh slides of each class, from one lab and
scanner, are the fix.

**Shared pieces for steps 2 to 4** (`wsi/`): `bags.py` (tissue tiles, frozen-encoder bags saved as `.npz`
without pickles), `abmil.py` (our own gated attention-MIL in NumPy, after Ilse et al. 2018,
https://arxiv.org/abs/1802.04712), `calibrate.py` (temperature scaling, ECE, the accept threshold).

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
| `panel/` | glioma panel markers from H&E, `entip-panel` command (see [PANEL.md](PANEL.md)) |
| `wsi/` | shared slide bags, attention-MIL, calibration (glioma workflow steps 1 to 4) |
| `triage/` | step 1: labels, IDC manifest, model, db writer, synthetic cohort, CLI |

## Open items

- Swap `store.py` for the db/ pillar's API once its schema lands (same method names); then Cell and Region rows
  come from db/ and scores are written back there.
- Cells must be supplied (from a segmenter upstream); this pillar does not segment.
- Probabilities are not calibrated beyond what the classifier gives; the gate checks ECE on the locked set.
- Joblib artifacts are only loaded from this service's own directory; do not load heads from elsewhere.
- Results so far are on synthetic data only. Nothing has been run with H-optimus-0 on real slides.
