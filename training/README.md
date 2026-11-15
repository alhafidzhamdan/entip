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

`entip-train triage` answers two questions per patient: is it tumour, and is the tumour glial? One
attention-MIL model gives six classes (`normal_brain`, `reactive`, `glioma`, `other_primary_cns`,
`metastasis`, `lymphoma`); tumour and glial probabilities are sums of those. Each patient either gets a
confident call or is sent to a pathologist. Anything other than a confident glioma call also goes to a
pathologist, because steps 2 to 4 only handle gliomas. Research use only.

```bash
pip install -e ".[idc,slides,foundation]"
entip-train triage demo                                   # synthetic cohort end to end, CPU, seconds
entip-train triage manifest --prep ffpe paxgene_paraffin --out manifest.csv     # IDC slide list, needs internet
entip-train triage build-bags --manifest manifest.csv --out bags/ --per-class 5 # trial: 5 slides per source
entip-train triage build-bags --manifest manifest.csv --out bags/               # everything; stop and rerun freely
entip-train triage train --manifest manifest.csv --bags bags/ --out models/triage [--db http://localhost:8000]
entip-train triage predict bags/<scan>.npz [more scans of the same patient] --model models/triage \
    --study-id SYN-0001 --db http://localhost:8000
```

`build-bags` downloads one slide, embeds it with H-optimus-0 (GPU or Apple MPS), saves `<SeriesInstanceUID>.npz`
and deletes the slide, so the 3.3 TB of the FFPE + GTEx manifest never sits on disk. It skips slides already
done and logs failures to `bags/failed.jsonl`. `extract` does the same for slides already on disk
(Edinburgh research copies).

**Checked on real slides (2026-11-11).** `build-bags` ran end to end on seven public slides from five
sources (GTEx, TCGA-GBM, TCGA-LGG, CCDI-MCI, HCMI) with the `handcrafted` test encoder. Found and fixed:
OpenSlide 4.0.1 refuses IDC series stored as JPEG 2000 with YBR_FULL_422 colour (376 of the 1,704 TCGA
diagnostic slides in the manifest), so `open_slide` falls back to wsidicom; scans at 0.25 to 0.26 µm/px need
antialiased resizing to 224 px; and one TCGA-GBM slide is partly out of focus, so every tile now carries a
focus measure. H-optimus-0 itself has not been run: the cloud sessions cannot reach Hugging Face.

**What it outputs.** Two records per patient in the step 5 contract (`entip.glioma_step/1`, `dx/contract/`):
`adequacy` (tissue area rule; 1 mm² is a placeholder to set; focus is reported, not judged) and `tissue_class`
(call, calibrated confidence, threshold, status, probabilities, a `calibration` block saying how the
confidence was calibrated and whether the threshold met its error target, p_tumour / p_glial, per-scan results, top
attention tiles as evidence, `order_test`). `predict --db` writes a `glioma_step` event, a whole-tissue `roi`
region per scan with that scan's call, the top tiles as `tile` regions, and predictions on those regions.
The model row is written once, at `train --db` or the first `predict --db`, with status `draft`; scoring
never rewrites it.

**How a patient's call is made.** With several scans, the most abnormal one decides (highest tumour
probability), so bland slides cannot outvote the one holding the tumour. The call is confident only if that
scan is above the threshold, is not flagged as unlike the training slides, and no other scan of the patient
confidently says something else.

**How confidence is set.** Five-fold cross-validation with patients kept together gives out-of-fold
probabilities. Temperature scaling is fitted on those. The accept threshold is fitted on one row per patient:
thresholds are tested from the most confident patient downwards with an exact binomial bound, stopping at the
first that fails (fixed-sequence testing, https://arxiv.org/abs/2110.01052), so that accepted calls are wrong
at most `--max-error` of the time (default 5%) with 95% confidence. At 5% this needs at least 59 error-free
patients before anything is accepted. If nothing passes, the model never accepts. The guarantee covers
patients like the validation ones only; probabilities reflect the training mix, not Edinburgh's.

**What it cannot know.** The model must put every slide in a class it was trained on, so an unseen tumour
type comes out as a wrong call. Two guards: a novelty check (`wsi/novelty.py`) sends slides far from every
training class to a pathologist, and the report's `unseen_classes` scores the slides of left-out classes and
counts how many would have been wrongly accepted. The novelty check is coarse (slide average); it will miss a
small unusual focus.

**Public data (IDC v25, counted 2026-11-10).** Glioma 5,433 slides, other primary CNS 856, normal brain 423
(GTEx cerebral cortex), metastasis 18, lymphoma 2, reactive 0. FFPE only: glioma 2,499, other primary CNS
550, no normal brain (GTEx is PAXgene-fixed and post-mortem). The manifests are in the project files under
`etip/datasets/triage/`. Classes with fewer than 5 patients are left out of training and listed in the
report, so the first public model covers glioma, other primary CNS and normal brain only.

**Shortcut checks.** Every public source here holds one class, so a model can learn the source (stain,
scanner, fixative) instead of the tissue. The training report includes `source_check` (how well the source can
be predicted within one class from the features) and `leave_one_source_out` (accuracy on a source the model
never saw). Read both before trusting a public-data result. Edinburgh slides of each class, from one lab and
scanner, are the fix.

**Memory.** Training keeps a seeded sample of tiles per slide (`--train-tiles`, default 1,024) as float16:
about 3 MB per slide with H-optimus-0, so 3,475 slides need about 11 GB. Prediction uses every tile.

**Shared pieces for steps 2 to 4** (`wsi/`): `bags.py` (tissue tiles, frozen-encoder bags saved as `.npz`
without pickles), `abmil.py` (our own gated attention-MIL in NumPy, after Ilse et al. 2018,
https://arxiv.org/abs/1802.04712), `calibrate.py` (temperature scaling, ECE, the accept threshold),
`novelty.py`.

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
| `wsi/` | shared slide bags, attention-MIL, calibration (glioma workflow steps 1 to 4) |
| `triage/` | step 1: labels, IDC manifest, model, db writer, synthetic cohort, CLI |

## Open items

- Swap `store.py` for the db/ pillar's API once its schema lands (same method names); then Cell and Region rows
  come from db/ and scores are written back there.
- Cells must be supplied (from a segmenter upstream); this pillar does not segment.
- Probabilities are not calibrated beyond what the classifier gives; the gate checks ECE on the locked set.
- Joblib artifacts are only loaded from this service's own directory; do not load heads from elsewhere.
- Results so far are on synthetic data only. Nothing has been run with H-optimus-0 on real slides.
