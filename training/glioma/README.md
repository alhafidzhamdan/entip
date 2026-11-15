# training/glioma/ — AI glioma workflow, steps 1 to 5

Package `entip_glioma`: pre-training (frozen H-optimus-0 tile bags), triage, high-grade features, panel markers
and methylation class. Step 6, the WHO CNS5 rules engine, is [`training/dx/`](../dx/README.md). The workflow
overview is in [`docs/glioma-workflow/`](../../docs/glioma-workflow/README.md). Research use only.

It uses the slide readers, encoders and ontology of the live-training pillar (`entip_train`,
[`platform/live-training/`](../../platform/live-training/README.md)), so install that first.

```bash
cd training/glioma
pip install -e ../../platform/live-training -e ".[dev]"
pytest -q
entip-glioma --help        # embed, triage, highgrade*, methyl; the same commands are added to entip-train
entip-panel --help         # step 4
```

## Step 1 of the AI glioma workflow: pre-training (frozen foundation model)

`entip-train embed` is where the foundation model is used. Each slide is cut into 112 µm tissue tiles (224 px at
0.5 µm/px) and every tile is embedded once with a frozen H-optimus-0, giving one bag per slide (`wsi/bags.py`).
Steps 2 to 5 train and predict from these bags only. Nothing is fine-tuned; each bag records its encoder version
and the later steps refuse bags from a different encoder. Each run appends one line per slide to
`index.jsonl` in the output folder. See [step 1](../../docs/glioma-workflow/step1-pretraining.md).

```bash
pip install -e ../../platform/live-training -e ".[idc,slides,foundation,dev]"
entip-train embed extract slide.svs [more slides] --out bags/                     # slides already on disk
entip-train embed build-bags --manifest manifest.csv --out bags/ --per-class 5    # trial: 5 slides per source
entip-train embed build-bags --manifest manifest.csv --out bags/                  # everything; stop and rerun freely
```

`embed build-bags` downloads one slide, embeds it with H-optimus-0 (GPU or Apple MPS), saves `<SeriesInstanceUID>.npz`
and deletes the slide, so the 3.3 TB of the FFPE + GTEx manifest never sits on disk. It skips slides already
done and logs failures to `bags/failed.jsonl`. `embed extract` does the same for slides already on disk
(Edinburgh research copies).

**Checked on real slides (2026-11-11).** `build-bags` ran end to end on seven public slides from five
sources (GTEx, TCGA-GBM, TCGA-LGG, CCDI-MCI, HCMI) with the `handcrafted` test encoder. Found and fixed:
OpenSlide 4.0.1 refuses IDC series stored as JPEG 2000 with YBR_FULL_422 colour (376 of the 1,704 TCGA
diagnostic slides in the manifest), so `open_slide` falls back to wsidicom; scans at 0.25 to 0.26 µm/px need
antialiased resizing to 224 px; and one TCGA-GBM slide is partly out of focus, so every tile now carries a
focus measure. H-optimus-0 itself has not been run: the cloud sessions cannot reach Hugging Face.

`entip-train triage extract` and `triage build-bags` still work as aliases.

## Step 2 of the AI glioma workflow: tumour and glial triage

`entip-train triage` answers two questions per patient: is it tumour, and is the tumour glial? One
attention-MIL model gives six classes (`normal_brain`, `reactive`, `glioma`, `other_primary_cns`,
`metastasis`, `lymphoma`); tumour and glial probabilities are sums of those. Each patient either gets a
confident call or is sent to a pathologist. Anything other than a confident glioma call also goes to a
pathologist, because steps 3 to 5 only handle gliomas. Research use only.

```bash
entip-train triage demo                                   # synthetic cohort end to end, CPU, seconds
entip-train triage manifest --prep ffpe paxgene_paraffin --out manifest.csv     # IDC slide list, needs internet
# bags/ from step 1 (entip-train embed build-bags)
entip-train triage train --manifest manifest.csv --bags bags/ --out models/triage [--db http://localhost:8000]
entip-train triage predict bags/<scan>.npz [more scans of the same patient] --model models/triage \
    --study-id SYN-0001 --db http://localhost:8000
```

**What it outputs.** Two records per patient in the step 6 contract (`entip.glioma_step/1`, `training/dx/contract/`):
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

**Shared pieces for steps 3 to 5** (`wsi/`): `bags.py` (tissue tiles, frozen-encoder bags saved as `.npz`
without pickles), `abmil.py` (our own gated attention-MIL in NumPy, after Ilse et al. 2018,
https://arxiv.org/abs/1802.04712), `calibrate.py` (temperature scaling, ECE, the accept threshold),
`novelty.py`.

## High-grade features (glioma workflow step 3)

`entip_glioma.highgrade`: one detector per feature, each returning an `entip.glioma_step/1` record (the step-6
contract in `training/dx/contract/README.md`) with a call, a calibrated confidence, a status and, when uncertain, a
pathologist review with the regions or cells to look at first.

| Finding | Detector | Input |
|---|---|---|
| `necrosis` | tile head on frozen embeddings → 8-connected foci → best focus score → slide-calibrated P(present) | tile bag (`wsi.bags.Bag`, from step 1) |
| `microvascular_proliferation` | same machinery, its own head | same bag |
| `mitotic_activity` | MitoEd candidates → P(at least *k* true mitoses in the whole specimen) | `mitoed predict` CSV |

- **Calls.** With p = P(present) and threshold t: `present` if p ≥ t, `absent` if 1 − p ≥ t, otherwise
  `indeterminate` and a pathologist review.
- **Two calibrations for the region heads.** Tile probabilities are Platt-calibrated in folds grouped by slide.
  The best of thousands of tiles is still not a slide probability, so `highgrade-calibrate` fits the slide
  score to present / absent on held-out specimens and, with 40 or more of them, sets the threshold so the
  error among accepted calls stays under 5% (Wilson upper bound, `wsi.calibrate.selective_threshold`).
- **Guards** set `status: uncertain` and order a review: slide score not calibrated, model not signed off (the
  default, so nothing is used unreviewed until it has been checked on pathologist-labelled slides), too few
  tissue tiles, too many tiles unlike the training tiles. An encoder or tile size different from training is
  refused.
- **Mitotic cut-off.** cIMPACT-NOW update 5 keeps "significant" mitotic activity for IDH-mutant astrocytoma
  grade 3, says no validated cut-off exists, and notes most neuropathologists use ≥ 2 mitoses in the whole
  specimen, or 1 in a very small biopsy ([Brat et al. 2020](https://doi.org/10.1007/s00401-020-02127-9)). Here
  `--min-mitoses` (default 2) and `--small-biopsy-mm2` (default off) are settings, not validated values. The same
  paper makes necrosis or microvascular proliferation (or CDKN2A/B homozygous deletion) grade 4 criteria for
  IDH-mutant astrocytoma; the step-6 engine applies those rules, not this step.
- **Which mitosis candidates count.** MitoEd lists every candidate, most of them mimics with a small p. Only
  those MitoEd calls `mitosis` or `uncertain` are counted (or p ≥ 0.2 for tables with no call), so thousands of
  low-p candidates cannot add up to a count. The same mitosis on two levels of one block is counted twice:
  pool scans of different blocks.
- **Database.** `actions.py` writes features, models (`draft`), tile and focus regions, counted mitosis
  candidates, predictions, then one `glioma_step` event with `detail = {study_id, records}`. Region ids carry
  the model id, so a new model version never overwrites an older one's rows. Checked against a live
  PostgreSQL 16 + PostGIS db with the seeded synthetic patients.

```bash
entip-train highgrade-demo                                    # synthetic specimen, prints the three records
entip-train highgrade-demo --db http://127.0.0.1:8705          # also posts the actions (scan SCN-0001-A1-01)
entip-train highgrade-train --feature necrosis --bag s1.npz --labels s1.json --out necrosis.joblib
entip-train highgrade-calibrate --head necrosis.joblib --specimens held_out.json
entip-train highgrade --study-id SYN-0001 --bag s1.npz --head necrosis.joblib --head mvp.joblib \
    --mitoed-cells SCN-0001-A1-01=cells.csv --db http://127.0.0.1:8705
```

Labels for `highgrade-train` are `{"positives": [polygon, ...], "negatives": [...]}` in µm in the scan frame;
`held_out.json` is a list of `{"bags": [paths], "present": 0 | 1}`. Bags come from the step-1 module
(`entip_glioma.wsi.bags.extract_bag`: tissue mask, 224 px tiles at 0.5 µm/px, H-optimus-0).

Not done yet: no head has been trained on real tissue. Candidate region labels are BraTS-Path 2024 and the Ivy
Glioblastoma Atlas ([Puchalski et al. 2018](https://doi.org/10.1126/science.aaf2666)); their licences have not
been checked against the NHS rule. MitoEd has not been run on CNS tumours (its training data, MIDOG++, has none
released). Thresholds must be set on held-out pathologist-labelled slides before `validated` is turned on.

## Layout

| Folder | What |
|---|---|
| `embed/` | step 1: slides → frozen H-optimus-0 tile bags, `entip-glioma embed` |
| `wsi/` | shared slide bags (written by step 1), attention-MIL, calibration, novelty check |
| `triage/` | step 2: labels, IDC manifest, model, db writer, synthetic cohort, CLI |
| `highgrade/` | step 3: necrosis, microvascular proliferation, mitotic activity |
| `panel/` | step 4: glioma panel markers from H&E, `entip-panel` (see [PANEL.md](PANEL.md)) |
| `methylation/` | step 5: methylation class from H&E (see [methylation.md](methylation.md)) |
| `cli.py` | `entip-glioma`, and the commands it adds to `entip-train` |

Panel markers that are switched on are set in [`configs/glioma_panel.json`](configs/glioma_panel.json).
