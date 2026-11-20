# Triage training runs (steps 1 and 2 on public slides)

End-to-end runs of the AI glioma workflow on public slides from the Imaging Data Commons (IDC):
manifest → TCGA IDH status → pick the slides in a run plan → step 1 (pre-training: frozen H-optimus-0 tile bags)
→ step 2 (triage training). Each run is a plan in `plans/`; the same scripts run any of them.

| Plan | Slides | What it is for |
|---|---|---|
| `test-run` | 10 (≈1 GB): 5 glioma, 5 normal brain, the smallest of at least 100 MB | Checks the pipeline. First run on Eddie 2026-11-15: worked end to end; the gate accepted nothing, as expected at this size |
| `diverse` | 250 (≈190 GB): see below | A first model across classes and sources |
| `stage-a` | 3,138 slides, 2,373 patients (≈2.7 TB): see below | The full public triage set |
| `diverse-astro4` | the 250 diverse slides plus every grade 4 IDH-mutant astrocytoma (65 new slides: all 22 TCGA patients with an FFPE slide, ≈42 GB, no size limits for this group; the other 13 have frozen sections only and are left out) | Bags for the step 3 grade picking model ([`highgrade_common/grade_mil.py`](../highgrade_common/README.md#grade-4-vs-grade-3-for-picking)), against the diverse set's grade 3 astrocytomas. Runs in the diverse folder, stage and embed only: `SKIP_TRAIN=1 RUN=diverse-astro4 WORK=$BASE/runs/triage-diverse bash training/runs/triage/eddie/submit.sh` |

## The diverse set

| Group | Class | Source | Patients |
|---|---|---|---|
| `gbm_idh_wildtype` | glioma | TCGA-GBM, IDH-wildtype only | 50 |
| `lgg_idh_mutant` | glioma | TCGA-LGG, IDH-mutant only | 50 |
| `other_primary_cns` | other primary CNS | CCDI-MCI | 50 |
| `non_cns_metastasis` | non-CNS / metastasis | the real brain metastases in IDC (CCDI-MCI, 3 FFPE), then TCGA lung (LUAD, LUSC), breast, melanoma, kidney, colon and rectum primaries | 50 |
| `normal_brain` | normal brain | GTEx cerebral cortex (PAXgene) | 50 |

- **IDH status** comes from cBioPortal's TCGA pan-glioma study (`lgggbm_tcga_pub`, Ceccarelli et al. 2016,
  `IDH_STATUS`), downloaded by `fetch_idh.py`. Patients with no recorded status are left out, never guessed.
  On 2026-11-15 this left out 55 IDH-mutant TCGA-GBM and 102 IDH-wildtype TCGA-LGG patients.
- **Non-CNS / metastasis is one category** (class key `metastasis` in the output contract): the few real brain
  metastases in IDC plus TCGA primaries of the commonest origins of brain metastasis (manifest option
  `--non-cns`). Most of it is not brain tissue, so the model may learn "not brain" for this class; Edinburgh
  metastases will sharpen it.
- **Inadequate** is not a class: adequacy stays the separate tissue-area rule. Slides with no tissue tiles are
  skipped from training and listed in the report (`slides_empty_bags`).
- **Slides** are chosen at random (seeded) between 100 MB and 2 GB, one per patient; groups with several
  collections are spread evenly across them. The 100 MB floor is there because the very smallest IDC slides are
  nearly blank (on the first Eddie run two 18 MB TCGA-GBM slides gave 0 and 2 tissue tiles).
- **What is still confounded.** Every slide is from a Leica scanner, and each class except glioma comes from a
  single source, so the classes can still be told apart partly by source. This run trains with
  leave-one-source-out on; read `source_check` and `leave_one_source_out` in the report with that in mind.

## Stage A

Every eligible public slide, checked on 2026-11-16 against IDC v25 and cBioPortal:

| Group | Source | Patients | Slides |
|---|---|---|---|
| `glioma_idh_wildtype` | TCGA-GBM, TCGA-LGG, CPTAC-GBM; IDH-wildtype confirmed | 414 | 809 |
| `glioma_idh_mutant` | TCGA-GBM, TCGA-LGG, CPTAC-GBM; IDH-mutant confirmed | 406 | 759 |
| `other_primary_cns` | CCDI-MCI | 537 | 553 |
| `normal_brain` | GTEx | 416 | 417 |
| `non_cns_metastasis` | TCGA and CPTAC primaries of 7 organs, one slide each | 600 | 600 |

- **Every glioma has a confirmed IDH status** (`fetch_idh.py`): TCGA from the pan-glioma study's curated
  `IDH_STATUS`; CPTAC-GBM from its exome calls (IDH1 R132 / IDH2 R172 = mutant; sequenced with neither = wildtype;
  another IDH variant = uncertain, left out). Gliomas without one are left out (293 patients' slides here), and
  CCDI-MCI's paediatric gliomas have no public IDH status, so none are included.
- CPTAC adds a second source for glioma and for every non-CNS organ. Its fixative is not recorded in IDC.
- All slides of each glioma, other CNS and normal brain patient are used (they share one cross-validation fold).
- Embedding about 3,100 slides is far beyond one GPU job: submit with `EMBED_SHARDS=8` so eight GPU tasks share
  the slides, and each skips what is done, so a task that runs out of time can be submitted again.

## On Eddie

Grid Engine, three jobs chained with `-hold_jid`. Compute nodes have no internet, so everything is downloaded in
the first job.

| Job | Queue | What |
|---|---|---|
| `1_stage.sh` | staging | IDC manifest, IDH status, pick the plan's slides, download them to `slides/<SeriesInstanceUID>/`, cache the H-optimus-0 weights |
| `2_embed.sh` | GPU | step 1: `entip-train embed extract` on every slide, offline |
| `3_train.sh` | CPU, 4 cores | step 2: `entip-train triage train` with the plan's options, then a short summary of the report |

```bash
git clone <this repository> /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip
bash training/runs/triage/eddie/setup.sh                 # once, on a login node: conda env in the group space
bash training/runs/triage/eddie/submit.sh                # the test run
RUN=diverse bash training/runs/triage/eddie/submit.sh    # the diverse set
RUN=stage-a EMBED_SHARDS=8 bash training/runs/triage/eddie/submit.sh   # stage A, embedding on 8 GPUs
qstat                                                    # watch them
RUN=diverse bash training/runs/triage/eddie/retrain.sh   # re-run only the training job, on the existing bags
```

The training report lists every patient's out-of-fold call (`oof_patients`, most confident first) and the
confidence rank of the first wrong call (`cv.gate.first_error_rank`). `python3 mistakes.py report-h-optimus-0.json`
prints the mistakes and the confusions; the train job runs it at the end. At the 5% limit the gate needs its 59
most confident patients to be right, so a mistake ranked within the top 59 blocks it on its own.

Settings are in `eddie/config.sh`, and each can be overridden from the environment. Defaults:

- `RUN`: `test-run`, which picks `plans/<RUN>.json` and the output folder `$BASE/runs/triage-<RUN>`
- `BASE`: `/exports/igmm/eddie/Glioblastoma-WGS/entip`, the group space. Everything below lives here, because
  scratch is cleared after a month and the home quota is too small for PyTorch.
- `REPO`: `/exports/igmm/eddie/Glioblastoma-WGS/scripts/entip`
- `WORK`: `$BASE/runs/triage-<RUN>` (plan copy, manifest, IDH table, slides, bags, model, report, `logs/`)
- `ENV_DIR`: `$BASE/envs/entip`, the conda env
- `HF_HOME`: `$BASE/hf`, the H-optimus-0 weights, downloaded once and reused by later runs
- conda cache and other tools' caches: `$BASE/cache/` (pip cache off); pip temp files: `$BASE/tmp`
- `HOME`: `$BASE/home`. Every Eddie script moves `HOME` here, so no tool can write to the real home folder
  (its quota is small). A Hugging Face login made from your normal shell is still picked up.
- `ENCODER`: `h-optimus-0` (`handcrafted` for a quick run that needs no GPU weights)
- `GPU_REQUEST`: `-q gpu -l gpu=1 -l a100=true -pe sharedmem 4 -l h_rss=16G` (one A100, 4 cores, 64 GB RAM; Eddie
  limits memory with `h_rss` per slot, at most 32 GB per slot). Use `-l l40s=true` for an L40S, or
  `-l h200=true` for an H200 (IGC users). `qstat -F gpu,gputype,h_rss -q gpu` shows what is free.
- `EMBED_SHARDS`: `1`. GPU tasks the embedding is split across (`-t 1-N` array job); each writes `index-<i>.jsonl`.
- `TRAIN_REQUEST`: `-pe sharedmem 8 -l h_rss=8G` for the CPU training job (8 cores, 64 GB). The embed job's cores
  (in `GPU_REQUEST`) are also its tile-reading workers. Every job limits its maths libraries to the cores it was given.

H-optimus-0 is a gated model: request access on its Hugging Face page, then log in once on Eddie with
`HF_HOME=$BASE/hf $BASE/envs/entip/bin/hf auth login` and `chmod 600 $BASE/hf/token`.

## Locally

`run_local.sh` does the same stages in one go with `entip-train embed build-bags` (download, embed, delete per
slide). It defaults to the `handcrafted` CPU test encoder, which needs no GPU or weights:

```bash
pip install -e platform/live-training -e "training/glioma[idc,slides]"
training/runs/triage/run_local.sh                                   # test run, handcrafted encoder
RUN=diverse ENCODER=h-optimus-0 training/runs/triage/run_local.sh   # needs [foundation] and a GPU or Apple MPS
```

Outputs go to `out/<RUN>/` here, which git ignores.
