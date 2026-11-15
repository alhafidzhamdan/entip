# Triage test run: 10 slides

A small end-to-end run of the AI glioma workflow on public slides from the Imaging Data Commons (IDC):
manifest → pick 10 slides → step 1 (pre-training: frozen tile bags) → step 2 (triage training). It checks that the
pipeline runs on real slides. It does not make a usable model: with 10 patients the accept gate accepts nothing,
because it needs at least 20 accepted patients.

**The 10 slides.** `pick_slides.py` takes the smallest slides of each class, one per patient: 5 glioma (FFPE) and
5 normal brain (GTEx, PAXgene-fixed). Triage training needs at least 5 patients per class, so 10 slides allow two
classes. When the manifest was built on 2026-11-15 this came to about 450 MB (5 TCGA-GBM slides, 5 GTEx).
Every glioma here comes from one source and every normal brain from another, so the classes can be told apart by
source alone; read `source_check` in the report with that in mind.

## On Eddie

Grid Engine, three jobs chained with `-hold_jid`. Compute nodes have no internet, so everything is downloaded in
the first job.

| Job | Queue | What |
|---|---|---|
| `1_stage.sh` | staging | IDC manifest, pick the 10 slides, download them to `slides/<SeriesInstanceUID>/`, cache the H-optimus-0 weights |
| `2_embed.sh` | GPU (A100) | step 1: `entip-train embed extract` on every slide, offline |
| `3_train.sh` | CPU, 4 cores | step 2: `entip-train triage train`, then a short summary of the report |

```bash
git clone <this repository> ~/entip && cd ~/entip
bash training/runs/triage-test-run/eddie/setup.sh       # once, on a login node: conda env in the group space
bash training/runs/triage-test-run/eddie/submit.sh      # submits the three jobs
qstat                                                   # watch them
```

Settings are in `eddie/config.sh`, and each can be overridden from the environment, for example
`WORK=/exports/igmm/eddie/Glioblastoma-WGS/entip/runs/trial2 ENCODER=handcrafted bash .../submit.sh` for a quick run that needs
no GPU weights. Defaults:

- `BASE`: `/exports/igmm/eddie/Glioblastoma-WGS/entip`, the group space. Everything below lives here, because
  scratch is cleared after a month and the home quota is too small for PyTorch.
- `REPO`: `~/entip`
- `WORK`: `$BASE/runs/triage-test-run` (slides, bags, model, report, `logs/`)
- `ENV_DIR`: `$BASE/envs/entip`, the conda env
- `HF_HOME`: `$BASE/hf`, the H-optimus-0 weights, downloaded once and reused by later runs
- conda and pip caches: `$BASE/cache/`
- `ENCODER`: `h-optimus-0`

Check on first use: the GPU request (`-q gpu`, `-pe gpu-a100 1`) and the `anaconda` module name follow Eddie's
documentation as we understand it; adjust them if your project uses a different GPU queue or module version. If
Hugging Face asks you to accept terms for H-optimus-0, log in once with a token before `submit.sh`.

## Locally

`run_local.sh` does the same four stages in one go with `entip-train embed build-bags` (download, embed, delete
per slide). It defaults to the `handcrafted` CPU test encoder, which needs no GPU or weights:

```bash
pip install -e platform/live-training -e "training/glioma[idc,slides]"
training/runs/triage-test-run/run_local.sh                       # handcrafted encoder
ENCODER=h-optimus-0 training/runs/triage-test-run/run_local.sh   # needs [foundation] and a GPU or Apple MPS
```

Outputs go to `out/` here, which git ignores.
