# External test: IPD-Brain

Scores a trained triage model on IPD-Brain (Chauhan et al. 2024, *Scientific Data*): H&E slides of gliomas from
Nizam's Institute of Medical Sciences, Hyderabad, India, scanned at 40x on a Morphle DigiPath 6T. 328 patients
with clinical records (168 glioblastoma, 88 astrocytoma, 72 oligodendroglioma; 484 slides) plus about 60 unlabelled
slides. **It is a test set only: no ENTIP model is ever trained on it** without the provider's written consent.

Why it is worth it: a population, lab, stain and scanner make the model has never seen, so it tests glioma recall
outside US and European data. Alongside the DBTA it gives two external centres.

## Terms

India Data's Data User Agreement: research and other lawful use, attribution, India's Digital Personal Data
Protection Act 2023, Indian law. **No commercial use without prior written consent from the provider**, which is
why this is test-only: a model trained on it would carry that limit. Keep the files in the group space and do not
copy them elsewhere. The Figshare copy (10.6084/m9.figshare.27186087) is CC BY 4.0 but only about 12 GB, so most
likely too low-resolution for 112 µm tiles at 0.5 µm/px. See `docs/data-sources.md`.

## What it does

| Step | Where | What |
|---|---|---|
| `make_manifest.py` | login node, seconds | clinical table + your download folder → `manifest.csv` (slide id, patient, `glioma`, diagnosis), `left_out.csv`, `slides/<id>.<ext>` links |
| `eddie/2_embed.sh` | GPU array | step 1 with the same frozen encoder as the model; each task takes every N-th slide and skips what is done |
| `eddie/3_evaluate.sh` | CPU | `entip-train triage evaluate`: each patient's call exactly as in use (same temperature, threshold and novelty check) |

Every labelled slide is `glioma`. The diagnosis column is the subtype plus the IHC result, for example
"glioblastoma (IDH1 R132H negative)", so the report's `per_diagnosis` shows recall by subtype and stain. IDH comes
from IDH1 R132H IHC only: a negative stain misses the rarer IDH mutations and is not a confirmed IDH-wildtype
(4 oligodendrogliomas are stain-negative). The 23 "glioblastomas" that are stain-positive would be astrocytoma,
IDH-mutant, grade 4 in WHO CNS5; for triage they are glioma either way.

With one class only, the report's `known` section is glioma recall (and how often a glioma was called normal brain
or not a tumour); balanced accuracy is not meaningful. The `gate` section is still the external check of the error
limit for the patients accepted as confident.

## On Eddie

1. Register on India Data, accept the Data User Agreement, and download the 40x version (about 800 GB) into
   `/exports/igmm/eddie/Glioblastoma-WGS/entip/data/ipd/`, with the clinical table there as `IPD_Brain.csv`
   (or set `IPD_DIR` and `IPD_META`).
2. Once the triage model you want to test exists (default: stage A):

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull
MODEL_RUN=stage-a EMBED_SHARDS=4 bash training/runs/ipd/eddie/submit.sh
```

`submit.sh` checks that the TIFFs state their pixel size. If they do not, it stops and asks for it: rerun with
`IPD_MPP=0.22` (the paper's value at 40x) in front.

Outputs go to `$BASE/runs/ipd/`. The bags are reusable for any later model trained with the same encoder: rerun
only `3_evaluate.sh` with another `MODEL_RUN`.
