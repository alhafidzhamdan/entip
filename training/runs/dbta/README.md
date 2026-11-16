# External test: Digital Brain Tumour Atlas

Scores a trained triage model on the Digital Brain Tumour Atlas (DBTA; Roetzer-Pejrimovsky et al. 2022,
*Scientific Data*): 3,115 H&E slides from 2,880 patients at the Medical University of Vienna, 126 diagnoses, scanned
on a Hamamatsu NanoZoomer 2.0 HT. **It is a test set only: no ENTIP model is ever trained on it.**

Why it is worth it: a lab, scanner and population the model has never seen, and the classes public IDC data could
not give us - 47 real metastases in the CNS, about 90 lymphomas (primary CNS lymphoma, counted as other primary CNS), and dozens of glioma and other CNS types beyond
adult diffuse gliomas (ependymomas, pilocytic astrocytomas, glioneuronal tumours, meningiomas, schwannomas, ...).

## Terms

Research only, under the EBRAINS Data Use Agreement for pseudonymised human data: cite the dataset's DOI, no
commercial use. Pseudonymised data are personal data under GDPR: keep the files in the group space, follow the
University's data protection steps, and do not copy them elsewhere. See `docs/data-sources.md`.

## What it does

| Step | Where | What |
|---|---|---|
| `make_manifest.py` | login node, seconds | DBTA sample table + your download folder → `manifest.csv` (uuid, patient, triage class, diagnosis), `left_out.csv`, `slides/<uuid>.ndpi` links |
| `eddie/2_embed.sh` | GPU array | step 1 with the same frozen encoder as the model; each task takes every N-th slide and skips what is done |
| `eddie/3_evaluate.sh` | CPU | `entip-train triage evaluate`: each patient's call exactly as in use (same temperature, threshold and novelty check) |

`dbta_classes.csv` maps every DBTA diagnosis to a triage class: 34 glial / glioneuronal diagnoses to glioma, 85 to
other primary CNS (neuronal-only, embryonal, pineal, choroid plexus, meningiomas, nerve sheath, melanocytic, germ cell,
sellar and mesenchymal tumours, and the 6 lymphomas in the WHO CNS5 haematolymphoid chapter), "Metastatic tumours" to
metastasis, the 2 systemic-type lymphomas (EBV-positive DLBCL NOS, follicular) to lymphoma (not trained, so scored
as an unseen class), and the 4 histiocytic diagnoses to none. **It is a judgement table: review it.** The 47
non-tumour controls are left out until someone sets each to normal brain or reactive (their kind is only in the
free-text comment).

## The report (`evaluation-<run>.json`)

- `known`: accuracy, balanced accuracy and recall for the classes the model was trained on, plus how often it got
  the triage questions themselves right (tumour or not, glial or not) and the share flagged as unlike training.
- `gate`: patients accepted (confident), errors among them and the exact 95% upper bound - the external check of the
  error limit set in cross-validation (`guarantee_held`).
- `unseen`: for lymphoma (and any class the model never saw), how often it was confidently called something else.
- `per_diagnosis`: accuracy and confident mistakes for each DBTA diagnosis. Expect circumscribed and paediatric-type
  gliomas, which are not in training, to do worse than adult diffuse gliomas.
- `confident_mistakes`: the patients to look at first.

## On Eddie

1. Download the dataset from its EBRAINS page (needs an EBRAINS account and the Data Use Agreement) into
   `/exports/igmm/eddie/Glioblastoma-WGS/entip/data/dbta/`, with the sample table as `metadata.csv` there
   (or set `DBTA_DIR` and `DBTA_META`).
2. Once the triage model you want to test exists (default: stage A):

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull
MODEL_RUN=stage-a EMBED_SHARDS=8 bash training/runs/dbta/eddie/submit.sh
```

Outputs go to `$BASE/runs/dbta/`. The bags are reusable for any later model trained with the same encoder: rerun
only `3_evaluate.sh` with another `MODEL_RUN`.
