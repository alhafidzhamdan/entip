# Step 2: triage (tumour? glial? enough tissue?)

[← workflow](README.md) · code `training/glioma/src/entip_glioma/triage/` and `wsi/` · PR #14 (branch `claude/project-thread-6u7wis`)

![Step 2 workflow](diagrams/2-triage.png)

## What it does

One attention-MIL model reads every tile of a patient's H&E slides and gives six classes: `normal_brain`,
`reactive`, `glioma`, `other_primary_cns`, `metastasis`, `lymphoma`. Tumour and glial probabilities are sums of
those. With several slides, the most abnormal one decides. Only a confident glioma call goes on to steps 3 to
5; everything else goes to a pathologist.

## Inputs and outputs

- **In:** FFPE H&E slides (resections and biopsies; frozen sections are a later, separate model).
- **Out:** two records, `adequacy` (tissue area; 1 mm² is a placeholder) and `tissue_class` (call, calibrated
  confidence, threshold, status, probabilities, top attention tiles as evidence), plus regions and predictions
  in the database.

## How it is trained

1. `triage manifest` lists labelled slides (stage A: public IDC data, 3 classes, 3,472 slides; stage B:
   Edinburgh, all 6 classes).
2. Bags from [step 1](step1-pretraining.md) (`entip-train embed build-bags`, GPU): one H-optimus-0 tile bag per
   slide.
3. `triage train` (CPU): gated attention-MIL (Ilse et al. 2018), 5-fold cross-validation with each patient's
   slides kept together, temperature scaling, then the accept threshold by an exact binomial bound.
4. The report includes a source check and leave-one-source-out runs, because each public source holds one class.

```bash
entip-train triage demo                     # synthetic, CPU, seconds
entip-train embed build-bags --manifest stage_a_all.csv --out bags/ --per-class 5    # step 1, trial
entip-train triage train --manifest stage_a_train.csv --bags bags/ --out models/triage
entip-train triage predict bags/<scan>.npz --model models/triage --study-id SYN-0001 --db http://localhost:8000
```

A 10-slide test run of steps 1 and 2, locally or as Eddie jobs: [`training/runs/triage-test-run/`](../../training/runs/triage-test-run/README.md).

Eddie jobs and the stage A dataset: project files `etip/datasets/triage/stage_a/` (`eddie_build_bags.sh` GPU,
`eddie_train.sh` CPU, README).

## Status and open items

- Tested on 7 real public slides with a test encoder; H-optimus-0 not yet run (Eddie trial pending).
- Every stage A slide was scanned on a Leica scanner; more scanners and classes are listed in the design doc.
- Metastasis, lymphoma and reactive need Edinburgh (or BRAIN UK) slides.
- To decide: adequacy area, treatment-effect and neurocytoma classes, error limit per class.
