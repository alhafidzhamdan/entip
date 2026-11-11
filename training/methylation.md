# Step 4: methylation class from H&E

Part of the AI diffuse glioma workflow (1 triage, 2 high-grade features, 3 panel, **4 methylation class**,
5 WHO CNS5 rules engine). Research use only: the output is a suggestion, and the model orders the
methylation array whenever it is not confident.

Code: `src/entip_train/methylation/`. It uses the shared slide pipeline from step 1 (`entip_train.wsi`:
tissue tiles, frozen H-optimus-0 bags, attention-MIL, calibration) and emits the step-5 finding record
(`entip.glioma_step/1`, see `dx/contract/README.md`).

## What it does

```
H&E slide(s) ─ wsi.bags.extract_bag (H-optimus-0, 112 µm tiles) ─> bag per slide
patient bags pooled ─ unlike the training slides? ─> call = no_match, order the array
                    ─ ensemble of ABMIL fold members, temperature-scaled ─> class probabilities
                    ─ summed per family ─> call = family (contract vocabulary)
family p >= threshold ? confident : uncertain -> order_test = methylation_array
subclass (e.g. "GBM, RTK II") only when the family is confident and the subclass reaches its own threshold
```

`not_assessed` (call `indeterminate`, order the array) when there are fewer than `min_tiles` tissue tiles,
the bags come from another encoder, or triage says the tissue is not glioma.

`no_match` mirrors what a methylation classifier does with an unfamiliar sample. It is a coarse check
(`novelty.py`: distance of the slide's mean embedding to its nearest training cases, cut-off at the 99th
percentile of held-out-site cases). It catches slides far from the training data; a tumour type that merely
resembles a trained class is not caught and relies on the confidence threshold and on triage.

The deployed model is the ensemble of the cross-validation members, so every part of it was tested on sites
it never saw. The thresholds were measured on single members, the threshold is chosen and its error measured
on the same held-out predictions, and all of it is TCGA: confirm on an external set (Edinburgh) before use.
The training report lists recall, share of confident calls and confident errors per family.

Thresholds are set on out-of-fold predictions where every fold holds whole TCGA tissue source sites, so
the model is always judged on sites it never saw. The threshold is the lowest confidence at which the
one-sided 95% upper bound on the error rate stays under `max_error` (default 5%). If none reaches it, the
model always orders the array. The share of cases above the threshold (coverage) is the number that
says how often the array could be skipped.

## Labels: methylation for the same tumour

| Option | How | Status |
|---|---|---|
| Classifier reports (`label --report`) | CSV `case_id,class,score`, calls under 0.9 dropped | Best for Edinburgh, if reports are kept |
| Own reference classifier (`label --ref-betas --ref-labels`) | Random forest on the most variable CpGs plus score calibration (fitted on out-of-bag votes), trained on a public reference set, applied to TCGA 450K betas | Default for TCGA |
| Clusters (`labels.cluster_labels`) | k-means on TCGA betas; unnamed until a neuropathologist maps them | Exploration only |

The reference set we have in mind is GEO **GSE90496**, the Capper et al. 2018 reference cohort: 2,682
tumours in 82 CNS classes ([OmicsDI record](https://www.omicsdi.org/dataset/geo/GSE90496); paper:
Capper et al., *Nature* 2018, [doi:10.1038/nature26000](https://doi.org/10.1038/nature26000)). Our random
forest and its score calibration follow the paper's approach (raw forest votes are rarely near 1, so the 0.9
cut-off applies to calibrated scores) but are our own scikit-learn code, not their classifier, so we do not
depend on the Heidelberg classifier software (its licence terms are not checked). **Not checked:** the
GSE90496 data-use terms (the OmicsDI page states none) and whether our classifier reproduces their calls. Pass
the sex-chromosome and SNP-affected probes from the array manifest with `--exclude-probes`; SNP (`rs`) and
non-CpG (`ch`) probes are always left out. The label step prints the classifier's out-of-bag accuracy.

## TCGA data

From the GBM plan (GDC counts checked 2026-11-10): TCGA-GBM 389 cases with diagnostic slides and 423 with
methylation (450K and older 27K arrays); TCGA-LGG 491 and 516. The exact number with both a DX slide and a
450K array is not yet counted: run `entip-train methyl pair` on the two GDC file lists. Downloads run on
the Mac or Eddie (the cloud session cannot reach the GDC API).

The slide and the methylation result come from the same tumour but not the same piece of tissue: TCGA
extracted DNA from a frozen portion, and for the FFPE diagnostic slides "the relationship or molecular
similarity of these sections to the tissues submitted for genomic analysis is not as precise" (Cooper et al.,
*J Pathol* 2018, [PMC6240356](https://pmc.ncbi.nlm.nih.gov/articles/PMC6240356)). Expect some label noise in
heterogeneous tumours. All diagnostic slides of a case (DX1, DX2, ...) are pooled into one bag; only primary
tumour samples (type 01) are used.

TCGA is mostly adult GBM and lower-grade glioma, so paediatric, H3-altered and rare classes will be thin
or absent. Classes with fewer than `--min-per-class` (default 10) labelled cases are dropped and listed in
the model's metrics; the model only learns what is left, and everything else falls back to the array.

Beta files: 450K, EPIC and EPIC v2 ids are put in one id space (EPIC v2 suffixes such as `_TC21` are dropped
and replicates averaged). A CpG missing in a few samples is kept (up to 10% missing) rather than dropped for
all. List only 450K or EPIC files from the GDC: the old 27K array shares too few CpGs, and the label step
refuses targets that cover under 90% of the classifier's CpGs.

## Run it

```bash
pip install -e ".[foundation]"
entip-train methyl demo                                  # synthetic, CPU, seconds
# GDC methylation files are named by UUID: give the GDC sample sheet so they map to barcodes
entip-train methyl pair --slides dx_slides.txt --methylation meth450k.txt \
    --methylation-sheet gdc_sample_sheet.tsv --out pairs.csv
entip-train methyl label --ref-betas ref/ --ref-labels ref_labels.csv \
    --target-betas tcga/ --target-sheet gdc_sample_sheet.tsv --out labels.csv
# bags: one <slide name without extension>.npz per slide, from entip_train.wsi.bags.extract_bag (H-optimus-0)
entip-train methyl train --labels labels.csv --pairs pairs.csv --bags bags/ --out models/mc-0.1.0
entip-train methyl predict --model models/mc-0.1.0 --bags SCN-1.npz --study-id SYN-0001 --db http://127.0.0.1:8705
```

`--db` posts up to three actions (the model row is skipped if already registered, so a promoted model is
never reset to `draft`): the model row (`draft`), a `genomic_finding` with marker
`methylation_class_ai` plus the top-attention tiles as `region` rows (`methylation_attention`), and the
`glioma_step` event the rules engine reads. The `_ai` marker never masks a real array result.

## Prior art

Hetairos (*Nature Cancer* 2026, [doi:10.1038/s43018-026-01186-3](https://doi.org/10.1038/s43018-026-01186-3))
predicts 102 methylation classes from H&E: 0.87 accuracy on high-confidence calls, with 50 to 70% of cases
reaching high confidence (as summarised in the GBM plan). We use it as a benchmark, not as code.

## Open

- Does Edinburgh keep IDATs or only classifier reports? Reports give labels directly; IDATs would let us
  run our own reference classifier on local cases too.
- Family mapping of each class needs a neuropathologist's check (`classes.py`; e.g. GBM MYCN is filed under
  paediatric-type HGG).
- Nothing has been run on real slides or real methylation data yet. All results are synthetic.
