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
patient bags pooled ─ ABMIL (multi-class) ─ temperature ─> class probabilities
                    ─ summed per family ─> call = family (contract vocabulary)
family p >= threshold ? confident : uncertain -> order_test = methylation_array
subclass (e.g. "GBM, RTK II") only reported when it reaches its own threshold
```

`not_assessed` (call `indeterminate`, order the array) when there are fewer than `min_tiles` tissue tiles,
the bags come from another encoder, or triage says the tissue is not glioma.

Thresholds are set on out-of-fold predictions where every fold holds whole TCGA tissue source sites, so
the model is always judged on sites it never saw. The threshold is the lowest confidence at which the
one-sided 95% upper bound on the error rate stays under `max_error` (default 5%). If none reaches it, the
model always orders the array. The share of cases above the threshold (coverage) is the number that
says how often the array could be skipped.

## Labels: methylation for the same tumour

| Option | How | Status |
|---|---|---|
| Classifier reports (`label --report`) | CSV `case_id,class,score`, calls under 0.9 dropped | Best for Edinburgh, if reports are kept |
| Own reference classifier (`label --ref-betas --ref-labels`) | Random forest on the most variable CpGs, trained on a public reference set, applied to TCGA 450K betas | Default for TCGA |
| Clusters (`labels.cluster_labels`) | k-means on TCGA betas; unnamed until a neuropathologist maps them | Exploration only |

The reference set we have in mind is GEO **GSE90496**, the Capper et al. 2018 reference cohort: 2,682
tumours in 82 CNS classes ([OmicsDI record](https://www.omicsdi.org/dataset/geo/GSE90496); paper:
Capper et al., *Nature* 2018, [doi:10.1038/nature26000](https://doi.org/10.1038/nature26000)). Our random
forest follows the paper's approach but is our own scikit-learn code, not their classifier, so we do not
depend on the Heidelberg classifier software (its licence terms are not checked). **Not checked:** the
GSE90496 data-use terms (the OmicsDI page states none) and whether our classifier reproduces their calls.

## TCGA data

From the GBM plan (GDC counts checked 2026-11-10): TCGA-GBM 389 cases with diagnostic slides and 423 with
methylation (450K and older 27K arrays); TCGA-LGG 491 and 516. The exact number with both a DX slide and a
450K array is not yet counted: run `entip-train methyl pair` on the two GDC file lists. Downloads run on
the Mac or Eddie (the cloud session cannot reach the GDC API).

TCGA is mostly adult GBM and lower-grade glioma, so paediatric, H3-altered and rare classes will be thin
or absent. The model only learns classes present in the labels; everything else falls back to the array.

## Run it

```bash
pip install -e ".[foundation]"
entip-train methyl demo                                  # synthetic, CPU, seconds
entip-train methyl pair --slides dx_slides.txt --methylation meth450k.txt --out pairs.csv
entip-train methyl label --ref-betas ref/ --ref-labels ref_labels.csv --target-betas tcga/ --out labels.csv
# bags: one <slide>.npz per slide from entip_train.wsi.bags.extract_bag with H-optimus-0
# labels.csv for train needs a 'slides' column (slide names joined by ';'), from pairs.csv
entip-train methyl train --labels labels.csv --bags bags/ --out models/mc-0.1.0
entip-train methyl predict --model models/mc-0.1.0 --bags SCN-1.npz --study-id SYN-0001 --db http://127.0.0.1:8705
```

`--db` posts three actions: the model row (`draft`), a `genomic_finding` with marker
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
