# AI glioma workflow: shared output contract

Steps 1 to 4 each emit **finding records** (`entip.glioma_step/1`). Step 5, the WHO CNS5 rules engine, reads
them and writes one **integrated diagnosis draft** (`entip.integrated_dx/1`) for the pathologist to sign off.

The source of truth is [`../src/entip_dx/contract.py`](../src/entip_dx/contract.py). The JSON Schemas in this
folder are generated from it (`entip-dx export-contract dx/contract`). Validate a record in Python with
`entip_dx.contract.validate_finding(rec)`.

## One finding record per finding

One record = one call about one patient: the triage class, one high-grade feature, one panel marker, or the
methylation class. A step that makes several calls emits several records.

```json
{
  "contract": "entip.glioma_step/1",
  "step": "high_grade_features",
  "finding": "necrosis",
  "study_id": "SYN-0001",
  "scan_ids": ["SCN-0001-A1-01"],
  "call": "present",
  "source": "ai",
  "confidence": 0.94,
  "threshold": 0.90,
  "status": "confident",
  "probabilities": {"present": 0.94, "absent": 0.06},
  "value": {"area_mm2": 3.1},
  "evidence": [{"kind": "region", "id": "SCN-0001-A1-01:r000017"}],
  "order_test": null,
  "reasons": ["palisading necrosis in 3 regions"],
  "model": {"model_id": "necrosis-head", "version": "0.1.0", "licence": "MIT"},
  "provenance": {"kind": "model", "actor": "necrosis-head", "version": "0.1.0"}
}
```

| Field | Rule |
|---|---|
| `step`, `finding`, `call` | From the table below. `call` may also be `indeterminate` (then `status` must be `uncertain`). |
| `source` | `ai` (default), `test` (a real lab result) or `pathologist` (a person's call). The engine always prefers `test`, then `pathologist`, then a confident `ai` call. |
| `confidence` | Calibrated probability that `call` is right. Required for `ai`. |
| `threshold` | The calibrated accept threshold for this finding and model version (set on held-out data at an agreed error rate). |
| `status` | `confident` (use it), `uncertain` (order the real test), `not_assessed`. If left out, it is `confident` when `confidence >= threshold`. |
| `order_test` | The real test to run when uncertain, from the test list. Filled from the default table if left out. |
| `calibration` | Optional: `{"method": "temperature" | "platt" | "isotonic" | "rule" | "none", "calibrated": bool, "validated": bool, "max_error": 0.05, "error_upper": float, "coverage": float, "n_heldout": int}`. Says how `confidence` was calibrated and whether the threshold met its error target on held-out data. |
| `evidence` | Pointers into the db: `{"kind": "region" | "cell" | "tile" | "prediction" | "scan" | ..., "id": ...}`, optional `x_um, y_um, w_um, h_um, weight`. |
| `model`, `provenance` | `model.model_id` is required for `ai`. Provenance as in the db (`kind`, `actor`, `version`). |

**Never in a record:** age, site, location, radiology, imaging, sex, or any identifier. Age, site and imaging
are a prior only: give them to your model if it uses them, but they go to the engine separately and never
appear in a report. `validate_finding` refuses those keys anywhere in a record.

## Findings and calls

| Step | `finding` | `call` | Default test when uncertain |
|---|---|---|---|
| `triage` | `adequacy` | `adequate`, `inadequate` | `additional_sampling` |
| | `tissue_class` | `normal_brain`, `reactive`, `glioma`, `other_primary_cns`, `metastasis`, `lymphoma` | `pathologist_review` |
| | `glioma_pattern` (optional) | `diffuse_astrocytic`, `oligodendroglial`, `circumscribed`, `ependymal` | `pathologist_review` |
| `high_grade_features` | `necrosis` | `present`, `absent` | `pathologist_review` |
| | `microvascular_proliferation` | `present`, `absent` | `pathologist_review` |
| | `mitotic_activity` | `present` (elevated), `absent` | `pathologist_review` |
| `glioma_panel` | `IDH` | `mutant`, `wildtype` | `IDH1_R132H_IHC` |
| | `ATRX` | `retained`, `lost` | `ATRX_IHC` |
| | `TP53` | `mutant`, `wildtype` | `p53_IHC` |
| | `1p19q` | `codeleted`, `not_codeleted` | `1p19q_testing` |
| | `CDKN2A` (CDKN2A/B) | `homozygous_deletion`, `retained` | `CDKN2A_B_testing` |
| | `EGFR` | `amplified`, `not_amplified` | `EGFR_amplification_testing` |
| | `chr7_chr10` | `gain7_loss10`, `absent` | `chr7_chr10_copy_number` |
| | `TERT` (promoter) | `mutant`, `wildtype` | `TERT_promoter_sequencing` |
| | `H3K27` | `altered`, `wildtype` | `H3K27M_IHC` |
| | `H3G34` | `mutant`, `wildtype` | `H3G34_testing` |
| | `MGMT` (promoter) | `methylated`, `unmethylated` | `MGMT_promoter_methylation` |
| `methylation_class` | `methylation_class` | `glioblastoma_idh_wildtype`, `astrocytoma_idh_mutant`, `oligodendroglioma_idh_mutant_codeleted`, `diffuse_midline_glioma_h3k27`, `diffuse_hemispheric_glioma_h3g34`, `pediatric_hgg_h3wt_idhwt`, `control_tissue`, `other`, `no_match` | `methylation_array` |

Panel marker names match `db` `genomic_finding.marker`. For methylation, `call` is the WHO-level family; put
the classifier's own class and version in `value` (`{"class": "GBM, RTK II", "classifier": "..."}`).

Tests (`order_test`): `pathologist_review`, `additional_sampling`, `IDH1_R132H_IHC`, `IDH_sequencing`,
`ATRX_IHC`, `p53_IHC`, `1p19q_testing`, `CDKN2A_B_testing`, `EGFR_amplification_testing`,
`chr7_chr10_copy_number`, `TERT_promoter_sequencing`, `H3K27M_IHC`, `H3K27me3_IHC`, `H3G34_testing`,
`MGMT_promoter_methylation`, `methylation_array`.

## Where records go in the database

Keep writing your own regions, predictions and models as you do now. In addition, post your finding records
as one event action so the engine can find them in the patient's log:

```json
{"verb": "glioma_step", "provenance": {"kind": "model", "actor": "<model_id>", "version": "<version>"},
 "detail": {"study_id": "SYN-0001", "records": [ <finding records> ]}}
```

The engine reads `GET /patients/{study_id}/actions` for `glioma_step` events (latest record per finding
wins) and reads real results from `genomic_finding` as `source: "test"`. Markers ending in `_ai` in
`genomic_finding` are ignored by the engine, so an AI value never masks or doubles a real result.

## Engine output: `entip.integrated_dx/1`

See [`integrated_dx.schema.json`](integrated_dx.schema.json). In short: the integrated diagnosis (WHO CNS5
type, CNS WHO grade, NOS/NEC), each report layer (histology, grade, molecular, methylation) with its source and
confidence, the WHO criteria it was `based_on` (with citations), the tests to order and why, conflicts, a draft
report in text, and `signoff.status = "awaiting_pathologist"`. It is posted as an event action with verb
`integrated_dx`. It never carries age, site or imaging.
