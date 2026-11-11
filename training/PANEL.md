# Glioma panel from H&E (`entip-panel`, step 3 of the AI glioma workflow)

Predicts each glioma panel marker from the H&E slides of a patient, with a calibrated confidence. A call is
used only when it passes every gate; otherwise the output names the real test to order. Research use only;
so far run on synthetic data only.

Markers (contract names from the step 5 engine, `dx/contract/README.md`): `IDH`, `ATRX`, `TP53`, `1p19q`,
`CDKN2A`, `EGFR`, `chr7_chr10`, `TERT`, `H3K27`, `H3G34`, `MGMT`. Which ones run is set in
[`configs/glioma_panel.json`](configs/glioma_panel.json), because Edinburgh's panel list is not confirmed yet.
A switched-off marker is still reported, as `not_assessed`.

## How it works

1. **Tiles** come from the shared pipeline in `entip_train/wsi/` (tissue tiles of 112 µm, frozen H-optimus-0
   embeddings, one `.npz` bag per slide). All diagnostic slides of a patient are pooled into one bag.
2. **One attention-MIL classifier per marker** (`wsi.abmil.ABMIL`, NumPy). Separate models let each marker
   use every patient that has its label; TCGA labels are patchy across markers.
3. **Calibration and accept thresholds**, on out-of-fold predictions from cross-validation that keeps each
   TCGA tissue source site in one fold: temperature scaling, then one threshold per call (`panel/gate.py`).
   A call is accepted only where two upper 95% bounds hold: the wrong-call rate among accepted calls of that
   class is within `max_error` (default 5%), and the share of patients of the *other* class accepted as that
   call is within `max_miss` (default 5%). The second rule stops a model passing on a rare alteration by
   calling everyone "wildtype". It also means a negative call cannot be accepted until at least 52 altered
   patients have been seen with none missed. The search runs from the most confident call downward and stops
   at the first failure. A call with no threshold always orders the test.
4. **Gates at prediction time**: same encoder and tile size as training; at least `min_tiles` tissue tiles;
   bag not out of distribution (mean-embedding distance within the 99% training quantile); prerequisite met
   (1p/19q only when IDH-mutant, from a real result or a confident IDH call; IDH-wildtype gives
   `not_assessed`); confidence at or above the threshold for that call. A 1p/19q call that rests on an AI IDH call says so.
5. **Output**: one `entip.glioma_step/1` record per marker, with `call`, `confidence`, `threshold`, `status`
   (`confident` / `uncertain` / `not_assessed`), `order_test`, `reasons`, and the five highest-attention tiles
   as `evidence`. Records are posted to the database as an event action, verb `glioma_step`, and each trained
   marker is registered as a `model` row. AI calls never go into `genomic_finding`, where they would mask the
   real result in `patient_row.molecular`.

## Run

```bash
cd training && pip install -e ".[dev]"
pytest -q tests/test_panel.py
entip-panel demo                     # synthetic: train, gate, predict, about a minute on CPU

# TCGA (on a machine with the bags)
entip-panel labels --cbioportal lgggbm_tcga_pub_samples.tsv --long tcga_extra_labels.csv --out labels.json
entip-panel train --bags bags/ --labels labels.json --out models/panel-0.2.0
entip-panel predict --model models/panel-0.2.0 --bags new_bags/ --scan-map scans.csv --db http://127.0.0.1:8705
```

## Labels

- `--cbioportal`: the sample table of cBioPortal study `lgggbm_tcga_pub` (TCGA pan-glioma, Ceccarelli et al.
  2016, [doi:10.1016/j.cell.2015.12.028](https://doi.org/10.1016/j.cell.2015.12.028)). Columns used, checked
  against the [cBioPortal API](https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-attributes) on
  10 Nov 2026: `IDH_STATUS`, `IDH_CODEL_SUBTYPE` (1p/19q), `ATRX_STATUS`, `TERT_PROMOTER_STATUS`,
  `MGMT_PROMOTER_STATUS`, `CHR_7_GAIN_CHR_10_LOSS`.
- `--long`: CSV `patient_id,finding,call` in contract words, for the markers that table does not carry
  (TP53, CDKN2A, EGFR, H3K27, H3G34, from mutation and copy-number calls) and later for Edinburgh results.
- Unrecognised values are left unlabelled and counted in the output, never guessed.

## Open items

- **Edinburgh panel list**: which markers the local panel actually tests (edit the config).
- **Proxies**: TCGA gives ATRX and TP53 *mutation*, while the panel uses ATRX and p53 *IHC*. Records say so
  in `reasons`. Edinburgh IHC results would be the proper labels.
- **Labels still to build** for TP53, CDKN2A/B, EGFR and H3 from TCGA mutation and GISTIC copy-number files.
  TCGA adult diffuse glioma has few H3-altered cases, so those markers may stay "order the test".
- Thresholds and calibration use the same out-of-fold predictions; an untouched test set (Edinburgh) is still
  needed before any real use.
- The value `IDHmut-codel` in `IDH_CODEL_SUBTYPE` is assumed by symmetry with `IDHmut-non-codel` (seen in the
  API); check the download.
- Nothing has been trained on real slides. All numbers so far are synthetic.
- A model that replaces a diagnostic test would be software as a medical device; the UK route is not checked.

## Licences

Code: this repository. NumPy and scikit-learn (BSD-3-Clause); encoder H-optimus-0 (Apache-2.0). No new
dependencies. TCGA labels and slides are open-access data under the GDC's own terms.
