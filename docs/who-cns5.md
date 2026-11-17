# WHO CNS5 in ENTIP

Every class and every diagnosis in ENTIP is stated in the terms of the WHO Classification of Tumours of the Central
Nervous System, 5th edition (2021), so neuropathologists can read the reports, error tables and review sheets in
their own vocabulary. The rules live in one place:
[`training/glioma/src/entip_glioma/triage/who_cns5.py`](../training/glioma/src/entip_glioma/triage/who_cns5.py).
They are a judgement table: review them, and edit the tables there rather than the code.

## Conventions

- **Grades** in Arabic numerals, within a type: "Astrocytoma, IDH-mutant, CNS WHO grade 3".
- **NOS** when the test a type needs was not done or is not available to us: "Oligodendroglioma, NOS" without
  1p/19q, "High-grade glioma, NOS" for a glioblastoma-like tumour without IDH status. **NEC** only when testing was
  done and fits no type; our sources rarely allow that, so we rarely use it.
- **Dropped or renamed terms are translated**: WHO 2016 "glioblastoma, IDH-mutant" becomes "Astrocytoma, IDH-mutant,
  CNS WHO grade 4"; "haemangiopericytoma" becomes "Solitary fibrous tumour"; "anaplastic ganglioglioma" (not a
  CNS5 type) becomes "Glioneuronal tumour, NOS".
- **Every diagnosis carries its basis** (`who_cns5_basis`): what the term rests on, for example "IDH and 1p/19q
  from TCGA pan-glioma; grade from histology; CDKN2A/B not assessed". A reader can see how far each diagnosis is
  integrated, and which ones a full molecular work-up could change.

## Triage classes

| Class | WHO CNS5 name | What it covers |
|---|---|---|
| `normal_brain` | Non-neoplastic CNS tissue (normal) | Normal brain parenchyma; not a tumour category |
| `reactive` | Non-neoplastic CNS tissue (reactive) | Gliosis, inflammation, infarct; not yet trained |
| `glioma` | Gliomas and glioneuronal tumours | Chapter 1 (gliomas, glioneuronal and neuronal tumours) without the neuronal tumours: adult-type diffuse, paediatric-type diffuse low- and high-grade, circumscribed astrocytic, glioneuronal and ependymal tumours |
| `other_primary_cns` | Other primary CNS tumours | Every other CNS5 chapter: neuronal, choroid plexus, embryonal, pineal, cranial and paraspinal nerve, meningiomas, mesenchymal non-meningothelial, melanocytic, haematolymphoid (primary CNS lymphomas), germ cell, sellar region |
| `metastasis` | Metastases to the CNS | The CNS5 chapter; in public training data, primary tumours of the commonest origins stand in for it |
| `lymphoma` | Secondary (systemic) lymphoma involving the CNS | Systemic lymphoma types; primary CNS lymphomas are counted as other primary CNS |

## Where each diagnosis comes from

| Data | Column | How the WHO CNS5 term is made |
|---|---|---|
| Training slides (IDC), `trial.csv` | `diagnosis`, `who_cns5_basis`; IDC's own term stays in `dx` | Confirmed label corrections first; then the collection's tissue (GTEx cortex, the non-CNS primaries); then TCGA and CPTAC gliomas from IDH, 1p/19q, histological grade, TERT promoter and +7/−10 (`fetch_idh.py`); then rules on IDC's ICD-O term (CCDI) |
| DBTA (Vienna) | `diagnosis`; DBTA's WHO 2016 term in `dbta_diagnosis` | `training/runs/dbta/dbta_classes.csv`, column `who_cns5`, one row per DBTA diagnosis |
| IPD-Brain (Hyderabad) | `diagnosis`, `who_cns5_basis` | Subtype, grade, IDH1 R132H stain and age: a negative stain counts as IDH-wildtype only for a glioblastoma over 55; 1p/19q was not tested, so oligodendrogliomas are NOS |

Training gliomas from TCGA come out as (patients in cBioPortal's pan-glioma study, before picking): glioblastoma,
IDH-wildtype (596); astrocytoma, IDH-mutant, grades 2–4; oligodendroglioma, IDH-mutant and 1p/19q-codeleted,
grades 2–3; and 34 IDH-wildtype lower-grade tumours without TERT promoter mutation or +7/−10 in the data, left as
"Diffuse glioma, IDH-wildtype, NOS" because EGFR and H3 were not assessed.

## Known limits

- **CDKN2A/B is not assessed** for IDH-mutant astrocytomas, so a few graded 2–3 here would be grade 4.
- **Paediatric tumours (CCDI)** have only their ICD-O term, so most are NOS at the molecular level (for example
  "Medulloblastoma, NOS, CNS WHO grade 4", "High-grade glioma, NOS").
- **Metastasis in training is a stand-in**: primary lung, breast, melanoma, kidney and colorectal tumours, not
  tumour in the brain. The DBTA's 47 metastases are the real test.
- **A source term no rule recognises** is shown as the class name with NOS and a basis ending in "review";
  `add_who_cns5.py` lists them so a rule can be added.

## Commands

New runs get the columns from `pick_slides.py`. For a run picked before 2026-11-17, refresh the molecular table
and add the columns in place (the original is kept as `trial.before-who-cns5.csv`):

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull && REPO=$PWD . training/runs/triage/eddie/config.sh && activate_env && W=/exports/igmm/eddie/Glioblastoma-WGS/entip/runs/triage-diverse && python training/runs/triage/fetch_idh.py $W/idh.tsv && python training/runs/triage/add_who_cns5.py $W/trial.csv --idh $W/idh.tsv
```

`mistakes.py` then shows each patient's WHO CNS5 diagnosis under its row.
