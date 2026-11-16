# Data sources and licences

Every dataset and model weight ENTIP uses or has considered, with its licence or terms. **Edit
`data-sources.csv`, then run `python docs/tools/render_data_sources.py`**; this page is generated from it.

The rule (`docs/pillar_specs.md`): every model, library and dataset must be usable in the NHS. MIT, BSD, Apache and
CC BY are open; non-commercial (NC), no-derivatives (ND) and copyleft items are research options only and must not
reach a deployable model. Software dependencies are listed in each package's `LICENCES.md`.

- **nhs_rule** - `open`: fine for a deployable model. `research only`: comparison or testing, never training a
  deployable model. `check`: terms not confirmed yet, so treat as research only until they are.
- **checked** - when the terms were last looked at. Licences change: re-check before a release.

This is a working register, not legal advice. Edinburgh Innovations and the data protection office have the final
word on commercial use and on personal data (pseudonymised data are still personal data under GDPR).

## Datasets

| Source | Licence / terms | Commercial | NHS rule | Status | Used for | Access | Checked | Notes |
|---|---|---|---|---|---|---|---|---|
| [CCDI-MCI H&E slides](https://portal.imaging.datacommons.cancer.gov/) (IDC) | CC BY 4.0 | yes with attribution | open | in use | Triage: other primary CNS; the 3 real brain metastases | Public download (idc-index) | 2026-11-16 | Paediatric; also has gliomas (not used yet) |
| [GTEx cerebral cortex H&E slides](https://portal.imaging.datacommons.cancer.gov/) (IDC) | CC BY 4.0 | yes with attribution | open | in use | Triage: normal brain | Public download (idc-index) | 2026-11-16 | PAXgene-fixed and post-mortem; the only normal brain source so far |
| Synthetic slides and patients (This repository) | PolyForm Noncommercial 1.0.0 | ours | open | in use | Tests and demos | In the repo | 2026-11-16 |  |
| [TCGA H&E slides (GBM; LGG; LUAD; LUSC; BRCA; SKCM; KIRC; COAD; READ)](https://portal.imaging.datacommons.cancer.gov/) (Imaging Data Commons (IDC v25)) | CC BY 3.0 (IDC license_short_name) | yes with attribution | open | in use | Triage: glioma (GBM IDH-wildtype; LGG IDH-mutant); non-CNS / metastasis (other organs) | Public download (idc-index) | 2026-11-16 | FFPE diagnostic slides; Leica and some Carl Zeiss scanners |
| [TCGA pan-glioma clinical table (lgggbm_tcga_pub; Ceccarelli et al. 2016)](https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub) (cBioPortal API) | TCGA open-access derived; cite the paper and cBioPortal | check | check | in use | IDH filter for the diverse triage set; panel labels (IDH; 1p/19q; ATRX; TERT; MGMT; +7/-10) | Public API | 2026-11-16 | No separate licence stated on the API |
| [HCMI and CGCI H&E slides](https://portal.imaging.datacommons.cancer.gov/) (IDC) | CC BY 4.0 | yes with attribution | open | in manifest | Triage: a few gliomas; metastases; lymphoma | Public download (idc-index) | 2026-11-16 | Too few patients to train on |
| MIDOG++ (via MitoEd) (MitoEd training data) | See MitoEd LICENCES.md | check | check | indirect | Mitosis candidates (step 3) through MitoEd | n/a | 2026-11-11 | Contains no brain tissue |
| Edinburgh / NHS Lothian slides (NHS Lothian) | Local ethics and information governance approvals | local approval | open once approved | planned | Training; calibration; validation; replacing proxies | Local approval | 2026-11-16 | The real target population; nothing in the repo |
| [GSE90496 methylation reference (Capper et al. 2018)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90496) (NCBI GEO) | Not stated (OmicsDI lists none) | check | check | planned | Labels for methylation class (step 5) | Public download (GEO) | 2026-11-10 | 2682 tumours in 82 classes; terms to confirm |
| [TCGA open-access molecular data (450K/EPIC betas; mutations; copy number; clinical)](https://gdc.cancer.gov/) (NCI Genomic Data Commons) | GDC open access; follow TCGA publication guidelines | yes | open | planned | Methylation (step 5) pairs; panel labels for TP53; CDKN2A/B; EGFR; H3 | Public download (GDC) | 2026-11-10 | DNA from a frozen portion; not the FFPE slide |
| [BRAIN-UK](https://brain-uk.org/) (UK Brain Archive Information Network) | Per approved application (NHS archive tissue) | check | check | candidate | UK/NHS tissue and slides for training and testing | Application | 2026-11-16 | Neuropath-AI's largest external test set (UCL) came through BRAIN-UK |
| [CPTAC H&E slides (GBM; COAD; LUAD; LSCC; BRCA; CCRCC; CM; others)](https://portal.imaging.datacommons.cancer.gov/) (IDC) | CC BY 4.0 | yes with attribution | open | candidate | Second source per non-CNS organ; third glioma source | Public download (idc-index) | 2026-11-16 | Fixative not recorded in IDC (prep 'unknown'); CPTAC-GBM needs IDH from a CPTAC study |
| [Children's Brain Tumor Network (CBTN)](https://cbtn.org/research-resources) (CBTN) | Per data access agreement | check | check | candidate | Paediatric CNS tumours | Application | 2026-11-16 | Used by Neuropath-AI and DEPLOY |
| [Digital Brain Tumour Atlas (Roetzer-Pejrimovsky et al. 2022)](https://search.kg.ebrains.eu/instances/8fc108ab-e2b4-406f-8999-60269dc1f994) (EBRAINS (Medical University of Vienna)) | EBRAINS Data Use Agreement for pseudonymised human data; cite DOI; no commercial use | no | research only | candidate | External test set (126 tumour types; Hamamatsu scanner) | EBRAINS account + DUA | 2026-11-16 | Personal data under GDPR; training on it may count as commercial derivative: ask the owners first |
| [BraTS-Path 2024](https://www.synapse.org/brats) (BraTS challenge) | CC BY-NC | no | research only | not used | Necrosis / MVP region labels (step 3) | Challenge registration | 2026-11-11 |  |
| [Ivy Glioblastoma Atlas (Ivy GAP)](https://glioblastoma.alleninstitute.org/) (Allen Institute) | Allen Institute terms of use (non-commercial) | no | research only | not used | Region labels (step 3) | Public download | 2026-11-11 |  |

## Model weights and code

| Source | Licence / terms | Commercial | NHS rule | Status | Used for | Access | Checked | Notes |
|---|---|---|---|---|---|---|---|---|
| [H-optimus-0 weights](https://huggingface.co/bioptimus/H-optimus-0) (Bioptimus (Hugging Face)) | Apache-2.0 (gated: accept terms on Hugging Face) | yes | open | in use | Step 1 tile encoder | Hugging Face login | 2026-11-16 |  |
| [timm vit_small_patch16_224 (AugReg) weights](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k) (Google via timm) | Apache-2.0 | yes | open | in use | Start point for the distilled student | Hugging Face (timm) | 2026-11-16 | Optional: --init scratch |
| ENTIP distilled student (This repository) | PolyForm Noncommercial 1.0.0 (ours); trained on H-optimus-0 outputs (Apache-2.0) | ours | open | built | Embedding without a large GPU | Produced by entip-glioma distill | 2026-11-16 |  |
| MitoEd (MitoEd project) | MIT code; weights per its LICENCES.md | check | check | indirect | Mitosis candidates (step 3) | n/a | 2026-11-11 | Cellpose-SAM weights were trained on CC BY-NC data |
| [CLAM](https://github.com/mahmoodlab/CLAM) (Mahmood lab) | GPL-3.0 | copyleft | not used | not used | None: ENTIP has its own attention-MIL | n/a | 2026-11-16 | Ideas from the paper may be reimplemented |
| [DEPLOY / Neuropath-AI](https://patents.google.com/patent/WO2025165914A1/en) (NIH (NCI) and Australian National University) | Non-commercial academic use; patent application WO2025165914A1 (pending) | no | research only | not used | None (comparison) | Code on request | 2026-11-16 | Claims cover predicting methylation from H&E then classifying; check before adding that to ENTIP |
| Heidelberg methylation classifier (DKFZ / Heidelberg) | Not checked | check | check | not used | None: ENTIP uses its own classifier | n/a | 2026-11-10 |  |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) (Microsoft (Hugging Face)) | Not checked | check | check | not used | Possible alternative encoder (used by Hetairos) | Hugging Face login | 2026-11-16 |  |
| [UNI; UNI2-h; CONCH; TITAN](https://huggingface.co/MahmoodLab) (Mahmood lab (Hugging Face)) | CC BY-NC-ND 4.0 | no | research only | not used | Comparison only | Hugging Face login | 2026-11-16 | Used by Neuropath-AI (UNI) |
| Virchow2; H-optimus-1 (Paige; Bioptimus) | CC BY-NC-ND 4.0 | no | research only | not used | Comparison only | Hugging Face login | 2026-11-11 |  |
