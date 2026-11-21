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
| [CPTAC H&E slides (GBM; COAD; LUAD; LSCC; BRCA; CCRCC; CM; others)](https://portal.imaging.datacommons.cancer.gov/) (IDC) | CC BY 4.0 | yes with attribution | open | planned | Second source per non-CNS organ; third glioma source | Public download (idc-index) | 2026-11-16 | Fixative not recorded in IDC (prep 'unknown'); CPTAC-GBM needs IDH from a CPTAC study |
| [CPTAC-GBM mutation calls (gbm_cptac_2021; Wang et al. 2021)](https://www.cbioportal.org/study/summary?id=gbm_cptac_2021) (cBioPortal API) | CPTAC open-access derived; cite the paper and cBioPortal | check | check | planned | IDH status of CPTAC-GBM gliomas (IDH1 R132 / IDH2 R172) | Public API | 2026-11-16 | No separate licence stated on the API |
| [Digital Brain Tumour Atlas (Roetzer-Pejrimovsky et al. 2022)](https://search.kg.ebrains.eu/instances/8fc108ab-e2b4-406f-8999-60269dc1f994) (EBRAINS (Medical University of Vienna)) | EBRAINS Data Use Agreement for pseudonymised human data; cite DOI; no commercial use | no | research only | planned | External test set only (never trained on): 126 tumour types; real metastases; lymphoma; Hamamatsu scanner (training/runs/dbta) | EBRAINS account + DUA | 2026-11-16 | Personal data under GDPR; training on it may count as commercial derivative: ask the owners first |
| Edinburgh / NHS Lothian slides (NHS Lothian) | Local ethics and information governance approvals | local approval | open once approved | planned | Local validation only (at a much later date); not for training | Local approval | 2026-11-16 | The real target population; nothing in the repo |
| [GSE90496 methylation reference (Capper et al. 2018)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90496) (NCBI GEO) | Not stated (OmicsDI lists none) | check | check | planned | Labels for methylation class (step 5) | Public download (GEO) | 2026-11-10 | 2682 tumours in 82 classes; terms to confirm |
| [IPD-Brain (Chauhan et al. 2024; Sci Data)](https://india-data.org/dataset-details/170acc68-1288-499e-9a91-b951e569e70d) (India Data (IHub-Data; IIIT Hyderabad); slides from NIMS Hyderabad) | India Data DUA: research and other lawful use; no commercial use without the provider's written consent; attribution; DPDPA 2023; Indian law | no (without written consent) | research only | planned | External test set only (never trained on): 328 glioma patients; 484 labelled slides; Indian population; Morphle DigiPath scanner (training/runs/ipd) | India Data account (registration) + DUA; 797 GB | 2026-11-16 | IDH from IHC (IDH1 R132H) only; no normal or other CNS; Figshare copy (10.6084/m9.figshare.27186087) is CC BY 4.0 but about 12 GB (likely downsampled); training needs written consent |
| [TCGA open-access molecular data (450K/EPIC betas; mutations; copy number; clinical)](https://gdc.cancer.gov/) (NCI Genomic Data Commons) | GDC open access; follow TCGA publication guidelines | yes | open | planned | Methylation (step 5) pairs; panel labels for TP53; CDKN2A/B; EGFR; H3 | Public download (GDC) | 2026-11-10 | DNA from a frozen portion; not the FFPE slide |
| [BRAIN-UK](https://brain-uk.org/) (UK Brain Archive Information Network) | Per approved application (NHS archive tissue) | check | check | candidate | UK/NHS tissue and slides for training and testing | Application | 2026-11-16 | Neuropath-AI's largest external test set (UCL) came through BRAIN-UK |
| [Children's Brain Tumor Network (CBTN)](https://cbtn.org/research-resources) (CBTN) | Per data access agreement | check | check | candidate | Paediatric CNS tumours | Application | 2026-11-16 | Used by Neuropath-AI and DEPLOY |
| [Multi-lab Stained and Multi-scanner Scanned Sarcoma WSI Dataset](https://bhchai.com/visualise_scan_stain_efffects/dataset.html) (Binghao Chai et al. (Kaggle)) | CC BY 4.0 (all 12 Kaggle parts) | yes with attribution | open | candidate | Testing robustness to stain and scanner variation | Public download (Kaggle; 12 parts; about 2.1 TB) | 2026-11-21 | 1,243 WSI of sarcomas and mimics: 14 subtypes, one case and block each; serial sections stained at 30 NHS labs and scanned on Aperio GT 450, Hamamatsu S360 and 3D-Histech P1000. Not CNS, and only 14 cases, so split by case and do not use for tumour-class training. Paper: https://doi.org/10.1002/2056-4538.70080 (its data statement not read) |
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
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) (Microsoft (Hugging Face)) | Apache-2.0 (per the Hugging Face model card) | licence yes; deployment out of scope | check | not used | Possible alternative encoder (used by Hetairos) | Hugging Face login | 2026-11-21 | Tile encoder plus a slide-level model (LongNet); both under the same licence. The model card puts any deployed use out of scope and says it is not for clinical care or decision-making |
| [UNI; UNI2-h; CONCH; TITAN](https://huggingface.co/MahmoodLab) (Mahmood lab (Hugging Face)) | CC BY-NC-ND 4.0 | no | research only | not used | Comparison only | Hugging Face login | 2026-11-16 | Used by Neuropath-AI (UNI) |
| Virchow2; H-optimus-1 (Paige; Bioptimus) | CC BY-NC-ND 4.0 | no | research only | not used | Comparison only | Hugging Face login | 2026-11-11 |  |
