# ENTIP against published brain-tumour classifiers

How the triage training set (stage A) and its external test sets compare in scale and design with published
H&E classifiers for CNS tumours. Figures for other studies are taken from their papers and are approximate where
sources differ (noted). Last checked 2026-11-16.

## Training and test data

| Study | Task | Training | External test | Test centres | Classes | Molecular truth |
|---|---|---|---|---|---|---|
| **ENTIP stage A** | Triage | 2,373 patients, 3,138 slides, about 30–35M tiles; TCGA, CPTAC, CCDI, GTEx (US) | DBTA: up to 2,880 patients (Vienna, Hamamatsu); IPD-Brain: 328 glioma patients (Hyderabad, Morphle) | 2 | 4 trained | IDH from sequencing (TCGA, CPTAC-GBM) for every training glioma |
| Integrated glioma classification (*Nat Commun* 2023, s41467-023-41195-9) | Adult-type diffuse glioma type, grade and genotype (WHO CNS5) | 1,362 patients (+340 validation, 289 internal test), 1 centre (Zhengzhou, China) | 305 + 328 patients | 2 (Henan, Beijing) | 6 (A2–4, O2–3, GBM) | IDH and TERT by Sanger sequencing; 1p/19q, CDKN2A, EGFR, +7/−10 tested |
| DEPLOY (Hoang et al., *Nat Med* 2024) | CNS tumour type | 1,796 patients (NCI) | about 2,150 patients, DBTA included | 3 cohorts | 10 | Methylation classes |
| Neuropath-AI (Lalchungnunga et al., *Lancet Oncol* 2026) | CNS tumour type | 5,835 samples, 3 sources | 5,516 samples | 4 | 9 families, 52 types | Methylation classes |
| Hetairos (Heidelberg, *Nat Cancer* 2026) | Methylation subtype from H&E | 4,961 cases, 6,115 slides | 111 prospective cases | 1 | 102 | Methylation classes |
| TITAN brain benchmark (EBRAINS) | Tumour type | about 3,100 public slides | none (benchmark split) | 0 | 30 | Diagnosis labels |
| IPD-Brain baseline (Chauhan et al., *Sci Data* 2024) | Glioma subtype | 328 patients, 1 centre (India) | none (cross-validation; TCGA as a second set) | 0 | 3 | IDH1 R132H by IHC only |

Patients per class: stage A about 400–800; the Chinese glioma study 445–1,676 per type (glioblastoma is 1,676
of its 2,624 patients); DEPLOY about 180; Neuropath-AI about 110 per type; Hetairos about 50.

Foundation models are trained on far more data, but without labels: H-optimus-0 on several hundred thousand slides,
Prov-GigaPath on about 171,000 slides (1.3 billion tiles), UNI on about 100,000 slides (about 100 million tiles).
ENTIP trains only a small attention model on top of a frozen encoder, so it needs far less.

## Where ENTIP matches the field

- **Scale.** About 2,400 training patients sits between DEPLOY (1,796) and Neuropath-AI (5,835), and with 4 classes
  instead of 10–102 each class has several times more patients.
- **External testing.** Two external centres (DBTA, IPD-Brain), like the Chinese glioma study and close to
  DEPLOY's three cohorts. The DBTA is also one of DEPLOY's test sets, so results can be set side by side at the
  coarse level (glial or not, tumour or not).
- **Test-to-train ratio.** The external patients outnumber the training ones (about 1.3:1), as in DEPLOY and
  Neuropath-AI; Hetairos's prospective test is far smaller.
- **Molecular truth for training gliomas.** Every training glioma has IDH from sequencing, the standard of the
  Chinese glioma study; IPD-Brain (IHC only) is therefore used to test, not to train.

## Where ENTIP falls short

- **Training sources.** All training data are public US collections, and normal brain (GTEx) and other primary CNS
  (CCDI) come from one source each. Neuropath-AI trained on three sources; the Chinese study on one centre but
  tested on two. Non-US training data would need an application (BRAIN-UK) or a partner centre.
- **"Other primary CNS" differs between training and test.** It is paediatric in training (CCDI) and mostly adult
  in the DBTA (meningiomas, schwannomas, pituitary tumours, about 90 primary CNS lymphomas). Expect this class to
  score worst there.
- **Class coverage.** No reactive tissue and no adult non-glial CNS tumours in training. The published type
  classifiers cover dozens of CNS types.
- **Grade and type within glioma.** Triage only says "glioma". The Chinese study and DEPLOY go further (type,
  grade, genotype); that belongs to later ENTIP steps, where the Chinese study is the closest comparison.

## Methods, briefly

The Chinese glioma study clusters tiles with k-means (9 clusters), keeps the 3 most useful clusters and trains a
CNN on them before combining tiles per patient: the approach before foundation models. ENTIP uses a frozen
pathology foundation model with attention over tiles; check each other paper's methods before quoting their
architecture. Licences differ: DEPLOY and Neuropath-AI are non-commercial with a pending patent (WO2025165914A1);
see `docs/data-sources.md`.

## Related method outside CNS tumours

**Pheno-MYCN** (Chai et al., bioRxiv preprint v2, 9 October 2026, not peer reviewed,
[doi:10.64898/2026.08.20.745848](https://doi.org/10.64898/2026.08.20.745848); checked 2026-11-21). It is left out of
the table above because it studies paediatric neuroblastoma, not a CNS classifier.

- **Task and data.** Predicts MYCN amplification (binary, from FISH) on 189 H&E slides from 86 patients at one
  centre (Great Ormond Street biobank): 49 amplified, 140 not; 68 primary, 121 relapse. No external test.
- **Method.** Frozen UNI tile embeddings, an attention MIL branch for the slide label, and a Gaussian mixture branch
  that groups tiles into 6 phenotypes, which experts then mapped to neuroblastoma morphology.
- **Results.** Mean test accuracy 0.889, F1 0.873 and AUC 0.900 over ten hold-out splits of 9 test slides each.
  Haemorrhagic and acellular necrosis clusters were enriched in amplified slides.
- **Relevance to ENTIP.** The closest published pattern to the grade_mil tile review: weak slide labels, then expert
  review of what the model picks out. Its phenotype clusters could complement ranking tiles by attention.
- **Caution.** Splits are by slide, so one patient's slides can fall on both sides. The authors justify this because
  status can differ within a patient (6 of 86) and report a check on fully held-out patients. ENTIP should still
  split by patient.

## Data access for the studies above

| Study | Slides available? |
|---|---|
| Chinese glioma study | No: on request to the authors, institutional review and a user agreement; 6 example patients with the code on CodeOcean; Sanger data in GSA-Human (HRA005239), access-controlled |
| DEPLOY | Its external tests include the DBTA (public, EBRAINS); other cohorts per the paper |
| IPD-Brain | Yes, after registration (India Data DUA: research use; no commercial use without consent) |
| DBTA | Yes, through EBRAINS after the Data Use Agreement (research only) |
| Pheno-MYCN | Slides restricted (Great Ormond Street biobank; any access under a non-commercial agreement). Code is GPL-3.0 at github.com/cbhindex/pheno_mycn; weights are non-commercial, following UNI's terms |
