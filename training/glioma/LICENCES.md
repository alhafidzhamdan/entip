# Licences (training/glioma/)

Al's rule: every component must be openly licensed so it can be deployed in the NHS. Research use only.

| Component | Use | Licence | Status |
|---|---|---|---|
| this package | code | PolyForm Noncommercial 1.0.0 (see ../../LICENSE.md) | ours; non-commercial use, commercial use needs permission |
| entip-train | slide readers, encoders (H-optimus-0), ontology | this repository; see [its LICENCES.md](../../platform/live-training/LICENCES.md) | ours |
| NumPy, scikit-learn, joblib | arrays, models, saving | BSD-3-Clause | open |
| SciPy | exact binomial bound for the accept threshold | BSD-3-Clause | open |
| idc-index, idc-index-data | triage manifest from the Imaging Data Commons (`[idc]`) | MIT (PyPI classifiers, checked 2026-11-10) | open |
| timm `vit_small_patch16_224` weights | ImageNet start point for the distilled student (`distill`, optional: `--init scratch`) | Apache-2.0 (Google AugReg via timm) | open |
| wsi/abmil.py | attention-MIL, our own NumPy code after Ilse et al. 2018 (arXiv 1802.04712) | MIT (this package) | open; no CLAM code |

Not used here: UNI, UNI2-h, Virchow2, H-optimus-1 (CC-BY-NC-ND), the Heidelberg classifier software and CLAM.

`highgrade/` adds no dependencies. It reads MitoEd output (MIT code; see MitoEd's LICENCES.md for its weights and
data). Region labels for real training are not chosen yet; check their licence before use.

Data: synthetic only in this repository (generated in `triage/synthetic.py`, `highgrade/synthetic.py` and the
tests). No patient data.

Triage training data (downloaded on Al's Mac or Eddie, never committed): IDC slides under CC BY 4.0
(GTEx, CCDI-MCI, CPTAC-GBM, HCMI, CGCI) and CC BY 3.0 (TCGA), as listed per slide in `license_short_name`.
Both need attribution in any publication or model card. ICDC-Glioma (canine) is excluded.

Step 5 (methylation class) uses only the packages above (scikit-learn random forest and k-means for labels).
Training data planned: TCGA diagnostic slides and 450K beta values (GDC open access); the GSE90496 reference set
(data-use terms not yet checked). The Heidelberg classifier software is not used.
