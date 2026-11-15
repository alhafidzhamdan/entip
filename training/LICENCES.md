# Licences (training/)

Al's rule: every component must be openly licensed so it can be deployed in the NHS. Research use only.

| Component | Use | Licence | Status |
|---|---|---|---|
| this package | code | MIT | open |
| NumPy | arrays | BSD-3-Clause | open |
| scikit-learn | live heads (logistic regression, histogram gradient boosting), CV | BSD-3-Clause | open |
| joblib | saving heads | BSD-3-Clause | open |
| FastAPI, Uvicorn | HTTP API (optional `[api]`) | MIT, BSD-3-Clause | open |
| PyTorch, timm, huggingface_hub | foundation encoder (optional `[foundation]`) | BSD-3-Clause, Apache-2.0, Apache-2.0 | open |
| H-optimus-0 weights | default encoder for real slides | Apache-2.0 (per MitoEd's LICENCES.md and licences.py) | open |
| XGBoost | optional head kind (`[xgboost]`) | Apache-2.0 | open |
| openslide-python / OpenSlide | optional WSI reader (`[slides]`), also reads DICOM slides from IDC | LGPL-2.1 | check with IG before deployment |
| wsidicom, pydicom | DICOM slides OpenSlide cannot read (JPEG 2000, YBR_FULL_422; about 650 TCGA series on IDC) (`[slides]`) | Apache-2.0, MIT (package metadata, checked 2026-11-11) | open |
| Pillow | antialiased tile resizing (`[slides]`) | MIT-CMU | open |
| SciPy | exact binomial bound for the accept threshold | BSD-3-Clause | open |
| idc-index, idc-index-data | triage manifest from the Imaging Data Commons (`[idc]`) | MIT (PyPI classifiers, checked 2026-11-10) | open |
| wsi/abmil.py | attention-MIL, our own NumPy code after Ilse et al. 2018 (arXiv 1802.04712) | MIT (this package) | open; no CLAM code |

Not used here: UNI, UNI2-h, Virchow2, H-optimus-1 (CC-BY-NC-ND) and Ultralytics (AGPL-3.0). Cells come in from
outside this pillar; if they come from Cellpose-SAM, note its weights were trained on CC-BY-NC data (see MitoEd).

Data: synthetic only in this repository (generated in `slides.py` and `triage/synthetic.py`). No patient data.

Triage training data (downloaded on Al's Mac or Eddie, never committed): IDC slides under CC BY 4.0
(GTEx, CCDI-MCI, CPTAC-GBM, HCMI, CGCI) and CC BY 3.0 (TCGA), as listed per slide in `license_short_name`.
Both need attribution in any publication or model card. ICDC-Glioma (canine) is excluded.

Step 4 (methylation class) uses only the packages above (scikit-learn random forest and k-means for labels).
Training data planned: TCGA diagnostic slides and 450K beta values (GDC open access); the GSE90496 reference set
(data-use terms not yet checked). The Heidelberg classifier software is not used.
