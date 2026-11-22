# Licences (platform/live-training/)

Al's rule: every component must be openly licensed so it can be deployed in the NHS. Research use only.

| Component | Use | Licence | Status |
|---|---|---|---|
| this package | code | PolyForm Noncommercial 1.0.0 (see ../../LICENSE.md) | ours; non-commercial use, commercial use needs permission |
| NumPy | arrays | BSD-3-Clause | open |
| scikit-learn | live heads (logistic regression, histogram gradient boosting), CV | BSD-3-Clause | open |
| joblib | saving heads | BSD-3-Clause | open |
| FastAPI, Uvicorn | HTTP API (optional `[api]`) | MIT, BSD-3-Clause | open |
| PyTorch, timm, huggingface_hub | foundation encoder (optional `[foundation]`) | BSD-3-Clause, Apache-2.0, Apache-2.0 | open |
| H-optimus-0 weights | default encoder for real slides | Apache-2.0 (Hugging Face model card) | open |
| XGBoost | optional head kind (`[xgboost]`) | Apache-2.0 | open |
| openslide-python / OpenSlide | optional WSI reader (`[slides]`), also reads DICOM slides from IDC | LGPL-2.1 | check with IG before deployment |
| wsidicom, pydicom | DICOM slides OpenSlide cannot read (JPEG 2000, YBR_FULL_422; about 650 TCGA series on IDC) (`[slides]`) | Apache-2.0, MIT (package metadata, checked 2026-11-11) | open |
| Pillow | antialiased tile resizing (`[slides]`) | MIT-CMU | open |

Not used here: UNI, UNI2-h, Virchow2, H-optimus-1 (CC-BY-NC-ND) and Ultralytics (AGPL-3.0). Cells come in from
outside this pillar; if they come from Cellpose-SAM, note its weights were trained on CC-BY-NC data.

Data: synthetic only in this repository (generated in `slides.py`). No patient data.
