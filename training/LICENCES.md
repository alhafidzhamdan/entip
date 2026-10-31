# Licences (training/)

Al's rule: every component must be openly licensed so it can be deployed in the NHS. Research use only.

| Component | Use | Licence | Status |
|---|---|---|---|
| this package | code | PolyForm Noncommercial 1.0.0 (repository licence, ../LICENSE.md) | set by Al |
| NumPy | arrays | BSD-3-Clause | open |
| scikit-learn | live heads (logistic regression, histogram gradient boosting), CV | BSD-3-Clause | open |
| joblib | saving heads | BSD-3-Clause | open |
| FastAPI, Uvicorn | HTTP API (optional `[api]`) | MIT, BSD-3-Clause | open |
| PyTorch, timm, huggingface_hub | foundation encoder (optional `[foundation]`) | BSD-3-Clause, Apache-2.0, Apache-2.0 | open |
| H-optimus-0 weights | default encoder for real slides | Apache-2.0 (per MitoEd's LICENCES.md and licences.py) | open |
| XGBoost | optional head kind (`[xgboost]`) | Apache-2.0 | open |
| Pillow | DZI tile reader (optional `[dzi]`) | MIT-CMU (HPND) | open |
| openslide-python / OpenSlide | optional WSI reader | LGPL-2.1 | check with IG before deployment |

Not used here: UNI, UNI2-h, Virchow2, H-optimus-1 (CC-BY-NC-ND) and Ultralytics (AGPL-3.0). Cells come in from
outside this pillar; if they come from Cellpose-SAM, note its weights were trained on CC-BY-NC data (see MitoEd).

Data: synthetic only (generated in `slides.py`). No patient data.
