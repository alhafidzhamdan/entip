<div align="center">
<img src="docs/assets/banner.svg" alt="ENTIP: Edinburgh Neuropathology Tissue Intelligence Platform" width="100%">
</div>

## AI models

**AI glioma workflow.** Six steps from H&E slides to a draft WHO CNS5 integrated diagnosis for a pathologist to sign off
([docs](docs/glioma-workflow/README.md)).

1. **Pre-training** (`training/glioma/.../embed/`): frozen H-optimus-0 tile embeddings, once per slide.
2. **Triage** (`training/glioma/.../triage/`): tumour? glial? enough tissue?
3. **High-grade features** (`training/glioma/.../highgrade/`): necrosis, microvascular proliferation, mitoses.
4. **Panel markers** (`training/glioma/.../panel/`): IDH, 1p/19q, ATRX, TP53, CDKN2A/B, EGFR and more.
5. **Methylation class** (`training/glioma/.../methylation/`): CNS methylation class from H&E.
6. **Integrated diagnosis** (`training/dx/`): WHO CNS5 rules engine and draft report.

## Platform

1. **Slide fork and link** (`platform/fork/`): clinical and research copies of each WSI, joined by a linkage code.
2. **Live training** (`platform/live-training/`): train any feature live, and retrain existing models.
3. **Spatial multi-omics** (`platform/spatial/`): spatial genomics registered to H&E.
4. **Clinician GUI** (`platform/gui/`): viewer and review for clinicians.
5. **Central database** (`platform/db/`): one row per patient, linked to slides, assays and model outputs.

Data sources, model weights and their licences: [docs/data-sources.md](docs/data-sources.md).
How the training and test sets compare with published classifiers: [docs/literature-comparison.md](docs/literature-comparison.md).

<sub>Research use only.</sub>
