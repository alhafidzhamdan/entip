# ENTIP

Edinburgh Neuropathology Tissue Intelligence Platform. Research use only; not a clinical tool.

Five pillars around one patient record:

1. `fork/` - slide fork and link: clinical and research copies of each WSI, joined by a linkage code.
2. `training/` - live training backend for any feature, plus gated retraining of existing models.
3. `spatial/` - spatial multi-omics backbone registered to H&E.
4. `gui/` - clinician-facing viewer and review.
5. `db/` - central database (the ontology): one row per patient, linked to slides, assays and model outputs.

Specs and shared rules: [docs/pillar_specs.md](docs/pillar_specs.md). MitoEd (mitosis module) lives in its own repo.

No patient data in this repository. Every dependency must be openly licensed (see the shared rules).
