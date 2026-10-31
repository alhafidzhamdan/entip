# Platform pillar specs (31 Oct 2026)

Map: https://claude.ai/artifact/HgDbFNVt6hqWWXiuDz1r6j (section "By your five pillars").
Platform plan: https://claude.ai/artifact/RGUwDZr1Yc8a1xaQV33uyn

## Shared rules (all pillars)
- Ontology objects: Patient (study_id, pseudonymised) > Specimen > Block > Section > Scan | Assay; plus Cell, Region, Transform, Model, Annotation.
- Positions in micrometres in a named frame. Any H&E-to-assay move goes through a stored Transform with its error in µm.
- Provenance on every object (person, or model + version). Changes are logged actions, not edits.
- Every dependency must pass Al's NHS open-licence rule: MIT / BSD / Apache preferred; NC, ND or AGPL only as flagged research options.
- Research use only. Synthetic or public data only; no patient data in any repo.
- Commits: author Al Hafidz Hamdan <alhafidz.hamdan@ed.ac.uk>, short one-line messages, no Claude co-author lines.
- Pillar 5 lands its schema first: it is the contract the others code against.

## P1 Slide fork and link
Scope: ingest a WSI; write a clinical copy and a pseudonymised research copy; mint study_id and a linkage key kept in a separate NHS-side store (CHI never in the research database).
Interfaces: creates Patient..Scan in P5.
First milestone: CLI that forks a synthetic slide and registers it in P5, key store kept apart.

## P2 Live training backend
Scope: per-scan embedding cache (H-optimus-0, Apache-2.0); tier 1 live heads per feature (cell / tile / region; XGBoost or scikit-learn; warm start; uncertainty queue |p-0.5| plus random share; background negatives for positives-only marks); tier 2 gated champion/challenger retrain (follow MitoEd's src/mitoed/train/retrain.py design).
Interfaces: reads Scan and Annotation from P5; writes Cell, Region, Model; HTTP API used by P4.
First milestone: train a head from ~20 clicks on a synthetic slide and return a ranked review queue.

## P3 Spatial multi-omics backbone
Scope: register Assays (Slide-tags, Visium, Xenium; AnnData) to their H&E Section through a Transform; export per-cell labels as training data for P2.
Interfaces: writes Assay, Transform and spot / nucleus objects to P5.
First milestone: load a public or synthetic AnnData, register it to an H&E with an affine transform, join a MitoEd-style per-spot table.

## P4 Clinician GUI
Scope: web slide viewer (OpenSeadragon, BSD), click-to-teach, uncertainty review queue, patient view. Any agent uses a local model only.
Interfaces: P5 read API; P2 training API.
First milestone: open a slide, click cells, watch the queue update.

## P5 Central database
Scope: PostgreSQL + PostGIS schema for the ontology, action log, one-row-per-patient view, API.
First milestone: migrations, a seeded synthetic patient, an API that returns the patient row with linked scans and assays.
