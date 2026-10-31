# gui: clinician-facing viewer and review (pillar 4)

A web app for looking at a patient's material and teaching the platform new features by clicking cells.
Research use only. Synthetic data only.

Two views over the same data, switched at the top of every page and remembered per user:

**Clinician view**: reading the output.
- **Cases**: one row per patient (pseudonymised study ID), diagnosis, key markers, model findings, review status.
- **Patient page**: a summary laid out like a WHO CNS5 integrated report (integrated diagnosis, grade,
  molecular information, methylation class), model findings with their source, slides, recent activity.
- **Slide page**: the finding as a number (e.g. mitotic figures per mm²), then a gallery of the model's
  detections cut from the slide. Agree (`1`) or disagree (`2`) with each; answers are saved as marks with the
  person's name and also teach the live head. Borderline cells just under the cut-off are one click away.

**Researcher view**: exploring and teaching.
- **Cohort**: sortable table, one row per patient, every molecular field and model value; copy or download CSV.
- **Patient page**: the full tissue tree (Specimen > Block > Section > Scan | Assay), all outputs and assays, full log.
- **Slide page**: layer toggles (model calls, marks, cell outlines, regions, queue); pick which model's calls to
  show (the live head or an existing model such as MitoEd) and move the threshold while a histogram and the
  count per mm² update; teach any feature with the review queue; export the per-cell table as CSV.

Layout ideas (candidate gallery, probability histogram with a threshold, layer list) were taken from looking
at TissueLab (github.com/zhihuanglab/TissueLab). No code, assets or text were copied: its Penn Academic licence
does not allow reuse.

Shared by both: an OpenSeadragon viewer with a µm scale bar and the cursor position in frame `scan:<scan_id>`.
In the researcher queue, keys are `1` yes, `2` no, `3`/`J` next, `K` previous, `M`/`Y`/`N` click modes.

Every mark is an appended action with the person's name and time; retractions are actions too.

## Run it

```bash
pip install numpy pillow                 # for the synthetic slides only
python gui/tools/make_mock_slides.py     # about 2-3 minutes; writes gui/web/mock/slides/ (not in git)
python gui/tools/make_mock_patients.py   # writes gui/web/mock/patients.json
python -m http.server 8000 -d gui/web    # then open http://localhost:8000
```

No build step and no external network calls: OpenSeadragon is vendored and fonts are system fonts, so it
runs on a closed NHS network.

## Mock and live modes

All data access is in `web/js/api.js`; `web/config.js` picks the mode.

- `mock` (default): five invented glioma cases. Cells carry synthetic 12-number embeddings that stand in
  for H-optimus-0 embeddings, and `web/js/head.js` runs a small logistic-regression head in the browser
  (warm start, background negatives when only positives are marked). Marks are kept in this browser only.
- `http`: the training backend in `training/` (routes in `training/README.md`: `/heads`,
  `/heads/{id}/labels`, `/queue`, `/predictions`) and the central database. Serve the GUI and set
  `trainBase` to the training service, started with `--cors <gui origin>`.
  The database routes below are provisional until `db/` lands. The `http` mode has not been run end to end yet.

| Route (database, provisional) | Returns |
|---|---|
| `GET /patients` | one row per patient: `study_id, sex, age_band, diagnosis, who_grade, review_status, molecular{}, n_scans, n_assays` |
| `GET /patients/{study_id}` | the row plus `specimens[] > blocks[] > sections[] > scans[] / assays[]` and `model_outputs[]` |
| `GET /scans/{scan_id}` | `scan_id, stain, mpp, objective, tile_source (DZI), study_id, diagnosis` |
| `GET /scans/{scan_id}/cells` | `{frame, mpp, cells: [{cell_id, x_um, y_um, r_um}], regions: [{kind, polygon_um}], source}` |
| `GET /scans/{scan_id}/outputs` | `[{model, version, feature, threshold, predictions: [{cell_id, p}]}]` per-cell outputs of existing models |
| `GET /actions?study_id=&scan_id=` / `POST /actions` | the append-only action log |

## Files

| Path | What |
|---|---|
| `web/index.html`, `web/css/app.css` | page and styles (light and dark) |
| `web/js/app.js` | worklist, patient page, slide workbench |
| `web/js/viewer.js` | OpenSeadragon plus a canvas overlay for cells, marks, model calls and the queue |
| `web/js/api.js` | mock and HTTP adapters |
| `web/js/head.js` | mock live head (mirrors training tier 1) |
| `tools/make_mock_slides.py` | synthetic H&E-like slides (Beer-Lambert stain model), DZI tiles, cell tables in µm, mock MitoEd detections (`mitoed.json`) |
| `tools/make_mock_patients.py` | synthetic patient records |
| `tests/` | `node --test gui/tests/*.test.js` and `python -m pytest gui/tests` |

## Licences

| Component | Licence |
|---|---|
| OpenSeadragon 5.0.1 (vendored, `web/vendor/openseadragon/`) | BSD-3-Clause |
| NumPy, Pillow (mock data tools only) | BSD-3-Clause, MIT-CMU (HPND) |

No other runtime dependencies. All mock data is generated; there is no patient data here.

## Not done yet

- Database routes: switch to the `db/` API when it lands.
- Tile and region features (only cell-level marking so far), and polygon drawing for regions.
- Sign-in: the name on marks is typed by the user; NHS single sign-on comes later.
- Assay overlays from `spatial/` (spots registered to the H&E) in the viewer.
- Retrain button for the gated champion/challenger job (`POST /heads/{id}/retrain`).
