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
- `http`: the central database (`db/`, contract in `db/schema/README.md`) and the training backend
  (`training/`). Open the GUI with `?api=http` added to the address, or set `mode` in `web/config.js`.
  Marks are written to the database first, as annotations with the person as provenance (retractions use
  verb `retract`), then sent to the training service for the live head. If the training service does not know
  the slide yet, the mark is still saved, the GUI says so, and the review queue comes from the database's
  uncertainty order for an existing model (`/predictions?order=uncertain`).

To try `http` mode with the mock cases:

```bash
# database (see db/README.md): PostgreSQL + PostGIS, then
entip-db migrate && entip-db serve                       # :8705
python gui/tools/load_mock_into_db.py --db http://127.0.0.1:8705
entip-train serve --encoder handcrafted --cors http://localhost:8000   # optional, :8100
python -m http.server 8000 -d gui/web                    # open http://localhost:8000/?api=http
```

Tested this way on 31 Oct 2026: patient list, patient tree, slide with cells, regions and MitoEd predictions
from the database, marks saved as annotations and shown in the patient's log. Live training on these slides
waits for the training service to read scans and cells from the database (its open item).

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
| `tools/load_mock_into_db.py` | load the mock cases into the database over its API, for `http` mode |
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
