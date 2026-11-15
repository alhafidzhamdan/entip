# spatial: pillar 3, spatial multi-omics backbone

Registers spatial assays (Visium, Xenium, Slide-tags or any AnnData) to their H&E section through a stored
Transform, summarises H&E cells under each spot, and turns paired spatial data into per-nucleus labels that
the live training backend (pillar 2) can learn from. Research use only. Synthetic data only in this repository.

## Try it

```bash
cd spatial
pip install -e '.[test]'
entip-spatial demo /tmp/entip-demo   # synthetic H&E + Xenium + Visium, every step, ~3 s
pytest -q
```

## Steps

1. **Load** (`loaders.py`). Each assay becomes an AnnData with `obsm["spatial_um"]` (micrometres, assay frame).
   - Visium: Space Ranger `outs/` (`filtered_feature_bc_matrix.h5`, `spatial/tissue_positions.csv` or
     `tissue_positions_list.csv`, `scalefactors_json.json`). Micrometres per full-res pixel =
     55 / `spot_diameter_fullres` (55 µm spots). Visium HD is not handled yet.
   - Xenium: `cell_feature_matrix.h5` + `cells.csv.gz` (or `cells.parquet`); centroids are already µm.
     Control probes are dropped.
   - Slide-tags / any `.h5ad`: positions in `obsm["spatial"]`; units must be given (`--um-per-unit`).
2. **Register** (`register.py`). Fit assay µm to H&E µm from ≥ 3 landmark pairs (affine, or similarity).
   Errors are stored in µm: fit RMSE, leave-one-out RMSE and the worst landmark. For single-cell assays the
   fit can be refined by trimmed ICP on mutual nearest neighbours between assay cells and H&E nuclei.
3. **Spot table** (`spots.py`). Per spot: H&E cells inside, counts per class (MitoEd `call`), max and mean of
   every `p_*` column, local mitosis density. Indexed by barcode, joined onto `adata.obs`.
4. **Labels for training** (`labels.py`), after the CytoFormer idea (rebuilt, no code copied):
   - single-cell assays: assay cells get a label from marker genes (or from an existing `obs` column);
     each H&E nucleus is matched to one assay cell (mutual nearest neighbour, gate = 3 × Transform error,
     clipped to 4 to 12 µm, ambiguity test) and inherits its label.
   - spot assays: nuclei inside a spot take the spot's dominant cell type when it is ≥ 80% of the spot
     (`--min-fraction`). These are weak labels; confidence = that fraction.
   - unmatched or ambiguous nuclei are left out, never called negative.
5. **Records** (`records.py`). Assay, Transform and SpatialObject records plus an append-only
   `actions.jsonl`, written as JSON Lines until the pillar 5 schema and API exist. `JsonlSink` is the only
   writer, so moving to the database means adding one sink.

## Inputs and outputs

| File | Columns |
|---|---|
| landmarks CSV | `assay_x_um, assay_y_um, he_x_um, he_y_um` (or `assay_x, assay_y` in native units) |
| H&E nuclei CSV | `cell_id, x_um, y_um` (+ `call`, `p_*` for MitoEd-style tables) in the scan frame |
| `transform.json` | matrix (3 × 3, assay → scan), `rmse_um`, `loo_rmse_um`, `max_err_um`, method, provenance |
| `training_labels.csv` | `cell_id, scan_id, x_um, y_um, label, confidence, label_source, assay_id, assay_barcode, match_dist_um, transform_id, transform_err_um` |

## Synthetic demo results (seed 0)

| | Xenium-like | Visium-like |
|---|---|---|
| Registration error vs truth | 0.04 µm (landmarks + ICP) | 5.1 µm (landmarks only; leave-one-out estimate 5.4 µm) |
| H&E nuclei labelled | 12,502 of 13,562 | 661 (only nuclei inside ≥ 80%-pure spots) |
| Label accuracy vs truth | 99.5% | 83% |

The synthetic tissue is easy by design (clean markers, same-section, mild noise). These numbers show the
code works; they say nothing about real tissue.

## Not done yet

- Real public data run (e.g. a 10x public Xenium or Visium dataset with its H&E) to check the loaders on
  real files; so far tested on synthetic files written to 10x's documented layout.
- Landmark picking in the GUI (pillar 4); image-based registration (no landmarks).
- Adjacent-section assays (Visium on a neighbouring section) need a non-rigid step; only affine here.
- Cell-type deconvolution of spots is expected as input (`--fractions`), not computed here.
- Swap `JsonlSink` for the pillar 5 API once the schema lands.

Dependencies and licences: [LICENCES.md](LICENCES.md).
