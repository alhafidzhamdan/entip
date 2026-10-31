# Licences: spatial (pillar 3)

Checked 31 Oct 2026 from installed package metadata. All runtime dependencies are permissive.

| Package | Licence | Role |
|---|---|---|
| numpy | BSD-3-Clause | arrays |
| scipy | BSD-3-Clause | sparse matrices, KD-trees |
| pandas | BSD-3-Clause | tables |
| anndata | BSD-3-Clause | assay container, `.h5ad` |
| h5py | BSD-3-Clause | 10x HDF5 matrices |
| pyarrow (optional) | Apache-2.0 | only for Xenium `cells.parquet` |
| pytest (test only) | MIT | tests |

Not used on purpose: CytoFormer (Penn Academic Software License: non-commercial, no redistribution);
the label-transfer idea is reimplemented from its description only. No scanpy / squidpy dependency
(both BSD-3, but not needed for this milestone).

Data: only synthetic data is generated and stored. When a real public dataset is used, record its
licence here before using it to train anything.
