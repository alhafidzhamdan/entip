# Step 3c: mitotic activity

[← step 3](step3-high-grade.md) · code `training/glioma/src/entip_glioma/highgrade/` (`--mitoed-cells`)
· training `training/runs/highgrade_mitosis/`

## What it decides

Are there at least *k* mitoses in the whole specimen: `present`, `absent` or `indeterminate`. Significant mitotic
activity makes an IDH-mutant astrocytoma grade 3, and brisk mitoses point to grade 3 in oligodendroglioma.

cIMPACT-NOW update 5 says no validated cut-off exists and that most neuropathologists use at least 2 mitoses in
the whole specimen, or 1 in a very small biopsy ([Brat et al. 2020](https://doi.org/10.1007/s00401-020-02127-9)).
Here `--min-mitoses` (default 2) and `--small-biopsy-mm2` (default off) are settings, not validated values.

## What counts

- **Mitosis:** a mitotic figure (prophase to telophase).
- **Look-alikes:** apoptotic and pyknotic nuclei, lymphocytes, karyorrhectic debris near necrosis, crush.

The unit is **one cell**, and the answer is a **count over the whole specimen**. That makes it a detection
problem: tile embeddings at 112 µm are the wrong tool.

## How

| | Now | Needed for commercial use |
|---|---|---|
| Candidates | MitoEd's candidate list per scan (`mitoed predict` CSV) | A candidate detector without non-commercial training data |
| Which count | Only candidates MitoEd calls `mitosis` or `uncertain` (or p ≥ 0.2 for tables without a call), so thousands of low-p mimics cannot add up | The same rule, retuned on glioma |
| Specimen count | P(at least *k* true mitoses), pooling scans of different blocks | The same |
| Labels | Candidates sampled at random within bands of p, labelled one cell at a time ([training module](../../training/runs/highgrade_mitosis/README.md)) | The same |
| Calibration | MitoEd's p recalibrated on glioma labels (`--mitosis-calibration`) | The same, for the new detector |

The same mitosis on two levels of one block is counted twice, so pool scans of different blocks only.

## Status and open items

- Synthetic only.
- MitoEd has not been run on CNS tumours: its training data (MIDOG++) has no brain.
- MitoEd's segmenter (Cellpose-SAM) was trained on CC BY-NC data, so a commercial count needs another one.
- Built: the labelling and recalibration module, tested on synthetic candidates. Not done: running MitoEd on the
  glioma slides.
