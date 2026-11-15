# Distil H-optimus-0 into a small student

A student encoder (default ViT-Small, about 22M parameters against H-optimus-0's 1.1B) trained to reproduce
H-optimus-0's tile embeddings, so new slides can be embedded on a CPU, a Mac or a modest GPU. Code:
`training/glioma/src/entip_glioma/distill/`; commands `entip-glioma distill tiles | train | evaluate`.

It distils from a finished triage run (`training/runs/triage/`): that run has already downloaded the slides and
embedded every tissue tile with H-optimus-0, so the teacher does not run again.

| Job | Queue | What |
|---|---|---|
| `1_tiles.sh` | CPU, 4 cores | Cut each teacher bag's tiles from its slide into a cache (up to 1,000 tiles per slide, 224 px; about 38 GB for 250 slides), next to the teacher's numbers |
| `2_train.sh` | GPU | Split by patient (20% held out per class), train the student on the teacher's numbers (MSE + cosine loss), keep the best by held-out cosine. Resumes from `last.pt` if cut short |
| `3_check.sh` | GPU | Embedding agreement on held-out patients; re-embed every slide with the student; retrain triage on the student's bags with the same plan; print both triage reports side by side (`comparison.txt`) |

```bash
cd /exports/igmm/eddie/Glioblastoma-WGS/scripts/entip && git pull
RUN=diverse GPU_REQUEST="-q gpu -l gpu=1 -l h200=true -pe sharedmem 4 -l h_rss=16G" bash training/runs/distill/eddie/submit.sh
```

`submit.sh` refuses to start until the triage run has finished (its `report-h-optimus-0.json` exists), and caches
the student's ImageNet start weights on the login node, since compute nodes have no internet. Outputs go to
`$BASE/runs/distill-<RUN>/`: `tiles/`, `student/student.pt` (89 MB) and `report.json`, `evaluation.json`,
`bags-student/`, `model-student/`, `report-student.json`, `comparison.txt`, `logs/`. Settings
(`STUDENT_ARCH`, `EPOCHS`, `MAX_TILES`, `DWORK`) are in `eddie/config.sh`.

**Using the student.** Any embedding command takes it as an encoder:
`entip-train embed extract slide.svs --out bags/ --encoder student:/path/to/student.pt`. Its bags carry the
student's own version string, so they are never mixed with H-optimus-0 bags: models for each step are retrained on
student bags (as `3_check.sh` does for triage) and must be revalidated before use.

**Reading the result.**

- *Embedding agreement* (tile cosine with the teacher on held-out patients) says how closely the student copies
  the teacher. Look at the 5th percentile and the per-class numbers too: a good average can hide a class it copies
  badly.
- *Triage side by side* is what decides it: if balanced accuracy, calibration and the gate are close to the
  teacher's, the student can replace H-optimus-0 for triage. The student has seen these slides' tiles (not their
  labels) in training, so this comparison flatters it a little; the real test is new slides, ideally Edinburgh's.
- Steps 3 to 5 depend on finer detail (mitoses, small necrotic foci) and need their own check before the student
  replaces the teacher there.

**Licences.** The student is our own model, trained on H-optimus-0 outputs (Apache-2.0). The default start weights
are timm's `vit_small_patch16_224` (AugReg ImageNet-21k/1k, Apache-2.0); `--init scratch` avoids them.
