# triage-app: triage a new slide with the distilled student

A small local app. Pick a whole-slide image, and it gives the triage call with the reasons and a map of where
the model looked. It runs on a Mac or a laptop: the distilled student (ViT-Small, about 22M parameters) embeds
the tissue tiles on the Apple GPU (MPS), CUDA or the CPU, and the triage model trained on that student's bags
makes the call. Research use only: not for diagnosis.

What it shows for each slide:

- **The call as a WHO CNS5 category** (normal, gliomas and glioneuronal tumours, other primary CNS tumours,
  metastases), and whether it is a **confident call** or **for the pathologist to classify**. A confident call
  needs the calibrated confidence at or above the gate's threshold, and a slide that is not unlike the
  training slides. Otherwise the reasons say why it was held back.
- **Every class probability**, tissue adequacy (area of tissue tiles), and the novelty check.
- **Where the model looked**: the slide with an attention heat map (toggle and opacity), the 16 most attended
  tiles boxed and shown full size, and how concentrated attention is (the share on the top 1% of tiles; an
  even spread gives 1%). Click a tile or a box to enlarge it.

Results are kept in `~/entip-triage/`, one folder per slide (`result.json`, the thumbnail, heat map, top tiles
and the student bag), and listed under **Recent**, including after a restart.

## Set up (once)

1. **Copy the models from Eddie** (under 100 MB for the student; the triage model is small):

   ```bash
   mkdir -p ~/entip-models && scp mhamdan@eddie.ecdf.ed.ac.uk:/exports/igmm/eddie/Glioblastoma-WGS/entip/runs/distill-diverse/student/student.pt ~/entip-models/ && scp -r mhamdan@eddie.ecdf.ed.ac.uk:/exports/igmm/eddie/Glioblastoma-WGS/entip/runs/distill-diverse/model-student ~/entip-models/
   ```

2. **A Python environment** (Python 3.10 or later), from the repository root:

   ```bash
   python3 -m venv ~/entip-venv && ~/entip-venv/bin/pip install -e "platform/live-training[foundation,slides]" -e training/glioma
   ```

   `slides` brings OpenSlide (`openslide-bin` bundles its libraries, so no Homebrew step) and wsidicom for DICOM
   folders. Nothing is downloaded at run time: the student's weights are all in `student.pt`.

## Run

```bash
~/entip-venv/bin/python platform/triage-app/app.py --student ~/entip-models/student.pt --model ~/entip-models/model-student --slides ~/Downloads
```

Open <http://127.0.0.1:8765>. `--slides` is the folder the slide browser opens in; you can also paste any path.
Slide files OpenSlide reads (.svs, .ndpi, .tiff, .mrxs, .scn and others) and DICOM series folders both work.

The app checks that the triage model was trained on bags from this exact student, and refuses to start if not.
Tile size and the tissue cut-off (`--tile-um 112`, `--min-tissue 0.5`) must match the training bags; the
defaults do.

## What this model is, and is not

- Trained on the **diverse set**: 250 patients from public IDC collections, four classes (normal brain,
  glioma, other primary CNS, metastasis). Reactive tissue and lymphoma are not trained, so the model cannot
  call them: such slides get one of the four classes, ideally with low confidence or a novelty flag.
- "Metastasis" was learnt from **primary** lung, breast, colorectal, kidney and skin tumours, not tumour in
  the brain.
- Out-of-fold on those 250 patients: balanced accuracy 0.976, and the gate accepted 248 at threshold 0.682
  (at most 5% error among accepted calls, 95% confidence). That is cross-validation on the training sources,
  **not an external validation**: the DBTA and IPD-Brain tests are still to come.
- Known weakness: IDH-mutant lower-grade gliomas, especially oligodendrogliomas, called "other primary CNS".
- The saved model was trained on all 250 patients: a slide from that set has been seen, so its result there
  is optimistic.

## Privacy

The app listens on 127.0.0.1 only and sends nothing anywhere. Its results folder holds crops of the slides you
triage: for patient slides, keep it on a University device or in the group space, as for the slides
themselves.
