"""Distil H-optimus-0 into a small student encoder, so new slides can be embedded without a large GPU.

The teacher's work is already done: every tile bag from step 1 holds H-optimus-0's 1,536 numbers for each tissue
tile and where that tile is. Distillation pairs those numbers with the tile images and trains a small network
(default ViT-Small, about 22M parameters against H-optimus-0's 1.1B) to reproduce them.

1. ``tiles``   (CPU) cut each bag's tiles out of its slide once, at the stored positions, into a per-slide array
               next to the teacher's numbers (``tiles.py``). Reading tiles from the slide every epoch would be
               slow, especially JPEG 2000 DICOM slides.
2. ``train``   (GPU) split by patient, train the student on standardised teacher numbers (MSE + cosine loss),
               keep the best checkpoint by held-out cosine similarity, save one ``student.pt`` (``train.py``).
3. ``evaluate`` per-tile and per-slide agreement with the teacher on held-out patients (``train.py``).

Then embed slides with ``entip-train embed extract --encoder student:student/student.pt`` and retrain or re-check
each step on those bags. The student's bags carry its own encoder version, so they are never mixed with teacher
bags. What decides whether the student can replace the teacher is the downstream comparison (triage accuracy,
calibration and gate on the same patients), not the embedding agreement alone. Research use only.
"""
