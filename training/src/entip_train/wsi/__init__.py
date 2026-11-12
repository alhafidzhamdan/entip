"""Shared slide-level pieces for the glioma workflow models (steps 1-4).

* ``bags``: tissue tiles at a fixed physical size, embedded once with a frozen encoder, saved per slide.
* ``abmil``: our own gated attention-MIL classifier (NumPy only), trained on those bags.
* ``calibrate``: temperature scaling, ECE and confidence thresholds that decide "accept" vs "order the test".
"""
