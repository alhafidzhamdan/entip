"""ENTIP pillar 3: spatial multi-omics backbone.

Loads spatial assays (Visium, Xenium, Slide-tags or any AnnData), registers
them to their H&E section through a stored Transform (error in micrometres),
joins per-spot tables, and exports per-cell labels as training data for the
live training backend (pillar 2). Research use only.
"""
__version__ = "0.1.0"
