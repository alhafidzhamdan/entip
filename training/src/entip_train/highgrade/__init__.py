"""Step 2 of the AI glioma workflow: high-grade feature detectors.

One detector per feature, each with its own model, confidence and fallback:

* ``necrosis``                    tile head on frozen embeddings -> connected regions -> slide call
* ``microvascular_proliferation`` same machinery, its own head and thresholds
* ``mitotic_activity``            reads MitoEd's per-candidate output and counts over the whole specimen

Every detector returns a :class:`~entip_train.highgrade.contract.FeatureResult`; ``actions.py`` turns the
results into ``POST /actions`` payloads for the central database. Research use only.
"""
from .contract import FeatureResult  # noqa: F401
