"""ENTIP AI glioma workflow, steps 1 to 5: pre-training (frozen foundation-model tile bags), triage, high-grade
features, panel markers and methylation class. Step 6, the rules engine, is ``entip_dx`` (``training/dx/``).

Uses the slide readers, encoders and ontology of the live-training pillar (``entip_train``,
``platform/live-training/``). Research use only."""
__version__ = "0.1.0"
