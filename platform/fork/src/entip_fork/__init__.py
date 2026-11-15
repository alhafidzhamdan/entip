"""ENTIP pillar 1: slide fork and link.

Each whole-slide image (WSI) is forked into:
- a clinical copy, kept byte-for-byte on the NHS side, and
- a pseudonymised research copy (pyramid only, label and macro images dropped,
  metadata rebuilt from an allow-list).

Both copies carry the same random linkage code. The table that joins the linkage
code back to CHI and accession lives only in the NHS-side key store.

Research use only. Synthetic or public data only.
"""

__version__ = "0.1.0"
