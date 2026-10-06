"""Ground-truth review: model-consensus drafts and the human correction app.

Ground truth for real panel photos is seeded by several models and corrected by a human:
``consensus`` merges the models' documents into one draft per scene with per-element
support, ``app`` serves the review page (tailnet only), and ``finalize`` turns a reviewed
draft into a strict ground-truth document with per-element provenance.
"""

from __future__ import annotations
