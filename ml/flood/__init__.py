"""Flood segmentation: dataset, model, losses, training, evaluation, inference.

Runs in the ML plane (ADR-001). The serving plane never imports this and never
sees torch -- inference happens in batch and writes artifacts that FastAPI then
reads.
"""

from __future__ import annotations
