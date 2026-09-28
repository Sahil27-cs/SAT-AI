"""Hazard analysis runners that need real acquired data (Phase 16).

The computations live in ``satai/hazards/``; these scripts are what feed them
real input and write the artifacts the serving plane reads. They are separate
because a computation should be testable without a network, and an acquisition
should be re-runnable without re-deriving the science.
"""
