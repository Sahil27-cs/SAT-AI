"""Hazard modules: wildfire, extreme weather and post-event damage.

What these are, and what they are not
-------------------------------------
Everything here is a **deterministic computation over observed inputs**, not a
trained model. dNBR is arithmetic on two reflectance stacks; a rainfall anomaly
is a percentile against a climatology; wind exposure along a track is a
distance-decay over supplied best-track points. None of it is fitted, none of it
is a forecast, and every envelope these modules produce carries
:class:`~satai.provenance.SourceKind.DERIVED` or ``INDEX`` rather than ``MODEL``.

That matters more here than it looks. A burn-severity map and a flood
segmentation look alike on a screen and mean different things: one is a
published index with fixed thresholds, the other is a model output with a
confidence. Presenting them identically is the failure the provenance contract
exists to prevent, so the distinction is carried in the type system rather than
in a caption.

Why these modules exist at all while the flood U-Net does not
-------------------------------------------------------------
They need no training data, no GPU and no labels — only the input rasters. That
makes them the part of the hazard surface that can be honestly completed now,
and it is why the wildfire and extreme-weather panels can show real computed
values while the deep flood model is still marked NOT TRAINED.

Each module states its own validation status. ``dnbr`` uses published USGS
severity breaks and is unvalidated over Indian dry deciduous and chir pine
forest; that caveat rides on the envelope rather than sitting in a docstring
nobody reads.
"""

from __future__ import annotations

__all__: list[str] = []
