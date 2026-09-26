"""Explainability — **not implemented**.

This package is empty. It is kept as a named placeholder rather than deleted
because the explanation contract it will have to satisfy already exists and is
enforced elsewhere, and that contract is the hard part:

* ``GET /api/v1/explanation`` returns ``available: false`` with a stated reason
  until an artifact exists, and never a plausible-sounding narrative;
* ``satai.agents.tools.get_model_explanation`` requires a stored ``drivers``
  array with real per-feature values, and raises ``DataUnavailableError``
  rather than returning anything when there is none;
* every explanation envelope carries "attributions describe what the model
  used, not physical causation" as a caveat.

So the surfaces are honest today: the system says it has no explanation,
because it has none. What is missing is the producer — the modality-ablation
and SHAP passes that would write those artifacts in the batch plane, which
depend on the flood U-Net that is itself unwritten (experiment C2 in
``docs/evaluation.md``).

Anything added here must write to ``satai.explanations`` from a batch run, not
compute on request: attribution over a segmentation model is not serving-plane
work (ADR-001).
"""
