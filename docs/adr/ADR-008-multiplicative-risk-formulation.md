# ADR-008: Multiplicative hazard–exposure–vulnerability risk

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 1

## Context

The system must turn hazard model outputs into a risk score a user can act on.
Two questions have to be answered: how hazard, exposure and vulnerability
combine into risk for one hazard, and how risks for different hazards combine
into a multi-hazard picture.

The common shortcut is to average normalised scores. It is simple, and it is
wrong in two separate ways.

**Averaging hazard with exposure** assigns substantial risk to an uninhabited
floodplain. A remote river meander that floods every year has high hazard and
close to zero risk, because there is nothing there to be harmed. Risk is not
hazard.

**Averaging across hazards** — `(flood + wildfire + cyclone) / 3` — treats
scores that are not commensurable as if they were. A flood score of 0.7 and a
fire score of 0.7 do not denote the same thing; they are outputs of different
models trained on different labels with different base rates. Their mean denotes
nothing at all.

## Decision

**Within a hazard**, follow the Crichton risk-triangle / UNDRR–IPCC
decomposition, multiplicatively:

```
R_h = H_h^α · E^β · V^γ
```

Multiplicative because it enforces the correct boundary condition: zero exposure
implies zero risk. Defaults `α = β = γ = 1.0`, configurable in
`configs/risk.yaml`, and **every published risk map must be accompanied by the
one-at-a-time sensitivity analysis** defined in that file. That is what converts
an arbitrary modelling choice into an inspectable, reported result rather than a
hidden assumption.

**Across hazards**, three outputs and no average:

1. **Risk vector** — per-hazard scores kept separate. The primary output.
2. **Dominant hazard** — `argmax` plus the margin to the runner-up, for map
   colouring.
3. **Compound risk** — computed *only* for pairs with a documented physical
   mechanism, listed in `configs/couplings.yaml`. Currently two:
   cyclone → flood (extreme rainfall and surge) and burn → flood (reduced
   infiltration on burned slopes, lagged and decaying). Flood and wildfire in the
   same window are not coupled and receive no compound term.

`configs/risk.yaml` sets `forbid_naive_average: true`, and `tests/test_configs.py`
fails the build if it is ever flipped.

## On vulnerability

`V` is a **proxy**, built from building density, road accessibility and
elevation relative to drainage. Fine-resolution socioeconomic vulnerability data
for India is not openly available. The config declares `is_proxy: true` and
carries a caveat stating plainly that the index omits income, age structure,
disability and tenure, and therefore systematically under-represents the
vulnerability of marginalised populations. That caveat is attached to every
envelope the risk engine emits and rendered in the UI. A test enforces its
presence.

Overstating a proxy as a measurement would be the dishonest move available here,
and it is the one most likely to go unchallenged.

## Alternatives considered

**Weighted additive.** Retained as an *ablation*, not the default, so the
sensitivity analysis can report how much the choice of functional form matters.
Rejected as default for the zero-exposure reason above.

**Learned risk (train a model to predict observed losses).** Scientifically the
most satisfying option. Rejected: it needs a loss database at fine spatial
resolution for India, which does not openly exist. Without it, a "learned" risk
model would be fitting noise while looking rigorous.

**Copula-based compound hazard.** Statistically principled for joint extremes.
Rejected for v1: it needs long joint records the project does not have, and it
would be a project in itself. Noted as future work.

## Consequences

**Buys:** a formulation with a defensible boundary condition, an explicit
literature lineage, and configuration transparent enough that a reader can
disagree with the weights and recompute. The sensitivity analysis turns the
weakest part of the design into a reported result.

**Costs:** the multiplicative form is sensitive to normalisation, since a near-
zero component drives the product toward zero. Mitigated by a configurable floor
and by percentile-based normalisation clipped at the 2nd and 98th percentiles.
Missing components are flagged and excluded, never silently filled — a missing
vulnerability proxy must not be imputed as zero, which would zero the risk.
