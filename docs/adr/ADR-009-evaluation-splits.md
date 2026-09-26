# ADR-009: Build leave-one-region-out splits; do not use Sen1Floods11's official ones

- **Status:** Accepted
- **Date:** 2026-09-22
- **Phase:** 2
- **Supersedes:** the evaluation protocol asserted in Phase 1 (README, ADR-007,
  `research-gap.md`), which described Sen1Floods11's official splits as
  region-disjoint. **That was wrong.**

## Context

Phase 1 and the Phase 0 gap analysis both stated that Sen1Floods11 ships
"region-disjoint train/val/test splits", and built a research claim on it: that
SAT-AI's spatial-generalisation result would be honest where the corpus
literature's was not.

`scripts/verify_sen1floods11.py`, run 2026-09-22, shows the claim is false.

| Event region | chips | train | valid | test |
|---|---:|---:|---:|---:|
| Bolivia | 15 | 0 | 0 | 0 |
| Ghana | 53 | 31 | 11 | 11 |
| **India** | **68** | **40** | **14** | **14** |
| Mekong | 30 | 18 | 6 | 6 |
| Nigeria | 18 | 10 | 4 | 4 |
| Pakistan | 28 | 16 | 6 | 6 |
| Paraguay | 67 | 39 | 14 | 14 |
| Somalia | 26 | 15 | 5 | 6 |
| Spain | 30 | 18 | 6 | 6 |
| Sri-Lanka | 42 | 24 | 9 | 9 |
| USA | 69 | 41 | 14 | 14 |

Every region except Bolivia appears in **all three** splits, at a consistent
~60/20/20. These are **chip-level random splits, stratified within region** —
not disjoint across region.

That matters because 512×512 chips at 10 m are ~5 km across, and chips from one
flood event come from the same acquisition, the same orbit, the same weather and
often adjacent ground. Training on some and testing on the rest measures
interpolation within an event, not generalisation to a new one. Spatial
autocorrelation leaks straight across the boundary, and reported IoU is
inflated by an amount nobody in the corpus quantifies.

Bolivia (15 chips, held out of all three files and distributed separately) is
the one genuinely region-disjoint evaluation the dataset ships with.

## Decision

**SAT-AI builds its own leave-one-region-out (LORO) splits** and reports both
protocols side by side.

1. **Primary protocol — LORO.** For each of the 11 regions: train on the other
   10, validate on a held-out region from the training set, test on the excluded
   region. Report per-region IoU/Dice and the distribution across folds. The
   spread between folds *is* a result: it measures how much performance depends
   on which flood you happened to train on.
2. **Secondary protocol — official splits.** Also train and evaluate exactly as
   the dataset intends, so SAT-AI's numbers can be placed next to published
   Sen1Floods11 figures.
3. **Report the gap between them.** `IoU_official − IoU_LORO` quantifies the
   inflation that the standard protocol carries on this dataset. As far as the
   Phase 0 corpus shows, nobody reports this. It is cheap to compute and it
   makes the comparability caveat concrete rather than rhetorical.
4. **India is a LORO fold, not a training region.** Under LORO, the India fold
   trains on the other ten regions and tests on India's 68 chips. That is
   exactly the domain-transfer measurement C3 needs, and it means the Indian
   labels are never both trained on and tested on.
5. **Bolivia stays a final untouched hold-out.** Used once, at the end, as a
   sanity check that no protocol decision quietly leaked.

## Alternatives considered

**Use the official splits and note the caveat in prose.** Cheapest, and what the
literature does. Rejected: the whole stated contribution is *measurement*. Using
a leaky protocol while criticising the field for not measuring leakage would be
indefensible, and an examiner who knows the dataset would find it immediately.

**Use only the Bolivia hold-out as the region-disjoint test.** Genuinely
disjoint, and what the dataset authors intended for that purpose. Rejected as
the sole protocol: 15 chips is far too small to support a confidence interval,
and one region cannot show the *variance* across regions, which is the
interesting quantity.

**Geographic block cross-validation within regions.** Rejected as unnecessary
here — region identity is already the natural blocking factor, and blocks within
a single flood event would still share weather and acquisition geometry.

**Re-annotate to enlarge the Indian sample.** Out of scope, and the project has
no capacity for expert flood annotation.

## Consequences

**Buys:** an evaluation protocol that is stronger than the primary corpus's, a
per-region variance estimate that comes free with LORO, a domain-transfer fold
(India) that C3 needs anyway, and a reportable number — the official-vs-LORO gap
— that did not exist before.

**Costs:** 11 training runs instead of one. Mitigated by the dataset's size: 446
hand-labelled chips is small, so a single fold trains in minutes on a Kaggle or
Colab GPU. Eleven folds is an afternoon, not a week.

**Expect worse headline numbers.** LORO IoU will be materially below published
Sen1Floods11 figures. That is the point, it must not be tuned away, and the
results chapter must say so before it says anything else.

**Also expect a small-data problem.** 446 chips total, ~400 per LORO fold, is
few for training a segmentation network from scratch. This raises the priority
of a pretrained encoder (ImageNet ResNet-34 at minimum) and makes the
Prithvi-EO foundation-model comparison a more valuable ablation than it looked
in Phase 1.

## Follow-up

- Correct every document that repeated the wrong claim: `README.md`,
  `docs/research-gap.md` §3/§4, `docs/limitations.md`, ADR-007.
- `scripts/verify_sen1floods11.py` now *derives* the split structure from the
  data and reports it, so this cannot be assumed again.
- Phase 4 implements the LORO splitter before the first baseline is trained.
