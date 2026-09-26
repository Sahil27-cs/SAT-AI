"""The grounding validator — the instrument behind contribution C1.

Phase 0 established the gap this fills. Shang et al. (2026, IEEE JSTARS §VI.D)
state that for remote-sensing AI agents, *"reasoning-consistency and
tool-invocation accuracy must be integrated into benchmarks to ensure logical
reliability"*, and that opaque reasoning limits adoption *"in high-risk
applications like territorial planning and emergency management"*. None of the
corpus papers that put a language model over Earth-observation outputs
(Aziz et al. 2025; Swain et al. 2025; Kundu et al. 2025) measures whether the
generated text is faithful to those outputs.

This module is the measurement.

**How it works.** Every value the serving plane returns is wrapped in a
:class:`~satai.provenance.ProvenanceEnvelope`. The union of
``groundable_values()`` across a turn's envelopes is the set of numbers the
response is permitted to state. After generation, every number is extracted
from the text and checked against that set. An ungrounded number is a
violation: logged, counted, and -- on the first occurrence -- fed back for one
regeneration attempt.

**Why deterministic extraction rather than a second model.** Using an LLM to
check an LLM makes the checker's reliability load-bearing and unmeasured.
Regex-based numeric extraction is exact, cheap, and its own failure modes
(missing a number, or flagging one that was legitimately derived) are
enumerable and tested below.

**What it deliberately does not do.** It does not judge whether the reasoning
is *good*, only whether the quantities are *traceable*. A response can be
perfectly grounded and still poorly argued. That is a different measurement and
this module does not pretend to make it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from satai.logging import get_logger
from satai.provenance import ProvenanceEnvelope

log = get_logger(__name__)

__all__ = [
    "GroundingReport",
    "GroundingValidator",
    "NumericClaim",
    "ViolationKind",
    "extract_numbers",
]

#: Numbers with optional thousands separators, decimals, percentages and signs.
_NUMBER_RE = re.compile(
    r"""(?<![\w.])            # not mid-identifier
    (?P<value>
        [-+]?
        (?:\d{1,3}(?:,\d{3})+ | \d+)   # 1,234,567 or 1234567
        (?:\.\d+)?                      # optional decimal
    )
    \s*(?P<suffix>%|percent|km2|km²|sq\s?km|km|m|dB|hours?|days?|hrs?)?
    (?![\w.]\d)""",
    re.VERBOSE | re.IGNORECASE,
)

#: Numbers that carry no factual claim. Years, list markers, small counts used
#: rhetorically ("three factors"), and the ordinals in "Sentinel-1"/"Sentinel-2".
#: Without this the violation rate is dominated by trivia and measures nothing.
_EXEMPT_EXACT = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0})
_YEAR_RANGE = (1900.0, 2100.0)

#: Contexts in which a number is structural rather than a claim about the world.
_STRUCTURAL_CONTEXT = re.compile(
    r"(sentinel[\s-]?|landsat[\s-]?|modis[\s-]?|viirs[\s-]?|noaa[\s-]?|"
    r"era|gpm|epsg:?\s?|adr[\s-]?|phase\s|step\s|band\s|figure\s|table\s|"
    r"^\s*\d+[.)]\s)",
    re.IGNORECASE,
)


class ViolationKind(StrEnum):
    """Why a response failed grounding. Counted separately in the C1 report."""

    UNGROUNDED_VALUE = "ungrounded_value"
    """A number appears that no tool returned."""

    OBSERVATION_PREDICTION_CONFUSION = "observation_prediction_confusion"
    """An observation described as a prediction, or the reverse. Scientifically
    the most serious class: it misrepresents what the system can do."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    """A categorical assertion (a warning, an evacuation instruction, an
    official status) with no tool output behind it."""

    MISSING_CAVEAT = "missing_caveat"
    """An envelope carried a caveat that the response dropped."""

    FABRICATED_AUTHORITY = "fabricated_authority"
    """An emergency number, agency instruction or official warning invented
    outright. Always a hard failure, never regenerated."""


@dataclass(frozen=True)
class NumericClaim:
    """One number found in a generated response."""

    value: float
    raw: str
    suffix: str | None
    position: int
    context: str

    @property
    def is_exempt(self) -> bool:
        """True for numbers that carry no factual claim about the world."""
        if _YEAR_RANGE[0] <= self.value <= _YEAR_RANGE[1] and self.value.is_integer():
            return True
        if self.value in _EXEMPT_EXACT and self.suffix is None:
            return True
        return bool(_STRUCTURAL_CONTEXT.search(self.context))


@dataclass
class GroundingReport:
    """The outcome of validating one response."""

    grounded: bool
    claims: list[NumericClaim] = field(default_factory=list)
    ungrounded: list[NumericClaim] = field(default_factory=list)
    violations: list[tuple[ViolationKind, str]] = field(default_factory=list)
    allowed_values: list[float] = field(default_factory=list)
    n_checked: int = 0

    @property
    def violation_kinds(self) -> set[ViolationKind]:
        return {kind for kind, _ in self.violations}

    @property
    def is_hard_failure(self) -> bool:
        """Fabricated authority is never regenerated -- it is returned as a refusal."""
        return ViolationKind.FABRICATED_AUTHORITY in self.violation_kinds

    def explain(self) -> str:
        if self.grounded:
            return f"grounded: {self.n_checked} numeric claims all traceable to tool output"
        lines = [f"NOT grounded: {len(self.violations)} violation(s)"]
        lines += [f"  [{kind}] {detail}" for kind, detail in self.violations]
        return "\n".join(lines)

    def feedback_for_regeneration(self) -> str:
        """Message handed back to the model for its one retry."""
        parts = ["Your previous response contained values not present in the tool results."]
        if self.ungrounded:
            listed = ", ".join(f"{c.raw}" for c in self.ungrounded[:8])
            parts.append(f"These numbers were not returned by any tool: {listed}.")
        permitted = ", ".join(f"{v:g}" for v in sorted(set(self.allowed_values))[:30])
        parts += [
            f"Permitted values: {permitted}.",
            "Rewrite using only those values. If the tools did not return a "
            "quantity the question asks for, say so plainly instead of estimating it.",
        ]
        return " ".join(parts)


def extract_numbers(text: str, *, context_chars: int = 48) -> list[NumericClaim]:
    """Find every numeric claim in generated text, with surrounding context."""
    claims: list[NumericClaim] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("value")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
        start = max(0, match.start() - context_chars)
        claims.append(
            NumericClaim(
                value=value,
                raw=match.group(0).strip(),
                suffix=(match.group("suffix") or "").strip().lower() or None,
                position=match.start(),
                context=text[start : match.end() + context_chars],
            )
        )
    return claims


class GroundingValidator:
    """Checks a generated response against the turn's provenance envelopes."""

    #: Phrases that assert official status. The system is a research prototype;
    #: it has no authority to issue any of these, so they are hard failures.
    FABRICATED_AUTHORITY_PATTERNS = (
        r"\bevacuat(e|ion)\s+(order|notice|is\s+(?:now\s+)?(?:in\s+effect|mandatory))",
        r"\bofficial\s+(warning|alert|advisory)\s+(has been|is)\s+issued",
        # India's emergency numbers are short -- 112, 1070, 1078 -- so a
        # minimum run length long enough for a mobile number would miss exactly
        # the ones a model is most likely to invent. Three digits is the bar,
        # and the lookahead stops "call 2 districts" from matching.
        r"\b(call|dial|contact)\s+(?:on\s+)?(?=[-+()\s]*\d{3})[-+()\d\s]{3,}",
        r"\b(IMD|NDMA|SDMA|NDRF)\s+has\s+(issued|declared|ordered)",
        r"\byou\s+(must|should)\s+evacuate\b",
    )

    #: An observation reported as a prediction, or a prediction as an
    #: observation. Both misrepresent what the system did.
    CONFUSION_PATTERNS = (
        (
            r"\b(predict\w*|forecast\w*|will\s+occur|expected\s+to\s+(?:flood|burn))\b",
            "observation",
        ),
        (r"\b(observed|detected|measured|recorded)\b", "prediction"),
    )

    def __init__(self, *, tolerance: float = 0.02) -> None:
        #: Relative tolerance. A response that rounds 0.8734 to "0.87" is
        #: grounded -- rounding is presentation, not invention. Too tight and
        #: the metric measures formatting; too loose and real fabrication slips
        #: through. 2 % admits sensible rounding and little else.
        self.tolerance = tolerance

    def collect_allowed(self, envelopes: list[ProvenanceEnvelope[Any]]) -> list[float]:
        """Every value the response is permitted to state, plus safe derivations.

        Percentages of a permitted probability are included: a model saying
        "87 %" for a returned 0.87 is faithful, and counting that as a
        violation would make the metric measure formatting rather than
        fidelity.
        """
        allowed: list[float] = []
        for env in envelopes:
            values = env.groundable_values()
            allowed.extend(values)
            allowed.extend(v * 100.0 for v in values if 0.0 <= v <= 1.0)
            allowed.extend(round(v, 2) for v in values)
            allowed.extend(round(v * 100.0, 1) for v in values if 0.0 <= v <= 1.0)
        return allowed

    def _is_grounded_value(self, value: float, allowed: list[float]) -> bool:
        for permitted in allowed:
            if permitted == 0.0:
                if abs(value) < 1e-9:
                    return True
                continue
            if abs(value - permitted) <= self.tolerance * abs(permitted):
                return True
        return False

    def validate(
        self,
        response: str,
        envelopes: list[ProvenanceEnvelope[Any]],
        *,
        check_caveats: bool = True,
    ) -> GroundingReport:
        """Validate a generated response. This is the C1 measurement."""
        allowed = self.collect_allowed(envelopes)
        claims = extract_numbers(response)
        checkable = [c for c in claims if not c.is_exempt]

        ungrounded = [c for c in checkable if not self._is_grounded_value(c.value, allowed)]
        violations: list[tuple[ViolationKind, str]] = [
            (
                ViolationKind.UNGROUNDED_VALUE,
                f"'{c.raw}' at position {c.position} is not traceable to any tool result",
            )
            for c in ungrounded
        ]

        for pattern in self.FABRICATED_AUTHORITY_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                violations.append(
                    (
                        ViolationKind.FABRICATED_AUTHORITY,
                        f"asserts official authority the system does not have: '{match.group(0)}'",
                    )
                )

        violations.extend(self._check_observation_prediction(response, envelopes))

        if check_caveats:
            violations.extend(self._check_caveats(response, envelopes))

        report = GroundingReport(
            grounded=not violations,
            claims=claims,
            ungrounded=ungrounded,
            violations=violations,
            allowed_values=allowed,
            n_checked=len(checkable),
        )

        if not report.grounded:
            log.warning(
                "grounding violation",
                extra={
                    "n_violations": len(violations),
                    "kinds": ",".join(sorted(report.violation_kinds)),
                },
            )
        return report

    #: Negations preceding a term. "not predictions" is the system stating the
    #: distinction CORRECTLY, so flagging it would penalise exactly the
    #: behaviour the benchmark rewards -- and would bias C1 against good
    #: responses. Found by experiment 9a (validator validation), which flagged
    #: a correct FIRMS answer as confused because it contained the word
    #: "predictions" inside the phrase "not predictions".
    _NEGATION_RE = re.compile(
        r"\b(not|never|isn'?t|aren'?t|no|rather\s+than|instead\s+of|"
        r"cannot|can'?t|does\s+not|doesn'?t|without)\b",
        re.IGNORECASE,
    )
    _SENTENCE_BREAK_RE = re.compile(r"[.!?;]\s|\n")

    def _is_negated(self, text: str, start: int) -> bool:
        """Whether the term at ``start`` sits inside a negation.

        The scope is the sentence, not a fixed character window. A window long
        enough for "not predictions" is too short for "not predictions of where
        a fire will occur", where the trailing "will occur" is the term that
        matches -- and that is an ordinary way to write the sentence, so a
        window would produce false positives on well-formed answers. The
        negation and the term it governs are in the same sentence either way.
        """
        breaks = [m.end() for m in self._SENTENCE_BREAK_RE.finditer(text[:start])]
        sentence_start = breaks[-1] if breaks else 0
        return bool(self._NEGATION_RE.search(text[sentence_start:start]))

    def _check_observation_prediction(
        self, response: str, envelopes: list[ProvenanceEnvelope[Any]]
    ) -> list[tuple[ViolationKind, str]]:
        """Catch an observation described as a prediction, and the reverse.

        The distinction SAT-AI is built around: a FIRMS thermal anomaly is a
        measurement at overpass time; a fire-danger score is a model output.
        Presenting either as the other misstates what the system can do.

        Negated occurrences are skipped. A response saying "these are
        observations, **not** predictions" is drawing the distinction, not
        confusing it.
        """
        has_observation = any(e.is_observation for e in envelopes)
        has_model = any(e.is_model_output for e in envelopes)
        out: list[tuple[ViolationKind, str]] = []

        for pattern, misrepresents in self.CONFUSION_PATTERNS:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                if self._is_negated(response, match.start()):
                    continue
                if misrepresents == "observation" and has_observation and not has_model:
                    out.append(
                        (
                            ViolationKind.OBSERVATION_PREDICTION_CONFUSION,
                            f"'{match.group(0)}' describes an observation as a prediction; "
                            f"the tools returned only measurements",
                        )
                    )
                    break
                if misrepresents == "prediction" and has_model and not has_observation:
                    out.append(
                        (
                            ViolationKind.OBSERVATION_PREDICTION_CONFUSION,
                            f"'{match.group(0)}' describes a model output as an observation; "
                            f"the tools returned only model results",
                        )
                    )
                    break
        return out

    #: Critical caveat classes, each with the trigger that identifies it in an
    #: envelope and the phrases that count as carrying it through.
    #:
    #: Matching on *distinctive phrases* rather than on any shared keyword was
    #: forced by experiment 9a: the original check extracted keywords from the
    #: caveat text and passed if ANY appeared, so a response containing the word
    #: "flood" satisfied a caveat about official warnings. That made the check
    #: pass almost unconditionally and miss a planted violation.
    CRITICAL_CAVEATS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "official_warning",
            ("not an official warning", "not official warnings", "prototype risk level"),
            ("not an official warning", "not official", "prototype", "imd", "ndma"),
        ),
        (
            "observation_not_prediction",
            ("not predictions", "not a complete census"),
            ("observation", "observed", "overpass", "not a prediction", "not predictions"),
        ),
        (
            "proxy_vulnerability",
            ("proxy",),
            ("proxy", "does not incorporate", "omits"),
        ),
        (
            "not_real_time",
            ("revisit-limited", "not real-time"),
            ("revisit", "not real-time", "days old", "batch"),
        ),
    )

    def _check_caveats(
        self, response: str, envelopes: list[ProvenanceEnvelope[Any]]
    ) -> list[tuple[ViolationKind, str]]:
        """Require that a critical caveat survives into the response.

        Only the classes in :attr:`CRITICAL_CAVEATS` are enforced. Requiring
        every caveat verbatim would make the metric measure verbosity rather
        than fidelity.
        """
        lowered = response.lower()
        out: list[tuple[ViolationKind, str]] = []
        seen: set[str] = set()

        for env in envelopes:
            for caveat in env.caveats:
                caveat_lower = caveat.lower()
                for name, triggers, acknowledgements in self.CRITICAL_CAVEATS:
                    if name in seen:
                        continue
                    if not any(t in caveat_lower for t in triggers):
                        continue
                    seen.add(name)
                    if not any(a in lowered for a in acknowledgements):
                        out.append(
                            (
                                ViolationKind.MISSING_CAVEAT,
                                f"response omits the '{name}' caveat carried by the "
                                f"tool result: '{caveat[:70]}...'",
                            )
                        )
        return out
