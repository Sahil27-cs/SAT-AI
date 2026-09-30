"""C1: the 28-question grounding benchmark, against the live deployment.

    python -m ml.experiments.run_c1_benchmark

Every question goes to the production chat endpoint, which runs the refusal
rules, the router, Gemini with native tool calling and the grounding check.
Nothing is simulated: the responses scored here are the ones a user would see.

Two things the scoring has to handle honestly
----------------------------------------------

**Tool names.** The benchmark (``satai.agents.benchmark``, version 1.0.0) was
written against the original nine-tool agent design. The deployed agent exposes
five tools that cover the same ground with coarser names. Tool-invocation
accuracy is therefore scored under the mapping in ``TOOL_MAP``, fixed in this
file before the run, not adjusted afterwards. Where a benchmark tool has no
deployed equivalent the tool check is recorded as not applicable rather than
passed.

**Capacity failures.** Gemini sometimes answers 503 or 429. When a turn stays
degraded after retries, its reply is verified tool output with no model in the
loop, so scoring it would measure the fallback rather than the model. Such
turns are recorded as *unmeasured* with the reason, and every rate is computed
over measured turns only, with the denominator reported.

Two grounding rates are reported, because they answer different questions:

* ``draft_violation_rate`` -- how often Gemini's own draft contained something
  the validator rejected. This is the property of the model.
* ``shown_violation_rate`` -- how often a user was shown an ungrounded answer.
  A rejected draft is replaced by verified tool output, so this measures the
  system, and it is what C1 claims about SAT-AI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satai.agents.benchmark import (  # noqa: E402
    BENCHMARK_VERSION,
    QUESTIONS,
    BenchmarkQuestion,
    check_content,
    detected_refusal,
    score_tool_calls,
)
from satai.paths import relative_to_repo  # noqa: E402

API = "https://sat-ai-api-chiragpednekar3-8808s-projects.vercel.app"
OUTPUT = REPO_ROOT / "ml" / "experiments" / "c1_benchmark" / "c1_benchmark_results.json"

#: Benchmark tool name -> deployed tool name. Fixed before the run.
#: ``None`` means the deployed agent has no equivalent, and the tool check for
#: that question is recorded as not applicable.
TOOL_MAP: dict[str, str | None] = {
    "get_flood_prediction": "get_hazard_result",
    "get_historical_risk": "get_hazard_result",
    "get_risk_map": "get_hazard_result",
    "get_active_fires": "get_hazard_result",
    "get_satellite_data": "get_hazard_result",
    "get_model_explanation": "get_model_info",
    "get_location_statistics": "get_study_area",
    "get_weather_data": None,
}

#: Notes that mean the model never answered: capacity, quota or transport.
TRANSIENT_MARKERS = ("503", "429", "high demand", "quota", "unreachable")


def mapped(question: BenchmarkQuestion) -> tuple[BenchmarkQuestion, bool]:
    """The question with its tool expectations translated to deployed names."""
    expected = [TOOL_MAP.get(name, name) for name in question.expected_tools]
    forbidden = [TOOL_MAP.get(name, name) for name in question.forbidden_tools]
    applicable = all(name is not None for name in expected)
    return (
        dataclasses.replace(
            question,
            expected_tools=tuple(dict.fromkeys(n for n in expected if n)),
            forbidden_tools=tuple(dict.fromkeys(n for n in forbidden if n)),
        ),
        applicable,
    )


def ask(message: str, *, attempts: int, pause: float) -> tuple[dict[str, Any] | None, float, str]:
    """One question to production, retried while the failure is transient."""
    last_reason = ""
    for attempt in range(attempts):
        started = time.perf_counter()
        request = urllib.request.Request(  # noqa: S310
            f"{API}/api/v1/chat",
            data=json.dumps({"message": message}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "SAT-AI-C1-benchmark"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                body: dict[str, Any] = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_reason = f"transport: {exc}"
            time.sleep(pause * (attempt + 1))
            continue
        latency = time.perf_counter() - started

        notes = " ".join(body.get("notes") or []).lower()
        transient = body.get("degraded") and any(m in notes for m in TRANSIENT_MARKERS)
        if not transient:
            return body, latency, ""
        last_reason = (body.get("notes") or ["degraded"])[0][:160]
        time.sleep(pause * (attempt + 1))
    return None, 0.0, last_reason


def score(question: BenchmarkQuestion, body: dict[str, Any], latency: float) -> dict[str, Any]:
    answer = body.get("answer", "")
    called = list(body.get("tools_called") or [])
    route = body.get("route_method", "")
    notes = body.get("notes") or []

    translated, tools_applicable = mapped(question)
    correct_tools, tool_problems = score_tool_calls(called, translated)
    content_ok, content_problems = check_content(answer, question)
    refused_ok = detected_refusal(answer) if question.must_refuse else True

    # A draft was rejected when the validator logged a violation; the answer
    # shown is then verified tool output rather than the draft.
    draft_rejected = any(str(n).startswith(("Grounding violation", "Hard failure")) for n in notes)
    model_in_loop = route == "gemini+tools"

    return {
        "id": question.id,
        "kind": question.kind.value,
        "question": question.question,
        "answer": answer,
        "route_method": route,
        "model_in_loop": model_in_loop,
        "tools_called": called,
        "tools_expected_deployed": list(translated.expected_tools),
        "tools_applicable": tools_applicable,
        "correct_tools": correct_tools if tools_applicable else None,
        "tool_problems": tool_problems if tools_applicable else [],
        "draft_rejected": draft_rejected,
        "shown_answer_grounded": bool(body.get("grounded", True)) or draft_rejected,
        "content_ok": content_ok,
        "content_problems": content_problems,
        "must_refuse": question.must_refuse,
        "refused_correctly": refused_ok,
        "passed": bool((correct_tools or not tools_applicable) and content_ok and refused_ok),
        "latency_s": round(latency, 2),
        "notes": notes,
    }


def rate(items: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(i[key]) for i in items if i.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--pause", type=float, default=12.0, help="seconds between questions")
    parser.add_argument("--only", default=None, help="comma-separated question ids")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep turns already scored in the results file and ask only the rest",
    )
    parser.add_argument(
        "--give-up-after",
        type=int,
        default=3,
        help="consecutive quota failures after which the remaining questions are not sent",
    )
    args = parser.parse_args(argv)

    health = json.loads(urllib.request.urlopen(f"{API}/health", timeout=60).read())  # noqa: S310
    model = health["checks"].get("llm_model")
    if health["checks"].get("llm") != "configured":
        print("BLOCKED: the deployment reports no configured language model.")
        return 2

    wanted = set(args.only.split(",")) if args.only else None
    questions = [q for q in QUESTIONS if wanted is None or q.id in wanted]
    scored: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    runs: list[str] = []
    if args.resume and OUTPUT.exists():
        # Earlier scored turns are kept exactly as they were measured, with
        # their own timestamps, rather than asked again. Only turns that were
        # never measured are sent.
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if previous.get("benchmark_version") == BENCHMARK_VERSION:
            scored = [r for r in previous.get("results", []) if wanted is None or r["id"] in wanted]
            runs = list(previous.get("runs") or [previous["run_at"]])
    done = {r["id"] for r in scored}
    questions = [q for q in questions if q.id not in done]

    print(
        f"C1 benchmark v{BENCHMARK_VERSION} against {API} ({model}), "
        f"{len(questions)} to ask, {len(done)} already measured\n"
    )
    quota_streak = 0
    for n, question in enumerate(questions, start=1):
        if quota_streak >= args.give_up_after:
            # A daily quota does not recover in minutes. Sending the rest would
            # only turn each of them into five more retries against a closed door.
            unmeasured.append(
                {
                    "id": question.id,
                    "kind": question.kind.value,
                    "reason": "not sent: model quota exhausted earlier in this run",
                }
            )
            print(f"  {n:>2} {question.id:<4} NOT SENT    quota exhausted")
            continue
        body, latency, reason = ask(question.question, attempts=args.attempts, pause=args.pause)
        if body is None:
            is_quota = "quota" in reason.lower() or "429" in reason
            quota_streak = quota_streak + 1 if is_quota else 0
            unmeasured.append({"id": question.id, "kind": question.kind.value, "reason": reason})
            print(f"  {n:>2} {question.id:<4} UNMEASURED  {reason[:70]}")
        else:
            quota_streak = 0
            result = score(question, body, latency)
            scored.append(result)
            flag = "pass" if result["passed"] else "FAIL"
            print(
                f"  {n:>2} {question.id:<4} {flag:<4} {result['route_method']:<13} "
                f"tools={','.join(result['tools_called']) or '-':<34} "
                f"draft_rejected={result['draft_rejected']!s:<5} {result['latency_s']:>5.1f}s"
            )
        time.sleep(args.pause)

    model_turns = [s for s in scored if s["model_in_loop"]]
    refusal_items = [s for s in scored if s["kind"] in {"must_refuse", "out_of_scope"}]
    latencies = [s["latency_s"] for s in scored if s["model_in_loop"]]
    report = {
        "experiment": "c1_grounding_benchmark",
        "contribution": "C1",
        "run_at": datetime.now(UTC).isoformat(),
        "benchmark_version": BENCHMARK_VERSION,
        "endpoint": API,
        "model": model,
        "runs": [*runs, datetime.now(UTC).isoformat()],
        "n_questions": len(done) + len(questions),
        "n_measured": len(scored),
        "n_unmeasured": len(unmeasured),
        "n_model_in_loop": len(model_turns),
        "metrics": {
            "draft_violation_rate": rate(model_turns, "draft_rejected"),
            "shown_violation_rate": (
                round(1 - rate(scored, "shown_answer_grounded"), 4)  # type: ignore[operator]
                if scored
                else None
            ),
            "tool_invocation_accuracy": rate(
                [s for s in scored if s["tools_applicable"]], "correct_tools"
            ),
            "refusal_accuracy": rate(refusal_items, "refused_correctly"),
            "content_check_pass_rate": rate(scored, "content_ok"),
            "overall_pass_rate": rate(scored, "passed"),
            "median_latency_s_model_turns": round(statistics.median(latencies), 2)
            if latencies
            else None,
        },
        "denominators": {
            "draft_violation_rate": len(model_turns),
            "tool_invocation_accuracy": sum(1 for s in scored if s["tools_applicable"]),
            "refusal_accuracy": len(refusal_items),
            "others": len(scored),
        },
        "tool_map": TOOL_MAP,
        "results": scored,
        "unmeasured": unmeasured,
        "caveats": [
            "Tool accuracy is scored under TOOL_MAP, fixed before the run: the "
            "benchmark predates the deployed five-tool agent.",
            "Turns that stayed degraded after retries (capacity or quota) are "
            "unmeasured, not scored; every rate states its denominator.",
            "Content checks are phrase matches written with the benchmark. They "
            "catch forbidden claims reliably and judge answer quality poorly.",
            "One run, one model version, one day. Model behaviour varies between "
            "runs and releases; this is a measurement, not a guarantee.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total = report["n_questions"]
    print(f"\n  measured {len(scored)}/{total}, model in the loop on {len(model_turns)}")
    for name, value in report["metrics"].items():
        print(f"  {name:<32} {value}")
    print(f"\nwritten to {relative_to_repo(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
