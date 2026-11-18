"""Routing accuracy evaluation for the supervisor agent.

The supervisor is the one component whose mistakes are expensive: a wrong
route means the customer's question is never answered, or is answered by an
agent that has no tools for it. This script measures that decision in
isolation, against a small labelled set of realistic messages.

Usage:
    python -m evaluation.routing_eval                 # run, print, write results
    python -m evaluation.routing_eval --threshold 0.9

Requires GROQ_API_KEY, since every case is one real supervisor call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

CASES_PATH = Path(__file__).with_name("routing_cases.jsonl")
RESULTS_PATH = Path(__file__).with_name("results.md")

ROUTES = ("music_catalog", "invoice_info", "both", "FINISH")


@dataclass(frozen=True)
class Case:
    """One labelled routing example."""

    message: str
    expected: str
    note: str = ""


@dataclass(frozen=True)
class Prediction:
    """The supervisor's answer for one case."""

    case: Case
    actual: str | None
    reasoning: str

    @property
    def correct(self) -> bool:
        """Whether the supervisor chose the labelled route."""
        return self.actual == self.case.expected


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Load the labelled cases from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        The parsed cases, in file order.

    Raises:
        ValueError: If a line is malformed or names an unknown route.
    """
    cases: list[Case] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number} is not valid JSON: {exc}") from exc

        message = str(record.get("message", "")).strip()
        expected = str(record.get("expected", ""))
        if not message:
            raise ValueError(f"{path.name}:{number} has an empty message")
        if expected not in ROUTES:
            raise ValueError(f"{path.name}:{number} has unknown route {expected!r}")

        cases.append(Case(message=message, expected=expected, note=str(record.get("note", ""))))
    return cases


def predict(cases: list[Case]) -> list[Prediction]:
    """Run every case through the supervisor, one live call each."""
    from langchain_core.messages import HumanMessage

    from src.agents.supervisor import build_supervisor_router

    router = build_supervisor_router()
    predictions: list[Prediction] = []
    for index, case in enumerate(cases, start=1):
        state = {
            "messages": [HumanMessage(content=case.message)],
            "loaded_memory": "No preferences on file yet.",
            "customer_id": "",
        }
        try:
            decision = router(state)
            prediction = Prediction(case=case, actual=decision.route, reasoning=decision.reasoning)
        except Exception as exc:  # noqa: BLE001 - a failed call is a failed case
            prediction = Prediction(case=case, actual=None, reasoning=f"error: {exc}")

        predictions.append(prediction)
        status = "ok" if prediction.correct else "MISS"
        print(f"[{index:02d}/{len(cases)}] {status:<4} {case.expected:<14} -> {prediction.actual}")
    return predictions


def render_markdown(predictions: list[Prediction], model: str) -> str:
    """Render a results document for humans and for the repository."""
    correct = sum(1 for prediction in predictions if prediction.correct)
    accuracy = correct / len(predictions)

    lines = [
        "# Routing evaluation results",
        "",
        f"- Model: `{model}`",
        f"- Cases: {len(predictions)}",
        f"- Correct: {correct}",
        f"- Accuracy: **{accuracy:.1%}**",
        f"- Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| # | Message | Expected | Actual | |",
        "|---|---|---|---|---|",
    ]
    for number, prediction in enumerate(predictions, start=1):
        message = prediction.case.message.replace("|", "\\|")
        mark = "pass" if prediction.correct else "**miss**"
        lines.append(
            f"| {number} | {message} | `{prediction.case.expected}` | "
            f"`{prediction.actual}` | {mark} |"
        )

    misses = [prediction for prediction in predictions if not prediction.correct]
    if misses:
        lines += ["", "## Misrouted cases", ""]
        for prediction in misses:
            lines.append(
                f"- `{prediction.case.expected}` → `{prediction.actual}`: "
                f"{prediction.case.message}  \n  {prediction.case.note}"
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="routing-eval",
        description="Measure supervisor routing accuracy against a labelled set.",
    )
    parser.add_argument("--threshold", type=float, default=0.85, help="minimum accuracy (0-1)")
    parser.add_argument(
        "--cases", type=Path, default=CASES_PATH, help="JSONL file of labelled cases"
    )
    parser.add_argument("--no-write", action="store_true", help="do not write results.md")
    args = parser.parse_args(argv)

    from src.config import get_settings

    settings = get_settings()
    if not settings.has_llm_credentials:
        print("GROQ_API_KEY is not set; the evaluation needs a live model.", file=sys.stderr)
        return 2

    cases = load_cases(args.cases)
    predictions = predict(cases)

    failed = [prediction for prediction in predictions if prediction.actual is None]
    if failed:
        print(
            f"\n{len(failed)} of {len(predictions)} calls failed; refusing to score them.",
            file=sys.stderr,
        )
        for prediction in failed[:3]:
            print(f"  - {prediction.reasoning}", file=sys.stderr)
        return 2

    correct = sum(1 for prediction in predictions if prediction.correct)
    accuracy = correct / len(predictions)

    totals = Counter(prediction.case.expected for prediction in predictions)
    hits = Counter(prediction.case.expected for prediction in predictions if prediction.correct)
    print("\nPer-route accuracy:")
    for route in ROUTES:
        if totals[route]:
            print(f"  {route:<14} {hits[route]}/{totals[route]}")
    print(f"\nOverall accuracy: {accuracy:.1%} ({correct}/{len(predictions)})")

    if not args.no_write:
        RESULTS_PATH.write_text(render_markdown(predictions, settings.model_name), encoding="utf-8")
        print(f"Wrote {RESULTS_PATH.relative_to(RESULTS_PATH.parent.parent)}")

    if accuracy < args.threshold:
        print(f"\nFAIL: below the {args.threshold:.0%} threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
