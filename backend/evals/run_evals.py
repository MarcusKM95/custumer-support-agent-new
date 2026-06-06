import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time

import httpx


DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
DEFAULT_REPORTS_DIR = Path(__file__).with_name("reports")


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_response(response: dict, expected: dict) -> dict:
    checks = []
    router = response.get("router") or {}
    escalation = response.get("escalation") or {}
    clarification = response.get("clarification") or {}
    verification = response.get("verification") or {}

    add_check(checks, "intent", router.get("intent"), expected.get("intent"))
    add_check(checks, "agent", response.get("agent"), expected.get("agent"))
    add_check(
        checks,
        "escalation_required",
        escalation.get("required"),
        expected.get("escalation_required"),
    )
    add_check(
        checks,
        "clarification_required",
        clarification.get("required"),
        expected.get("clarification_required"),
    )
    add_check(
        checks,
        "escalation_queue",
        escalation.get("queue"),
        expected.get("escalation_queue"),
    )
    add_check(
        checks,
        "escalation_priority",
        escalation.get("priority"),
        expected.get("escalation_priority"),
    )
    add_check(
        checks,
        "verification_supported",
        verification.get("supported"),
        expected.get("verification_supported"),
    )

    expected_sections = expected.get("source_sections")
    if expected_sections is not None:
        actual_sections = {
            source.get("section")
            for source in response.get("sources", [])
            if source.get("section")
        }
        checks.append(
            {
                "name": "source_sections",
                "passed": bool(actual_sections.intersection(expected_sections)),
                "expected": expected_sections,
                "actual": sorted(actual_sections),
            }
        )

    answer = response.get("answer", "")
    expected_terms = expected.get("answer_contains_any")
    if expected_terms is not None:
        normalized_answer = answer.casefold()
        checks.append(
            {
                "name": "answer_contains_any",
                "passed": any(term.casefold() in normalized_answer for term in expected_terms),
                "expected": expected_terms,
                "actual": answer,
            }
        )

    required_checks = [check for check in checks if check["expected"] is not None]
    passed_count = sum(check["passed"] for check in required_checks)
    score = passed_count / len(required_checks) if required_checks else 1.0

    return {
        "passed": all(check["passed"] for check in required_checks),
        "score": score,
        "checks": required_checks,
    }


def add_check(checks: list[dict], name: str, actual, expected) -> None:
    if expected is None:
        return

    checks.append(
        {
            "name": name,
            "passed": actual == expected,
            "expected": expected,
            "actual": actual,
        }
    )


def run_case(client: httpx.Client, case: dict) -> dict:
    conversation_id = None
    responses = []
    started_at = time.perf_counter()

    for message in case["messages"]:
        response = client.post(
            "/chat",
            json={
                "message": message,
                "conversation_id": conversation_id,
            },
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("error"):
            raise RuntimeError(payload["error"])

        conversation_id = payload.get("conversation_id")
        responses.append(payload)

    duration_seconds = time.perf_counter() - started_at
    final_response = responses[-1]
    evaluation = evaluate_response(final_response, case["expected"])

    return {
        "id": case["id"],
        "category": case["category"],
        "messages": case["messages"],
        "conversation_id": conversation_id,
        "duration_seconds": round(duration_seconds, 3),
        "evaluation": evaluation,
        "response": final_response,
    }


def build_summary(results: list[dict]) -> dict:
    category_results = defaultdict(list)
    for result in results:
        category_results[result["category"]].append(result)

    categories = {}
    for category, entries in sorted(category_results.items()):
        categories[category] = summarize_results(entries)

    return {
        **summarize_results(results),
        "categories": categories,
    }


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(result["evaluation"]["passed"] for result in results)
    average_score = (
        sum(result["evaluation"]["score"] for result in results) / total if total else 0.0
    )
    average_duration = (
        sum(result["duration_seconds"] for result in results) / total if total else 0.0
    )

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": round(average_score, 4),
        "average_duration_seconds": round(average_duration, 3),
    }


def write_reports(report: dict, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = reports_dir / f"evaluation-{timestamp}.json"
    markdown_path = reports_dir / f"evaluation-{timestamp}.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Agent Evaluation Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- API: `{report['base_url']}`",
        f"- Cases: **{summary['total']}**",
        f"- Passed: **{summary['passed']}**",
        f"- Failed: **{summary['failed']}**",
        f"- Pass rate: **{summary['pass_rate']:.1%}**",
        f"- Average criterion score: **{summary['average_score']:.1%}**",
        f"- Average duration: **{summary['average_duration_seconds']}s**",
        "",
        "## Categories",
        "",
        "| Category | Passed | Total | Pass rate | Avg. score |",
        "|---|---:|---:|---:|---:|",
    ]

    for category, category_summary in summary["categories"].items():
        lines.append(
            f"| {category} | {category_summary['passed']} | "
            f"{category_summary['total']} | {category_summary['pass_rate']:.1%} | "
            f"{category_summary['average_score']:.1%} |"
        )

    lines.extend(["", "## Cases", ""])
    for result in report["results"]:
        status = "PASS" if result["evaluation"]["passed"] else "FAIL"
        lines.extend(
            [
                f"### {status}: `{result['id']}`",
                "",
                f"- Category: `{result['category']}`",
                f"- Duration: `{result['duration_seconds']}s`",
                f"- Score: `{result['evaluation']['score']:.1%}`",
                "",
            ]
        )

        failed_checks = [
            check for check in result["evaluation"]["checks"] if not check["passed"]
        ]
        if failed_checks:
            lines.append("Failed checks:")
            for check in failed_checks:
                lines.append(
                    f"- `{check['name']}` expected `{check['expected']}`, "
                    f"got `{check['actual']}`"
                )
            lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the live support agent API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)

    if args.case_ids:
        selected_ids = set(args.case_ids)
        cases = [case for case in cases if case["id"] in selected_ids]

    if args.limit is not None:
        cases = cases[: args.limit]

    if not cases:
        print("No evaluation cases selected.", file=sys.stderr)
        return 2

    results = []
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']}...", end=" ", flush=True)
            try:
                result = run_case(client, case)
            except Exception as exc:
                result = {
                    "id": case["id"],
                    "category": case["category"],
                    "messages": case["messages"],
                    "duration_seconds": 0.0,
                    "evaluation": {
                        "passed": False,
                        "score": 0.0,
                        "checks": [
                            {
                                "name": "execution",
                                "passed": False,
                                "expected": "successful response",
                                "actual": str(exc),
                            }
                        ],
                    },
                    "response": None,
                }
            results.append(result)
            print("PASS" if result["evaluation"]["passed"] else "FAIL")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "summary": build_summary(results),
        "results": results,
    }
    json_path, markdown_path = write_reports(report, args.reports_dir)

    summary = report["summary"]
    print()
    print(
        f"Passed {summary['passed']}/{summary['total']} "
        f"({summary['pass_rate']:.1%}), average score {summary['average_score']:.1%}"
    )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
