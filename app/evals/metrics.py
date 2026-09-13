"""Golden-set metric scores by category and average over time.

Scores production traces (and durable score points) against golden cases using
tool/route oracles. Categories: family, severity, source, expected_tool.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone

from app.evals.runner import score_case
from app.observability.model import public_question

_NORMALIZE_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    cleaned = public_question(text or "").lower()
    cleaned = _NORMALIZE_RE.sub(" ", cleaned)
    return _SPACE_RE.sub(" ", cleaned).strip()


def prompts_match(case_input: str, trace_question: str) -> bool:
    left = normalize_prompt(case_input)
    right = normalize_prompt(trace_question)
    if not left or not right:
        return False
    if left == right:
        return True
    if left in right or right in left:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
    return overlap >= 0.72


def _day_key(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = str(value)
    return text[:10]


def score_trace_against_case(case: dict, trace: dict) -> dict:
    """Return 0–1 score for a trace vs a golden case (tool/route focused)."""
    actual = {
        "tools": trace.get("tools") or [],
        "route": _trace_route(trace),
        "response": "",
    }
    # Text oracles need the model answer; public traces omit it, so only enforce
    # tool/route expectations here.
    slim = {
        "expected_tool": case.get("expected_tool"),
        "expected_route": case.get("expected_route"),
        "oracle": "code",
        "must_include": [],
        "must_not_include": [],
    }
    failures = score_case(slim, actual)
    tool_score = 1.0 if not failures else max(0.0, 1.0 - 0.5 * len(failures))

    outcome = (trace.get("outcome") or "success").lower()
    if outcome == "success":
        outcome_score = 1.0
    elif outcome in {"tool_refused", "guardrail"}:
        outcome_score = 0.4
    else:
        outcome_score = 0.15

    triad = ((trace.get("rag_triage") or {}).get("scores")) or {}
    triad_vals = [
        float(triad[key])
        for key in ("context_relevance", "answer_faithfulness", "answer_relevance")
        if triad.get(key) is not None
    ]
    triad_score = sum(triad_vals) / len(triad_vals) if triad_vals else None

    # Weight contract correctness highest; blend triad when present.
    if triad_score is None:
        score = round(0.75 * tool_score + 0.25 * outcome_score, 3)
    else:
        score = round(0.55 * tool_score + 0.2 * outcome_score + 0.25 * triad_score, 3)

    return {
        "case_id": case.get("id"),
        "trace_id": trace.get("id"),
        "score": score,
        "passed": not failures,
        "failures": failures,
        "family": case.get("family") or "unknown",
        "severity": case.get("severity") or "unknown",
        "source": case.get("source") or "unknown",
        "expected_tool": case.get("expected_tool") or "none",
        "started_at": trace.get("started_at"),
        "day": _day_key(trace.get("started_at")),
        "triad": triad or None,
    }


def _trace_route(trace: dict) -> str | None:
    route = trace.get("route")
    if route == "portfolio":
        return "portfolio_agent"
    if route == "greeting":
        return "general_responder"
    if route in {"portfolio_agent", "general_responder"}:
        return route
    return None


def match_scores(cases: list[dict], traces: list[dict]) -> list[dict]:
    indexed = [(case, normalize_prompt(case.get("input") or "")) for case in cases if case.get("id")]
    results = []
    for trace in traces:
        question = trace.get("question") or ""
        if not question:
            continue
        for case, _norm in indexed:
            if prompts_match(case.get("input") or "", question):
                results.append(score_trace_against_case(case, trace))
                break
    return results


def _bucket_avg(points: list[dict], field: str) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for point in points:
        key = str(point.get(field) or "unknown")
        groups[key].append(float(point.get("score") or 0))
    return {
        key: {
            "avg": round(sum(vals) / len(vals), 3) if vals else 0.0,
            "n": len(vals),
            "pass_rate": round(
                sum(1 for point in points if str(point.get(field) or "unknown") == key and point.get("passed"))
                / len(vals),
                3,
            )
            if vals
            else 0.0,
        }
        for key, vals in sorted(groups.items())
    }


def _case_avg(points: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        groups[str(point.get("case_id") or "?")].append(point)
    rows = []
    for case_id, items in groups.items():
        scores = [float(item.get("score") or 0) for item in items]
        latest = max(items, key=lambda item: item.get("started_at") or "")
        rows.append(
            {
                "id": case_id,
                "avg_score": round(sum(scores) / len(scores), 3),
                "n": len(scores),
                "pass_rate": round(sum(1 for item in items if item.get("passed")) / len(items), 3),
                "family": latest.get("family"),
                "severity": latest.get("severity"),
                "source": latest.get("source"),
                "expected_tool": latest.get("expected_tool"),
                "latest_at": latest.get("started_at"),
            }
        )
    rows.sort(key=lambda row: row.get("avg_score") or 0)
    return rows


def _over_time(points: list[dict]) -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for point in points:
        groups[point.get("day") or _day_key(None)].append(float(point.get("score") or 0))
    series = []
    for day in sorted(groups.keys()):
        vals = groups[day]
        series.append(
            {
                "date": day,
                "avg": round(sum(vals) / len(vals), 3),
                "n": len(vals),
            }
        )
    return series


def aggregate_golden_metrics(
    cases: list[dict],
    traces: list[dict],
    durable_points: list[dict] | None = None,
) -> dict:
    live = match_scores(cases, traces)
    durable = list(durable_points or [])
    # Prefer live score for same trace+case; keep durable history otherwise.
    merged: dict[str, dict] = {}
    for point in durable + live:
        key = f"{point.get('case_id')}::{point.get('trace_id') or point.get('day')}"
        existing = merged.get(key)
        if not existing or (point.get("started_at") or "") >= (existing.get("started_at") or ""):
            merged[key] = point
    points = list(merged.values())
    points.sort(key=lambda item: item.get("started_at") or "")

    overall = [float(p.get("score") or 0) for p in points]
    return {
        "overall_avg": round(sum(overall) / len(overall), 3) if overall else 0.0,
        "pass_rate": round(sum(1 for p in points if p.get("passed")) / len(points), 3) if points else 0.0,
        "samples": len(points),
        "matched_traces": len({p.get("trace_id") for p in points if p.get("trace_id")}),
        "cases_covered": len({p.get("case_id") for p in points if p.get("case_id")}),
        "cases_total": len(cases),
        "by_family": _bucket_avg(points, "family"),
        "by_severity": _bucket_avg(points, "severity"),
        "by_source": _bucket_avg(points, "source"),
        "by_expected_tool": _bucket_avg(points, "expected_tool"),
        "over_time": _over_time(points),
        "case_scores": _case_avg(points),
        "note": (
            "Scores match production traces to golden prompts (tool/route oracles). "
            "Average over time is daily mean of matched scores."
        ),
    }
