import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
# Prefer the image-shipped copy under app/evals (Dockerfile does not copy tests/).
PUBLIC_SET_CANDIDATES = (
    APP_ROOT / "golden_set.public.json",
    ROOT / "tests" / "evals" / "golden_set.public.json",
)
PRIVATE_SET = ROOT / "tests" / "evals" / "golden_set.private.json"
SCHEMA_CANDIDATES = (
    APP_ROOT / "schema.json",
    ROOT / "tests" / "evals" / "schema.json",
)


def _first_existing(paths) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_cases() -> list[dict]:
    cases = []
    public = _first_existing(PUBLIC_SET_CANDIDATES)
    for path in (public, PRIVATE_SET if PRIVATE_SET.exists() else None):
        if path is None or not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            cases.extend(data)
    return cases


def load_public_cases() -> list[dict]:
    """Packaged public cases plus any durable production extras."""
    packaged: list[dict] = []
    path = _first_existing(PUBLIC_SET_CANDIDATES)
    if path is not None:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            packaged = data
    merged: dict[str, dict] = {case.get("id"): case for case in packaged if case.get("id")}
    try:
        from app.evals.durable import get_golden_store

        for case in get_golden_store().list_cases():
            case_id = case.get("id")
            if case_id:
                merged[case_id] = case
    except Exception:
        pass
    # Preserve packaged order, then durable-only extras.
    ordered = []
    seen = set()
    for case in packaged:
        case_id = case.get("id")
        if case_id and case_id in merged and case_id not in seen:
            ordered.append(merged[case_id])
            seen.add(case_id)
    for case_id, case in merged.items():
        if case_id not in seen:
            ordered.append(case)
    return ordered


def public_case(case: dict) -> dict:
    """Fields safe to show on the public observability page."""
    from app.observability.model import public_question, redact

    return {
        "id": case.get("id"),
        "family": case.get("family"),
        "severity": case.get("severity"),
        "source": case.get("source"),
        "oracle": case.get("oracle"),
        "expected_tool": case.get("expected_tool"),
        "expected_route": case.get("expected_route"),
        "eval_focus": case.get("eval_focus"),
        "input": public_question(case.get("input") or ""),
        "prior_turns": [
            {
                "role": turn.get("role"),
                "content": public_question(turn.get("content") or "")
                if turn.get("role") == "user"
                else redact(turn.get("content") or ""),
            }
            for turn in (case.get("prior_turns") or [])
        ],
        "must_include": [redact(item) for item in (case.get("must_include") or [])],
        "must_not_include": [redact(item) for item in (case.get("must_not_include") or [])],
        "notes": redact(case.get("notes") or ""),
    }


def score_case(case: dict, actual: dict) -> list[str]:
    failures = []
    expected_tool = case.get("expected_tool")
    tools = actual.get("tools") or []
    if expected_tool == "none" and tools:
        failures.append(f"expected no tools, called {tools}")
    elif expected_tool and expected_tool != "none" and expected_tool not in tools:
        failures.append(f"expected tool {expected_tool}, called {tools}")

    expected_route = case.get("expected_route")
    if expected_route and actual.get("route") and actual.get("route") != expected_route:
        failures.append(f"expected route {expected_route}, got {actual.get('route')}")

    text = actual.get("response") or ""
    oracle = case.get("oracle") or "code"
    if oracle == "exact":
        expected = case.get("expected") or ""
        if expected and text != expected:
            failures.append("response did not match exactly")
    elif oracle == "regex":
        for pattern in case.get("must_include") or []:
            if re.search(pattern, text) is None:
                failures.append(f"regex missed {pattern!r}")
        for pattern in case.get("must_not_include") or []:
            if re.search(pattern, text):
                failures.append(f"forbidden pattern {pattern!r}")
    else:
        for needle in case.get("must_include") or []:
            if needle not in text:
                failures.append(f"missing {needle!r}")
        for needle in case.get("must_not_include") or []:
            if needle in text:
                failures.append(f"forbidden {needle!r}")
    return failures


def route_case(case: dict) -> str | None:
    """Score deterministic routing helpers for multi-turn cases without calling the model."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.runtime.route import choose_route

    messages = []
    for turn in case.get("prior_turns") or []:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=case.get("input") or ""))
    return choose_route(messages) or None


def run() -> int:
    cases = load_cases()
    if not cases:
        print("Golden set is empty. No cases to score.")
        return 0
    schema_path = _first_existing(SCHEMA_CANDIDATES)
    if schema_path is None:
        print("Golden schema missing.")
        return 1
    schema = json.loads(schema_path.read_text())
    required = schema["required"]
    missing = []
    route_failures = []
    for case in cases:
        for field in required:
            if case.get(field) in (None, ""):
                missing.append(f"{case.get('id', '?')}.{field}")
        expected_route = case.get("expected_route")
        if expected_route:
            actual_route = route_case(case)
            # choose_route returns "" when the LLM router should decide.
            if actual_route and actual_route != expected_route:
                route_failures.append(f"{case.get('id')}: expected {expected_route}, got {actual_route}")
            if not actual_route and expected_route == "portfolio_agent" and case.get("prior_turns"):
                route_failures.append(f"{case.get('id')}: expected portfolio follow-up route")
    if missing:
        print("Cases missing required fields:", ", ".join(missing))
        return 1
    if route_failures:
        print("Route helper failures:", "; ".join(route_failures))
        return 1
    print(f"Scaffold check passed for {len(cases)} cases. Model execution is still offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
