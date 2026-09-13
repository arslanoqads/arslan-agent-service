import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SET = ROOT / "tests" / "evals" / "golden_set.public.json"
PRIVATE_SET = ROOT / "tests" / "evals" / "golden_set.private.json"
SCHEMA = ROOT / "tests" / "evals" / "schema.json"


def load_cases() -> list[dict]:
    cases = []
    for path in (PUBLIC_SET, PRIVATE_SET):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            cases.extend(data)
    return cases


def score_case(case: dict, actual: dict) -> list[str]:
    failures = []
    expected_tool = case.get("expected_tool")
    tools = actual.get("tools") or []
    if expected_tool == "none" and tools:
        failures.append(f"expected no tools, called {tools}")
    elif expected_tool and expected_tool != "none" and expected_tool not in tools:
        failures.append(f"expected tool {expected_tool}, called {tools}")

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


def run() -> int:
    cases = load_cases()
    if not cases:
        print("Golden set is empty. No cases to score.")
        return 0
    schema = json.loads(SCHEMA.read_text())
    required = schema["required"]
    missing = []
    for case in cases:
        for field in required:
            if case.get(field) in (None, ""):
                missing.append(f"{case.get('id', '?')}.{field}")
    if missing:
        print("Cases missing required fields:", ", ".join(missing))
        return 1
    print("Scaffold check passed. Cases are not executed against the model in this build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
