"""RAG evaluation triad for observability triage.

Core triad (LLM-judge or human):
  a. Context relevance  — fetched the right documents
  b. Answer faithfulness — stuck to facts in those documents
  c. Answer relevance   — addressed the user's question

Scores below are cheap production proxies from traces, not full LLM judges.
"""

from __future__ import annotations

import re

from app.observability.model import RETRIEVAL_TOOLS

CITATION_RE = re.compile(
    r"\b(?:resume|bio)\b.{0,12}\bv?\d+|\bpage\s+\d+\b|\bv\d+\s*,\s*page\s+\d+\b",
    re.IGNORECASE,
)
ABSTAIN_RE = re.compile(
    r"\b("
    r"i don['’]?t know|do not know|don['’]?t have (enough )?evidence|"
    r"no (resume |bio )?(evidence|excerpts?)|not (found|in) (the )?(resume|bio|documents?)|"
    r"could(?: not|n't) find|insufficient evidence|outside (the )?(resume|bio|corpus)"
    r")\b",
    re.IGNORECASE,
)

# Fix playbook tied to each triad edge (what this product ships vs plans).
PLAYBOOK = [
    {
        "dimension": "context_relevance",
        "fix": "Hybrid keyword + semantic retrieval",
        "status": "active",
        "detail": "BM25 keyword search fused with dense embeddings — helps with IDs and exact phrases.",
    },
    {
        "dimension": "context_relevance",
        "fix": "Query rewrite for vague asks",
        "status": "active",
        "detail": "Rewrite short/ambiguous portfolio questions before retrieval.",
    },
    {
        "dimension": "context_relevance",
        "fix": "Two-step retrieval",
        "status": "active",
        "detail": "BM25 narrows candidates; dense scoring picks the top chunks.",
    },
    {
        "dimension": "answer_faithfulness",
        "fix": "Chain of verification (CoVe)",
        "status": "partial",
        "detail": "Prompt asks the model to extract claims and verify each against tool evidence.",
    },
    {
        "dimension": "answer_faithfulness",
        "fix": "Abstain when evidence is missing",
        "status": "active",
        "detail": "Instructed to say it does not know rather than invent resume facts.",
    },
    {
        "dimension": "answer_faithfulness",
        "fix": "Citation enforcement",
        "status": "active",
        "detail": "Answers from retrieval must cite resume/bio version and page.",
    },
    {
        "dimension": "answer_relevance",
        "fix": "Role prompting",
        "status": "active",
        "detail": "Portfolio-assistant system prompt plus hop-specific role instructions.",
    },
    {
        "dimension": "answer_relevance",
        "fix": "Multi-step planning",
        "status": "active",
        "detail": "Independent hops for profile, email, calendar, JD, and links in one turn.",
    },
]

DIAGNOSIS_PATTERNS = {
    "A": {
        "label": "Retrieval failure cascading",
        "context_relevance": "low",
        "answer_faithfulness": "low",
        "answer_relevance": "low",
    },
    "B": {
        "label": "Generator ignoring good context",
        "context_relevance": "high",
        "answer_faithfulness": "low",
        "answer_relevance": "medium",
    },
    "C": {
        "label": "Faithful summary of wrong docs",
        "context_relevance": "low",
        "answer_faithfulness": "high",
        "answer_relevance": "low",
    },
    "D": {
        "label": "Off-topic but accurate answer",
        "context_relevance": "high",
        "answer_faithfulness": "high",
        "answer_relevance": "low",
    },
    "E": {
        "label": "Healthy RAG turn",
        "context_relevance": "high",
        "answer_faithfulness": "high",
        "answer_relevance": "high",
    },
}


def _band(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def has_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text or ""))


def abstained(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text or ""))


def annotate_rag_triage(trace: dict, answer: str | None = None) -> dict | None:
    """Attach per-turn RAG triad signals when retrieval tools ran."""
    tools = set(trace.get("tools") or [])
    if not tools.intersection(RETRIEVAL_TOOLS):
        return None

    tool_status = {item.get("name"): item.get("status") for item in (trace.get("tool_status") or [])}
    retrieval_names = tools.intersection(RETRIEVAL_TOOLS)
    retrieval_ok = all(tool_status.get(name) == "ok" for name in retrieval_names) if retrieval_names else False
    budget = trace.get("context_budget") or {}
    cut = budget.get("cut") or []
    if isinstance(cut, bool):
        cut_retrieved = bool(cut)
    else:
        cut_retrieved = "retrieved" in cut
    text = answer or ""
    signals = {
        "retrieval": True,
        "retrieval_ok": retrieval_ok,
        "retrieved_tokens": int(budget.get("retrieved") or 0),
        "context_cut": cut_retrieved,
        "has_citation": has_citation(text),
        "abstained": abstained(text),
        "multi_hop": int(trace.get("loop_count") or 0) > 1 or len(tools) > 1,
        "outcome": trace.get("outcome") or "success",
        "stop_reason": trace.get("stop_reason"),
    }
    scores = score_turn(signals)
    payload = {**signals, "scores": scores}
    trace["rag_triage"] = payload
    return payload


def score_turn(signals: dict) -> dict:
    """Proxy scores for one retrieval turn (0–1)."""
    retrieval_ok = bool(signals.get("retrieval_ok"))
    tokens = int(signals.get("retrieved_tokens") or 0)
    cut = bool(signals.get("context_cut"))
    cited = bool(signals.get("has_citation"))
    said_no = bool(signals.get("abstained"))
    outcome = signals.get("outcome") or "success"
    multi_hop = bool(signals.get("multi_hop"))
    stop = signals.get("stop_reason")

    context = 0.0
    if retrieval_ok:
        context += 0.45
    if tokens >= 20:
        context += 0.35
    elif tokens > 0:
        context += 0.15
    if not cut:
        context += 0.2

    faith = 0.0
    if said_no:
        faith += 0.55  # honest abstain beats invented facts
    if cited:
        faith += 0.45
    elif retrieval_ok and tokens >= 20 and not said_no:
        faith += 0.15  # had evidence but weak grounding signal

    relevance = 0.0
    if outcome == "success":
        relevance += 0.55
    elif outcome in {"tool_refused"}:
        relevance += 0.25
    if stop not in {"token_ceiling", "injection"}:
        relevance += 0.2
    if multi_hop:
        relevance += 0.15
    elif retrieval_ok:
        relevance += 0.1
    if cited or said_no:
        relevance += 0.1

    return {
        "context_relevance": round(min(1.0, context), 3),
        "answer_faithfulness": round(min(1.0, faith), 3),
        "answer_relevance": round(min(1.0, relevance), 3),
    }


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def diagnose(scores: dict) -> dict:
    bands = {key: _band(float(scores.get(key) or 0)) for key in (
        "context_relevance",
        "answer_faithfulness",
        "answer_relevance",
    )}
    for code, pattern in DIAGNOSIS_PATTERNS.items():
        if (
            bands["context_relevance"] == pattern["context_relevance"]
            and bands["answer_faithfulness"] == pattern["answer_faithfulness"]
            and bands["answer_relevance"] == pattern["answer_relevance"]
        ):
            return {"code": code, "label": pattern["label"], "bands": bands}
    # Nearest healthy / failure fallback
    if all(band == "high" for band in bands.values()):
        return {"code": "E", "label": DIAGNOSIS_PATTERNS["E"]["label"], "bands": bands}
    if bands["context_relevance"] == "low":
        return {"code": "A" if bands["answer_faithfulness"] == "low" else "C", "label": DIAGNOSIS_PATTERNS["A" if bands["answer_faithfulness"] == "low" else "C"]["label"], "bands": bands}
    if bands["answer_faithfulness"] == "low":
        return {"code": "B", "label": DIAGNOSIS_PATTERNS["B"]["label"], "bands": bands}
    if bands["answer_relevance"] == "low":
        return {"code": "D", "label": DIAGNOSIS_PATTERNS["D"]["label"], "bands": bands}
    return {"code": "E", "label": DIAGNOSIS_PATTERNS["E"]["label"], "bands": bands}


def fallback_signals(trace: dict) -> dict | None:
    """Build signals for older traces that lack rag_triage annotations."""
    tools = set(trace.get("tools") or [])
    if not tools.intersection(RETRIEVAL_TOOLS):
        return None
    if trace.get("rag_triage"):
        return trace["rag_triage"]
    tool_status = {item.get("name"): item.get("status") for item in (trace.get("tool_status") or [])}
    retrieval_names = tools.intersection(RETRIEVAL_TOOLS)
    budget = trace.get("context_budget") or {}
    cut = budget.get("cut") or []
    cut_retrieved = bool(cut) if isinstance(cut, bool) else "retrieved" in cut
    signals = {
        "retrieval": True,
        "retrieval_ok": all(tool_status.get(name) == "ok" for name in retrieval_names),
        "retrieved_tokens": int(budget.get("retrieved") or 0),
        "context_cut": cut_retrieved,
        "has_citation": False,
        "abstained": False,
        "multi_hop": int(trace.get("loop_count") or 0) > 1 or len(tools) > 1,
        "outcome": trace.get("outcome") or "success",
        "stop_reason": trace.get("stop_reason"),
    }
    signals["scores"] = score_turn(signals)
    return signals


def aggregate_triad(traces: list[dict]) -> dict:
    """Legacy proxy rollup over all retrieval turns (kept for citation/abstain rates)."""
    turns = []
    for trace in traces:
        signals = fallback_signals(trace)
        if signals:
            turns.append(signals)

    context_scores = [float((t.get("scores") or {}).get("context_relevance") or 0) for t in turns]
    faith_scores = [float((t.get("scores") or {}).get("answer_faithfulness") or 0) for t in turns]
    relevance_scores = [float((t.get("scores") or {}).get("answer_relevance") or 0) for t in turns]
    scores = {
        "context_relevance": _avg(context_scores),
        "answer_faithfulness": _avg(faith_scores),
        "answer_relevance": _avg(relevance_scores),
    }
    return {
        "method": "proxy",
        "note": "Operational proxies from retrieval turns. Prefer golden-set triad scores in the RAG triage panel.",
        "scores": scores,
        "bands": {key: _band(value) for key, value in scores.items()},
        "diagnosis": diagnose(scores) if turns else {"code": "—", "label": "No retrieval turns yet", "bands": {}},
        "patterns": DIAGNOSIS_PATTERNS,
        "playbook": PLAYBOOK,
        "samples": len(turns),
        "citation_rate": _avg([1.0 if t.get("has_citation") else 0.0 for t in turns]),
        "abstain_rate": _avg([1.0 if t.get("abstained") else 0.0 for t in turns]),
        "multi_hop_rate": _avg([1.0 if t.get("multi_hop") else 0.0 for t in turns]),
    }


FOCUS_KEYS = ("context_relevance", "answer_faithfulness", "answer_relevance")


def score_rag_golden_case(case: dict, trace: dict) -> dict:
    """Score one RAG golden case against a matched production/demo trace."""
    from app.evals.metrics import score_trace_against_case

    focus = case.get("eval_focus") or "contract"
    contract = score_trace_against_case(case, trace)
    signals = fallback_signals(trace) or {}
    triad_scores = (signals.get("scores") or score_turn(signals)) if signals else {
        "context_relevance": 0.0,
        "answer_faithfulness": 0.0,
        "answer_relevance": 0.0,
    }

    # Dimension-specific expectations from the RAG golden label.
    if focus == "context_relevance":
        dim = float(triad_scores.get("context_relevance") or 0)
        if signals.get("retrieval_ok") and int(signals.get("retrieved_tokens") or 0) >= 20:
            dim = max(dim, 0.7)
        if signals.get("context_cut"):
            dim = min(dim, 0.45)
    elif focus == "answer_faithfulness":
        if case.get("family") == "NEG" or "abstain" in (case.get("id") or ""):
            dim = 1.0 if signals.get("abstained") else 0.15
        elif "citation" in (case.get("id") or "") or "citations" in (case.get("input") or "").lower():
            dim = 1.0 if signals.get("has_citation") else float(triad_scores.get("answer_faithfulness") or 0) * 0.4
        else:
            dim = float(triad_scores.get("answer_faithfulness") or 0)
    elif focus == "answer_relevance":
        dim = float(triad_scores.get("answer_relevance") or 0)
        if contract.get("passed"):
            dim = max(dim, 0.75)
        else:
            dim = min(dim, 0.4)
    else:
        dim = float(contract.get("score") or 0)

    # Blend contract correctness with the focused triad dimension.
    score = round(0.4 * float(contract.get("score") or 0) + 0.6 * dim, 3)
    return {
        "case_id": case.get("id"),
        "focus": focus,
        "score": score,
        "passed": score >= 0.7 and bool(contract.get("passed")),
        "family": case.get("family"),
        "severity": case.get("severity"),
        "source": case.get("source"),
        "input": case.get("input"),
        "notes": case.get("notes"),
        "trace_id": trace.get("id"),
        "started_at": trace.get("started_at"),
        "day": (trace.get("started_at") or "")[:10],
    }


def aggregate_rag_golden_triad(
    cases: list[dict],
    traces: list[dict],
    durable_points: list[dict] | None = None,
) -> dict:
    """Score the triad against RAG golden cases (eval_focus = triad edge)."""
    from app.evals.metrics import prompts_match

    rag_cases = [
        case
        for case in cases
        if (case.get("eval_focus") or "") in FOCUS_KEYS
    ]
    # Match latest trace per case (prefer live traces, then durable score points).
    case_results: dict[str, dict] = {}
    for case in rag_cases:
        best = None
        for trace in traces:
            if not prompts_match(case.get("input") or "", trace.get("question") or ""):
                continue
            point = score_rag_golden_case(case, trace)
            if not best or (point.get("started_at") or "") >= (best.get("started_at") or ""):
                best = point
        if best:
            case_results[case["id"]] = best

    # Durable points may already encode RAG scores with focus.
    for point in durable_points or []:
        case_id = point.get("case_id")
        focus = point.get("focus") or point.get("eval_focus")
        if not case_id or focus not in FOCUS_KEYS:
            continue
        existing = case_results.get(case_id)
        if not existing or (point.get("started_at") or "") >= (existing.get("started_at") or ""):
            case_results[case_id] = {
                "case_id": case_id,
                "focus": focus,
                "score": float(point.get("score") or 0),
                "passed": bool(point.get("passed")),
                "family": point.get("family"),
                "severity": point.get("severity"),
                "source": point.get("source"),
                "input": point.get("input"),
                "notes": point.get("notes"),
                "trace_id": point.get("trace_id"),
                "started_at": point.get("started_at"),
                "day": point.get("day") or (point.get("started_at") or "")[:10],
            }

    # Fill unmatched labeled cases as unscored placeholders (count in coverage).
    for case in rag_cases:
        if case["id"] in case_results:
            continue
        case_results[case["id"]] = {
            "case_id": case["id"],
            "focus": case.get("eval_focus"),
            "score": None,
            "passed": False,
            "family": case.get("family"),
            "severity": case.get("severity"),
            "source": case.get("source"),
            "input": case.get("input"),
            "notes": case.get("notes"),
            "trace_id": None,
            "started_at": None,
            "day": None,
            "unscored": True,
        }

    by_focus: dict[str, list[float]] = {key: [] for key in FOCUS_KEYS}
    scored_cases = []
    for row in case_results.values():
        scored_cases.append(row)
        if row.get("score") is None:
            continue
        focus = row.get("focus")
        if focus in by_focus:
            by_focus[focus].append(float(row["score"]))

    scores = {key: _avg(vals) for key, vals in by_focus.items()}
    return {
        "method": "rag_golden_set",
        "note": (
            "The three scores are averages over the RAG golden set: labeled cases for "
            "context relevance, faithfulness, and answer relevance. Match production/demo "
            "traces to those prompts, then score the focused triad edge."
        ),
        "scores": scores,
        "bands": {key: _band(value) for key, value in scores.items()},
        "diagnosis": diagnose(scores) if any(by_focus.values()) else {
            "code": "—",
            "label": "No scored RAG golden cases yet",
            "bands": {},
        },
        "patterns": DIAGNOSIS_PATTERNS,
        "playbook": PLAYBOOK,
        "playbook_explained": (
            "Not a score. When a triad metric is weak, these are the engineering fixes to try "
            "(hybrid search, rewrite, CoVe, abstain, citations, role prompts, multi-step). "
            "Status shows what this product already ships."
        ),
        "cases": sorted(scored_cases, key=lambda row: (row.get("focus") or "", row.get("case_id") or "")),
        "cases_total": len(rag_cases),
        "cases_scored": sum(1 for row in scored_cases if row.get("score") is not None),
        "samples": sum(len(vals) for vals in by_focus.values()),
        "by_focus_n": {key: len(vals) for key, vals in by_focus.items()},
    }
