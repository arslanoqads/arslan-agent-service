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
    """Roll retrieval turns into triad averages + playbook for the dashboard."""
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
        "note": "Proxy scores from retrieval traces (citations, abstains, cuts, outcomes). Replace with LLM-judge/human labels for golden-set gates.",
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
