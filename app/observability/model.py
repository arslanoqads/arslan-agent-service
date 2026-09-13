import re
import uuid
from datetime import datetime, timezone

EMAIL_REDACTION = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_REDACTION = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_REDACTION = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

RETRIEVAL_TOOLS = {"query_arslan_profile", "match_role_evidence"}
ROUTER_NODES = {"supervisor"}

_BLOCKED_QUESTION_MARKERS = (
    "ignore previous",
    "system prompt",
    "jailbreak",
    "__import__",
    "os.system",
    "eval(",
    "exec(",
    "<script",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return str(value)


def new_trace(thread_id: str, question: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "thread_id": thread_id,
        "question": question,
        "status": "running",
        "started_at": now_iso(),
        "ended_at": None,
        "duration_ms": None,
        "loop_count": 0,
        "attempt": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tools": [],
        "error": None,
        "error_kind": None,
        "route": "unknown",
        "stop_reason": None,
        "pipeline_version": "1",
        "prompt_version": "1",
        "model": "gpt-4o",
        "cost_usd": 0.0,
        "cache": None,
        "context_budget": {},
        "tool_status": [],
        "spans": [],
    }


def new_span(
    *,
    name: str,
    kind: str,
    loop_index: int,
    attempt: int,
    parent_id: str | None,
    context: str = "",
    started_perf: float,
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "parent_id": parent_id,
        "name": name,
        "kind": kind,
        "started_at": now_iso(),
        "ended_at": None,
        "ttft_ms": None,
        "loop_index": loop_index,
        "attempt": attempt,
        "input_tokens": 0,
        "output_tokens": 0,
        "context": context,
        "output": "",
        "error": None,
        "status": "running",
        "duration_ms": None,
        "_started_perf": started_perf,
    }


def span_kind(name: str, default: str) -> str:
    if name in ROUTER_NODES:
        return "router"
    if name in RETRIEVAL_TOOLS:
        return "retrieval"
    return default


def close_span(span: dict, *, ended_perf: float, output: str = "", error: str | None = None) -> None:
    span["ended_at"] = now_iso()
    span["duration_ms"] = round((ended_perf - span.get("_started_perf", ended_perf)) * 1000)
    span["output"] = output
    span["error"] = error
    span["status"] = "error" if error else "ok"
    span.pop("_started_perf", None)


def persisted_trace(trace: dict) -> dict:
    saved = dict(trace)
    saved["spans"] = []
    for span in trace.get("spans") or []:
        item = {key: value for key, value in span.items() if key != "_started_perf"}
        saved["spans"].append(item)
    return saved


def redact(value: str) -> str:
    text = EMAIL_REDACTION.sub("[redacted-email]", value or "")
    text = PHONE_REDACTION.sub("[redacted-phone]", text)
    return SSN_REDACTION.sub("[redacted]", text)


def public_question(question: str) -> str:
    text = (question or "").strip()
    lowered = text.lower()
    if any(marker in lowered for marker in _BLOCKED_QUESTION_MARKERS):
        return "[redacted: blocked request]"
    cleaned = redact(text)
    if len(cleaned) > 140:
        cleaned = cleaned[:137].rstrip() + "..."
    return cleaned


def public_span(span: dict) -> dict:
    return {
        "name": redact(span.get("name") or ""),
        "kind": span.get("kind"),
        "status": span.get("status"),
        "loop_index": span.get("loop_index", 0),
        "attempt": span.get("attempt", 0),
        "ttft_ms": span.get("ttft_ms"),
        "duration_ms": span.get("duration_ms"),
        "input_tokens": span.get("input_tokens", 0),
        "output_tokens": span.get("output_tokens", 0),
    }


def public_trace(trace: dict) -> dict:
    """Sanitized trace for public chat and the public observability dashboard."""
    return {
        "id": trace.get("id") or str(uuid.uuid4()),
        "thread_id": trace.get("thread_id") or "",
        "started_at": _as_text(trace.get("started_at")),
        "ended_at": _as_text(trace.get("ended_at")),
        "status": trace.get("status") or "unknown",
        "question": public_question(trace.get("question") or ""),
        "loop_count": trace.get("loop_count", 0),
        "attempt": trace.get("attempt", 0),
        "input_tokens": trace.get("input_tokens", 0),
        "output_tokens": trace.get("output_tokens", 0),
        "duration_ms": trace.get("duration_ms"),
        "model": trace.get("model"),
        "pipeline_version": trace.get("pipeline_version"),
        "prompt_version": trace.get("prompt_version"),
        "route": trace.get("route") or "unknown",
        "stop_reason": trace.get("stop_reason"),
        "error_kind": trace.get("error_kind"),
        "cost_usd": trace.get("cost_usd") or 0.0,
        "tools": [redact(name) for name in trace.get("tools", []) or []],
        "tool_status": [
            {"name": redact(item.get("name") or ""), "status": item.get("status")}
            for item in trace.get("tool_status") or []
        ],
        "context_budget": {
            key: value
            for key, value in (trace.get("context_budget") or {}).items()
            if key in {"system", "tools", "retrieved", "history", "reserved", "used", "window", "cut"}
        },
        "cache": None
        if not trace.get("cache")
        else {
            "kind": trace["cache"].get("kind"),
            "similarity": trace["cache"].get("similarity"),
        },
        "spans": [public_span(span) for span in trace.get("spans", []) or []],
    }
