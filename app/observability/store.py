import json
import os
import sqlite3
from pathlib import Path
from threading import Lock


class MemoryTraceStore:
    """Process-local ring buffer so recent (including failed) traces stay listable."""

    def __init__(self, limit: int = 200):
        self.limit = limit
        self._lock = Lock()
        self._items: dict[str, dict] = {}
        self._order: list[str] = []

    def save(self, trace: dict) -> None:
        trace_id = trace.get("id")
        if not trace_id:
            return
        with self._lock:
            self._items[trace_id] = dict(trace)
            if trace_id in self._order:
                self._order.remove(trace_id)
            self._order.append(trace_id)
            while len(self._order) > self.limit:
                old = self._order.pop(0)
                self._items.pop(old, None)

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            item = self._items.get(trace_id)
            return dict(item) if item else None

    def list_traces(self, limit: int = 50) -> list[dict]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [dict(self._items[trace_id]) for trace_id in ids if trace_id in self._items]


class SqliteTraceStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    started_at TEXT,
                    status TEXT,
                    payload TEXT NOT NULL
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.path)

    def save(self, trace: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (id, started_at, status, payload) VALUES (?, ?, ?, ?)",
                (trace["id"], trace.get("started_at"), trace.get("status"), json.dumps(trace)),
            )

    def get(self, trace_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload FROM traces WHERE id = ?", (trace_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_traces(self, limit: int = 50) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM traces ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]


class FirestoreTraceStore:
    def __init__(self):
        from google.cloud import firestore

        self.client = firestore.Client()
        self.collection = self.client.collection("agent_traces")

    def save(self, trace: dict) -> None:
        payload = dict(trace)
        for key in ("started_at", "ended_at"):
            value = payload.get(key)
            if value is not None and not isinstance(value, str) and hasattr(value, "isoformat"):
                payload[key] = value.isoformat()
        for span in payload.get("spans") or []:
            for key in ("started_at", "ended_at"):
                value = span.get(key)
                if value is not None and not isinstance(value, str) and hasattr(value, "isoformat"):
                    span[key] = value.isoformat()
        # Drop non-JSON-safe leftovers before write.
        self.collection.document(trace["id"]).set(json.loads(json.dumps(payload, default=str)))

    def get(self, trace_id: str) -> dict | None:
        try:
            snap = self.collection.document(trace_id).get()
        except Exception:
            return None
        return snap.to_dict() if snap.exists else None

    def list_traces(self, limit: int = 50) -> list[dict]:
        try:
            docs = list(self.collection.limit(max(limit, 100)).stream())
        except Exception:
            return []
        items = []
        for doc in docs:
            data = doc.to_dict() or {}
            if not data.get("id"):
                data["id"] = doc.id
            items.append(data)
        items.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return items[:limit]


class CompositeTraceStore:
    """Always write memory; also write durable backend when available."""

    def __init__(self, durable, memory: MemoryTraceStore | None = None):
        self.durable = durable
        self.memory = memory or MemoryTraceStore()
        self.backend = type(durable).__name__

    def save(self, trace: dict) -> None:
        self.memory.save(trace)
        try:
            self.durable.save(trace)
        except Exception:
            return

    def get(self, trace_id: str) -> dict | None:
        try:
            found = self.durable.get(trace_id)
            if found:
                return found
        except Exception:
            pass
        return self.memory.get(trace_id)

    def list_traces(self, limit: int = 50) -> list[dict]:
        durable_items: list[dict] = []
        try:
            durable_items = self.durable.list_traces(limit) or []
        except Exception:
            durable_items = []
        memory_items = self.memory.list_traces(limit)
        merged: dict[str, dict] = {}
        for item in durable_items + memory_items:
            trace_id = item.get("id")
            if not trace_id:
                continue
            existing = merged.get(trace_id)
            if not existing or (item.get("started_at") or "") >= (existing.get("started_at") or ""):
                merged[trace_id] = item
        items = list(merged.values())
        items.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return items[:limit]


_store = None
_memory = MemoryTraceStore()


def get_store():
    global _store
    if _store is not None:
        return _store
    if os.getenv("K_SERVICE") or os.getenv("TRACE_BACKEND") == "firestore":
        try:
            _store = CompositeTraceStore(FirestoreTraceStore(), _memory)
            return _store
        except Exception:
            durable = SqliteTraceStore(os.getenv("TRACE_SQLITE_PATH", "/tmp/traces.sqlite"))
            _store = CompositeTraceStore(durable, _memory)
            return _store
    path = os.getenv("TRACE_SQLITE_PATH", "data/traces.sqlite")
    _store = CompositeTraceStore(SqliteTraceStore(path), _memory)
    return _store


def _counts(traces: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in traces:
        name = item.get(field)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _per_success(traces: list[dict]) -> float:
    ok = [item for item in traces if (item.get("outcome") or "success") == "success"]
    if not ok:
        return 0
    return round(sum(item.get("cost_usd") or 0 for item in ok) / len(ok), 6)


def classify_outcome(trace: dict) -> str:
    if trace.get("outcome"):
        return trace["outcome"]
    if trace.get("status") == "error":
        return "processing_error"
    if trace.get("error_kind") == "guardrail" or (trace.get("stop_reason") or "").startswith(
        ("budget", "injection")
    ):
        return "guardrail"
    tool_statuses = [item.get("status") for item in trace.get("tool_status") or []]
    if "error" in tool_statuses:
        return "tool_error"
    if "refused" in tool_statuses:
        return "tool_refused"
    if trace.get("status") == "running":
        return "running"
    return "success"


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return round(ordered[index], 2)


def _ttfts(trace: dict) -> list[float]:
    values = []
    for span in trace.get("spans") or []:
        if span.get("kind") == "llm" and span.get("ttft_ms") is not None:
            values.append(float(span["ttft_ms"]))
    return values


def conversation_metrics(traces: list[dict]) -> dict:
    """Session-aware dashboard metrics for the observability page."""
    finished = [item for item in traces if item.get("status") != "running"]
    sessions: dict[str, list[dict]] = {}
    for item in finished:
        key = item.get("thread_id") or item.get("id") or "unknown"
        sessions.setdefault(key, []).append(item)

    tool_count_hist = {"0": 0, "1": 0, "2": 0, "3+": 0}
    session_rows = []
    tool_failures = {}
    outcomes = {}
    latencies = []
    loops = []
    retries = []
    ttfts = []
    input_tokens = []
    output_tokens = []
    costs = []
    rag_calls = 0
    rag_retrieved = []
    rag_cuts = 0
    turns_with_retrieval = 0

    for session_id, turns in sessions.items():
        tools_unique = set()
        tool_calls = 0
        failed_tools = 0
        session_in = 0
        session_out = 0
        session_cost = 0.0
        session_latency = 0
        session_loops = 0
        session_retries = 0
        session_ttft = []
        for turn in turns:
            names = turn.get("tools") or []
            tools_unique.update(names)
            tool_calls += len(names)
            for status in turn.get("tool_status") or []:
                if status.get("status") == "error":
                    failed_tools += 1
                    tool_name = status.get("name") or "unknown"
                    tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1
            outcome = classify_outcome(turn)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            latencies.append(float(turn.get("duration_ms") or 0))
            loops.append(float(turn.get("loop_count") or 0))
            attempt = float(turn.get("attempt") or 0)
            retries.append(max(0.0, attempt - 1))
            session_retries += max(0, int(attempt) - 1)
            session_latency += int(turn.get("duration_ms") or 0)
            session_loops += int(turn.get("loop_count") or 0)
            session_in += int(turn.get("input_tokens") or 0)
            session_out += int(turn.get("output_tokens") or 0)
            session_cost += float(turn.get("cost_usd") or 0)
            input_tokens.append(float(turn.get("input_tokens") or 0))
            output_tokens.append(float(turn.get("output_tokens") or 0))
            costs.append(float(turn.get("cost_usd") or 0))
            ttft_vals = _ttfts(turn)
            ttfts.extend(ttft_vals)
            session_ttft.extend(ttft_vals)
            budget = turn.get("context_budget") or {}
            retrieval_tools = {"query_arslan_profile", "match_role_evidence"}
            if retrieval_tools.intersection(names):
                rag_calls += 1
                turns_with_retrieval += 1
                rag_retrieved.append(float(budget.get("retrieved") or 0))
                if budget.get("cut"):
                    rag_cuts += 1
        count_key = str(len(tools_unique)) if len(tools_unique) < 3 else "3+"
        if count_key not in tool_count_hist:
            count_key = "3+"
        tool_count_hist[count_key] = tool_count_hist.get(count_key, 0) + 1
        session_rows.append(
            {
                "thread_id": session_id,
                "turns": len(turns),
                "tools_unique": len(tools_unique),
                "tool_calls": tool_calls,
                "tool_failures": failed_tools,
                "input_tokens": session_in,
                "output_tokens": session_out,
                "tokens": session_in + session_out,
                "cost_usd": round(session_cost, 6),
                "latency_ms": session_latency,
                "loops": session_loops,
                "retries": session_retries,
                "avg_ttft_ms": _avg(session_ttft),
            }
        )

    session_rows.sort(key=lambda row: row.get("cost_usd") or 0, reverse=True)
    return {
        "totals": {
            "sessions": len(sessions),
            "turns": len(finished),
            "tool_failures": sum(tool_failures.values()),
            "input_tokens": int(sum(input_tokens)),
            "output_tokens": int(sum(output_tokens)),
            "tokens": int(sum(input_tokens) + sum(output_tokens)),
            "cost_usd": round(sum(costs), 6),
        },
        "tools_per_session": tool_count_hist,
        "outcomes": outcomes,
        "tool_failures": tool_failures,
        "latency": {
            "avg_ms": _avg(latencies),
            "p50_ms": _percentile(latencies, 50),
            "p95_ms": _percentile(latencies, 95),
        },
        "ttft": {
            "avg_ms": _avg(ttfts),
            "p50_ms": _percentile(ttfts, 50),
            "p95_ms": _percentile(ttfts, 95),
            "samples": len(ttfts),
        },
        "loops": {"avg": _avg(loops), "p95": _percentile(loops, 95)},
        "retries": {"avg": _avg(retries), "total": int(sum(retries))},
        "tokens": {
            "avg_input": _avg(input_tokens),
            "avg_output": _avg(output_tokens),
            "avg_total": _avg([a + b for a, b in zip(input_tokens, output_tokens)]) if finished else 0,
        },
        "cost": {
            "avg_per_turn": _avg(costs),
            "avg_per_session": _avg([row["cost_usd"] for row in session_rows]),
            "total": round(sum(costs), 6),
        },
        "rag": {
            "retrieval_turns": turns_with_retrieval,
            "retrieval_rate": round(turns_with_retrieval / len(finished), 3) if finished else 0,
            "avg_retrieved_tokens": _avg(rag_retrieved),
            "context_cut_rate": round(rag_cuts / turns_with_retrieval, 3) if turns_with_retrieval else 0,
            "rag_tool_calls": rag_calls,
        },
        "sessions": session_rows[:50],
    }


def summary(traces: list[dict]) -> dict:
    finished = [item for item in traces if item.get("status") != "running"]
    count = len(finished)
    empty = {
        "requests": 0,
        "avg_duration_ms": 0,
        "avg_tokens": 0,
        "failure_rate": 0,
        "tools": {},
        "error_kinds": {},
        "stop_reasons": {},
        "outcomes": {},
        "cost_per_request": 0,
        "cost_per_success": 0,
        "unknown_routes": 0,
        "conversation": conversation_metrics([]),
    }
    if not count:
        return empty
    durations = [item.get("duration_ms") or 0 for item in finished]
    tokens = [
        (item.get("input_tokens") or 0) + (item.get("output_tokens") or 0) for item in finished
    ]
    outcomes = {}
    failures = 0
    for item in finished:
        outcome = classify_outcome(item)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome in {"processing_error", "tool_error"}:
            failures += 1
    tools: dict[str, int] = {}
    for item in finished:
        for name in item.get("tools") or []:
            tools[name] = tools.get(name, 0) + 1
    return {
        "requests": count,
        "avg_duration_ms": round(sum(durations) / count),
        "avg_tokens": round(sum(tokens) / count),
        "failure_rate": round(failures / count, 3),
        "tools": tools,
        "error_kinds": _counts(finished, "error_kind"),
        "stop_reasons": _counts(finished, "stop_reason"),
        "outcomes": outcomes,
        "cost_per_request": round(sum(item.get("cost_usd") or 0 for item in finished) / count, 6),
        "cost_per_success": _per_success(finished),
        "unknown_routes": sum(1 for item in finished if (item.get("route") or "unknown") == "unknown"),
        "conversation": conversation_metrics(finished),
    }
