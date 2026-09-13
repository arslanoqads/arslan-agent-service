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


def summary(traces: list[dict]) -> dict:
    finished = [item for item in traces if item.get("status") != "running"]
    count = len(finished)
    if not count:
        return {
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
        }
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
    }
