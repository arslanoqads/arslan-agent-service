import json
import os
import sqlite3
from pathlib import Path
from threading import Lock


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
        self.collection.document(trace["id"]).set(trace)

    def get(self, trace_id: str) -> dict | None:
        snap = self.collection.document(trace_id).get()
        return snap.to_dict() if snap.exists else None

    def list_traces(self, limit: int = 50) -> list[dict]:
        docs = (
            self.collection.order_by("started_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        return [doc.to_dict() for doc in docs]


def get_store():
    if os.getenv("K_SERVICE") or os.getenv("TRACE_BACKEND") == "firestore":
        try:
            return FirestoreTraceStore()
        except Exception:
            return SqliteTraceStore(os.getenv("TRACE_SQLITE_PATH", "/tmp/traces.sqlite"))
    path = os.getenv("TRACE_SQLITE_PATH", "data/traces.sqlite")
    return SqliteTraceStore(path)


def _counts(traces: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in traces:
        name = item.get(field)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _per_success(traces: list[dict]) -> float:
    ok = [item for item in traces if item.get("status") == "ok"]
    if not ok:
        return 0
    return round(sum(item.get("cost_usd") or 0 for item in ok) / len(ok), 6)


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
            "cost_per_request": 0,
            "cost_per_success": 0,
            "unknown_routes": 0,
        }
    durations = [item.get("duration_ms") or 0 for item in finished]
    tokens = [
        (item.get("input_tokens") or 0) + (item.get("output_tokens") or 0) for item in finished
    ]
    failures = sum(1 for item in finished if item.get("status") == "error")
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
        "cost_per_request": round(sum(item.get("cost_usd") or 0 for item in finished) / count, 6),
        "cost_per_success": _per_success(finished),
        "unknown_routes": sum(1 for item in finished if (item.get("route") or "unknown") == "unknown"),
    }
