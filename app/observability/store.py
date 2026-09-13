import json
import logging
import os
import sqlite3
from pathlib import Path
from threading import Lock

from app.observability.rag_triad import aggregate_rag_golden_triad, aggregate_triad

logger = logging.getLogger(__name__)

DEFAULT_LIST_LIMIT = 250
MEMORY_LIMIT = 600


class MemoryTraceStore:
    """Process-local ring buffer so recent (including failed) traces stay listable."""

    def __init__(self, limit: int = MEMORY_LIMIT):
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

    def list_traces(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict]:
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

    def list_traces(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict]:
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
        payload = _json_safe(trace)
        self.collection.document(trace["id"]).set(payload)

    def get(self, trace_id: str) -> dict | None:
        try:
            snap = self.collection.document(trace_id).get()
        except Exception:
            return None
        return snap.to_dict() if snap.exists else None

    def list_traces(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict]:
        try:
            from google.cloud import firestore

            docs = list(
                self.collection.order_by("started_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream()
            )
        except Exception:
            # Fallback when the started_at index is missing or order_by fails.
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


class GcsTraceStore:
    """Durable JSON archive in the private RAG bucket — survives Cloud Run redeploys."""

    PREFIX = "observability/traces"

    def __init__(self, bucket_name: str | None = None):
        from google.cloud import storage

        self.bucket_name = bucket_name or os.getenv("RAG_GCS_BUCKET")
        if not self.bucket_name:
            raise RuntimeError("RAG_GCS_BUCKET is required for GCS trace archive")
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def _blob(self, trace_id: str):
        return self.bucket.blob(f"{self.PREFIX}/{trace_id}.json")

    def save(self, trace: dict) -> None:
        payload = _json_safe(trace)
        self._blob(trace["id"]).upload_from_string(
            json.dumps(payload),
            content_type="application/json",
        )

    def get(self, trace_id: str) -> dict | None:
        blob = self._blob(trace_id)
        try:
            if not blob.exists():
                return None
            return json.loads(blob.download_as_text())
        except Exception:
            return None

    def list_traces(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict]:
        items = []
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=f"{self.PREFIX}/")
            for blob in blobs:
                if not blob.name.endswith(".json"):
                    continue
                try:
                    data = json.loads(blob.download_as_text())
                except Exception:
                    continue
                if isinstance(data, dict):
                    if not data.get("id"):
                        data["id"] = Path(blob.name).stem
                    items.append(data)
        except Exception:
            return []
        items.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return items[:limit]


class CompositeTraceStore:
    """Memory plus one or more durable backends. Never drop a save just because one backend fails."""

    def __init__(self, durables: list, memory: MemoryTraceStore | None = None):
        if not isinstance(durables, list):
            durables = [durables]
        self.durables = [item for item in durables if item is not None]
        self.memory = memory or MemoryTraceStore()
        self.backend = "+".join(type(item).__name__ for item in self.durables) or "MemoryTraceStore"
        self.last_errors: dict[str, str] = {}
        self.write_counts: dict[str, int] = {type(item).__name__: 0 for item in self.durables}
        self.write_counts["MemoryTraceStore"] = 0

    @property
    def durable(self):
        return self.durables[0] if self.durables else None

    @durable.setter
    def durable(self, value) -> None:
        # Tests replace a single durable backend.
        self.durables = [value]
        self.backend = type(value).__name__
        self.write_counts[type(value).__name__] = self.write_counts.get(type(value).__name__, 0)

    def save(self, trace: dict) -> None:
        self.memory.save(trace)
        self.write_counts["MemoryTraceStore"] = self.write_counts.get("MemoryTraceStore", 0) + 1
        for backend in self.durables:
            name = type(backend).__name__
            try:
                backend.save(trace)
                self.write_counts[name] = self.write_counts.get(name, 0) + 1
                self.last_errors.pop(name, None)
            except Exception as exc:
                self.last_errors[name] = str(exc)
                logger.warning("trace durable save failed backend=%s error=%s", name, exc)

    def get(self, trace_id: str) -> dict | None:
        for backend in self.durables:
            try:
                found = backend.get(trace_id)
                if found:
                    return found
            except Exception as exc:
                self.last_errors[type(backend).__name__] = str(exc)
        return self.memory.get(trace_id)

    def list_traces(self, limit: int = DEFAULT_LIST_LIMIT) -> list[dict]:
        merged: dict[str, dict] = {}
        for backend in list(self.durables) + [self.memory]:
            name = type(backend).__name__
            try:
                items = backend.list_traces(limit) or []
            except Exception as exc:
                self.last_errors[name] = str(exc)
                logger.warning("trace list failed backend=%s error=%s", name, exc)
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                trace_id = item.get("id")
                if not trace_id:
                    continue
                existing = merged.get(trace_id)
                if not existing or (item.get("started_at") or "") >= (existing.get("started_at") or ""):
                    merged[trace_id] = item
        items = list(merged.values())
        items.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return items[:limit]

    def persistence_status(self) -> dict:
        return {
            "backend": self.backend,
            "durable_backends": [type(item).__name__ for item in self.durables],
            "write_counts": dict(self.write_counts),
            "last_errors": dict(self.last_errors),
            "ok": not self.last_errors,
        }


def _json_safe(trace: dict) -> dict:
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
    return json.loads(json.dumps(payload, default=str))


_store = None
_memory = MemoryTraceStore()


def _build_cloud_durables() -> list:
    durables = []
    try:
        durables.append(FirestoreTraceStore())
    except Exception as exc:
        logger.warning("firestore trace store unavailable: %s", exc)
    bucket = os.getenv("RAG_GCS_BUCKET")
    if bucket:
        try:
            durables.append(GcsTraceStore(bucket))
        except Exception as exc:
            logger.warning("gcs trace archive unavailable: %s", exc)
    return durables


def get_store():
    global _store
    if _store is not None:
        return _store
    if os.getenv("K_SERVICE") or os.getenv("TRACE_BACKEND") == "firestore":
        durables = _build_cloud_durables()
        if durables:
            _store = CompositeTraceStore(durables, _memory)
            return _store
        # Last resort on Cloud Run — still better than memory-only, but /tmp is ephemeral.
        durable = SqliteTraceStore(os.getenv("TRACE_SQLITE_PATH", "/tmp/traces.sqlite"))
        _store = CompositeTraceStore([durable], _memory)
        return _store
    if os.getenv("TRACE_BACKEND") == "sqlite" or os.getenv("TRACE_SQLITE_PATH"):
        path = os.getenv("TRACE_SQLITE_PATH", "data/traces.sqlite")
        _store = CompositeTraceStore([SqliteTraceStore(path)], _memory)
        return _store
    path = os.getenv("TRACE_SQLITE_PATH", "data/traces.sqlite")
    durables: list = [SqliteTraceStore(path)]
    if os.getenv("RAG_GCS_BUCKET"):
        try:
            durables.append(GcsTraceStore())
        except Exception:
            pass
    _store = CompositeTraceStore(durables, _memory)
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


CONVERSION_TOOLS = {"send_resume_email", "schedule_intro_call"}
TRACKED_SUCCESS_TOOLS = (
    "send_resume_email",
    "schedule_intro_call",
    "match_role_evidence",
    "query_arslan_profile",
    "get_social_links",
)
RETRIEVAL_TOOLS = {"query_arslan_profile", "match_role_evidence"}
SLOW_TURN_MS = 3500


def _successful_tools(turn: dict) -> set[str]:
    names = set()
    for status in turn.get("tool_status") or []:
        if status.get("status") == "ok" and status.get("name"):
            names.add(status["name"])
    return names


def _cache_kind(turn: dict) -> str | None:
    cache = turn.get("cache") or {}
    kind = str(cache.get("kind") or "").lower()
    if kind in {"exact", "semantic"}:
        return kind
    stop = (turn.get("stop_reason") or "").lower()
    if stop.startswith("cache_exact") or stop == "exact_cache":
        return "exact"
    if stop.startswith("cache_semantic") or stop == "semantic_cache":
        return "semantic"
    return None


def _cache_metrics(finished: list[dict]) -> dict:
    """Exact/semantic cache hit rates and estimated USD avoided."""
    exact_hits = 0
    semantic_hits = 0
    uncached_costs: list[float] = []
    stamped_exact = 0.0
    stamped_semantic = 0.0
    stamped_exact_n = 0
    stamped_semantic_n = 0
    for turn in finished:
        kind = _cache_kind(turn)
        cache = turn.get("cache") or {}
        avoided = cache.get("avoided_cost_usd")
        if kind == "exact":
            exact_hits += 1
            if avoided is not None:
                stamped_exact += float(avoided)
                stamped_exact_n += 1
        elif kind == "semantic":
            semantic_hits += 1
            if avoided is not None:
                stamped_semantic += float(avoided)
                stamped_semantic_n += 1
        else:
            cost = float(turn.get("cost_usd") or 0)
            if cost > 0:
                uncached_costs.append(cost)

    turn_count = len(finished)
    hits = exact_hits + semantic_hits
    avg_uncached = _avg(uncached_costs)
    # Prefer per-hit stamped savings when present; else hits × avg uncached turn cost.
    exact_savings = (
        round(stamped_exact, 6)
        if stamped_exact_n == exact_hits and exact_hits
        else round(exact_hits * avg_uncached, 6)
    )
    semantic_savings = (
        round(stamped_semantic, 6)
        if stamped_semantic_n == semantic_hits and semantic_hits
        else round(semantic_hits * avg_uncached, 6)
    )
    return {
        "turns": turn_count,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "hits": hits,
        "uncached_turns": max(0, turn_count - hits),
        "exact_rate": round(exact_hits / turn_count, 3) if turn_count else 0.0,
        "semantic_rate": round(semantic_hits / turn_count, 3) if turn_count else 0.0,
        "hit_rate": round(hits / turn_count, 3) if turn_count else 0.0,
        "avg_uncached_cost_usd": avg_uncached,
        "exact_savings_usd": exact_savings,
        "semantic_savings_usd": semantic_savings,
        "estimated_savings_usd": round(exact_savings + semantic_savings, 6),
        "note": (
            "Cache rates are share of finished turns served from exact or semantic answer cache. "
            "Savings estimate avoided LLM spend as hits × average cost of uncached turns "
            "(or stamped avoided_cost_usd when present)."
        ),
    }


def _session_business(turns: list[dict], latency_p95: float) -> dict:
    successful: set[str] = set()
    tool_failure = False
    hit_budget = False
    injection = False
    slow = False
    rag_weak = False
    retrieval_used = False
    for turn in turns:
        successful |= _successful_tools(turn)
        for status in turn.get("tool_status") or []:
            if status.get("status") == "error":
                tool_failure = True
        stop = (turn.get("stop_reason") or "").lower()
        if stop.startswith("budget") or stop == "budget":
            hit_budget = True
        if stop.startswith("injection") or (
            turn.get("error_kind") == "guardrail" and "injection" in stop
        ):
            injection = True
        if float(turn.get("duration_ms") or 0) >= max(SLOW_TURN_MS, latency_p95):
            slow = True
        tools = set(turn.get("tools") or [])
        if tools & RETRIEVAL_TOOLS:
            retrieval_used = True
            triage = turn.get("rag_triage") or {}
            scores = triage.get("scores") or {}
            budget = turn.get("context_budget") or {}
            cut = budget.get("cut") or []
            cut_retrieved = bool(cut) if isinstance(cut, bool) else "retrieved" in cut
            if (
                cut_retrieved
                or float(scores.get("context_relevance") or 1) < 0.45
                or float(scores.get("answer_faithfulness") or 1) < 0.45
                or float(scores.get("answer_relevance") or 1) < 0.45
                or (triage and not triage.get("retrieval_ok", True))
            ):
                rag_weak = True

    resume = "send_resume_email" in successful
    appointment = "schedule_intro_call" in successful
    converted = resume or appointment
    browse_only = (
        not converted
        and not hit_budget
        and not tool_failure
        and not injection
        and not slow
        and not rag_weak
    )
    return {
        "converted": converted,
        "resume": resume,
        "appointment": appointment,
        "both": resume and appointment,
        "successful_tools": successful,
        "hit_budget": hit_budget,
        "tool_failure": tool_failure,
        "slow": slow,
        "rag_weak": rag_weak and not converted,
        "injection": injection,
        "browse_only": browse_only,
        "retrieval_used": retrieval_used,
    }


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

    # First pass for latency p95 used in slow-session attribution.
    for turns in sessions.values():
        for turn in turns:
            latencies.append(float(turn.get("duration_ms") or 0))
    latency_p95 = _percentile(latencies, 95) if latencies else float(SLOW_TURN_MS)
    latencies = []

    converted = 0
    resume_ok = 0
    appointment_ok = 0
    both_ok = 0
    why_not = {
        "hit_message_budget": 0,
        "tool_failure": 0,
        "slow_latency": 0,
        "rag_weak": 0,
        "injection_or_guardrail": 0,
        "browse_only": 0,
    }
    success_tool_sessions = {name: 0 for name in TRACKED_SUCCESS_TOOLS}
    successful_tool_counts = []
    budget_hits_total = 0

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
            if RETRIEVAL_TOOLS.intersection(names):
                rag_calls += 1
                turns_with_retrieval += 1
                rag_retrieved.append(float(budget.get("retrieved") or 0))
                if budget.get("cut"):
                    rag_cuts += 1

        biz = _session_business(turns, latency_p95)
        if biz["converted"]:
            converted += 1
        if biz["resume"]:
            resume_ok += 1
        if biz["appointment"]:
            appointment_ok += 1
        if biz["both"]:
            both_ok += 1
        if biz["hit_budget"]:
            budget_hits_total += 1
        if not biz["converted"]:
            if biz["hit_budget"]:
                why_not["hit_message_budget"] += 1
            if biz["tool_failure"]:
                why_not["tool_failure"] += 1
            if biz["slow"]:
                why_not["slow_latency"] += 1
            if biz["rag_weak"]:
                why_not["rag_weak"] += 1
            if biz["injection"]:
                why_not["injection_or_guardrail"] += 1
            if biz["browse_only"]:
                why_not["browse_only"] += 1

        for name in TRACKED_SUCCESS_TOOLS:
            if name in biz["successful_tools"]:
                success_tool_sessions[name] += 1
        successful_tool_counts.append(len(biz["successful_tools"]))

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
                "converted": biz["converted"],
                "resume": biz["resume"],
                "appointment": biz["appointment"],
            }
        )

    session_count = len(sessions) or 0
    non_converted = max(0, session_count - converted)
    why_not_rates = {
        key: round(value / non_converted, 3) if non_converted else 0.0
        for key, value in why_not.items()
    }
    successful_tool_mix = {
        name: {
            "sessions": success_tool_sessions[name],
            "rate": round(success_tool_sessions[name] / session_count, 3) if session_count else 0.0,
        }
        for name in TRACKED_SUCCESS_TOOLS
    }

    session_rows.sort(key=lambda row: row.get("cost_usd") or 0, reverse=True)
    return {
        "totals": {
            "sessions": session_count,
            "turns": len(finished),
            "tool_failures": sum(tool_failures.values()),
            "input_tokens": int(sum(input_tokens)),
            "output_tokens": int(sum(output_tokens)),
            "tokens": int(sum(input_tokens) + sum(output_tokens)),
            "cost_usd": round(sum(costs), 6),
        },
        "business": {
            "primary": "resume_or_appointment",
            "note": (
                "Primary success = session where send_resume_email or schedule_intro_call "
                "completed successfully. Secondary metrics explain non-conversion."
            ),
            "sessions": session_count,
            "converted_sessions": converted,
            "non_converted_sessions": non_converted,
            "conversion_rate": round(converted / session_count, 3) if session_count else 0.0,
            "resume_sessions": resume_ok,
            "appointment_sessions": appointment_ok,
            "both_sessions": both_ok,
            "resume_rate": round(resume_ok / session_count, 3) if session_count else 0.0,
            "appointment_rate": round(appointment_ok / session_count, 3) if session_count else 0.0,
            "both_rate": round(both_ok / session_count, 3) if session_count else 0.0,
            "hit_message_budget_sessions": budget_hits_total,
            "hit_message_budget_rate": round(budget_hits_total / session_count, 3) if session_count else 0.0,
            "why_not_converted": why_not,
            "why_not_converted_rates": why_not_rates,
            "successful_tool_mix": successful_tool_mix,
            "avg_successful_tools_per_session": _avg([float(v) for v in successful_tool_counts]),
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
        "cache": _cache_metrics(finished),
        "rag": {
            "retrieval_turns": turns_with_retrieval,
            "retrieval_rate": round(turns_with_retrieval / len(finished), 3) if finished else 0,
            "avg_retrieved_tokens": _avg(rag_retrieved),
            "context_cut_rate": round(rag_cuts / turns_with_retrieval, 3) if turns_with_retrieval else 0,
            "rag_tool_calls": rag_calls,
            "triad": _rag_triad_for_traces(finished),
        },
        "sessions": session_rows[:50],
    }


def _rag_triad_for_traces(traces: list[dict]) -> dict:
    """Prefer RAG golden-set triad scores; fall back to operational proxies."""
    try:
        from app.evals.durable import get_golden_store
        from app.evals.runner import load_public_cases

        cases = load_public_cases()
        durable = []
        try:
            durable = get_golden_store().list_scores()
        except Exception:
            durable = []
        golden = aggregate_rag_golden_triad(cases, traces, durable)
        proxy = aggregate_triad(traces)
        # Keep operational rates from proxy; overwrite scores with golden-set method.
        return {
            **proxy,
            **golden,
            "citation_rate": proxy.get("citation_rate"),
            "abstain_rate": proxy.get("abstain_rate"),
            "multi_hop_rate": proxy.get("multi_hop_rate"),
            "proxy_samples": proxy.get("samples"),
        }
    except Exception:
        return aggregate_triad(traces)


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
