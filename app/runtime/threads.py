"""Durable chat history so follow-ups survive Cloud Run instance hops."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from threading import Lock

from langchain_core.messages import AIMessage, HumanMessage

MAX_TURNS = 12


class MemoryThreadStore:
    def __init__(self):
        self._lock = Lock()
        self._threads: dict[str, list[dict]] = {}

    def load(self, thread_id: str) -> list[dict]:
        with self._lock:
            return list(self._threads.get(thread_id) or [])

    def save(self, thread_id: str, turns: list[dict]) -> None:
        with self._lock:
            self._threads[thread_id] = list(turns[-MAX_TURNS:])


class SqliteThreadStore:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.path)

    def load(self, thread_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            return []
        data = json.loads(row[0])
        return list(data.get("turns") or [])

    def save(self, thread_id: str, turns: list[dict]) -> None:
        payload = json.dumps({"turns": list(turns[-MAX_TURNS:])})
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO threads (id, payload) VALUES (?, ?)",
                (thread_id, payload),
            )


class FirestoreThreadStore:
    def __init__(self):
        from google.cloud import firestore

        self.client = firestore.Client()
        self.collection = self.client.collection("agent_threads")

    def load(self, thread_id: str) -> list[dict]:
        try:
            snap = self.collection.document(thread_id).get()
        except Exception:
            return []
        if not snap.exists:
            return []
        data = snap.to_dict() or {}
        return list(data.get("turns") or [])

    def save(self, thread_id: str, turns: list[dict]) -> None:
        try:
            self.collection.document(thread_id).set({"turns": list(turns[-MAX_TURNS:])})
        except Exception:
            return


_store = None
_memory = MemoryThreadStore()


def get_thread_store():
    global _store
    if _store is not None:
        return _store
    if os.getenv("K_SERVICE") or os.getenv("TRACE_BACKEND") == "firestore":
        try:
            _store = FirestoreThreadStore()
            return _store
        except Exception:
            path = os.getenv("TRACE_SQLITE_PATH", "/tmp/traces.sqlite")
            _store = SqliteThreadStore(path.replace("traces.sqlite", "threads.sqlite"))
            return _store
    path = os.getenv("THREAD_SQLITE_PATH") or os.getenv("TRACE_SQLITE_PATH", "data/traces.sqlite")
    if path.endswith("traces.sqlite"):
        path = path.replace("traces.sqlite", "threads.sqlite")
    else:
        path = "data/threads.sqlite"
    _store = SqliteThreadStore(path)
    return _store


def load_turns(thread_id: str) -> list[dict]:
    turns = get_thread_store().load(thread_id)
    if turns:
        return turns
    return _memory.load(thread_id)


def save_turns(thread_id: str, turns: list[dict]) -> None:
    trimmed = list(turns[-MAX_TURNS:])
    _memory.save(thread_id, trimmed)
    try:
        get_thread_store().save(thread_id, trimmed)
    except Exception:
        return


def append_turn(thread_id: str, user_text: str, assistant_text: str) -> list[dict]:
    turns = load_turns(thread_id)
    turns.append({"role": "user", "content": user_text})
    turns.append({"role": "assistant", "content": assistant_text})
    save_turns(thread_id, turns)
    return turns


def turns_as_messages(turns: list[dict]) -> list:
    messages = []
    for turn in turns:
        role = (turn.get("role") or "").lower()
        content = turn.get("content") or ""
        if role in {"user", "human"}:
            messages.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
    return messages
