"""Durable golden-set extras that survive Cloud Run redeploys.

Packaged public cases live in app/evals/golden_set.public.json (shipped in the image).
Production-added / private extras are dual-written to Firestore + GCS when available.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

GCS_PREFIX = "observability/golden"
FIRESTORE_COLLECTION = "agent_golden_cases"


class MemoryGoldenStore:
    def __init__(self):
        self._lock = Lock()
        self._items: dict[str, dict] = {}

    def save(self, case: dict) -> None:
        case_id = case.get("id")
        if not case_id:
            return
        with self._lock:
            self._items[case_id] = dict(case)

    def list_cases(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._items.values()]


class FirestoreGoldenStore:
    def __init__(self):
        from google.cloud import firestore

        self.client = firestore.Client()
        self.collection = self.client.collection(FIRESTORE_COLLECTION)

    def save(self, case: dict) -> None:
        payload = json.loads(json.dumps(case, default=str))
        self.collection.document(case["id"]).set(payload)

    def list_cases(self) -> list[dict]:
        try:
            docs = list(self.collection.stream())
        except Exception:
            return []
        items = []
        for doc in docs:
            data = doc.to_dict() or {}
            if not data.get("id"):
                data["id"] = doc.id
            items.append(data)
        return items


class GcsGoldenStore:
    def __init__(self, bucket_name: str | None = None):
        from google.cloud import storage

        self.bucket_name = bucket_name or os.getenv("RAG_GCS_BUCKET")
        if not self.bucket_name:
            raise RuntimeError("RAG_GCS_BUCKET is required for GCS golden archive")
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

    def save(self, case: dict) -> None:
        payload = json.loads(json.dumps(case, default=str))
        blob = self.bucket.blob(f"{GCS_PREFIX}/{case['id']}.json")
        blob.upload_from_string(json.dumps(payload), content_type="application/json")

    def list_cases(self) -> list[dict]:
        items = []
        try:
            for blob in self.client.list_blobs(self.bucket_name, prefix=f"{GCS_PREFIX}/"):
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
        return items


class CompositeGoldenStore:
    def __init__(self, durables: list | None = None, memory: MemoryGoldenStore | None = None):
        self.memory = memory or MemoryGoldenStore()
        self.durables = durables or []
        self.last_errors: dict[str, str] = {}

    def save(self, case: dict) -> None:
        self.memory.save(case)
        for backend in self.durables:
            name = type(backend).__name__
            try:
                backend.save(case)
                self.last_errors.pop(name, None)
            except Exception as exc:
                self.last_errors[name] = str(exc)
                logger.warning("golden durable save failed backend=%s error=%s", name, exc)

    def list_cases(self) -> list[dict]:
        merged: dict[str, dict] = {}
        for backend in list(self.durables) + [self.memory]:
            name = type(backend).__name__
            try:
                items = backend.list_cases() or []
            except Exception as exc:
                self.last_errors[name] = str(exc)
                continue
            for item in items:
                case_id = item.get("id")
                if case_id:
                    merged[case_id] = item
        return list(merged.values())

    def persistence_status(self) -> dict:
        return {
            "durable_backends": [type(item).__name__ for item in self.durables],
            "last_errors": dict(self.last_errors),
            "ok": not self.last_errors,
        }


_store: CompositeGoldenStore | None = None


def get_golden_store() -> CompositeGoldenStore:
    global _store
    if _store is not None:
        return _store
    durables = []
    if os.getenv("K_SERVICE") or os.getenv("TRACE_BACKEND") == "firestore" or os.getenv("RAG_GCS_BUCKET"):
        try:
            durables.append(FirestoreGoldenStore())
        except Exception as exc:
            logger.warning("firestore golden store unavailable: %s", exc)
        if os.getenv("RAG_GCS_BUCKET"):
            try:
                durables.append(GcsGoldenStore())
            except Exception as exc:
                logger.warning("gcs golden archive unavailable: %s", exc)
    _store = CompositeGoldenStore(durables)
    return _store
