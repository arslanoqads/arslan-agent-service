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
GCS_SCORE_PREFIX = "observability/golden_scores"
FIRESTORE_COLLECTION = "agent_golden_cases"
FIRESTORE_SCORE_COLLECTION = "agent_golden_scores"


class MemoryGoldenStore:
    def __init__(self):
        self._lock = Lock()
        self._items: dict[str, dict] = {}
        self._scores: dict[str, dict] = {}

    def save(self, case: dict) -> None:
        case_id = case.get("id")
        if not case_id:
            return
        with self._lock:
            self._items[case_id] = dict(case)

    def list_cases(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._items.values()]

    def save_score(self, point: dict) -> None:
        key = point.get("id") or f"{point.get('case_id')}::{point.get('trace_id') or point.get('day')}"
        with self._lock:
            self._scores[key] = dict(point)

    def list_scores(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._scores.values()]


class FirestoreGoldenStore:
    def __init__(self):
        from google.cloud import firestore

        self.client = firestore.Client()
        self.collection = self.client.collection(FIRESTORE_COLLECTION)
        self.scores = self.client.collection(FIRESTORE_SCORE_COLLECTION)

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

    def save_score(self, point: dict) -> None:
        payload = json.loads(json.dumps(point, default=str))
        doc_id = point.get("id") or f"{point.get('case_id')}__{point.get('trace_id') or point.get('day')}"
        self.scores.document(str(doc_id).replace("/", "_")).set(payload)

    def list_scores(self) -> list[dict]:
        try:
            docs = list(self.scores.stream())
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
                if not blob.name.endswith(".json") or "/golden_scores/" in blob.name:
                    continue
                # Only direct case objects under observability/golden/*.json
                relative = blob.name[len(GCS_PREFIX) + 1 :]
                if "/" in relative:
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

    def save_score(self, point: dict) -> None:
        payload = json.loads(json.dumps(point, default=str))
        doc_id = point.get("id") or f"{point.get('case_id')}__{point.get('trace_id') or point.get('day')}"
        blob = self.bucket.blob(f"{GCS_SCORE_PREFIX}/{doc_id}.json")
        blob.upload_from_string(json.dumps(payload), content_type="application/json")

    def list_scores(self) -> list[dict]:
        items = []
        try:
            for blob in self.client.list_blobs(self.bucket_name, prefix=f"{GCS_SCORE_PREFIX}/"):
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

    def save_score(self, point: dict) -> None:
        payload = dict(point)
        if not payload.get("id"):
            payload["id"] = f"{payload.get('case_id')}__{payload.get('trace_id') or payload.get('day')}"
        self.memory.save_score(payload)
        for backend in self.durables:
            name = type(backend).__name__
            if not hasattr(backend, "save_score"):
                continue
            try:
                backend.save_score(payload)
                self.last_errors.pop(f"{name}.scores", None)
            except Exception as exc:
                self.last_errors[f"{name}.scores"] = str(exc)
                logger.warning("golden score save failed backend=%s error=%s", name, exc)

    def list_scores(self) -> list[dict]:
        merged: dict[str, dict] = {}
        for backend in list(self.durables) + [self.memory]:
            name = type(backend).__name__
            if not hasattr(backend, "list_scores"):
                continue
            try:
                items = backend.list_scores() or []
            except Exception as exc:
                self.last_errors[f"{name}.scores"] = str(exc)
                continue
            for item in items:
                key = item.get("id") or f"{item.get('case_id')}::{item.get('trace_id') or item.get('day')}"
                existing = merged.get(key)
                if not existing or (item.get("started_at") or "") >= (existing.get("started_at") or ""):
                    merged[key] = item
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
