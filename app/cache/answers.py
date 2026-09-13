import os
import re
from math import sqrt

from app.guardrails import looks_like_injection

EXACT_CACHE_ENABLED = "EXACT_CACHE_ENABLED"
SEMANTIC_CACHE_ENABLED = "SEMANTIC_CACHE_ENABLED"
SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))

_exact: dict[str, dict] = {}
_semantic: list[dict] = []

ACTION_MARKERS = (
    "email",
    "gmail",
    "calendar",
    "book a",
    "intro call",
    "schedule",
    "job description",
    "compare this role",
    "match this",
)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def cacheable(text: str) -> bool:
    if looks_like_injection(text):
        return False
    cleaned = normalize_question(text)
    if not cleaned or len(cleaned) < 12:
        return False
    # Short confirmations and follow-ups are session-bound, never global cache keys.
    if cleaned in {"yes", "yep", "yeah", "y", "ok", "okay", "sure", "please", "proceed", "go ahead", "do it", "book it", "confirm"}:
        return False
    lowered = cleaned
    return not any(marker in lowered for marker in ACTION_MARKERS)


def _enabled(name: str) -> bool:
    return os.getenv(name, "1") != "0"


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def invalidate() -> None:
    _exact.clear()
    _semantic.clear()


def lookup(question: str, fingerprint: str, embed=None) -> dict | None:
    if not cacheable(question):
        return None
    key = normalize_question(question)
    if _enabled(EXACT_CACHE_ENABLED):
        hit = _exact.get(key)
        if hit and hit["fingerprint"] == fingerprint:
            return {"kind": "exact", "answer": hit["answer"], "similarity": 1.0}
    if _enabled(SEMANTIC_CACHE_ENABLED) and embed is not None:
        vector = embed(question)
        best = None
        best_score = 0.0
        for item in _semantic:
            if item["fingerprint"] != fingerprint:
                continue
            score = _cosine(vector, item["embedding"])
            if score > best_score:
                best = item
                best_score = score
        if best and best_score >= SIMILARITY_THRESHOLD:
            return {"kind": "semantic", "answer": best["answer"], "similarity": round(best_score, 4)}
    return None


def store(question: str, answer: str, fingerprint: str, embed=None) -> None:
    if not cacheable(question) or not answer:
        return
    key = normalize_question(question)
    _exact[key] = {"answer": answer, "fingerprint": fingerprint}
    if embed is None:
        return
    _semantic.append(
        {
            "question": key,
            "answer": answer,
            "fingerprint": fingerprint,
            "embedding": embed(question),
        }
    )
