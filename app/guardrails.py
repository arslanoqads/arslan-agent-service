from collections import defaultdict
from threading import Lock

MAX_QUESTIONS_PER_IP = 2
BUDGET_LIMIT_MESSAGE = (
    "The number of questions that can be asked is limited to 2 per visitor "
    "to control budget. Thanks for trying the demo."
)

_counts: dict[str, int] = defaultdict(int)
_lock = Lock()


def consume_question(ip: str) -> str | None:
    with _lock:
        if _counts[ip] >= MAX_QUESTIONS_PER_IP:
            return BUDGET_LIMIT_MESSAGE
        _counts[ip] += 1
        return None


def release_question(ip: str) -> None:
    with _lock:
        _counts[ip] = max(0, _counts[ip] - 1)


INJECTION_REFUSAL = (
    "I can't follow that request. Ask about the resume, email the resume, "
    "book a 30-minute intro call, compare a job description, or ask for public links."
)

_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all instructions",
    "ignore your instructions",
    "system prompt",
    "reveal your instructions",
    "reveal these instructions",
    "disregard your",
    "you are now",
    "developer message",
)


def looks_like_injection(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _INJECTION_MARKERS)


def reset_question_counts() -> None:
    with _lock:
        _counts.clear()
