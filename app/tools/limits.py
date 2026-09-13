import re
from contextvars import ContextVar

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default_session")
current_user_message: ContextVar[str] = ContextVar("current_user_message", default="")

_email_sent: set[str] = set()
_calendar_booked: set[str] = set()
_tool_calls: dict[str, list[str]] = {}
MAX_TOOL_CALLS = 4


def thread_key() -> str:
    return current_thread_id.get()


def mark_email_sent(thread_id: str | None = None) -> str | None:
    key = thread_id or thread_key()
    if key in _email_sent:
        return "A resume email was already sent in this chat. One email per conversation."
    _email_sent.add(key)
    return None


def mark_calendar_booked(thread_id: str | None = None) -> str | None:
    key = thread_id or thread_key()
    if key in _calendar_booked:
        return "An intro call was already booked in this chat. One meeting per conversation."
    _calendar_booked.add(key)
    return None


def release_email(thread_id: str | None = None) -> None:
    _email_sent.discard(thread_id or thread_key())


def release_calendar(thread_id: str | None = None) -> None:
    _calendar_booked.discard(thread_id or thread_key())


def note_tool_call(name: str, signature: str) -> str | None:
    key = thread_key()
    calls = _tool_calls.setdefault(key, [])
    if signature in calls:
        return "already_sent"
    if len(calls) >= MAX_TOOL_CALLS:
        return "tool_cap"
    calls.append(signature)
    return None


def reset_thread_limits() -> None:
    _email_sent.clear()
    _calendar_booked.clear()
    _tool_calls.clear()
