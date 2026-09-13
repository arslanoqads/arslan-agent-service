import re
import time
from contextvars import ContextVar

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="default_session")
current_user_message: ContextVar[str] = ContextVar("current_user_message", default="")

MAX_EMAILS_PER_SESSION = 10
MAX_CALENDAR_PER_SESSION = 10
MAX_JD_PER_SESSION = 1
SESSION_IDLE_SECONDS = 15 * 60
MAX_TOOL_CALLS = 40

# thread_id -> {emails, calendars, jd_matches, last_active, signatures}
_sessions: dict[str, dict] = {}
_now = time.time


def thread_key() -> str:
    return current_thread_id.get()


def _session(thread_id: str | None = None) -> dict:
    key = thread_id or thread_key()
    now = _now()
    state = _sessions.get(key)
    if state is None or (now - float(state.get("last_active") or 0)) > SESSION_IDLE_SECONDS:
        state = {"emails": 0, "calendars": 0, "jd_matches": 0, "last_active": now, "signatures": []}
        _sessions[key] = state
    else:
        state["last_active"] = now
        state.setdefault("jd_matches", 0)
    return state


def touch_session(thread_id: str | None = None) -> None:
    _session(thread_id)


def mark_email_sent(thread_id: str | None = None) -> str | None:
    state = _session(thread_id)
    if state["emails"] >= MAX_EMAILS_PER_SESSION:
        return (
            f"This session already sent {MAX_EMAILS_PER_SESSION} resume emails. "
            f"Session action limits reset after {SESSION_IDLE_SECONDS // 60} minutes of inactivity."
        )
    state["emails"] += 1
    return None


def mark_calendar_booked(thread_id: str | None = None) -> str | None:
    state = _session(thread_id)
    if state["calendars"] >= MAX_CALENDAR_PER_SESSION:
        return (
            f"This session already booked {MAX_CALENDAR_PER_SESSION} intro calls. "
            f"Session action limits reset after {SESSION_IDLE_SECONDS // 60} minutes of inactivity."
        )
    state["calendars"] += 1
    return None


def release_email(thread_id: str | None = None) -> None:
    state = _session(thread_id)
    if state["emails"] > 0:
        state["emails"] -= 1


def release_calendar(thread_id: str | None = None) -> None:
    state = _session(thread_id)
    if state["calendars"] > 0:
        state["calendars"] -= 1


def mark_jd_match(thread_id: str | None = None) -> str | None:
    state = _session(thread_id)
    if state["jd_matches"] >= MAX_JD_PER_SESSION:
        minutes = SESSION_IDLE_SECONDS // 60
        return (
            "This session already ran a job-description comparison. "
            f"Please wait for the next session (after {minutes} minutes of inactivity) "
            "before comparing another role."
        )
    state["jd_matches"] += 1
    return None


def release_jd_match(thread_id: str | None = None) -> None:
    state = _session(thread_id)
    if state["jd_matches"] > 0:
        state["jd_matches"] -= 1


def note_tool_call(name: str, signature: str) -> str | None:
    state = _session()
    calls = state["signatures"]
    if len(calls) >= MAX_TOOL_CALLS:
        return "tool_cap"
    calls.append(f"{name}:{signature}")
    return None


def email_count(thread_id: str | None = None) -> int:
    return int(_session(thread_id)["emails"])


def calendar_count(thread_id: str | None = None) -> int:
    return int(_session(thread_id)["calendars"])


def reset_thread_limits() -> None:
    _sessions.clear()
