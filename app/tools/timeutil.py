"""Resolve intro-call start times in America/New_York."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
_TIME_RE = re.compile(
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?",
    re.IGNORECASE,
)


def now_et(now: datetime | None = None) -> datetime:
    current = now or datetime.now(ET)
    if current.tzinfo is None:
        return current.replace(tzinfo=ET)
    return current.astimezone(ET)


def clock_context(now: datetime | None = None) -> str:
    current = now_et(now)
    tomorrow = current + timedelta(days=1)
    return (
        f"Current time in America/New_York (ET): {current.isoformat()} ({current.strftime('%A')}). "
        f"Today's date is {current.date().isoformat()}. "
        f"Tomorrow's date is {tomorrow.date().isoformat()} ({tomorrow.strftime('%A')}). "
        "When the visitor says tomorrow, use tomorrow's date, never today's."
    )


def _apply_ampm(hour: int, ampm: str | None) -> int:
    marker = (ampm or "").lower().replace(".", "")
    if marker == "pm" and hour < 12:
        return hour + 12
    if marker == "am" and hour == 12:
        return 0
    return hour


def _parse_clock(text: str) -> tuple[int, int] | None:
    match = _TIME_RE.search(text or "")
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if hour > 24 or minute > 59:
        return None
    hour = _apply_ampm(hour, match.group("ampm"))
    if hour == 24:
        hour = 0
    return hour, minute


def resolve_intro_start(start_time: str, *, now: datetime | None = None) -> datetime:
    """
    Accept ISO-8601 or short phrases like 'tomorrow at 3pm ET'.
    Relative words are resolved against America/New_York.
    """
    current = now_et(now)
    text = (start_time or "").strip()
    if not text:
        raise ValueError("missing start time")

    lowered = text.lower()
    wants_tomorrow = bool(_TOMORROW_RE.search(lowered))
    wants_today = bool(_TODAY_RE.search(lowered))

    parsed: datetime | None = None
    iso_candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ET)
        else:
            parsed = parsed.astimezone(ET)
    except ValueError:
        parsed = None

    if parsed is None:
        clock = _parse_clock(text)
        if clock is None:
            raise ValueError(f"unrecognized start time: {text}")
        hour, minute = clock
        base = current.date()
        if wants_tomorrow:
            base = (current + timedelta(days=1)).date()
        parsed = datetime(base.year, base.month, base.day, hour, minute, tzinfo=ET)

    # If the visitor said tomorrow but the model emitted today's date, bump one day.
    if wants_tomorrow and parsed.date() == current.date():
        parsed = parsed + timedelta(days=1)

    # Bare clock with no day: if that time already passed today, use tomorrow.
    if parsed is not None and not wants_today and not wants_tomorrow:
        # ISO without relative words: if clearly in the past by >1 minute, and same calendar day,
        # prefer tomorrow when the source text looks like a clock-only phrase.
        clock_only = bool(_parse_clock(text)) and "T" not in text and not re.search(r"\d{4}-\d{2}-\d{2}", text)
        if clock_only and parsed <= current:
            parsed = parsed + timedelta(days=1)

    return parsed.astimezone(ET)
