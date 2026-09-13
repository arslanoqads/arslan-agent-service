"""Hop planning for multi-step portfolio actions."""

from __future__ import annotations

import re

from app.runtime.route import is_confirmation, is_greeting, is_links_request

EMAIL_RE = re.compile(
    r"\b(email|e-mail|send(?:\s+me)?\s+(?:the\s+|your\s+|my\s+)?resume)\b",
    re.IGNORECASE,
)
CALENDAR_RE = re.compile(
    r"\b(book|schedule|meeting|intro call|calendar|invite|call with me|set up)\b",
    re.IGNORECASE,
)
JD_RE = re.compile(r"\b(job description|jd\b|compare this role|match this role)\b", re.IGNORECASE)
PROFILE_RE = re.compile(
    r"\b(resume|background|experience|who is|what (?:has|have)|tell me about|skills?)\b",
    re.IGNORECASE,
)


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content or "")


def conversation_blob(messages: list) -> str:
    parts = []
    for message in messages:
        role = getattr(message, "type", "") or message.__class__.__name__.lower()
        parts.append(f"{role}: {_message_text(message)}")
    return "\n".join(parts)


def plan_hops(messages: list) -> list[str]:
    """
    Split a turn into independent hops so email and calendar never share one tool call batch.
    Order is stable: email -> calendar -> jd -> links -> profile.
    """
    if not messages:
        return []
    blob = conversation_blob(messages)
    last = _message_text(messages[-1]).strip()
    if is_greeting(last) and len(messages) == 1:
        return []

    hops: list[str] = []
    if EMAIL_RE.search(blob):
        hops.append("email")
    if CALENDAR_RE.search(blob):
        hops.append("calendar")
    if JD_RE.search(blob):
        hops.append("jd")
    if is_links_request(last) or is_links_request(blob):
        hops.append("links")
    if not hops and (PROFILE_RE.search(blob) or is_confirmation(last) or len(last) > 0):
        # Default Q&A / follow-up lands on profile retrieval hop.
        if not is_greeting(last):
            hops.append("profile")
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered = []
    for hop in hops:
        if hop not in seen:
            seen.add(hop)
            ordered.append(hop)
    return ordered
