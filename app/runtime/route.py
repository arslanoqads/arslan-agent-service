import re

GREETING_RE = re.compile(r"^(hi|hello|hey|thanks|thank you|good morning|good afternoon)[!. ]*$", re.IGNORECASE)
LINK_RE = re.compile(r"\b(links?|linkedin|instagram|substack|website)\b", re.IGNORECASE)

DEGRADED_MESSAGE = "The model is unavailable right now. Try again in a moment."
TOKEN_CEILING_MESSAGE = "This turn hit the token ceiling before it finished."
FIRST_TOKEN_BUDGET_MS = 8000


def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match((text or "").strip()))


def is_links_request(text: str) -> bool:
    lowered = (text or "").lower()
    if any(word in lowered for word in ("email", "book", "calendar", "job")):
        return False
    return bool(LINK_RE.search(text))


def is_provider_failure(detail: str) -> bool:
    text = (detail or "").lower()
    return any(
        needle in text
        for needle in (
            "429",
            "timeout",
            "timed out",
            "rate limit",
            "unavailable",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "service unavailable",
            "connection error",
        )
    )
