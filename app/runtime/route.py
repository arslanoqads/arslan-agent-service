import re

GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|good morning|good afternoon)[!. ]*$",
    re.IGNORECASE,
)
LINK_RE = re.compile(r"\b(links?|linkedin|instagram|substack|website)\b", re.IGNORECASE)
CONFIRMATION_RE = re.compile(
    r"^(yes|yep|yeah|y|ok|okay|sure|please|proceed|go ahead|do it|book it|confirm)[!. ]*$",
    re.IGNORECASE,
)

DEGRADED_MESSAGE = "The model is unavailable right now. Try again in a moment."
TOKEN_CEILING_MESSAGE = "This turn hit the token ceiling before it finished."
FIRST_TOKEN_BUDGET_MS = 8000


def is_greeting(text: str) -> bool:
    return bool(GREETING_RE.match((text or "").strip()))


def is_confirmation(text: str) -> bool:
    return bool(CONFIRMATION_RE.match((text or "").strip()))


def is_links_request(text: str) -> bool:
    lowered = (text or "").lower()
    if any(word in lowered for word in ("email", "book", "calendar", "job", "meeting", "call")):
        return False
    return bool(LINK_RE.search(text or ""))


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content or "")


def choose_route(messages: list) -> str:
    """Deterministic overrides before the LLM router. Empty string means ask the model."""
    if not messages:
        return ""
    last = messages[-1]
    text = _message_text(last)
    prior_humans = [
        message
        for message in messages[:-1]
        if getattr(message, "type", "") == "human" or message.__class__.__name__ == "HumanMessage"
    ]
    if is_confirmation(text):
        return "portfolio_agent"
    if prior_humans and not is_greeting(text):
        return "portfolio_agent"
    if is_greeting(text) and not prior_humans:
        return "general_responder"
    return ""


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
