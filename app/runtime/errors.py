from app.runtime.route import DEGRADED_MESSAGE, is_provider_failure

GENERIC_MESSAGE = "Something went wrong on my side. Please try that question again."
MESSAGE_SHAPE_MESSAGE = (
    "I hit a temporary processing error on that turn. Please ask again in a new message."
)
GOOGLE_MESSAGE = (
    "Email and calendar are not fully configured yet. You can still ask resume questions "
    "or request public links."
)
RETRIEVAL_MESSAGE = (
    "I could not read the resume documents just now. Please try again in a moment."
)


def public_error_message(detail: str) -> str:
    text = (detail or "").lower()
    if (
        "tool_calls" in text
        or "role 'tool'" in text
        or "messages with role" in text
        or "invalid parameter: messages" in text
    ):
        return MESSAGE_SHAPE_MESSAGE
    if "google is not configured" in text or "refresh_token" in text or "oauth" in text:
        return GOOGLE_MESSAGE
    if "no resume" in text or "pdf" in text and "not found" in text:
        return RETRIEVAL_MESSAGE
    if is_provider_failure(text) or "error code:" in text or "openai" in text:
        return DEGRADED_MESSAGE
    if not text:
        return GENERIC_MESSAGE
    # Never leak raw provider JSON or stack traces to the visitor.
    if "{" in text or "traceback" in text or "file \"" in text:
        return GENERIC_MESSAGE
    if len(text) > 220:
        return GENERIC_MESSAGE
    return GENERIC_MESSAGE
