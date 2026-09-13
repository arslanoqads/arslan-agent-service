from contextvars import ContextVar

current_context_budget: ContextVar[dict] = ContextVar("current_context_budget", default={})

OUTPUT_RESERVE_TOKENS = 800
HISTORY_TOKEN_CAP = 1200
RETRIEVED_TOKEN_CAP = 900
WINDOW_TOKENS = 8000


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4) if text else 0


def sandwich(chunks: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = (chunk.get("text") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    ranked = sorted(unique, key=lambda item: item.get("score", 0), reverse=True)
    if len(ranked) < 3:
        return ranked
    best, second, *rest = ranked
    return [best, *rest, second]


def citation_label(chunk: dict) -> str:
    return f"{chunk.get('doc_type', 'document')} v{chunk.get('version', 1)}, page {chunk.get('page', 1)}"


def format_quotes(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        text = " ".join((chunk.get("text") or "").split())
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        lines.append(f"- {citation_label(chunk)}: \"{text}\"")
    return "\n".join(lines)


def _trim_to_tokens(text: str, cap: int) -> tuple[str, bool]:
    if estimate_tokens(text) <= cap:
        return text, False
    return text[: cap * 4], True


def compact_history(messages: list[str]) -> list[str]:
    kept = []
    for text in messages:
        if "MATCH EVIDENCE" in text or text.startswith("Public links:"):
            kept.append("Earlier tool result omitted.")
            continue
        if len(text) > 400:
            kept.append(text[:240].rstrip() + "...")
            continue
        kept.append(text)
    return kept


def assemble(system: str, tools_text: str, quotes: str, history: list[str]) -> dict:
    history = compact_history(history)
    history_text = "\n".join(history)
    quotes, quotes_cut = _trim_to_tokens(quotes, RETRIEVED_TOKEN_CAP)
    history_text, history_cut = _trim_to_tokens(history_text, HISTORY_TOKEN_CAP)
    used = estimate_tokens(system) + estimate_tokens(tools_text) + estimate_tokens(quotes) + estimate_tokens(history_text)
    room = WINDOW_TOKENS - OUTPUT_RESERVE_TOKENS
    cut = []
    if used > room:
        history_text, history_cut = "", True
        used = estimate_tokens(system) + estimate_tokens(tools_text) + estimate_tokens(quotes)
    if used > room and quotes:
        quotes, quotes_cut = "", True
        used = estimate_tokens(system) + estimate_tokens(tools_text)
    if history_cut:
        cut.append("history")
    if quotes_cut:
        cut.append("retrieved")
    return {
        "quotes": quotes,
        "history": history_text,
        "context_budget": {
            "system": estimate_tokens(system),
            "tools": estimate_tokens(tools_text),
            "retrieved": estimate_tokens(quotes),
            "history": estimate_tokens(history_text),
            "reserved": OUTPUT_RESERVE_TOKENS,
            "used": used,
            "window": WINDOW_TOKENS,
            "cut": cut,
        },
    }
