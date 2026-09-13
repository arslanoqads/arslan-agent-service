PRICES = {
    "gpt-4o": {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000},
}


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES.get(model or "", PRICES["gpt-4o"])
    return round((input_tokens * price["input"]) + (output_tokens * price["output"]), 6)


def classify_error(detail: str) -> str:
    text = (detail or "").lower()
    if "429" in text or "timeout" in text or "rate limit" in text:
        return "provider"
    if "no resume excerpts" in text or "were retrieved" in text:
        return "retrieval_empty"
    if "could not" in text:
        return "tool"
    return "provider"
