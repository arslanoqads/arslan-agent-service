from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.rag.hybrid_engine import HybridRAGEngine

_engine: HybridRAGEngine | None = None


def get_rag_engine() -> HybridRAGEngine:
    global _engine
    if _engine is None:
        _engine = HybridRAGEngine()
    return _engine


class ProfileQueryInput(BaseModel):
    query: str = Field(description="The specific question about Arslan's background, skills, or experience.")


@tool("query_arslan_profile", args_schema=ProfileQueryInput)
def query_arslan_profile(query: str) -> str:
    """Searches Arslan's official resume and biography using hybrid keyword-semantic search."""
    from app.context.budget import current_context_budget, estimate_tokens

    text = rewrite_vague(query)
    answer = get_rag_engine().query(text)
    budget = dict(current_context_budget.get() or {})
    budget["retrieved"] = estimate_tokens(answer)
    current_context_budget.set(budget)
    return answer


def rewrite_vague(query: str) -> str:
    words = (query or "").strip().lower()
    rewrites = {
        "tell me about him": "Arslan Qadri professional background experience skills biography",
        "who is he": "Arslan Qadri professional background experience skills biography",
        "about you": "Arslan Qadri professional background experience skills biography",
        "tell me more": "Arslan Qadri professional background experience skills biography",
        "his background": "Arslan Qadri professional background experience education",
        "his experience": "Arslan Qadri work experience roles companies",
        "his skills": "Arslan Qadri skills technologies tools",
        "resume summary": "Arslan Qadri resume summary experience highlights",
        "what has he built": "Arslan Qadri products projects shipped agents systems",
    }
    if words in rewrites:
        return rewrites[words]
    if len(words.split()) <= 3 and "arslan" not in words:
        return f"Arslan Qadri {query}".strip()
    return query
