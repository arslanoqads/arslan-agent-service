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
    if words in {"tell me about him", "who is he", "about you", "tell me more"}:
        return "Arslan Qadri professional background experience skills biography"
    return query
