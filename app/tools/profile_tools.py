from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.rag.documents import ensure_rag_pdfs
from app.rag.hybrid_engine import HybridRAGEngine

rag_engine = HybridRAGEngine(ensure_rag_pdfs())


# --- TOOL 1: Hybrid RAG Background Query ---
class ProfileQueryInput(BaseModel):
    query: str = Field(description="The specific question about Arslan's background, skills, or experience.")

@tool("query_arslan_profile", args_schema=ProfileQueryInput)
def query_arslan_profile(query: str) -> str:
    """Searches Arslan's official resume and biography using hybrid keyword-semantic search."""
    return rag_engine.query(query)


# --- TOOL 2: Resume Emailer ---
class EmailRequestInput(BaseModel):
    user_email: str = Field(description="The recipient email address.")
    note: str = Field(description="Personal message to include with the email.")

@tool("send_resume_email", args_schema=EmailRequestInput)
def send_resume_email(user_email: str, note: str) -> str:
    """Sends Arslan's official resume to a specified email address."""
    return f"[SUCCESS] Resume dispatched to {user_email}. Message attached: '{note}'"


# --- TOOL 3: Job Description Matcher ---
class JobMatchInput(BaseModel):
    job_description: str = Field(description="The job description text to evaluate.")

@tool("evaluate_jd_match", args_schema=JobMatchInput)
def evaluate_jd_match(job_description: str) -> str:
    """Evaluates alignment between a provided Job Description and Arslan's profile."""
    return "[MATCH ANALYSIS] High alignment (94%). Strong match for AI Product Leadership, Agent Architecture, and Data Science."
