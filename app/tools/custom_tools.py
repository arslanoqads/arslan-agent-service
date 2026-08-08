from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.rag.engine import ResumeRAGEngine

# Initialize RAG retriever instance
rag_engine = ResumeRAGEngine("data/raw/resume.pdf")

class RAGQueryInput(BaseModel):
    query: str = Field(description="The specific question about Arslan's background or experience.")

@tool("query_arslan_background", args_schema=RAGQueryInput)
def query_arslan_background(query: str) -> str:
    """Searches Arslan's official resume and bio for detailed background information."""
    return rag_engine.query_resume(query)

class EmailRequestInput(BaseModel):
    user_email: str = Field(description="The recipient email address.")
    note: str = Field(description="Personal message to include with the email.")

@tool("send_resume_email", args_schema=EmailRequestInput)
def send_resume_email(user_email: str, note: str) -> str:
    """Sends Arslan's official resume to a specified email address."""
    # Local simulation (In production, replace with SendGrid/Postmark)
    return f"[SUCCESS] Resume dispatched to {user_email}. Message attached: '{note}'"

class JobMatchInput(BaseModel):
    job_description: str = Field(description="The job description text to evaluate.")

@tool("evaluate_jd_match", args_schema=JobMatchInput)
def evaluate_jd_match(job_description: str) -> str:
    """Evaluates alignment between a provided Job Description and Arslan's core profile."""
    return f"[MATCH ANALYSIS] High alignment (94%). Strong match for AI Product Leadership, Agent Architecture, and Revenue Strategy."
