import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.guardrails import INJECTION_REFUSAL, looks_like_injection
from app.tools.google_client import create_calendar_event, send_gmail
from app.tools.limits import (
    EMAIL_RE,
    current_user_message,
    mark_calendar_booked,
    mark_email_sent,
    note_tool_call,
    release_calendar,
    release_email,
)
from app.tools.profile_tools import get_rag_engine

ET = ZoneInfo("America/New_York")

DEFAULT_LINKS = {
    "website": "https://qads.us",
    "linkedin": "https://www.linkedin.com/in/maqadri",
}


def social_links() -> dict[str, str]:
    links = {
        "website": os.getenv("PUBLIC_WEBSITE_URL", DEFAULT_LINKS["website"]),
        "linkedin": os.getenv("PUBLIC_LINKEDIN_URL", DEFAULT_LINKS["linkedin"]),
        "instagram": os.getenv("PUBLIC_INSTAGRAM_URL", "").strip(),
        "substack": os.getenv("PUBLIC_SUBSTACK_URL", "").strip(),
    }
    return {name: url for name, url in links.items() if url}


def format_social_links() -> str:
    lines = [f"{name}: {url}" for name, url in social_links().items()]
    if not lines:
        return "No public links are configured."
    return "Public links:\n" + "\n".join(lines)


def short_quote(chunk: str) -> str:
    text = " ".join(chunk.split())
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    return f'"{text}"'


def match_role_from_chunks(job_description: str, chunks: list[str]) -> str:
    quotes = [short_quote(chunk) for chunk in chunks if chunk and chunk.strip()]
    if not quotes:
        return (
            "MATCH EVIDENCE\nNo resume excerpts were retrieved.\n\n"
            "GAPS\nCannot assess fit. Do not invent a match score or percentage."
        )
    quoted = "\n".join(f"- {quote}" for quote in quotes[:3])
    return (
        "MATCH EVIDENCE\n"
        "Use only these quotes from retrieved resume excerpts. Do not invent a match score or percentage.\n\n"
        f"{quoted}\n\n"
        "GAPS\n"
        "If the job description asks for something not present in the quotes, say it was not found. "
        "Do not add a percentage."
    )


def parse_start(start_time: str) -> datetime:
    parsed = datetime.fromisoformat(start_time.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed.astimezone(ET)


def validate_slot(start: datetime) -> str | None:
    if start <= datetime.now(ET):
        return "That time is in the past. Ask for a future weekday between 9:00 and 17:00 ET."
    if start.weekday() >= 5:
        return "Intro calls are weekdays only, 9:00-17:00 ET."
    if start.hour < 9 or start.hour >= 17 or (start.hour == 16 and start.minute > 30):
        return "Intro calls are weekdays 9:00-17:00 ET. The 30-minute slot must finish by 17:00."
    return None


class EmailInput(BaseModel):
    user_email: str = Field(description="The recipient email address.")
    note: str = Field(default="", description="Optional short note from the visitor.")


@tool("send_resume_email", args_schema=EmailInput)
def send_resume_email(user_email: str, note: str = "") -> str:
    """Emails Arslan's resume PDF from his Gmail to the visitor. One send per chat."""
    if os.getenv("ACTIONS_ENABLED", "1") == "0":
        return "Email and calendar actions are turned off."
    if looks_like_injection(current_user_message.get()) or looks_like_injection(note):
        return INJECTION_REFUSAL
    if not EMAIL_RE.match(user_email.strip()):
        return "Cannot send the resume. Provide a valid email address."
    blocked = note_tool_call("send_resume_email", user_email.strip().lower())
    if blocked == "already_sent":
        return "A resume email was already sent in this chat. One email per conversation."
    if blocked == "tool_cap":
        return "This turn hit the tool call limit."
    blocked = mark_email_sent()
    if blocked:
        return blocked
    try:
        sent = send_gmail(user_email.strip(), note)
    except Exception as exc:
        release_email()
        return f"Could not send the resume email: {exc}"
    return sent


class CalendarInput(BaseModel):
    visitor_email: str = Field(description="Email address that should receive the calendar invite.")
    start_time: str = Field(description="Start time in ISO 8601, for example 2026-09-15T10:00:00-04:00.")


@tool("schedule_intro_call", args_schema=CalendarInput)
def schedule_intro_call(visitor_email: str, start_time: str) -> str:
    """Books a fixed 30-minute intro call on Arslan's Google Calendar and emails an invite.
    Always use 30 minutes even if the visitor asked for a shorter or longer slot.
    """
    if os.getenv("ACTIONS_ENABLED", "1") == "0":
        return "Email and calendar actions are turned off."
    if looks_like_injection(current_user_message.get()) or looks_like_injection(start_time):
        return INJECTION_REFUSAL
    if not EMAIL_RE.match(visitor_email.strip()):
        return "Cannot book a call. Provide a valid email address."
    try:
        start = parse_start(start_time)
    except ValueError:
        return "Cannot book a call. Provide a start time in ISO 8601 format."
    slot_error = validate_slot(start)
    if slot_error:
        return slot_error
    blocked = note_tool_call("schedule_intro_call", f"{visitor_email.strip().lower()}|{start.isoformat()}")
    if blocked == "already_sent":
        return "An intro call was already booked in this chat. One meeting per conversation."
    if blocked == "tool_cap":
        return "This turn hit the tool call limit."
    blocked = mark_calendar_booked()
    if blocked:
        return blocked
    end = start + timedelta(minutes=30)
    try:
        booked = create_calendar_event(
            visitor_email.strip(),
            start.isoformat(),
            end.isoformat(),
            "30-minute intro call with Arslan Qadri.\n" + format_social_links(),
        )
    except Exception as exc:
        release_calendar()
        return f"Could not book the intro call: {exc}"
    return booked


class JobMatchInput(BaseModel):
    job_description: str = Field(description="The job description text to compare with the resume.")


@tool("match_role_evidence", args_schema=JobMatchInput)
def match_role_evidence(job_description: str) -> str:
    """Retrieves resume evidence for a job description. Does not invent a match score."""
    text = (job_description or "").strip()
    if not text:
        return "Provide a job description to compare against the resume."
    chunks = [chunk["text"] for chunk in get_rag_engine().retrieve_chunks(text[:1500], job_match=True)]
    return match_role_from_chunks(text, chunks)


@tool("get_social_links")
def get_social_links() -> str:
    """Returns Arslan's public website, LinkedIn, Instagram, and Substack links. No phone or private email."""
    return format_social_links()
