import base64
import os
from email.message import EmailMessage
from pathlib import Path

SENDER = os.getenv("GOOGLE_SENDER_EMAIL", "thatqadri@gmail.com")
EMAIL_SUBJECT = "Arslan Qadri resume"
EVENT_TITLE = "Intro call with Arslan Qadri"


def credentials():
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    refresh_token = os.getenv("GOOGLE_REFRESH_TOKEN")
    if not client_id or not client_secret or not refresh_token:
        raise RuntimeError(
            "Google is not configured. Set GOOGLE_OAUTH_CLIENT_ID, "
            "GOOGLE_OAUTH_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN."
        )
    from google.oauth2.credentials import Credentials

    return Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )


def resume_pdf_bytes() -> tuple[bytes, str]:
    from app.rag.corpus import latest_resume

    latest = latest_resume()
    if not latest:
        raise FileNotFoundError("No resume PDF is available to attach.")
    path = Path(latest["path"])
    return path.read_bytes(), path.name


def build_resume_message(to_email: str, note: str) -> str:
    pdf_bytes, filename = resume_pdf_bytes()
    message = EmailMessage()
    message["To"] = to_email
    message["From"] = SENDER
    message["Subject"] = EMAIL_SUBJECT
    body = (
        "Please find Arslan Qadri's resume attached.\n\n"
        "Website: https://qads.us\n"
        "LinkedIn: https://www.linkedin.com/in/maqadri\n"
    )
    cleaned = " ".join((note or "").split())[:400]
    if cleaned:
        body += f"\nNote from the visitor:\n{cleaned}\n"
    message.set_content(body)
    message.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


def send_gmail(to_email: str, note: str) -> str:
    raw = build_resume_message(to_email, note)
    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=credentials(), cache_discovery=False)
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    message_id = sent.get("id", "unknown")
    return f"Resume emailed to {to_email}. Gmail message id: {message_id}."


def calendar_event_body(visitor_email: str, start_iso: str, end_iso: str, description: str) -> dict:
    return {
        "summary": EVENT_TITLE,
        "description": description,
        "start": {"dateTime": start_iso, "timeZone": "America/New_York"},
        "end": {"dateTime": end_iso, "timeZone": "America/New_York"},
        "attendees": [{"email": visitor_email}],
    }


def create_calendar_event(visitor_email: str, start_iso: str, end_iso: str, description: str) -> str:
    from googleapiclient.discovery import build

    service = build("calendar", "v3", credentials=credentials(), cache_discovery=False)
    event = calendar_event_body(visitor_email, start_iso, end_iso, description)
    created = (
        service.events()
        .insert(calendarId="primary", body=event, sendUpdates="all")
        .execute()
    )
    link = created.get("htmlLink", "")
    return f"Booked a 30-minute intro call. Invite sent to {visitor_email}. {link}".strip()
