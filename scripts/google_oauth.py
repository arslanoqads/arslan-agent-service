"""One-time local helper. Prints a Google refresh token. Do not commit the token."""

import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]


def main() -> None:
    client_path = Path(os.getenv("GOOGLE_OAUTH_CLIENT_FILE", "client_secret.json"))
    if not client_path.exists():
        raise SystemExit(f"Missing {client_path}. Download the OAuth desktop client JSON first.")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
    creds = flow.run_local_server(port=0)
    print("Add this value to Secret Manager as GOOGLE_REFRESH_TOKEN. Do not commit it.")
    print(creds.refresh_token)


if __name__ == "__main__":
    main()
