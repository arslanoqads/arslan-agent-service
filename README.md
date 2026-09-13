# Arslan Agent Service

Portfolio FastAPI service with one resume agent, Gmail/Calendar actions, a chat trace panel, and a private traces dashboard. Deployed on Google Cloud Run.

## Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
# keep resume.pdf + bio.pdf in data/raw/ (gitignored — not public)
uvicorn app.main:app --reload --port 8080
```

Open http://localhost:8080 for the chat UI.

## Private PDFs

`data/raw/*.pdf` are **not** in Git. Locally they stay on your machine.

Drop additional resumes in `data/raw/resumes/v2.pdf`, `v3.pdf`, and so on. The current `data/raw/resume.pdf` stays version 1. The biography stays at `data/raw/bio.pdf`, or later versions in `data/raw/bio/`. New files are indexed beside the old ones. Search uses the latest resume plus the current biography unless the question asks about an earlier resume. Email attaches only the latest resume.

On Cloud Run, the app downloads them from a **private GCS bucket** (`RAG_GCS_BUCKET`).

One-time Cloud Shell setup:

```bash
PROJECT_ID="project-a4383faa-cfc2-4119-8d5"
PROJECT_NUMBER="59038284696"
BUCKET="arslan-agent-rag-docs"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud storage buckets create "gs://${BUCKET}" --project="${PROJECT_ID}" --location=us-central1
gcloud storage cp data/raw/resume.pdf data/raw/bio.pdf "gs://${BUCKET}/"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.objectViewer"
```

Then add GitHub Actions secret:

| Secret | Value |
|---|---|
| `RAG_GCS_BUCKET` | `arslan-agent-rag-docs` |

## Actions

- `query_arslan_profile` searches the private resume and bio.
- `send_resume_email` sends `resume.pdf` from `thatqadri@gmail.com`. One send per chat. Invalid addresses are refused.
- `schedule_intro_call` creates a 30-minute Google Calendar invite, weekdays 9:00–17:00 ET. One booking per chat.
- `match_role_evidence` returns retrieved resume excerpts. It does not invent a match percentage.
- `get_social_links` returns website and LinkedIn. Instagram and Substack are omitted until those URLs are set.

Google send/book needs a user OAuth refresh token. A service account cannot send as a consumer Gmail address.

One-time setup, from a machine with `gcloud` signed in as `thatqadri@gmail.com`:

1. In the GCP project, create an OAuth desktop client. Download the JSON as `client_secret.json` (gitignored).
2. Enable the Gmail API and the Google Calendar API.
3. Print a refresh token. Scopes are `gmail.send` and `calendar.events`. Do not commit the token.

```bash
pip install google-auth-oauthlib
python scripts/google_oauth.py
```

4. Store the Google values and the observability token in Secret Manager. Instagram and Substack stay empty until you have the real URLs.

```bash
export GOOGLE_OAUTH_CLIENT_ID="..."
export GOOGLE_OAUTH_CLIENT_SECRET="..."
export GOOGLE_REFRESH_TOKEN="..."
export GOOGLE_SENDER_EMAIL="thatqadri@gmail.com"
export PUBLIC_WEBSITE_URL="https://qads.us"
export PUBLIC_LINKEDIN_URL="https://www.linkedin.com/in/maqadri"
export OBSERVABILITY_TOKEN="choose-a-long-random-string"
bash scripts/setup_google_secrets.sh
```

5. Let the Cloud Run runtime service account read those secrets, and let it write traces:

```bash
PROJECT_NUMBER="59038284696"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for name in GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET GOOGLE_REFRESH_TOKEN GOOGLE_SENDER_EMAIL OBSERVABILITY_TOKEN; do
  gcloud secrets add-iam-policy-binding "${name}" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor"
done
gcloud projects add-iam-policy-binding project-a4383faa-cfc2-4119-8d5 \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/datastore.user"
```

Do this before the next merge to `main`. Cloud Run deploy reads those Secret Manager names and fails if they are missing.

Private traces: open `/traces` for the public observability dashboard. Emails, phones, resume text, tool arguments, and raw errors are stripped before display.

## Golden set

The eval files are empty on purpose. After this build, create cases using `tests/evals/CHECKLIST.md`. Do not put resume facts in Git. Public policy cases go in `tests/evals/golden_set.public.json`. Private labels go in `tests/evals/golden_set.private.json`, which is gitignored.

## Live

- **Chat UI:** https://arslan-agent-service-59038284696.us-central1.run.app/
- **Repo:** https://github.com/arslanoqads/arslan-agent-service
- **API docs:** https://arslan-agent-service-59038284696.us-central1.run.app/docs
- **Health:** https://arslan-agent-service-59038284696.us-central1.run.app/health

## Deploy flow

Push to `develop` → PR → merge `main` → GitHub Actions CD → Cloud Run.
