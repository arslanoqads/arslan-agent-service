# Arslan Agent Service

Multi-agent FastAPI service (profile + system tools) with a web chat UI, deployed on Google Cloud Run.

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

## Live

- **Chat UI:** https://arslan-agent-service-59038284696.us-central1.run.app/
- **Repo:** https://github.com/arslanoqads/arslan-agent-service
- **API docs:** https://arslan-agent-service-59038284696.us-central1.run.app/docs
- **Health:** https://arslan-agent-service-59038284696.us-central1.run.app/health

## Deploy flow

Push to `develop` → PR → merge `main` → GitHub Actions CD → Cloud Run.
