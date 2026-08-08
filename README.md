# Arslan Agent Service

Multi-agent FastAPI service (profile + system tools) deployed on Google Cloud Run.

## Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
uvicorn app.main:app --reload --port 8080
```

## Live

- API: https://arslan-agent-service-59038284696.us-central1.run.app/
- Docs UI: https://arslan-agent-service-59038284696.us-central1.run.app/docs

## Deploy flow

Push to `develop` → PR → merge `main` → GitHub Actions CD → Cloud Run.
