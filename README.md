# Arslan Agent Service

Multi-agent FastAPI service (profile + system tools) with a web chat UI, deployed on Google Cloud Run.

## Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
uvicorn app.main:app --reload --port 8080
```

Open http://localhost:8080 for the chat UI.

## Live

- **Chat UI:** https://arslan-agent-service-59038284696.us-central1.run.app/
- **Repo:** https://github.com/arslanoqads/arslan-agent-service
- **API docs:** https://arslan-agent-service-59038284696.us-central1.run.app/docs
- **Health:** https://arslan-agent-service-59038284696.us-central1.run.app/health

## Deploy flow

Push to `develop` → PR → merge `main` → GitHub Actions CD → Cloud Run.
