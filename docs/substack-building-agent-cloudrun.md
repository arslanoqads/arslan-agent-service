# From Zero to a Live Agent on Cloud Run

*Lots of tutorials teach you how to build agents. Almost none walk you through deploying one end to end. This post is about that missing middle—getting from a working local demo to a real URL, with Git and GCP in the loop.*

---

I’m not going to paste the entire codebase here. You don’t need that. If you want the details, clone the repo and let your coding agent fill in the gaps.

What I *am* showing is the **sequence of steps** that gets you from zero to one: a basic agent running locally, then a team-style Git flow, then a trial GCP deploy that updates when you merge a pull request.

**Try it:** [live chat UI](https://arslan-agent-service-59038284696.us-central1.run.app/)  
**Read the code:** [github.com/arslanoqads/arslan-agent-service](https://github.com/arslanoqads/arslan-agent-service)

This agent is **basic on purpose**. No serious guardrails, no auth, no production hardening. It simply works well enough to prove the path.

---

## The point

Building the agent is the fun part. Shipping it is where people stall:

- Where do secrets live?
- How does a teammate get the same code?
- How does “merge” become “new version in the cloud”?

This project answers those with a boring, copyable path:

```text
local agent → GitHub (develop / main) → GitHub Actions → Cloud Run
```

Once you’ve done that once, improvising is easy. **Zero to one is the hard step.**

---

## Step 1 — Set up the agent locally

Think of the product as three pieces glued together:

1. **RAG** — answers grounded in documents  
2. **Tools** — actions the model can call  
3. **Chatbot** — FastAPI + a simple UI in front  

### RAG

I indexed my **resume** and **bio** PDFs. Hybrid retrieval (keyword + embeddings) pulls relevant chunks when the profile agent needs them.

You don’t need a fancy vector database for v1. In-memory is enough to learn the loop.

Those PDFs stay **private**. They are not in the public GitHub repo. Locally they live on my machine under `data/raw/`. In the cloud they live in a **private GCS bucket**, and the app downloads them at startup.

### Tools

I added a few **dummy tools** around retrieval and profile actions (search background, pretend to email a resume, rough JD match).

I also added **system tools**—battery status, RAM, temp files. They’re irrelevant to a portfolio bot. That’s fine. The point was to show a second agent with a different toolset, not to monitor laptops in production.

### Chatbot

A thin API (`POST /chat`) and a small web UI at `/`. Locally:

```bash
uvicorn app.main:app --reload --port 8080
```

Open localhost, ask a question, confirm the graph routes and tools fire. That’s your baseline.

A supervisor sits on top and routes to a profile agent or a system agent. High level, not magic:

```text
User message → Supervisor → Profile agent (RAG/tools)
                          → System agent (system tools)
                          → General reply
```

---

## Step 2 — Dummy what won’t work in the cloud

On a Mac, system metrics can be real. On **Cloud Run**, there is no laptop battery.

So for GCP I **dummied those tool results**—random simulated RAM/battery/file paths. Same tool names, cloud-safe behavior.

That’s a useful habit: design tools for the environment they’ll run in. Don’t discover it after deploy.

Secrets stay out of Git. Locally use `.env`. In GCP, put `OPENAI_API_KEY` in Secret Manager and inject it into Cloud Run.

---

## Step 3 — Set up Git like a small team in prod

Even solo, use the same shape teams use in production.

**`develop`** is where you integrate work.  
**`main`** is production—and merging to it triggers deploy.

Optional feature branches can come later. For this project, working on `develop` and opening a pull request into `main` is enough.

Why bother? Because **CI/CD teaches the production habit**:

- You don’t SSH into a box to “update the bot”
- You push code, open a pull request, merge when checks look good
- The pipeline builds a Docker image and deploys it

Protect `main` so it only updates through PRs. That’s how prod teams avoid “I fixed it live and forgot to commit.”

Day-to-day:

```text
edit locally → commit → push develop → pull request into main → merge
```

---

## Step 4 — Set up GCP on a trial account

Use a Google Cloud trial project. You need roughly:

1. A **GCP project** (billing/trial credits are fine for demos)  
2. **Secret Manager** for the OpenAI key  
3. A **private Cloud Storage bucket** for resume/bio PDFs  
4. **Artifact Registry** for Docker images  
5. **Cloud Run** to run the container  
6. A **service account** GitHub Actions can impersonate  
7. **Workload Identity Federation** so GitHub talks to GCP without downloading a long-lived JSON key  

You’re not managing a VM. Cloud Run is the “instance”: it runs your container, scales, and gives you a HTTPS URL.

The Dockerfile stays minimal—install deps, copy the app, start uvicorn on `$PORT`. Documents come from the private bucket at runtime, not from the public repo.

---

## Step 5 — Connect local → Git → GCP

This is the part tutorials skip.

**Your laptop** holds the code, local `.env`, and local PDFs.  
**GitHub** holds the repo, PRs, Actions, and GCP auth secrets (project ID, region, service account, WIF provider, bucket name).  
**GCP** holds the OpenAI secret, private PDF bucket, image registry, and Cloud Run service.

Wire them once:

- Point `git remote` at your GitHub repo  
- Add the GitHub Actions secrets for GCP auth + `RAG_GCS_BUCKET`  
- Give the Cloud Run runtime service account permission to read the PDF bucket  
- Deploy Cloud Run so it reads `OPENAI_API_KEY` from Secret Manager  

After that, your laptop never needs to talk to Cloud Run directly for deploys. **Git is the control plane.**

---

## Step 6 — Push, PR, deploy

```text
git push origin develop
```

Open a **pull request** from `develop` into `main`. CI can lint/build. When you merge:

1. CD builds the Docker image  
2. Pushes it to Artifact Registry  
3. Deploys a new Cloud Run revision  

Refresh the public URL. That’s the whole loop.

The chat UI is for humans. `/docs` is there if you want the raw API. The UI also links to the repo so readers can inspect how it’s wired.

---

## What this is—and isn’t

**Is:** a complete 0→1 path for RAG + tools + chatbot + Git CI/CD + Cloud Run.

**Isn’t:** a secure, multi-tenant, guarded agent platform. No rate limits, no prompt firewall, stubbed email/JD tools, simulated system metrics in the cloud.

Improvise from here: real email, auth, better memory, stricter routing, scrubbing old secrets from git history. Those are 1→N problems.

Getting something live that updates on merge is the N=1 problem. This gets you there.

---

## Recap

1. Build a basic agent locally (RAG + tools + chat)  
2. Keep PDFs and API keys out of the public repo  
3. Dummy cloud-hostile tools  
4. Use Git branches and pull requests like a tiny prod team  
5. Stand up a trial GCP project (secrets, private bucket, registry, Cloud Run)  
6. Connect GitHub ↔ GCP with Workload Identity  
7. Push → PR → merge → image deploys  

Clone the repo, ask your coding agent for the file-level details, and run the same path on your own docs.

Zero to one. Then make it yours.
