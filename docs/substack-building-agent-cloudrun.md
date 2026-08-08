# From Laptop to Cloud Run: Building a Multi-Agent Chatbot with RAG, Tools, and Git-Based Deploy

*A practical, high-level walkthrough of building a FastAPI + LangGraph agent that answers from your resume, calls tools, and ships to Google Cloud Run every time you merge to `main`.*

---

Most “AI chatbot” tutorials stop at a notebook. This one goes further: you’ll build a small **multi-agent** service with **hybrid RAG** and **tools**, wrap it in **FastAPI**, then deploy it to **Cloud Run** using **GitHub Actions**—so a merge to `main` becomes a live URL.

This is the architecture behind a real portfolio agent: ask about someone’s background, simulate system diagnostics, keep a conversation thread, and ship updates without SSHing into a server.

---

## What you’ll build

A single HTTP service with two capabilities:

1. **Profile agent** — answers questions from resume/bio PDFs (RAG), can “send resume” and score a job description.
2. **System agent** — handles OS-style questions (RAM, battery, temp files). In the cloud these return **simulated** results (Cloud Run isn’t your MacBook).

A **supervisor** routes each user message to the right agent—or to a light general responder for greetings.

```text
User → FastAPI /chat
         ↓
      Supervisor
      /    |    \
Profile  System  General
  ↕         ↕
 Tools     Tools
```

Live shape of the product:

- `GET /` → **chat UI** (for article readers / demos)  
- `POST /chat` → `{ "message": "...", "thread_id": "..." }`  
- `GET /health` → health JSON  
- Bonus: FastAPI’s `/docs` for raw API tryouts  
- Repo link in the UI header: [github.com/arslanoqads/arslan-agent-service](https://github.com/arslanoqads/arslan-agent-service)

---

## Project shape (keep it boring)

```text
app/
  main.py              # FastAPI gateway
  config/settings.py   # load .env, require OPENAI_API_KEY
  agent/graph.py       # LangGraph: supervisor + agents
  rag/hybrid_engine.py # BM25 + embeddings RAG
  tools/
    profile_tools.py
    system_tools.py
data/raw/              # resume.pdf, bio.pdf
Dockerfile
.github/workflows/
  ci.yml
  cd.yml
```

Two ideas matter more than folder names:

1. **Secrets never go in Git** (`.env` is gitignored; production uses Secret Manager).
2. **`main` is production**; you work on `develop`, then PR into `main`.

---

## Part 1 — Hybrid RAG (answers grounded in PDFs)

RAG = retrieve relevant chunks from your documents, then let the model answer with that context.

**Hybrid** means you combine:

- **Dense search** — OpenAI embeddings + vector similarity (good for meaning: “AI product leadership”)
- **Sparse search** — BM25 keywords (good for exact terms: company names, titles)

Then blend them with an ensemble retriever:

```python
# Conceptual core of HybridRAGEngine
splits = text_splitter.split_documents(pdf_docs)

dense = InMemoryVectorStore.from_documents(
    splits, embedding=OpenAIEmbeddings()
).as_retriever(search_kwargs={"k": 3})

bm25 = BM25Retriever.from_documents(splits)
bm25.k = 3

retriever = EnsembleRetriever(
    retrievers=[bm25, dense],
    weights=[0.5, 0.5],
)
```

Expose that as a **tool** the profile agent can call:

```python
@tool("query_arslan_profile")
def query_arslan_profile(query: str) -> str:
    """Search resume + bio with hybrid RAG."""
    return rag_engine.query(query)
```

**Why a tool, not “stuff the whole PDF into the prompt”?**  
Tools keep retrieval optional and on-demand. The model decides when it needs documents. That scales better as docs grow.

---

## Part 2 — Tools (actions, not just text)

Besides RAG, the profile agent can have lightweight actions:

- `send_resume_email` — simulate dispatch (swap for SendGrid later)
- `evaluate_jd_match` — structured “fit” response for a job description

The system agent exposes diagnostics. On a laptop you might use `psutil`; **on Cloud Run, simulate**:

```python
@tool("get_ram_usage")
def get_ram_usage() -> str:
    percent = round(random.uniform(35.0, 82.0), 1)
    return f"[SIMULATED] RAM Usage: {percent}% used ..."
```

That’s an underrated production lesson: **design tools for the environment they’ll run in.**

---

## Part 3 — LangGraph: supervisor + sub-agents

LangGraph is a state machine for agents. Shared state is mostly the message list:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str
```

Each specialist is a node: call an LLM with tools bound, return a new message. A `ToolNode` executes tool calls; `tools_condition` loops until the model stops calling tools.

The supervisor only **routes**:

```python
# Route to profile_agent | system_agent | general_responder
decision = supervisor_llm.invoke([supervisor_prompt] + state["messages"])
return {"next_node": decision.next_destination}
```

Wire the graph:

```python
builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", supervisor_router, {...})
builder.add_conditional_edges("profile_agent", tools_condition, {...})
builder.add_edge("profile_tools", "profile_agent")
# same pattern for system_agent ...
```

**Greeting trap:** if “FINISH” goes straight to `END`, `/chat` may echo the user message. Prefer a small `general_responder` node that actually writes a reply.

---

## Part 4 — FastAPI gateway + session memory

Keep the HTTP layer thin. Load secrets first, compile the graph with a checkpointer, invoke with a `thread_id`:

```python
import app.config.settings  # load_dotenv + validate OPENAI_API_KEY
from app.agent.graph import builder

memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    config = {"configurable": {"thread_id": query.thread_id}}
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=query.message)]},
        config=config,
    )
    return {"response": result["messages"][-1].content, "thread_id": query.thread_id}
```

Local run:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Try `http://localhost:8080` for the chat UI, or `/docs` for the raw API explorer.

Serve the UI from FastAPI so one Cloud Run service hosts both page and API:

```python
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def chat_ui():
    return FileResponse(STATIC_DIR / "index.html")
```

The page posts to `/chat` and links out to the GitHub repo so readers can use the bot and inspect the code.

---

## Part 5 — Containerize for Cloud Run

Cloud Run runs a container and injects `PORT`. A minimal Dockerfile:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data/raw ./data/raw
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
```

Ship PDFs in the image for v1 (or later pull from GCS). Keep `.env` out of the image; pass `OPENAI_API_KEY` at runtime from Secret Manager.

---

## Part 6 — Git branches (the human workflow)

Think in three lanes:

| Branch | Role |
|--------|------|
| `main` | Production (triggers deploy) |
| `develop` | Integration draft |
| `feat/...` | Optional one-feature side path |

Solo flow that works:

1. Edit on `develop`  
2. Commit + push  
3. Open PR → `main`  
4. Merge when CI is green  
5. CD deploys  

Protect `main` so you can’t push production by accident: require PRs + status checks.

---

## Part 7 — GCP once, then forget the servers

One-time cloud setup (high level):

1. **GCP project** + enable Cloud Run, Artifact Registry, Secret Manager, IAM APIs  
2. **Secret Manager** secret: `OPENAI_API_KEY`  
3. **Artifact Registry** Docker repo (e.g. `arslan-agent` in `us-central1`)  
4. **Service account** for GitHub Actions (`github-runner`) with rights to push images and deploy Cloud Run  
5. **Workload Identity Federation** so GitHub can impersonate that SA **without** a downloaded JSON key  
6. Grant the Cloud Run runtime SA **Secret Accessor** on `OPENAI_API_KEY`

In GitHub → Settings → Secrets, store only GCP auth values:

- `GCP_PROJECT_ID`  
- `GCP_REGION`  
- `GCP_SERVICE_ACCOUNT`  
- `GCP_WORKLOAD_IDENTITY_PROVIDER`  

Do **not** duplicate the OpenAI key in GitHub if Cloud Run reads Secret Manager.

---

## Part 8 — CI/CD: merge to `main` = new revision

**CI** (on PRs / `develop`): install deps, `compileall`, smoke-import cloud-safe tools, `docker build`.

**CD** (on push to `main`):

```yaml
# Conceptual CD steps
- uses: google-github-actions/auth@v2   # Workload Identity
- gcloud auth configure-docker ...
- docker build && docker push
- uses: google-github-actions/deploy-cloudrun@v2
  with:
    image: ...
    flags: >-
      --allow-unauthenticated
      --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest
```

After a green CD run you get a URL like:

`https://YOUR-SERVICE-xxxxx.us-central1.run.app/`

Open that URL for the chat UI (with a **View repo** button). Or call the API:

```bash
curl -X POST https://YOUR-SERVICE/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Arslan known for?","thread_id":"demo1"}'
```

---

## Day-2 loop (what you do forever after)

```text
change code → commit → push develop → PR to main → merge → wait for CD → test /docs
```

No SSH. No “remember the deploy command.” The pipeline *is* the deploy command.

---

## What this teaches (beyond the demo)

1. **Agents are graphs**, not one giant prompt.  
2. **RAG belongs behind a tool** when the model should choose when to retrieve.  
3. **Environment-aware tools** (real locally, simulated in Cloud Run) prevent silent production failures.  
4. **FastAPI is the door**; LangGraph is the brain.  
5. **Git + WIF + Cloud Run** is enough to run a serious demo without babysitting VMs.  
6. **Secrets and PDFs** need an explicit plan (Secret Manager + image/GCS)—Git alone isn’t enough.

---

## Stretch goals

- Real email via SendGrid/Postmark  
- Persist checkpointer state (Redis / Postgres) instead of in-memory  
- Move PDFs to GCS and load at startup  
- Add a tiny chat UI in front of `/chat`  
- Feature branches when you and a teammate ship B and C in parallel  

---

## Closing

You don’t need a platform team to ship an agent. You need a clear graph, grounded retrieval, honest tool design, a thin API, and a boring path from `git push` to Cloud Run.

Build it once. Protect `main`. Merge with confidence. Then iterate in public—one green CD at a time.

---

*If you build your own version, start with one agent + one RAG tool + `/chat`, then add the supervisor. Complexity is easier to add than to debug.*
