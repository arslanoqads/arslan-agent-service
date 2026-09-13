import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import app.config.settings  # Load .env and validate OPENAI_API_KEY before graph init
from app.guardrails import (
    BUDGET_LIMIT_MESSAGE,
    INJECTION_REFUSAL,
    assess_message,
    consume_question,
    enforce_limit,
    release_question,
)
from app.evals.durable import get_golden_store
from app.evals.metrics import aggregate_golden_metrics, match_scores
from app.evals.runner import load_public_cases, public_case
from app.observability.model import public_question, public_trace
from app.observability.seed import seed_observability
from app.observability.store import DEFAULT_LIST_LIMIT, classify_outcome, get_store, summary
from app.runtime.errors import public_error_message
from app.runtime.runner import record_guardrail, stream_turn

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Arslan Portfolio Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_store = get_store()


class ChatQuery(BaseModel):
    message: str
    thread_id: str = Field(default="default_session")


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    trace: dict | None = None


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def require_observability(request: Request) -> None:
    expected = os.getenv("OBSERVABILITY_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Observability token is not configured.")
    provided = request.headers.get("x-observability-token") or request.query_params.get("token")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
def chat_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/traces")
def traces_page():
    return FileResponse(STATIC_DIR / "traces.html")


@app.get("/health")
def health_check():
    return {"status": "active", "service": "Arslan Portfolio Assistant"}


async def run_chat(query: ChatQuery, request: Request):
    ip = client_ip(request)
    blocked = consume_question(ip)
    if blocked:
        trace = record_guardrail(query.thread_id, query.message, "budget", BUDGET_LIMIT_MESSAGE)
        yield {
            "type": "done",
            "response": BUDGET_LIMIT_MESSAGE,
            "trace": public_trace(trace),
            "trace_id": trace["id"],
        }
        return

    unsafe, category = assess_message(query.message)
    if unsafe:
        enforce_limit(ip)
        refusal = INJECTION_REFUSAL
        trace = record_guardrail(query.thread_id, query.message, f"injection:{category}", refusal)
        yield {
            "type": "done",
            "response": refusal,
            "trace": public_trace(trace),
            "trace_id": trace["id"],
        }
        return

    try:
        async for event in stream_turn(query.message, query.thread_id):
            if event.get("type") == "error":
                release_question(ip)
                friendly = public_error_message(event.get("detail") or "")
                yield {
                    "type": "done",
                    "response": friendly,
                    "trace": event.get("trace"),
                    "trace_id": event.get("trace_id"),
                }
                return
            yield event
    except Exception as exc:
        release_question(ip)
        yield {
            "type": "done",
            "response": public_error_message(str(exc)),
            "trace": None,
            "trace_id": None,
        }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: ChatQuery, request: Request):
    final = None
    async for event in run_chat(query, request):
        if event["type"] in {"done", "error"}:
            final = event
    if not final:
        raise HTTPException(status_code=500, detail="Chat failed")
    if final["type"] == "error":
        return ChatResponse(
            response=public_error_message(final.get("detail") or ""),
            thread_id=query.thread_id,
            trace=final.get("trace"),
        )
    return ChatResponse(response=final["response"], thread_id=query.thread_id, trace=final.get("trace"))


@app.post("/chat/stream")
async def chat_stream(query: ChatQuery, request: Request):
    async def generate():
        async for event in run_chat(query, request):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/observability/traces")
def list_traces(request: Request):
    # Auto-seed a rich demo week when the durable store is empty (showcase).
    try:
        seed_observability(force=False)
    except Exception:
        pass
    try:
        raw = _store.list_traces(DEFAULT_LIST_LIMIT) or []
    except Exception:
        raw = []
    traces = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("id") == "demo-seed-marker" or item.get("stop_reason") == "demo_seed":
            continue
        try:
            item = dict(item)
            item["outcome"] = classify_outcome(item)
            traces.append(public_trace(item))
        except Exception:
            continue
    backend = getattr(_store, "backend", type(_store).__name__)
    persistence = {}
    if hasattr(_store, "persistence_status"):
        try:
            persistence = _store.persistence_status()
        except Exception:
            persistence = {}
    return {
        "traces": traces,
        "summary": summary(traces),
        "backend": backend,
        "persistence": persistence,
        "privacy": (
            "Public showcase view. Emails, phones, resume text, tool arguments, "
            "and raw errors are removed. Failed tool turns are listed with outcome=tool_error."
        ),
    }


@app.post("/observability/seed-demo")
def seed_demo(request: Request):
    """Token-protected: (re)populate demo sessions for the observability dashboard."""
    require_observability(request)
    force = (request.query_params.get("force") or "").lower() in {"1", "true", "yes"}
    result = seed_observability(force=force, sessions=55)
    return result


@app.get("/observability/golden-set")
def list_golden_set():
    try:
        seed_observability(force=False)
    except Exception:
        pass
    raw_cases = load_public_cases()
    cases = [public_case(case) for case in raw_cases]
    families = {}
    severities = {}
    sources = {}
    for case in cases:
        families[case["family"]] = families.get(case["family"], 0) + 1
        severities[case["severity"]] = severities.get(case["severity"], 0) + 1
        sources[case["source"]] = sources.get(case["source"], 0) + 1
    persistence = {}
    durable_scores = []
    try:
        store = get_golden_store()
        persistence = store.persistence_status()
        durable_scores = store.list_scores()
    except Exception:
        persistence = {}
    try:
        traces = _store.list_traces(DEFAULT_LIST_LIMIT) or []
    except Exception:
        traces = []
    metrics = aggregate_golden_metrics(raw_cases, traces, durable_scores)
    # Attach per-case avg onto the public list for the UI badges.
    score_by_id = {row["id"]: row for row in metrics.get("case_scores") or []}
    for case in cases:
        row = score_by_id.get(case.get("id") or "")
        if row:
            case["avg_score"] = row.get("avg_score")
            case["score_n"] = row.get("n")
            case["pass_rate"] = row.get("pass_rate")
    return {
        "cases": cases,
        "summary": {
            "count": len(cases),
            "families": families,
            "severities": severities,
            "sources": sources,
        },
        "metrics": metrics,
        "persistence": persistence,
    }


@app.get("/observability/traces/{trace_id}")
def get_trace(trace_id: str):
    try:
        trace = _store.get(trace_id)
    except Exception:
        trace = None
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    trace = dict(trace)
    trace["outcome"] = classify_outcome(trace)
    return public_trace(trace)


@app.post("/observability/traces/{trace_id}/eval-stub")
def save_eval_stub(trace_id: str, request: Request):
    require_observability(request)
    trace = _store.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    case = {
        "id": f"trace-{trace_id[:8]}",
        "family": "POS",
        "input": public_question(trace.get("question") or ""),
        "expected_tool": "none",
        "tools_called": trace.get("tools") or [],
        "must_include": [],
        "must_not_include": [],
        "oracle": "code",
        "severity": "major",
        "source": "production",
        "stop_reason": trace.get("stop_reason"),
    }
    # Durable first — Cloud Run container disk is ephemeral and tests/ is not in the image.
    get_golden_store().save(case)
    path = Path("tests/evals/golden_set.private.json")
    try:
        cases = []
        if path.exists():
            cases = json.loads(path.read_text())
        cases = [item for item in cases if item.get("id") != case["id"]]
        cases.append(case)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cases, indent=2) + "\n")
    except Exception:
        pass
    return {"ok": True, "case": public_case(case)}
