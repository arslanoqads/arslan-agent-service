from collections import defaultdict
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import app.config.settings  # Load .env and validate OPENAI_API_KEY before graph init
from app.agent.graph import builder

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_QUESTIONS_PER_IP = 2
BUDGET_LIMIT_MESSAGE = (
    "The number of questions that can be asked is limited to 2 per visitor "
    "to control budget. Thanks for trying the demo."
)

app = FastAPI(title="Multi-Agent Portfolio & System Assistant API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)

_ip_question_counts: dict[str, int] = defaultdict(int)
_ip_lock = Lock()


class ChatQuery(BaseModel):
    message: str = Field(description="User message or query.")
    thread_id: str = Field(
        default="default_session",
        description="Unique session ID to maintain conversation history across turns.",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@app.get("/")
def chat_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "active", "service": "Multi-Agent System Assistant"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: ChatQuery, request: Request):
    ip = _client_ip(request)
    with _ip_lock:
        used = _ip_question_counts[ip]
        if used >= MAX_QUESTIONS_PER_IP:
            return ChatResponse(response=BUDGET_LIMIT_MESSAGE, thread_id=query.thread_id)
        _ip_question_counts[ip] = used + 1

    try:
        config = {"configurable": {"thread_id": query.thread_id}}
        inputs = {"messages": [HumanMessage(content=query.message)]}
        result = agent_graph.invoke(inputs, config=config)
        final_message = result["messages"][-1].content
        return ChatResponse(response=final_message, thread_id=query.thread_id)
    except Exception as e:
        with _ip_lock:
            _ip_question_counts[ip] = max(0, _ip_question_counts[ip] - 1)
        raise HTTPException(status_code=500, detail=str(e))
