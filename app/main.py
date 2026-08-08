from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import app.config.settings  # Load .env and validate OPENAI_API_KEY before graph init
from app.agent.graph import builder

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Multi-Agent Portfolio & System Assistant API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)


class ChatQuery(BaseModel):
    message: str = Field(description="User message or query.")
    thread_id: str = Field(
        default="default_session",
        description="Unique session ID to maintain conversation history across turns.",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.get("/")
def chat_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "active", "service": "Multi-Agent System Assistant"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: ChatQuery):
    try:
        config = {"configurable": {"thread_id": query.thread_id}}
        inputs = {"messages": [HumanMessage(content=query.message)]}
        result = agent_graph.invoke(inputs, config=config)
        final_message = result["messages"][-1].content
        return ChatResponse(response=final_message, thread_id=query.thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
