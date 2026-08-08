from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import app.config.settings  # Load .env and validate OPENAI_API_KEY before graph init
from app.agent.graph import builder

app = FastAPI(title="Multi-Agent Portfolio & System Assistant API")

# 1. Attach In-Memory Checkpointer to maintain session memory
memory = MemorySaver()
agent_graph = builder.compile(checkpointer=memory)


# 2. Updated Request/Response Schemas
class ChatQuery(BaseModel):
    message: str = Field(description="User message or query.")
    thread_id: str = Field(
        default="default_session",
        description="Unique session ID to maintain conversation history across turns."
    )

class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.get("/")
def read_root():
    return {"status": "active", "service": "Multi-Agent System Assistant"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: ChatQuery):
    try:
        # Pass the thread_id into the graph config for state persistence
        config = {"configurable": {"thread_id": query.thread_id}}
        inputs = {"messages": [HumanMessage(content=query.message)]}
        
        # Invoke graph execution
        result = agent_graph.invoke(inputs, config=config)
        
        # Extract the final message content from state
        final_message = result["messages"][-1].content
        
        return ChatResponse(
            response=final_message,
            thread_id=query.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
