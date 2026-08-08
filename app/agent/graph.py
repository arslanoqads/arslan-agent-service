from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.profile_tools import (
    query_arslan_profile,
    send_resume_email,
    evaluate_jd_match,
)
from app.tools.system_tools import (
    get_ram_usage,
    get_battery_status,
    create_timestamp_temp_file,
)


# =====================================================================
# 1. SHARED AGENT STATE
# =====================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next_node: str


# =====================================================================
# 2. SUB-AGENT 1: PROFILE AGENT (Arslan's Background)
# =====================================================================
profile_tools = [query_arslan_profile, send_resume_email, evaluate_jd_match]
profile_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(profile_tools)

def profile_agent_node(state: State):
    system_prompt = SystemMessage(
        content="You are a profile specialist focused on Arslan's professional background, resume, bio, and job fit."
    )
    response = profile_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# =====================================================================
# 3. SUB-AGENT 2: SYSTEM AGENT (Computer & OS Diagnostics)
# =====================================================================
system_tools = [get_ram_usage, get_battery_status, create_timestamp_temp_file]
system_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(system_tools)

def system_agent_node(state: State):
    system_prompt = SystemMessage(
        content="You are an OS and hardware diagnostic specialist focused on system memory, battery metrics, and local file generation."
    )
    response = system_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# =====================================================================
# 4. GENERAL RESPONDER (greetings / direct answers)
# =====================================================================
general_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def general_responder_node(state: State):
    system_prompt = SystemMessage(
        content=(
            "You are the front desk for Arslan's multi-agent assistant. "
            "Handle greetings and light chitchat briefly. "
            "Mention you can help with Arslan's background/resume or local system diagnostics when relevant."
        )
    )
    response = general_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# =====================================================================
# 5. SUPERVISOR ROUTER NODE
# =====================================================================
class RouterOutput(BaseModel):
    next_destination: Literal["profile_agent", "system_agent", "general_responder"] = Field(
        description=(
            "Route to profile_agent for Arslan/resume/bio/email/JD questions, "
            "system_agent for computer/OS tasks, or general_responder for greetings and simple chat."
        )
    )

supervisor_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(RouterOutput)

def supervisor_node(state: State):
    supervisor_prompt = SystemMessage(
        content="""You are the Supervisor Orchestrator. Analyze the user request and route to the correct agent:
        - Route to 'profile_agent' if the query is about Arslan's resume, bio, work background, email requests, or job descriptions.
        - Route to 'system_agent' if the query asks about computer RAM, battery usage, OS metrics, or creating local temp files.
        - Route to 'general_responder' for greetings, thanks, or other simple conversation that needs a direct reply."""
    )
    decision = supervisor_llm.invoke([supervisor_prompt] + state["messages"])
    return {"next_node": decision.next_destination}


# =====================================================================
# 6. GRAPH CONSTRUCTION
# =====================================================================
builder = StateGraph(State)

builder.add_node("supervisor", supervisor_node)
builder.add_node("profile_agent", profile_agent_node)
builder.add_node("profile_tools", ToolNode(profile_tools))
builder.add_node("system_agent", system_agent_node)
builder.add_node("system_tools", ToolNode(system_tools))
builder.add_node("general_responder", general_responder_node)

builder.add_edge(START, "supervisor")

def supervisor_router(state: State):
    return state["next_node"]

builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "profile_agent": "profile_agent",
        "system_agent": "system_agent",
        "general_responder": "general_responder",
    },
)

builder.add_edge("general_responder", END)

builder.add_conditional_edges("profile_agent", tools_condition, {"tools": "profile_tools", END: END})
builder.add_edge("profile_tools", "profile_agent")

builder.add_conditional_edges("system_agent", tools_condition, {"tools": "system_tools", END: END})
builder.add_edge("system_tools", "system_agent")
