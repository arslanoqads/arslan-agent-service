import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import MODEL_NAME, PIPELINE_VERSION, PROMPT_VERSION, RECURSION_LIMIT, SYSTEM_PROMPT, TOOLS_BRIEF, TOKEN_CEILING, builder
from app.cache.answers import cacheable, lookup, store as store_answer
from app.context.budget import assemble, current_context_budget
from app.guardrails import INJECTION_REFUSAL, looks_like_injection
from app.observability.cost import classify_error, estimate_usd
from app.observability.model import close_span, new_span, new_trace, persisted_trace, public_trace, span_kind
from app.observability.store import get_store
from app.rag.corpus import corpus_fingerprint
from app.runtime.route import DEGRADED_MESSAGE, TOKEN_CEILING_MESSAGE, is_greeting, is_links_request, is_provider_failure
from app.tools.actions import format_social_links
from app.tools.limits import current_thread_id, current_user_message

_graph = builder.compile(checkpointer=MemorySaver())
_store = None


def store():
    global _store
    if _store is None:
        _store = get_store()
    return _store


def save_trace(trace: dict) -> None:
    try:
        store().save(persisted_trace(trace))
    except Exception:
        return


def _usage(output) -> tuple[int, int]:
    meta = getattr(output, "usage_metadata", None) or {}
    if not meta and hasattr(output, "response_metadata"):
        meta = (output.response_metadata or {}).get("token_usage") or {}
    return int(meta.get("input_tokens") or meta.get("prompt_tokens") or 0), int(
        meta.get("output_tokens") or meta.get("completion_tokens") or 0
    )


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(value if content is None else content)


def _open_span(trace, open_spans, run_id, **kwargs) -> dict:
    parent_id = None
    for span in reversed(trace["spans"]):
        if span.get("status") == "running" and span.get("kind") in {"llm", "router"}:
            parent_id = span.get("id")
            break
    span = new_span(parent_id=parent_id, started_perf=time.perf_counter(), **kwargs)
    open_spans[run_id] = span
    trace["spans"].append(span)
    return span


def finish_trace(trace: dict, *, status: str, started: float, error: str | None = None) -> None:
    trace["status"] = status
    trace["error"] = error
    trace["pipeline_version"] = PIPELINE_VERSION
    trace["prompt_version"] = PROMPT_VERSION
    trace["model"] = MODEL_NAME
    trace["context_budget"] = current_context_budget.get() or trace.get("context_budget") or {}
    trace["cost_usd"] = estimate_usd(MODEL_NAME, trace.get("input_tokens") or 0, trace.get("output_tokens") or 0)
    trace["ended_at"] = datetime.now(timezone.utc).isoformat()
    trace["duration_ms"] = round((time.perf_counter() - started) * 1000)


def _embed(text: str):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings().embed_query(text)


def _stamp_budget(trace: dict) -> None:
    budget = current_context_budget.get()
    if budget:
        trace["context_budget"] = budget


def complete_short(trace: dict, *, started: float, name: str, kind: str, output: str, stop_reason: str, route: str):
    span = new_span(
        name=name,
        kind=kind,
        loop_index=0,
        attempt=0,
        parent_id=None,
        context="",
        started_perf=time.perf_counter(),
    )
    close_span(span, ended_perf=time.perf_counter(), output=output)
    trace["spans"].append(span)
    trace["route"] = route
    trace["stop_reason"] = stop_reason
    if kind == "guardrail":
        trace["error_kind"] = "guardrail"
    finish_trace(trace, status="ok", started=started)
    save_trace(trace)
    return {"type": "done", "response": output, "trace": public_trace(trace), "trace_id": trace["id"]}


def record_guardrail(thread_id: str, question: str, name: str, output: str):
    started = time.perf_counter()
    trace = new_trace(thread_id, question)
    span = new_span(
        name=name,
        kind="guardrail",
        loop_index=0,
        attempt=0,
        parent_id=None,
        context="",
        started_perf=started,
    )
    close_span(span, ended_perf=time.perf_counter(), output=output)
    trace["spans"].append(span)
    trace["stop_reason"] = name
    trace["route"] = "unknown"
    trace["error_kind"] = "guardrail"
    finish_trace(trace, status="ok", started=started)
    save_trace(trace)
    return trace


async def stream_turn(message: str, thread_id: str):
    trace = new_trace(thread_id, message)
    thread_token = current_thread_id.set(thread_id)
    message_token = current_user_message.set(message)
    started = time.perf_counter()
    open_spans: dict[str, dict] = {}
    loop_index = 0
    packed = assemble(SYSTEM_PROMPT, TOOLS_BRIEF, "", [])
    current_context_budget.set(packed["context_budget"])
    trace["context_budget"] = packed["context_budget"]
    trace["pipeline_version"] = PIPELINE_VERSION
    trace["prompt_version"] = PROMPT_VERSION
    save_trace(trace)
    yield {"type": "trace", "trace": public_trace(trace)}

    if looks_like_injection(message):
        span = new_span(
            name="injection",
            kind="guardrail",
            loop_index=0,
            attempt=0,
            parent_id=None,
            context="",
            started_perf=time.perf_counter(),
        )
        close_span(span, ended_perf=time.perf_counter(), output=INJECTION_REFUSAL)
        trace["spans"].append(span)
        trace["error_kind"] = "guardrail"
        trace["route"] = "greeting"
        trace["stop_reason"] = "injection"
        finish_trace(trace, status="ok", started=started)
        save_trace(trace)
        yield {
            "type": "done",
            "response": INJECTION_REFUSAL,
            "trace": public_trace(trace),
            "trace_id": trace["id"],
        }
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    if is_greeting(message):
        yield complete_short(
            trace,
            started=started,
            name="general_responder",
            kind="llm",
            output="Hello. I can answer from the resume, email it, book a 30-minute intro call, compare a job description, or share public links.",
            stop_reason=None,
            route="greeting",
        )
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    if is_links_request(message):
        yield complete_short(
            trace,
            started=started,
            name="get_social_links",
            kind="tool",
            output=format_social_links(),
            stop_reason=None,
            route="greeting",
        )
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    fingerprint = corpus_fingerprint()
    cached = lookup(message, fingerprint)
    if cached:
        trace["cache"] = cached
        trace["tools"] = []
        yield complete_short(
            trace,
            started=started,
            name="cache",
            kind="retrieval",
            output=cached["answer"],
            stop_reason=None,
            route="portfolio",
        )
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    trace["route"] = "portfolio"
    try:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
        async for event in _graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            name = event.get("name") or "step"
            run_id = event.get("run_id") or name
            if kind == "on_chat_model_start":
                _open_span(
                    trace,
                    open_spans,
                    run_id,
                    name=name,
                    kind="llm",
                    loop_index=loop_index,
                    attempt=trace["attempt"] + 1,
                    context=_text(event.get("data", {}).get("input")),
                )
                trace["attempt"] += 1
            elif kind == "on_chat_model_stream" and run_id in open_spans:
                if open_spans[run_id]["ttft_ms"] is None:
                    open_spans[run_id]["ttft_ms"] = round(
                        (time.perf_counter() - open_spans[run_id]["_started_perf"]) * 1000
                    )
                token_text = _text(event.get("data", {}).get("chunk"))
                if token_text:
                    yield {"type": "token", "text": token_text}
                continue
            elif kind == "on_chat_model_end" and run_id in open_spans:
                span = open_spans.pop(run_id)
                output = event.get("data", {}).get("output")
                prompt_tokens, completion_tokens = _usage(output)
                span["input_tokens"] = prompt_tokens
                span["output_tokens"] = completion_tokens
                close_span(span, ended_perf=time.perf_counter(), output=_text(output))
                trace["input_tokens"] += prompt_tokens
                trace["output_tokens"] += completion_tokens
            elif kind == "on_tool_start":
                loop_index += 1
                tool_name = name
                _open_span(
                    trace,
                    open_spans,
                    run_id,
                    name=tool_name,
                    kind=span_kind(tool_name, "tool"),
                    loop_index=loop_index,
                    attempt=trace["attempt"],
                    context=_text(event.get("data", {}).get("input")),
                )
                if tool_name not in trace["tools"]:
                    trace["tools"].append(tool_name)
                trace["loop_count"] = loop_index
            elif kind == "on_tool_end" and run_id in open_spans:
                span = open_spans.pop(run_id)
                output = _text(event.get("data", {}).get("output"))
                error = output if output.lower().startswith("could not") else None
                close_span(span, ended_perf=time.perf_counter(), output=output, error=error)
                status = "error" if error else "ok"
                if output.startswith("Cannot") or "turned off" in output or "already" in output:
                    status = "refused"
                trace["tool_status"].append({"name": span["name"], "status": status})
                _stamp_budget(trace)
            elif kind == "on_chain_start" and name in {"supervisor", "portfolio_agent", "general_responder"}:
                _open_span(
                    trace,
                    open_spans,
                    run_id,
                    name=name,
                    kind=span_kind(name, "llm"),
                    loop_index=loop_index,
                    attempt=trace["attempt"],
                    context="",
                )
            elif kind == "on_chain_end" and run_id in open_spans:
                span = open_spans.pop(run_id)
                close_span(span, ended_perf=time.perf_counter())
            else:
                continue
            _stamp_budget(trace)
            if (trace.get("input_tokens") or 0) + (trace.get("output_tokens") or 0) > TOKEN_CEILING:
                trace["stop_reason"] = "token_ceiling"
                trace["error_kind"] = "guardrail"
                finish_trace(trace, status="error", started=started, error=TOKEN_CEILING_MESSAGE)
                save_trace(trace)
                yield {"type": "done", "response": TOKEN_CEILING_MESSAGE, "trace": public_trace(trace), "trace_id": trace["id"]}
                return
            save_trace(trace)
            yield {"type": "trace", "trace": public_trace(trace)}

        state = await _graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages") or []
        answer = _text(messages[-1]) if messages else ""
        if cacheable(message) and answer:
            try:
                store_answer(message, answer, fingerprint, embed=_embed)
            except Exception:
                store_answer(message, answer, fingerprint)
        finish_trace(trace, status="ok", started=started)
        save_trace(trace)
        yield {"type": "done", "response": answer, "trace": public_trace(trace), "trace_id": trace["id"]}
    except Exception as exc:
        if is_provider_failure(str(exc)):
            trace["stop_reason"] = "fallback"
            trace["error_kind"] = "provider"
            yield complete_short(
                trace,
                started=started,
                name="fallback",
                kind="guardrail",
                output=DEGRADED_MESSAGE,
                stop_reason="fallback",
                route="portfolio",
            )
            return
        for span in open_spans.values():
            close_span(span, ended_perf=time.perf_counter(), error=str(exc))
        trace["error_kind"] = classify_error(str(exc))
        trace["stop_reason"] = trace.get("stop_reason") or "tool_error"
        finish_trace(trace, status="error", started=started, error=str(exc))
        save_trace(trace)
        yield {"type": "error", "detail": str(exc), "trace": public_trace(trace)}
    finally:
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
