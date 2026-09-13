import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from app.agent.graph import MODEL_NAME, PIPELINE_VERSION, PROMPT_VERSION, RECURSION_LIMIT, SYSTEM_PROMPT, TOOLS_BRIEF, TOKEN_CEILING, builder
from app.cache.answers import cacheable, lookup, store as store_answer
from app.context.budget import assemble, current_context_budget
from app.evals.durable import get_golden_store
from app.evals.metrics import match_scores
from app.evals.runner import load_public_cases
from app.guardrails import INJECTION_REFUSAL, looks_like_injection
from app.observability.cost import classify_error, estimate_usd
from app.observability.model import close_span, new_span, new_trace, persisted_trace, public_trace, span_kind
from app.observability.rag_triad import annotate_rag_triage
from app.observability.store import get_store
from app.rag.corpus import corpus_fingerprint
from app.runtime.errors import public_error_message
from app.runtime.route import DEGRADED_MESSAGE, TOKEN_CEILING_MESSAGE, is_greeting, is_links_request, is_provider_failure
from app.runtime.threads import append_turn, load_turns, turns_as_messages
from app.tools.actions import format_social_links
from app.tools.limits import current_thread_id, current_user_message

# No process-local checkpointer: Cloud Run hops would drop MemorySaver state.
# Durable turns are loaded from the thread store on every request.
_graph = builder.compile()
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
    try:
        _record_golden_scores(trace)
    except Exception:
        return


def _record_golden_scores(trace: dict) -> None:
    if not trace.get("question") or not trace.get("id"):
        return
    cases = load_public_cases()
    for point in match_scores(cases, [trace]):
        # Tag RAG golden focus when the matched case is triad-labeled.
        matched = next((case for case in cases if case.get("id") == point.get("case_id")), None)
        if matched and matched.get("eval_focus") in {
            "context_relevance",
            "answer_faithfulness",
            "answer_relevance",
        }:
            from app.observability.rag_triad import score_rag_golden_case

            point = score_rag_golden_case(matched, trace)
            point["id"] = f"{point['case_id']}__{point.get('trace_id')}"
        get_golden_store().save_score(point)


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
    tool_statuses = [item.get("status") for item in trace.get("tool_status") or []]
    if status == "error":
        trace["outcome"] = "processing_error"
    elif trace.get("error_kind") == "guardrail":
        trace["outcome"] = "guardrail"
    elif "error" in tool_statuses:
        trace["outcome"] = "tool_error"
        # Keep visitor-facing chat ok, but mark the stored trace as failed for observability.
        trace["status"] = "error"
        if not trace.get("error_kind"):
            trace["error_kind"] = "tool_error"
    elif "refused" in tool_statuses:
        trace["outcome"] = "tool_refused"
    else:
        trace["outcome"] = "success"


def _embed(text: str):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings().embed_query(text)


def _stamp_budget(trace: dict) -> None:
    budget = current_context_budget.get()
    if budget:
        trace["context_budget"] = budget


def complete_short(trace: dict, *, started: float, name: str, kind: str, output: str, stop_reason: str, route: str, thread_id: str | None = None, message: str | None = None):
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
    if thread_id and message:
        append_turn(thread_id, message, output)
    return {"type": "done", "response": output, "trace": public_trace(trace), "trace_id": trace["id"]}


def record_guardrail(thread_id: str, question: str, name: str, output: str, *, client_ip: str | None = None):
    started = time.perf_counter()
    trace = new_trace(thread_id, question, client_ip=client_ip)
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


async def stream_turn(message: str, thread_id: str, *, client_ip: str | None = None):
    trace = new_trace(thread_id, message, client_ip=client_ip)
    thread_token = current_thread_id.set(thread_id)
    message_token = current_user_message.set(message)
    started = time.perf_counter()
    open_spans: dict[str, dict] = {}
    loop_index = 0
    prior_turns = load_turns(thread_id)
    history_messages = turns_as_messages(prior_turns)
    packed = assemble(SYSTEM_PROMPT, TOOLS_BRIEF, "", [turn.get("content") or "" for turn in prior_turns])
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

    # Standalone greeting only when the thread is empty. "yes" after a booking offer must not reset.
    if is_greeting(message) and not prior_turns:
        yield complete_short(
            trace,
            started=started,
            name="general_responder",
            kind="llm",
            output="Hello. I can answer from the resume, email it, book a 30-minute intro call, compare a job description, or share public links.",
            stop_reason=None,
            route="greeting",
            thread_id=thread_id,
            message=message,
        )
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    if is_links_request(message) and not prior_turns:
        yield complete_short(
            trace,
            started=started,
            name="get_social_links",
            kind="tool",
            output=format_social_links(),
            stop_reason=None,
            route="greeting",
            thread_id=thread_id,
            message=message,
        )
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    fingerprint = corpus_fingerprint()
    # Never serve a cached answer into an active conversation — follow-ups need history.
    cached = None if prior_turns else lookup(message, fingerprint, embed=_embed)
    if cached:
        cache_kind = cached.get("kind") or "exact"
        trace["cache"] = {"kind": cache_kind, "similarity": cached.get("similarity")}
        trace["tools"] = []
        yield complete_short(
            trace,
            started=started,
            name=f"{cache_kind}_cache",
            kind="cache",
            output=cached["answer"],
            stop_reason=f"cache_{cache_kind}",
            route="portfolio",
            thread_id=thread_id,
            message=message,
        )
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
        return

    trace["route"] = "portfolio"
    last_ai_text = ""
    try:
        config = {"recursion_limit": RECURSION_LIMIT}
        graph_input = {
            "messages": history_messages + [HumanMessage(content=message)],
            "hops": [],
            "hop_results": [],
            "active_hop": "",
            "next_node": "",
        }
        async for event in _graph.astream_events(
            graph_input,
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
                text = _text(output)
                close_span(span, ended_perf=time.perf_counter(), output=text)
                trace["input_tokens"] += prompt_tokens
                trace["output_tokens"] += completion_tokens
                if text and not getattr(output, "tool_calls", None):
                    last_ai_text = text
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
                if (
                    output.startswith("Cannot")
                    or "turned off" in output
                    or "already" in output
                    or "in the past" in output.lower()
                    or "weekdays only" in output.lower()
                ):
                    status = "refused"
                trace["tool_status"].append({"name": span["name"], "status": status})
                _stamp_budget(trace)
            elif kind == "on_chain_start" and name in {
                "supervisor",
                "hop_entry",
                "hop_agent",
                "hop_done",
                "compose",
                "general_responder",
                "portfolio_agent",
            }:
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

        answer = last_ai_text
        if not answer:
            result = await _graph.ainvoke(graph_input, config=config)
            messages = result.get("messages") or []
            answer = _text(messages[-1]) if messages else ""

        if cacheable(message) and answer and not prior_turns:
            try:
                store_answer(message, answer, fingerprint, embed=_embed)
            except Exception:
                store_answer(message, answer, fingerprint)
        append_turn(thread_id, message, answer)
        finish_trace(trace, status="ok", started=started)
        annotate_rag_triage(trace, answer)
        save_trace(trace)
        yield {"type": "done", "response": answer, "trace": public_trace(trace), "trace_id": trace["id"]}
    except Exception as exc:
        detail = str(exc)
        friendly = public_error_message(detail)
        for span in open_spans.values():
            close_span(span, ended_perf=time.perf_counter(), error=detail)
        if is_provider_failure(detail):
            stop_reason = "fallback"
            error_kind = "provider"
        else:
            stop_reason = "tool_error"
            error_kind = classify_error(detail)
        trace["stop_reason"] = stop_reason
        trace["error_kind"] = error_kind
        trace["route"] = trace.get("route") or "portfolio"
        finish_trace(trace, status="error", started=started, error=detail)
        annotate_rag_triage(trace, friendly)
        save_trace(trace)
        append_turn(thread_id, message, friendly)
        # Visitors get a calm reply; the private trace keeps the raw error.
        yield {
            "type": "done",
            "response": friendly,
            "trace": public_trace(trace),
            "trace_id": trace["id"],
        }
    finally:
        current_thread_id.reset(thread_token)
        current_user_message.reset(message_token)
