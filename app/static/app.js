(() => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const sendBtn = document.getElementById("send");
  const messages = document.getElementById("messages");
  const suggestions = document.getElementById("suggestions");
  const traceSteps = document.getElementById("trace-steps");
  const traceSummary = document.getElementById("trace-summary");
  const budgetBar = document.getElementById("budget-bar");
  const budgetLegend = document.getElementById("budget-legend");
  const toolStatus = document.getElementById("tool-status");

  const threadId =
    crypto.randomUUID?.() ||
    `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  function addMessage(role, text) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }

  function renderBudget(budget) {
    if (!budgetBar || !budgetLegend) return;
    const windowSize = budget.window || 1;
    const parts = [
      ["system", "budget-system"],
      ["tools", "budget-tools"],
      ["retrieved", "budget-retrieved"],
      ["history", "budget-history"],
      ["reserved", "budget-reserved"],
    ];
    budgetBar.replaceChildren();
    parts.forEach(([key, className]) => {
      const value = budget[key] || 0;
      if (!value) return;
      const segment = document.createElement("span");
      segment.className = className;
      segment.style.width = `${Math.max(2, (value / windowSize) * 100)}%`;
      budgetBar.appendChild(segment);
    });
    const used = (budget.system || 0) + (budget.tools || 0) + (budget.retrieved || 0) + (budget.history || 0);
    const percent = Math.round((used / windowSize) * 100);
    const cut = (budget.cut || []).length ? ` · cut ${budget.cut.join(", ")}` : "";
    budgetLegend.textContent = `Context ${used}/${windowSize} (${percent}%) · retrieved ${budget.retrieved || 0} · history ${budget.history || 0}${cut}`;
  }

  function renderTrace(trace) {
    if (!trace || !traceSteps) return;
    const tokens = (trace.input_tokens || 0) + (trace.output_tokens || 0);
    const late = trace.spans?.some((span) => span.kind === "llm" && span.ttft_ms != null && span.ttft_ms > 8000);
    traceSummary.textContent = `${trace.status || "running"} · ${trace.route || "unknown"} · ${trace.loop_count || 0} loops · ${tokens} tokens · ${trace.duration_ms ?? "—"} ms${late ? " · first token over budget" : ""}`;
    renderBudget(trace.context_budget || {});
    traceSteps.replaceChildren();
    (trace.spans || []).forEach((span) => {
      const item = document.createElement("li");
      const ttft = span.ttft_ms == null ? "" : ` · ttft ${span.ttft_ms} ms`;
      const duration = span.duration_ms == null ? "" : ` · ${span.duration_ms} ms`;
      item.textContent = `${span.kind} · ${span.name} · ${span.status} · loop ${span.loop_index || 0} · attempt ${span.attempt || 0}${ttft}${duration}`;
      traceSteps.appendChild(item);
    });
    if (toolStatus) {
      toolStatus.textContent = (trace.tool_status || []).map((item) => `${item.name} ${item.status}`).join(" · ");
    }
  }

  addMessage(
    "system",
    "Try a suggestion below. Demo limit: 2 questions per visitor to control budget."
  );

  suggestions?.addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-prompt]");
    if (!btn) return;
    input.value = btn.dataset.prompt;
    input.focus();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addMessage("user", message);
    input.value = "";
    sendBtn.disabled = true;
    const loading = addMessage("loading", "");
    traceSteps.replaceChildren();
    traceSummary.textContent = "Running…";

    try {
      const res = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: threadId }),
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        loading.className = "msg error";
        const detail = typeof data.detail === "string" ? data.detail : "Something went wrong. Please try again.";
        loading.textContent = detail.includes("Error code") || detail.includes("{")
          ? "Something went wrong. Please try again."
          : detail;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      while (!finished) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((row) => row.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));
          if (event.trace) renderTrace(event.trace);
          if (event.type === "token" && event.text) loading.textContent += event.text;
          if (event.type === "done") {
            loading.className = "msg assistant";
            loading.textContent = event.response || loading.textContent || "(empty response)";
            finished = true;
          } else if (event.type === "error") {
            loading.className = "msg error";
            loading.textContent = event.detail || "Something went wrong. Please try again.";
            finished = true;
          }
        }
      }
      if (!finished) {
        loading.remove();
        addMessage("error", "The stream ended before a reply.");
      }
    } catch (err) {
      loading.remove();
      addMessage("error", err?.message || "Network error");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });
})();
