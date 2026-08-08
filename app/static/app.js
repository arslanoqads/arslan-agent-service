(() => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const sendBtn = document.getElementById("send");
  const messages = document.getElementById("messages");
  const suggestions = document.getElementById("suggestions");

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
    const loading = addMessage("loading", "Thinking…");

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: threadId }),
      });

      const data = await res.json().catch(() => ({}));
      loading.remove();

      if (!res.ok) {
        addMessage("error", data.detail || `Request failed (${res.status})`);
        return;
      }

      addMessage("assistant", data.response || "(empty response)");
    } catch (err) {
      loading.remove();
      addMessage("error", err?.message || "Network error");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });
})();
