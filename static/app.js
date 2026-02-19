const el = (id) => document.getElementById(id);

const statusEl = el("status");
const questionEl = el("question");
const sessionEl = el("sessionId");
const modelEl = el("modelOverride");
const askBtn = el("askBtn");
const clearBtn = el("clearBtn");
const copyBtn = el("copyBtn");

const typeBadge = el("typeBadge");
const modelBadge = el("modelBadge");
const confidenceBadge = el("confidenceBadge");
const rootCauseEl = el("rootCause");
const explanationEl = el("explanation");
const fixEl = el("fix");
const evidenceEl = el("evidence");
const sourcesEl = el("sources");

let lastResponseText = "";

const setStatus = (text) => {
  statusEl.textContent = text;
};

const setBadge = (badge, text, color) => {
  badge.textContent = text;
  badge.style.borderColor = color || "";
  badge.style.color = color || "";
};

const renderList = (target, items) => {
  target.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "(none)";
    target.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  });
};

const formatResponseText = (dx) => {
  return [
    `Root Cause: ${dx.root_cause}`,
    `Explanation: ${dx.explanation}`,
    `Fix: ${dx.recommended_fix}`,
    `Confidence: ${Math.round(dx.confidence * 100)}%`,
    `Model: ${dx.model_used}`,
  ].join("\n\n");
};

const diagnose = async () => {
  const question = questionEl.value.trim();
  if (!question) return;

  askBtn.disabled = true;
  setStatus("Diagnosing...");

  const payload = {
    question,
    session_id: sessionEl.value.trim() || "web",
    force_model: modelEl.value || null,
  };

  try {
    const res = await fetch("/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Request failed");
    }

    const data = await res.json();
    const dx = data.diagnosis;

    rootCauseEl.textContent = dx.root_cause || "";
    explanationEl.textContent = dx.explanation || "";
    fixEl.textContent = dx.recommended_fix || "";

    const responseType = dx.response_type || "diagnostic";
    setBadge(typeBadge, responseType, responseType === "diagnostic" ? "#45d483" : "#f7c948");
    setBadge(modelBadge, dx.model_used || "model", "#9aa7bd");
    setBadge(confidenceBadge, `${Math.round((dx.confidence || 0) * 100)}%`, "#9aa7bd");

    renderList(evidenceEl, dx.evidence_snippets || []);
    renderList(sourcesEl, dx.sources || []);

    lastResponseText = formatResponseText(dx);
    setStatus("Ready");
  } catch (err) {
    setStatus("Error");
    rootCauseEl.textContent = "Request failed";
    explanationEl.textContent = err.message;
    fixEl.textContent = "";
    renderList(evidenceEl, []);
    renderList(sourcesEl, []);
  } finally {
    askBtn.disabled = false;
  }
};

askBtn.addEventListener("click", diagnose);
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    diagnose();
  }
});

clearBtn.addEventListener("click", () => {
  const sessionId = sessionEl.value.trim() || "web";
  questionEl.value = "";
  fetch("/memory/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  })
    .then(() => setStatus("Memory cleared"))
    .catch(() => setStatus("Memory clear failed"));
});

copyBtn.addEventListener("click", async () => {
  if (!lastResponseText) return;
  try {
    await navigator.clipboard.writeText(lastResponseText);
    setStatus("Copied response");
    setTimeout(() => setStatus("Ready"), 1500);
  } catch (err) {
    setStatus("Copy failed");
  }
});
