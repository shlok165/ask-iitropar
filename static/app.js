"use strict";

(() => {
  const $ = (sel, el = document) => el.querySelector(sel);
  const messagesEl = $("#messages");
  const welcome = $("#welcome");
  const form = $("#composer");
  const input = $("#input");
  const sendBtn = $("#send");
  const newChatBtn = $("#new-chat");
  const tpl = $("#tpl-assistant");

  const HISTORY_MESSAGES = 6; // last 3 question/answer pairs go along for follow-up questions
  const ROUTE_LABELS = {
    live: "Live from iitrpr.ac.in",
    table: "Institute directory",
    retrieval: "Official pages & documents",
    refuse: "Privacy",
    web: "🌐 From the web",
  };
  const SCOPE_LABELS = { official: "IIT Ropar websites", open: "Wider web, social media excluded" };
  const TYPE_LABELS = { live: "Live data", api: "Official data", site: "Official page", pdf: "PDF", web: "Web" };

  let history = [];
  let controller = null; // AbortController of the answer being streamed

  marked.setOptions({ gfm: true, breaks: false });
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });

  // --- rendering ------------------------------------------------------------------
  const escapeAttr = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

  function domainOf(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; }
  }

  // The model ends its answer with its own "Sources" list; the page shows source cards instead.
  function stripSourcesSection(text) {
    const m = text.match(/\n[ \t]*(?:#{1,4}[ \t]*|\*\*)?Sources[ \t]*:?[ \t]*(?:\*\*)?[ \t]*:?[ \t]*(?:\n|$)/i);
    return m ? text.slice(0, m.index) : text;
  }

  // "[2]" / "[1, 4]" -> small numbered links to the source
  function linkCitations(text, byN, cited) {
    return text.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\](?!\()/g, (whole, nums) => {
      const ns = nums.split(/\s*,\s*/).map(Number);
      if (!ns.every((n) => byN[n])) return whole;
      ns.forEach((n) => cited.add(n));
      return ns.map((n) => `<sup class="cite"><a href="${escapeAttr(byN[n].url)}" title="${escapeAttr(byN[n].title)}">${n}</a></sup>`).join("");
    });
  }

  function renderAnswer(li, text, sources) {
    const byN = Object.fromEntries(sources.map((s) => [s.n, s]));
    const cited = new Set();
    const body = stripSourcesSection(text);
    // numbers the model listed only in its own (hidden) "Sources" section still count as cited
    for (const m of text.slice(body.length).matchAll(/\[(\d{1,2})\]/g)) if (byN[m[1]]) cited.add(Number(m[1]));
    const md = linkCitations(body, byN, cited);
    $(".bubble", li).innerHTML = DOMPurify.sanitize(marked.parse(md));
    return cited;
  }

  function sourceItem(s) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.className = "src";
    a.href = s.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    const n = Object.assign(document.createElement("span"), { className: "n", textContent: s.n });
    const t = Object.assign(document.createElement("span"), { className: "t", textContent: s.title });
    const d = Object.assign(document.createElement("span"), {
      className: "d",
      textContent: [TYPE_LABELS[s.source_type] || s.source_type, domainOf(s.url)].filter(Boolean).join(" · "),
    });
    a.append(n, t, d);
    li.append(a);
    return li;
  }

  function renderSources(li, sources, cited) {
    if (!sources.length) return;
    const box = $(".sources", li);
    const main = cited.size ? sources.filter((s) => cited.has(s.n)) : sources.slice(0, 4);
    const more = sources.filter((s) => !main.includes(s));
    $(":scope > ol", box).replaceChildren(...main.map(sourceItem));
    const details = $("details", box);
    if (more.length) {
      $("summary", details).textContent = `${more.length} more source${more.length > 1 ? "s" : ""} checked`;
      $("ol", details).replaceChildren(...more.map(sourceItem));
      details.hidden = false;
    }
    box.hidden = false;
  }

  function badge(text, cls = "") {
    return Object.assign(document.createElement("span"), { className: `badge ${cls}`.trim(), textContent: text });
  }

  function renderMeta(li, meta, done) {
    const el = $(".meta", li);
    const badges = [];
    if (ROUTE_LABELS[meta.route]) badges.push(badge(ROUTE_LABELS[meta.route], meta.route === "web" ? "warn" : ""));
    if (meta.scope) badges.push(badge(SCOPE_LABELS[meta.scope], "subtle"));
    if (meta.cached || done?.cached) badges.push(badge("Saved answer", "subtle"));
    if (done?.notice && done.notice !== "limit_reached") badges.push(badge(done.notice, "warn"));
    el.replaceChildren(...badges);
  }

  function addUser(text) {
    const li = document.createElement("li");
    li.className = "msg user";
    const bubble = Object.assign(document.createElement("div"), { className: "bubble", textContent: text });
    li.append(bubble);
    messagesEl.append(li);
  }

  const nearBottom = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  const scrollToBottom = () => window.scrollTo({ top: document.documentElement.scrollHeight });

  // --- streaming -------------------------------------------------------------------
  async function* sseEvents(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const data = chunk.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
        if (data) yield JSON.parse(data);
      }
    }
  }

  // opts.web = "official" | "open" answers from a web search; opts.trigger says which button asked for it
  async function ask(question, opts = {}) {
    question = question.trim();
    if (!question || controller) return;
    welcome.hidden = true;
    newChatBtn.hidden = false;
    addUser(opts.web ? `🌐 Search ${opts.web === "open" ? "the wider web" : "IIT Ropar's websites"}: ${question}` : question);
    const li = tpl.content.firstElementChild.cloneNode(true);
    messagesEl.append(li);
    scrollToBottom();
    setBusy(true);

    controller = new AbortController();
    let text = "";
    let sources = [];
    let meta = null;
    let done = null;
    let finished = false;
    let painting = false;
    const paint = () => {
      painting = false;
      if (finished) return;
      const stick = nearBottom();
      renderAnswer(li, text, sources);
      if (stick) scrollToBottom();
    };

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history: history.slice(-HISTORY_MESSAGES), web: opts.web ?? null,
          trigger: opts.trigger ?? null }),
        signal: controller.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(res.status === 422 ? "Please shorten your question (up to 1000 characters)." : `Server error (${res.status}). Please try again.`);
      }
      for await (const ev of sseEvents(res.body)) {
        if (ev.type === "meta") {
          meta = ev;
          sources = ev.sources || [];
          renderMeta(li, meta);
        } else if (ev.type === "delta") {
          text += ev.text;
          if (!painting) { painting = true; requestAnimationFrame(paint); }
        } else if (ev.type === "done") {
          done = ev;
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }
      finished = true;
      const cited = renderAnswer(li, text, sources);
      if (meta) renderMeta(li, meta, done);
      if (meta?.route !== "refuse") renderSources(li, sources, cited);
      renderOffer(li, question, meta);
      setupFeedback(li, question, text, meta, sources);
      history.push({ role: "user", content: question }, { role: "assistant", content: stripSourcesSection(text).slice(0, 2000) });
      history = history.slice(-HISTORY_MESSAGES);
    } catch (err) {
      finished = true;
      if (err.name === "AbortError") {
        renderAnswer(li, text ? `${text}\n\n_(stopped)_` : "_(stopped)_", sources);
      } else {
        li.classList.add("error");
        $(".bubble", li).textContent = err.message || "Something went wrong. Please try again.";
      }
    } finally {
      controller = null;
      setBusy(false);
      if (nearBottom()) scrollToBottom();
      input.focus();
    }
  }

  // --- web search offers -------------------------------------------------------------------
  // The next web step for an answer: official sites first, then the wider web, then nothing.
  function nextWebScope(meta) {
    if (!meta || meta.route === "refuse") return null;
    if (meta.route !== "web") return "official";
    return meta.scope === "official" ? "open" : null;
  }

  function renderOffer(li, question, meta) {
    const scope = nextWebScope(meta);
    const lowConfidence = meta?.offer_web;
    const webWithNext = meta?.route === "web" && scope;
    if (!scope || !(lowConfidence || webWithNext)) return;
    const offer = $(".offer", li);
    const note = document.createElement("span");
    note.textContent = lowConfidence
      ? "I'm not confident IIT Ropar's official data covers this. "
      : "Not what you needed? ";
    const btn = Object.assign(document.createElement("button"), {
      type: "button",
      className: "chip",
      textContent: scope === "official" ? "🌐 Search IIT Ropar's websites" : "🌐 Search the wider web",
    });
    btn.addEventListener("click", () => {
      btn.disabled = true;
      ask(question, { web: scope, trigger: scope === "official" ? "offer" : "wider" });
    });
    offer.replaceChildren(note, btn);
    offer.hidden = false;
  }

  // --- feedback ----------------------------------------------------------------------
  function setupFeedback(li, question, answer, meta, sources) {
    const actions = $(".actions", li);
    const status = $(".status", actions);
    actions.hidden = false;
    actions.querySelectorAll(".vote").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const rating = btn.dataset.rating;
        actions.querySelectorAll(".vote").forEach((b) => { b.disabled = true; });
        btn.classList.add("on");
        const scope = rating === "down" ? nextWebScope(meta) : null;
        status.textContent = rating === "up" ? "Thanks!"
          : scope ? "Thanks. Searching the web for a better answer…" : "Thanks, this helps us find gaps in the data.";
        try {
          await fetch("/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rating, question, answer: answer.slice(0, 8000), route: meta?.route ?? null,
              sources: sources.map((s) => s.url).slice(0, 30) }),
          });
        } catch { /* feedback is best-effort */ }
        if (scope) {
          $(".offer", li).hidden = true;
          ask(question, { web: scope, trigger: "thumbs_down" });
        }
      });
    });
  }

  // --- composer ------------------------------------------------------------------------
  function setBusy(on) {
    document.body.classList.toggle("busy", on);
    sendBtn.setAttribute("aria-label", on ? "Stop answering" : "Send question");
    sendBtn.title = on ? "Stop" : "Send";
    updateSend();
  }

  function updateSend() {
    sendBtn.disabled = !controller && !input.value.trim();
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (controller) { controller.abort(); return; }
    const q = input.value;
    input.value = "";
    autosize();
    ask(q);
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener("input", () => { autosize(); updateSend(); });

  $("#examples").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) ask(chip.textContent);
  });

  newChatBtn.addEventListener("click", () => {
    if (controller) controller.abort();
    history = [];
    messagesEl.replaceChildren();
    welcome.hidden = false;
    newChatBtn.hidden = true;
    input.focus();
  });

  updateSend();
  input.focus();
})();
