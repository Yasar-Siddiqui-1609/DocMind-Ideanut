// ============================================================
// DOCMIND AI — app.js  (ES Module)
// ============================================================
import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

// ── Hardcoded API Key (no user input needed) ─────────────────
const GEMINI_API_KEY = "AQ.Ab8RN6LhyGik6uirGBj6gt2EQGPWWqE5upMgAL5ZXiIL_t6YNQ";

// ── State ────────────────────────────────────────────────────
const state = {
  apiKey: GEMINI_API_KEY,
  docs: JSON.parse(localStorage.getItem("docmind_docs") || "[]"),
  activeDocId: null,
  highlightsOn: true,
  localMode: false,
};

// ── Helpers ──────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const genAI = () => new GoogleGenerativeAI(state.apiKey);

function saveDocsToStorage() {
  localStorage.setItem("docmind_docs", JSON.stringify(
    state.docs.map(d => ({ ...d, chunks: undefined, embeddings: undefined }))
  ));
}

// ── Splash → App (goes straight in, no API modal) ────────────
window.addEventListener("load", () => {
  setTimeout(() => {
    $('splash-screen').style.opacity = "0";
    setTimeout(() => {
      $("splash-screen").remove();
      showMainApp();
    }, 500);
  }, 2200);
});

function showMainApp() {
  $("main-app").classList.remove("hidden");
  renderDocList();
  showScreen("screen-welcome");
}

// ── API Key (modal removed — key is hardcoded) ───────────────
window.showApiModal = () => {}; // no-op
window.saveApiKey = () => {};
window.toggleKeyVisibility = () => {};

// ── Screens & Nav ────────────────────────────────────────────
window.showScreen = (id) => {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  const s = $(id);
  if (s) { s.classList.remove("hidden"); s.classList.add("active"); }
  ["home","summary","chat","viewer"].forEach(n => {
    const btn = $(`nav-${n}`);
    if (btn) btn.classList.remove("active");
  });
  const map = { "screen-welcome":"home","screen-summary":"summary","screen-chat":"chat","screen-viewer":"viewer" };
  if (map[id]) $(`nav-${map[id]}`)?.classList.add("active");
  toggleSidebar(false);
};

window.navTo = (name) => {
  if (!state.activeDocId && name !== "welcome") return;
  const map = { welcome:"screen-welcome", summary:"screen-summary", chat:"screen-chat", viewer:"screen-viewer" };
  showScreen(map[name]);
};

window.toggleSidebar = (force) => {
  const sb = $("sidebar"), ov = $("sidebar-overlay");
  const open = force !== undefined ? force : !sb.classList.contains("open");
  sb.classList.toggle("open", open);
  ov.classList.toggle("hidden", !open);
};

// ── File Upload ──────────────────────────────────────────────
window.handleFileUpload = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = "";
  if (!state.apiKey) { showApiModal(); return; }
  toggleSidebar(false);
  showScreen("screen-processing");
  try {
    await processDocument(file);
  } catch (err) {
    alert("Error processing document: " + err.message);
    showScreen("screen-welcome");
  }
};

// ── Text Extraction ──────────────────────────────────────────
async function extractText(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (ext === "txt") {
    return await file.text();
  }
  if (ext === "docx") {
    const ab = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer: ab });
    return result.value;
  }
  if (ext === "pdf") {
    return await extractPdfText(file);
  }
  throw new Error("Unsupported file type");
}

async function extractPdfText(file) {
  const ab = await file.arrayBuffer();
  const pdfjsLib = await import("https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.min.mjs");
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.worker.min.mjs";
  const pdf = await pdfjsLib.getDocument({ data: ab }).promise;
  let text = "";
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map(it => it.str).join(" ") + "\n\n";
  }
  return text;
}

// ── Chunking ─────────────────────────────────────────────────
function chunkText(text, size = 400, overlap = 60) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    chunks.push(words.slice(i, i + size).join(" "));
    i += size - overlap;
  }
  return chunks;
}

// ── Embeddings via Gemini ────────────────────────────────────
async function embedTexts(texts) {
  const model = genAI().getGenerativeModel({ model: "text-embedding-004" });
  const results = [];
  for (const text of texts) {
    const r = await model.embedContent(text);
    results.push(r.embedding.values);
  }
  return results;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]**2; nb += b[i]**2; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}

async function retrieveTopK(queryEmbedding, doc, k = 4) {
  return doc.embeddings
    .map((emb, i) => ({ score: cosine(queryEmbedding, emb), chunk: doc.chunks[i] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// ── Local Fallback (no API needed) ───────────────────────────
const STOPWORDS = new Set(['the','a','an','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','to','of','in','for','on','with','at','by','from','up','about','into','this','that','these','those','it','its','they','them','their','we','our','you','your','he','she','his','her','i','me','my','and','or','but','if','as','not','no','so','than','then','when','where','what','which','who','how','all','some','each','more','most','also','just','out','are','been']);

function localSummarize(text) {
  const sentences = (text.match(/[^.!?\n]{20,}[.!?]/g) || []).filter(s => s.trim().length > 20);
  const words = text.toLowerCase().replace(/[^a-z\s]/g,' ').split(/\s+/).filter(w => w.length > 3 && !STOPWORDS.has(w));
  const freq = {};
  words.forEach(w => freq[w] = (freq[w]||0)+1);
  const topWords = Object.entries(freq).sort((a,b)=>b[1]-a[1]).map(([w])=>w);
  const concepts = topWords.slice(0,8).map(w=>w.charAt(0).toUpperCase()+w.slice(1));
  const para1 = sentences.slice(0,3).join(' ').trim();
  const para2 = sentences.slice(Math.floor(sentences.length/2), Math.floor(sentences.length/2)+2).join(' ').trim();
  const tldr = [para1, para2].filter(Boolean).join(' ');
  const glossary = topWords.slice(0,5).map(term => {
    const ctx = sentences.find(s=>s.toLowerCase().includes(term)) || 'Key term in this document.';
    return { term: term.charAt(0).toUpperCase()+term.slice(1), definition: ctx.trim().slice(0,150) };
  });
  const insightKw = ['important','key','must','should','critical','essential','result','therefore','conclusion','ensure','require','note'];
  let insights = sentences.filter(s=>insightKw.some(k=>s.toLowerCase().includes(k))).slice(0,4).map(s=>s.trim());
  if (!insights.length) insights = sentences.filter((_,i)=>i%Math.max(1,Math.floor(sentences.length/4))===0).slice(0,4).map(s=>s.trim());
  return { tldr: tldr||text.slice(0,300), concepts, glossary, insights: insights.slice(0,4) };
}

function localRetrieve(query, chunks, k=4) {
  const qw = new Set(query.toLowerCase().replace(/[^a-z\s]/g,' ').split(/\s+/).filter(w=>w.length>2&&!STOPWORDS.has(w)));
  if (!qw.size) return chunks.slice(0,k).map(chunk=>({chunk,score:0.5}));
  return chunks.map(chunk=>{
    const cw = chunk.toLowerCase().replace(/[^a-z\s]/g,' ').split(/\s+/);
    return {chunk, score: cw.filter(w=>qw.has(w)).length / (qw.size+0.1)};
  }).sort((a,b)=>b.score-a.score).slice(0,k);
}

function localAnswer(query, topChunks) {
  if (!topChunks.length || topChunks[0].score < 0.08) return {text:'Not covered in this document.',oos:true};
  const qw = query.toLowerCase().split(/\s+/).filter(w=>w.length>2);
  const allText = topChunks.slice(0,2).map(r=>r.chunk).join(' ');
  const sents = (allText.match(/[^.!?]{15,}[.!?]/g)||[]);
  const relevant = sents.map(s=>({s,sc:qw.filter(w=>s.toLowerCase().includes(w)).length})).sort((a,b)=>b.sc-a.sc).filter(x=>x.sc>0).slice(0,3).map(x=>x.s.trim());
  const answer = relevant.length ? relevant.join(' ') : topChunks[0].chunk.slice(0,400);
  return {text:answer, oos:false, chunk:topChunks[0].chunk};
}

// ── Summary Generation ───────────────────────────────────────
async function generateSummary(text) {
  try {
    const model = genAI().getGenerativeModel({ model: "gemini-2.0-flash" });
    const sample = text.slice(0, 12000);
    const prompt = `Analyze this document and respond ONLY with valid JSON (no markdown, no code fences):
{"tldr":"2-3 sentence summary","concepts":["c1","c2","c3","c4","c5"],"glossary":[{"term":"t1","definition":"d1"},{"term":"t2","definition":"d2"},{"term":"t3","definition":"d3"}],"insights":["i1","i2","i3","i4"]}

Document:
${sample}`;
    const result = await model.generateContent(prompt);
    let raw = result.response.text().trim().replace(/```json|```/g, "").trim();
    return JSON.parse(raw);
  } catch {
    state.localMode = true;
    return localSummarize(text);
  }
}

// ── Process Document ─────────────────────────────────────────
async function processDocument(file) {
  const steps = ["step-1","step-2","step-3","step-4","step-5"];
  const setStep = (i) => {
    steps.forEach((s,j) => {
      $(s)?.classList.toggle("active", j === i);
      if (j < i) $(s)?.classList.add("done");
    });
    $("progress-bar").style.width = `${(i/4)*100}%`;
  };

  $("processing-title").textContent = `Processing "${file.name}"...`;

  // Step 1 – Extract
  setStep(0);
  const text = await extractText(file);

  // Step 2 – Chunk
  setStep(1);
  const chunks = chunkText(text);

  // Step 3 – Embed (try API, fallback gracefully)
  setStep(2);
  let embeddings = null;
  try {
    $("processing-subtitle").textContent = `Computing embeddings (${chunks.length} chunks)...`;
    embeddings = await embedTexts(chunks);
  } catch {
    state.localMode = true;
    $("processing-subtitle").textContent = "API unavailable — switching to local mode...";
    await new Promise(r => setTimeout(r, 600));
  }

  // Step 4 – Summarize
  setStep(3);
  $("processing-subtitle").textContent = "Generating structured summary...";
  const summary = await generateSummary(text);

  // Step 5 – Done
  setStep(4);

  const docId = Date.now().toString();
  const ext = file.name.split(".").pop().toUpperCase();
  const wordCount = text.split(/\s+/).length;
  const doc = {
    id: docId, name: file.name, ext, text, chunks, embeddings, summary,
    wordCount, addedAt: new Date().toLocaleString(), chatHistory: []
  };

  state.docs.push(doc);
  saveDocsToStorage();
  setActiveDoc(docId, doc);
  await new Promise(r => setTimeout(r, 600));
  renderDocList();
  showSummaryScreen(doc);
}

// ── Active Document ──────────────────────────────────────────
function setActiveDoc(id, doc) {
  state.activeDocId = id;
  $("active-doc-badge").classList.remove("hidden");
  $("active-doc-name").textContent = doc.name.split(".")[0];
  ["nav-summary","nav-chat","nav-viewer"].forEach(n => {
    const btn = $(n);
    if (btn) btn.disabled = false;
  });
}

function getActiveDoc() {
  return state.docs.find(d => d.id === state.activeDocId);
}

// ── Render Sidebar ───────────────────────────────────────────
function renderDocList() {
  const container = $("sidebar-docs");
  const msg = $("no-docs-msg");
  if (!state.docs.length) { msg?.classList.remove("hidden"); return; }
  msg?.classList.add("hidden");
  container.innerHTML = `<p class="no-docs-msg hidden"></p>`;
  state.docs.forEach(doc => {
    const icons = { PDF:"📄", DOCX:"📝", TXT:"📃" };
    const div = document.createElement("div");
    div.className = "doc-item" + (doc.id === state.activeDocId ? " active" : "");
    div.id = `doc-item-${doc.id}`;
    div.innerHTML = `
      <div class="doc-icon">${icons[doc.ext]||"📄"}</div>
      <div class="doc-info">
        <div class="doc-name">${doc.name}</div>
        <div class="doc-meta">${doc.wordCount.toLocaleString()} words · ${doc.addedAt}</div>
      </div>
      <button class="doc-delete" onclick="deleteDoc('${doc.id}',event)" title="Delete">🗑</button>
    `;
    div.addEventListener("click", () => switchDoc(doc.id));
    container.appendChild(div);
  });
}

window.deleteDoc = (id, e) => {
  e.stopPropagation();
  if (!confirm("Delete this document?")) return;
  state.docs = state.docs.filter(d => d.id !== id);
  saveDocsToStorage();
  if (state.activeDocId === id) {
    state.activeDocId = null;
    $("active-doc-badge").classList.add("hidden");
    ["nav-summary","nav-chat","nav-viewer"].forEach(n => { const b=$(n); if(b) b.disabled=true; });
    showScreen("screen-welcome");
  }
  renderDocList();
};

function switchDoc(id) {
  const doc = state.docs.find(d => d.id === id);
  if (!doc) return;
  setActiveDoc(id, doc);
  renderDocList();
  showSummaryScreen(doc);
  toggleSidebar(false);
}

// ── Summary Screen ───────────────────────────────────────────
function showSummaryScreen(doc) {
  $("summary-doc-type").textContent = doc.ext;
  $("summary-doc-title").textContent = doc.name;
  $("summary-doc-meta").textContent = `${doc.wordCount.toLocaleString()} words · ${doc.chunks.length} chunks indexed`;
  $("summary-tldr").textContent = doc.summary.tldr;

  // Concepts
  $("summary-concepts").innerHTML = doc.summary.concepts
    .map(c => `<span class="concept-tag">${c}</span>`).join("");

  // Glossary
  $("summary-glossary").innerHTML = doc.summary.glossary
    .map(g => `<div class="glossary-item"><div class="glossary-term">${g.term}</div><div class="glossary-def">${g.definition}</div></div>`).join("");

  // Insights
  $("summary-insights").innerHTML = doc.summary.insights
    .map(i => `<li>${i}</li>`).join("");

  // Chat header
  $("chat-doc-name").textContent = doc.name;
  $("chat-chunk-count").textContent = `${doc.chunks.length} chunks indexed`;

  // Viewer
  $("viewer-doc-title").textContent = doc.name;
  renderViewer(doc);

  showScreen("screen-summary");
}

// ── Document Viewer ──────────────────────────────────────────
function renderViewer(doc) {
  const terms = doc.summary.glossary.map(g => g.term);
  const concepts = doc.summary.concepts;
  let html = escapeHtml(doc.text.slice(0, 30000));
  if (state.highlightsOn) {
    terms.forEach(t => {
      const re = new RegExp(`\\b(${escapeRe(t)})\\b`, "gi");
      html = html.replace(re, '<mark class="term-highlight">$1</mark>');
    });
    concepts.forEach(c => {
      const re = new RegExp(`\\b(${escapeRe(c)})\\b`, "gi");
      html = html.replace(re, '<mark class="concept-highlight">$1</mark>');
    });
  }
  $("viewer-content").innerHTML = `<p style="white-space:pre-wrap;font-family:inherit">${html}</p>`;
}

window.toggleHighlights = () => {
  state.highlightsOn = !state.highlightsOn;
  $("highlight-toggle").textContent = state.highlightsOn ? "✨" : "○";
  const doc = getActiveDoc();
  if (doc) renderViewer(doc);
};

function escapeHtml(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");
}

// ── Chat ─────────────────────────────────────────────────────
let chatRendering = false;
const OUT_OF_SCOPE_THRESHOLD = 0.28;

window.handleChatKeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
};

window.autoResizeTextarea = (el) => {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
};

window.sendSuggestedQuery = (q) => {
  $("chat-input").value = q;
  sendChatMessage();
};

window.sendChatMessage = async () => {
  const input = $("chat-input");
  const query = input.value.trim();
  if (!query || chatRendering) return;
  const doc = getActiveDoc();
  if (!doc) return;

  input.value = "";
  input.style.height = "auto";
  chatRendering = true;
  $("send-btn").disabled = true;

  appendUserMsg(query);
  const typingId = appendTyping();

  try {
    let results;
    if (!state.localMode && doc.embeddings) {
      try {
        const model = genAI().getGenerativeModel({ model: "text-embedding-004" });
        const qEmb = (await model.embedContent(query)).embedding.values;
        results = await retrieveTopK(qEmb, doc);
      } catch {
        state.localMode = true;
        results = localRetrieve(query, doc.chunks);
      }
    } else {
      results = localRetrieve(query, doc.chunks);
    }
    removeTyping(typingId);
    const bestScore = results[0]?.score || 0;
    if (state.localMode) {
      const r = localAnswer(query, results);
      appendAiMsg(r.text, r.oos ? null : r.chunk, r.oos);
    } else if (bestScore < OUT_OF_SCOPE_THRESHOLD) {
      appendAiMsg("Not covered in this document.", null, true);
    } else {
      const context = results.map(r => r.chunk).join("\n\n---\n\n");
      const answer = await generateAnswer(query, context, doc.name);
      appendAiMsg(answer, results[0].chunk);
    }
  } catch (err) {
    removeTyping(typingId);
    const r = localAnswer(query, localRetrieve(query, doc.chunks));
    appendAiMsg(r.text, r.oos ? null : r.chunk, r.oos);
  }

  chatRendering = false;
  $("send-btn").disabled = false;
};

async function generateAnswer(query, context, docName) {
  const model = genAI().getGenerativeModel({ model: "gemini-2.0-flash" });
  const prompt = `You are a helpful document assistant for "${docName}". Answer ONLY based on the context below. If the answer is not in the context, respond with exactly: "Not covered in this document."

Context:
${context}

Question: ${query}

Answer:`;
  const result = await model.generateContent(prompt);
  return result.response.text().trim();
}

function appendUserMsg(text) {
  const row = document.createElement("div");
  row.className = "user-msg-row";
  row.innerHTML = `<div class="user-avatar">👤</div><div class="msg-bubble user-bubble">${escapeHtml(text)}</div>`;
  $("chat-messages").appendChild(row);
  scrollChat();
}

function appendAiMsg(text, sourceChunk, isOOS = false) {
  const row = document.createElement("div");
  row.className = "ai-msg-row";
  let inner = "";
  if (isOOS || text.toLowerCase().includes("not covered")) {
    inner += `<div class="out-of-scope-badge">🚫 Not covered in this document</div>`;
  }
  inner += `<p>${escapeHtml(text).replace(/\n/g,"<br>")}</p>`;
  if (sourceChunk && !isOOS) {
    inner += `<div class="source-snippet"><div class="source-label">📎 Source excerpt</div>${escapeHtml(sourceChunk.slice(0,220))}...</div>`;
  }
  row.innerHTML = `<div class="ai-avatar">🧠</div><div class="msg-bubble ai-bubble">${inner}</div>`;
  $("chat-messages").appendChild(row);
  scrollChat();
}

function appendTyping() {
  const id = "typing-" + Date.now();
  const row = document.createElement("div");
  row.className = "ai-msg-row"; row.id = id;
  row.innerHTML = `<div class="ai-avatar">🧠</div><div class="msg-bubble ai-bubble"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
  $("chat-messages").appendChild(row);
  scrollChat();
  return id;
}

function removeTyping(id) { $(id)?.remove(); }
function scrollChat() {
  const m = $("chat-messages");
  m.scrollTop = m.scrollHeight;
}

// ── PDF Export ───────────────────────────────────────────────
window.exportSummaryPDF = () => {
  const doc = getActiveDoc();
  if (!doc) return;
  const s = doc.summary;
  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Summary – ${doc.name}</title>
<style>
  body{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;color:#1a1a2e;padding:0 20px}
  h1{color:#667eea;font-size:1.8rem;margin-bottom:4px}
  .meta{color:#666;font-size:.85rem;margin-bottom:32px}
  h2{color:#764ba2;font-size:1.1rem;border-bottom:2px solid #eee;padding-bottom:6px;margin:24px 0 12px}
  .tldr{background:#f0f0ff;padding:16px;border-radius:8px;line-height:1.7;border-left:4px solid #667eea}
  .chip{display:inline-block;padding:4px 12px;background:#ede9fe;border-radius:99px;margin:3px;font-size:.85rem;color:#764ba2}
  .gitem{margin-bottom:12px;padding:10px;background:#f9f9ff;border-radius:6px;border-left:3px solid #667eea}
  .term{font-weight:700;margin-bottom:4px}
  .def{color:#444;font-size:.9rem}
  li{margin-bottom:8px;line-height:1.6}
  .footer{margin-top:40px;text-align:center;color:#999;font-size:.8rem;border-top:1px solid #eee;padding-top:16px}
</style></head><body>
<h1>📄 ${doc.name}</h1>
<p class="meta">Generated by DocMind AI · ${new Date().toLocaleString()}</p>
<h2>⚡ TL;DR</h2><div class="tldr">${s.tldr}</div>
<h2>💡 Key Concepts</h2><div>${s.concepts.map(c=>`<span class="chip">${c}</span>`).join("")}</div>
<h2>📖 Terminology Glossary</h2>${s.glossary.map(g=>`<div class="gitem"><div class="term">${g.term}</div><div class="def">${g.definition}</div></div>`).join("")}
<h2>🎯 Actionable Insights</h2><ul>${s.insights.map(i=>`<li>${i}</li>`).join("")}</ul>
<div class="footer">Generated by DocMind AI | RAG-Powered Document Intelligence</div>
</body></html>`;

  const w = window.open("", "_blank");
  w.document.write(html);
  w.document.close();
  setTimeout(() => w.print(), 500);
};
