// Page wiring: pick a model, download it, and hand it to the right panel.
//
// Which panels a model gets is decided by its `kind` in the manifest, not by a
// list kept here — export a new specialist and it shows up with the panel its
// head implies (a classifier gets Classify, a generator gets Generate).

import { ChatEngine } from "./engine.js";
import { OnnxLM } from "./model.js";
import { ACTIONS, SnakeGame } from "./snake.js";

const MODELS_URL = "models";
const $ = (id) => document.getElementById(id);

const ui = {
  model: $("model"),
  load: $("load"),
  status: $("status"),
  hint: $("backend-hint"),
  progress: $("progress"),
  bar: $("bar"),
  progressText: $("progress-text"),
  workspace: $("workspace"),
  tabs: $("tabs"),
};

let manifest = null;
let lm = null;
let engine = null;
let history = [];

// ---------------------------------------------------------------- onnxruntime

/**
 * Start the runtime. WebGPU is tried first where the browser has it and falls
 * back to WASM, which every browser here can run; threads need the
 * cross-origin isolation headers `sodachat.web` sends (see web.py).
 */
async function bootRuntime() {
  const ort = await import("../vendor/ort/ort.webgpu.bundle.min.mjs");
  globalThis.ort = ort;
  // Absolute, not "vendor/ort/": the runtime loads its WASM glue with a dynamic
  // import, and a bare relative path is a *module specifier* there — which
  // resolves against nothing and fails before any backend starts.
  ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;
  ort.env.logLevel = "error";
  const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  ui.hint.textContent =
    (providers[0] === "webgpu" ? "webgpu" : "wasm") +
    (ort.env.wasm.numThreads > 1 ? ` · ${ort.env.wasm.numThreads} threads` : " · 1 thread");
  return providers;
}

// -------------------------------------------------------------------- loading

async function loadManifest() {
  const response = await fetch(`${MODELS_URL}/manifest.json`);
  if (!response.ok) {
    throw new Error(
      "no models/manifest.json — run `python -m sodachat.export_onnx` to build one",
    );
  }
  manifest = await response.json();
  for (const [name, spec] of Object.entries(manifest.models)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${spec.label} · ${(spec.bytes / 1e6).toFixed(0)} MB`;
    ui.model.append(option);
  }
  ui.status.textContent = `${Object.keys(manifest.models).length} models available.`;
}

async function loadModel(providers) {
  const name = ui.model.value;
  const spec = manifest.models[name];
  ui.load.disabled = true;
  ui.model.disabled = true;
  ui.progress.hidden = false;
  ui.status.textContent = `Downloading ${spec.file}…`;

  const onProgress = (loaded, total) => {
    const pct = total ? (loaded / total) * 100 : 0;
    ui.bar.style.width = `${pct.toFixed(1)}%`;
    ui.progressText.textContent = `${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
  };

  try {
    lm = await tryProviders(name, providers, onProgress);
  } catch (error) {
    ui.status.textContent = `Could not load ${name}: ${error.message}`;
    ui.load.disabled = false;
    ui.model.disabled = false;
    return;
  }
  ui.progress.hidden = true;
  ui.load.disabled = false;
  ui.model.disabled = false;
  ui.status.textContent =
    `${spec.label} — ${(spec.params / 1e6).toFixed(1)}M parameters, ` +
    `${spec.config.n_layer}L/${spec.config.n_embd}d, ${spec.config.block_size}-token context` +
    (spec.quantization !== "none" ? `, ${spec.quantization}` : "");

  engine = spec.kind === "chat" ? new ChatEngine(lm) : null;
  history = [];
  buildTabs(spec);
  ui.workspace.hidden = false;
}

/** WebGPU is worth trying and not worth failing over; drop to WASM if it errors. */
async function tryProviders(name, providers, onProgress) {
  try {
    return await OnnxLM.load(MODELS_URL, name, manifest, { onProgress, providers });
  } catch (error) {
    if (providers.length < 2) throw error;
    console.warn("falling back to wasm:", error);
    ui.hint.textContent = "wasm (webgpu unavailable)";
    return OnnxLM.load(MODELS_URL, name, manifest, { onProgress, providers: ["wasm"] });
  }
}

// ----------------------------------------------------------------------- tabs

const TABS = {
  chat: { label: "Chat", kinds: ["chat"] },
  classify: { label: "Classify", kinds: ["classify"] },
  generate: { label: "Generate", kinds: ["generate"] },
  play: { label: "Play snake", kinds: ["move"] },
};

function buildTabs(spec) {
  const available = Object.entries(TABS).filter(([, tab]) => tab.kinds.includes(spec.kind));
  ui.tabs.replaceChildren();
  if (!available.length) {
    ui.status.textContent += ` — no panel knows what to do with kind "${spec.kind}"`;
    return;
  }
  for (const [id, tab] of available) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = tab.label;
    button.onclick = () => showTab(id);
    button.dataset.tab = id;
    ui.tabs.append(button);
  }
  showTab(available[0][0]);
  if (spec.kind === "classify") {
    $("classify-hint").textContent =
      `Reads a ${spec.labels.length}-way head at the final token: ${spec.labels.join(", ")}.`;
  }
  if (spec.kind === "generate") {
    const headers = spec.frame?.headers;
    $("generate-hint").textContent = headers
      ? "Continues your prompt through the codegen expert and the shared LM head."
      : "Thinks step by step, then commits to an answer.";
    $("generate-lang-row").hidden = !headers;
    if (headers) {
      $("generate-lang").replaceChildren(
        ...Object.keys(headers).map((lang) => new Option(lang, lang)),
      );
    }
  }
  if (spec.kind === "move") resetGame();
}

function showTab(id) {
  for (const section of document.querySelectorAll(".tab")) {
    section.hidden = section.dataset.tab !== id;
  }
  for (const button of ui.tabs.children) {
    button.setAttribute("aria-selected", String(button.dataset.tab === id));
  }
}

// ----------------------------------------------------------------------- chat

function addTurn(who, text, meta = "") {
  const turn = document.createElement("div");
  turn.className = `turn ${who}`;
  turn.innerHTML = `<div class="who">${who === "you" ? "you" : "bot"}</div>`;
  turn.append(text);
  if (meta) {
    const line = document.createElement("div");
    line.className = "meta";
    line.textContent = meta;
    turn.append(line);
  }
  $("transcript").append(turn);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return turn;
}

$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = $("message").value.trim();
  if (!message || !engine) return;
  $("message").value = "";
  $("send").disabled = true;
  addTurn("you", message);
  const pending = addTurn("bot", "…", "thinking");

  const reply = await engine.reply(message, {
    history,
    onEvent: (event) => {
      pending.lastChild.textContent =
        event.stage === "sampling"
          ? `sampling candidate ${event.done}/${event.total} (${event.kept} kept)`
          : event.stage === "ranking"
            ? `ranking ${event.total} candidates by relevance`
            : "reading the prompt";
      // Inference runs on this thread, so nothing repaints unless we let it.
      return yieldToPaint();
    },
  });

  pending.innerHTML = '<div class="who">bot</div>';
  pending.append(reply.text);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = reply.score ? `${reply.source} · rel ${reply.score.toFixed(2)}` : reply.source;
  pending.append(meta);
  history.push(message, reply.text);
  $("send").disabled = false;
  $("message").focus();
});

// ------------------------------------------------------------------- classify

$("classify-run").addEventListener("click", async () => {
  const frame = lm.spec.frame ?? {};
  let value = $("classify-input").value;
  if (!value.trim()) return;
  if (frame.collapse_whitespace) value = value.split(/\s+/).join(" ").trim();
  if (frame.max_chars) value = value.slice(0, frame.max_chars);
  $("classify-run").disabled = true;
  $("classify-out").textContent = "…";
  const { label, confidence } = await lm.classify(
    (frame.template ?? "{input}").replace("{input}", value),
  );
  $("classify-out").textContent = `${label} · ${(confidence * 100).toFixed(1)}% confident`;
  $("classify-run").disabled = false;
});

// ------------------------------------------------------------------- generate

$("generate-run").addEventListener("click", async () => {
  const spec = lm.spec;
  const frame = spec.frame ?? {};
  let value = $("generate-input").value;
  if (!value.trim()) return;
  if (frame.collapse_whitespace) value = value.split(/\s+/).join(" ").trim();
  const header = frame.headers?.[$("generate-lang").value] ?? "";
  const prompt = (frame.template ?? "{input}")
    .replace("{header}", header)
    .replace("{input}", value);

  $("generate-run").disabled = true;
  $("generate-status").textContent = "generating…";
  $("generate-out").textContent = "";
  const prefilled = await lm.prefill(prompt, { reserve: frame.sampling?.max_new_tokens ?? 128 });
  const { ids } = await lm.generate(prefilled, {
    ...(frame.sampling ?? {}),
    maxNew: frame.sampling?.max_new_tokens,
    // A reasoning chain and a function body both contain newlines: unlike a
    // chat line, these stop only at <|end|>.
    stop: new Set([spec.specials.end]),
  });

  // Split on the token id, not the text: the tokenizer drops special tokens
  // when it decodes, so there would be nothing left to split on (reason.py).
  const splitId = frame.split_on ? frame.tokens?.[frame.split_on] : undefined;
  const at = splitId === undefined ? -1 : ids.indexOf(splitId);
  $("generate-out").textContent =
    at < 0
      ? lm.tok.decode(ids)
      : `${lm.tok.decode(ids.slice(0, at)).trim()}\n\nanswer: ${lm.tok.decode(ids.slice(at + 1)).trim()}`;
  $("generate-status").textContent = `${ids.length} tokens`;
  $("generate-run").disabled = false;
});

// ----------------------------------------------------------------------- play

let game = null;
let playing = false;

function resetGame() {
  game = new SnakeGame();
  playing = false;
  $("play-toggle").textContent = "Play";
  drawGame();
}

function drawGame() {
  $("board").textContent = game.modelBoard();
  $("play-status").textContent = game.done
    ? `game over — score ${game.score}`
    : `score ${game.score}`;
}

async function playStep() {
  if (!game || game.done) return;
  // The play frame from `expert._play_doc`, ending on the <|act|> token the
  // action head was trained to fire at.
  const goal = "eat the food";
  const legal = ACTIONS.filter((a) => game.legal(a));
  const prompt = `<|game|> snake | goal: ${goal}\n${game.modelBoard()}\n<|act|>`;
  const move = await lm.move(prompt, legal);
  game.step(move);
  drawGame();
}

$("play-step").addEventListener("click", playStep);
$("play-reset").addEventListener("click", resetGame);
$("play-toggle").addEventListener("click", async () => {
  playing = !playing;
  $("play-toggle").textContent = playing ? "Pause" : "Play";
  while (playing && game && !game.done) {
    await playStep();
    await yieldToPaint();
  }
  playing = false;
  $("play-toggle").textContent = "Play";
});

/** Hand the browser a turn to repaint between forward passes. */
const yieldToPaint = () => new Promise((resolve) => setTimeout(resolve, 0));

// ----------------------------------------------------------------------- boot

const providers = await bootRuntime().catch((error) => {
  ui.status.textContent = `onnxruntime failed to start: ${error.message}`;
  throw error;
});
await loadManifest().catch((error) => {
  ui.status.textContent = error.message;
});
ui.load.addEventListener("click", () => loadModel(providers));
