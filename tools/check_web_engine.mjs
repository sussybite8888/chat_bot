// End-to-end check of the browser runtime against PyTorch.
//
//   python tools/dump_engine_cases.py && node tools/check_web_engine.mjs
//
// Runs `web/js/model.js` — the same file the page loads — under Node's copy of
// onnxruntime-web, and compares it to values `sodachat`'s PyTorch wrappers
// produced. Greedy ids must match exactly; log-probs, confidences and moves to
// within fp32 noise. If this passes, the browser is running the same model the
// terminal is, not merely a similar one.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";

import { ChatEngine } from "../web/js/engine.js";
import { OnnxLM } from "../web/js/model.js";
import { Tokenizer } from "../web/js/tokenizer.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODELS = join(root, "web", "models");

ort.env.wasm.wasmPaths = join(root, "node_modules", "onnxruntime-web", "dist") + "/";
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "error";
globalThis.ort = ort;

const GREEDY_TOKENS = 24; // matches tools/dump_engine_cases.py
const LOGPROB_TOLERANCE = 2e-3;
const CONFIDENCE_TOLERANCE = 2e-3;

const manifest = JSON.parse(readFileSync(join(MODELS, "manifest.json"), "utf8"));
const cases = JSON.parse(readFileSync(join(MODELS, "engine-cases.json"), "utf8"));

let checked = 0;
let failed = 0;
const fail = (what, expected, got) => {
  console.error(`FAIL ${what}\n  python ${expected}\n  js     ${got}`);
  failed++;
};

async function loadLocal(name) {
  const spec = manifest.models[name];
  const session = await ort.InferenceSession.create(
    new Uint8Array(readFileSync(join(MODELS, spec.file))),
    { executionProviders: ["wasm"], graphOptimizationLevel: "all" },
  );
  const tokenizer = Tokenizer.fromJSON(readFileSync(join(MODELS, spec.tokenizer), "utf8"));
  return new OnnxLM(session, spec, tokenizer);
}

for (const [name, group] of Object.entries(cases)) {
  if (!manifest.models[name]) {
    console.error(`FAIL ${name} is in the fixtures but not the manifest — re-export`);
    failed++;
    continue;
  }
  const lm = await loadLocal(name);
  process.stdout.write(`${name}: `);

  for (const { prompt, ids } of group.greedy ?? []) {
    const prefilled = await lm.prefill(prompt, { reserve: GREEDY_TOKENS });
    const got = await lm.generate(prefilled, { greedy: true, maxNew: GREEDY_TOKENS });
    if (got.ids.length !== ids.length || got.ids.some((id, i) => id !== ids[i])) {
      fail(`${name} greedy ${JSON.stringify(prompt.slice(-32))}`, ids, got.ids);
    }
    checked++;
  }

  for (const { context, continuation, value } of group.logprob ?? []) {
    const got = await lm.logprob(context, continuation);
    if (!(Math.abs(got - value) < LOGPROB_TOLERANCE)) {
      fail(`${name} logprob ${JSON.stringify(continuation)}`, value, got);
    }
    checked++;
    // The shared-prefill path must agree with the direct one, or a reply would
    // be ranked by numbers that no longer mean what MMI expects.
    const prefilled = await lm.prefill(context, { reserve: 0 });
    const shared = await lm.logprobFrom(prefilled, continuation);
    if (!(Math.abs(shared - got) < LOGPROB_TOLERANCE)) {
      fail(`${name} logprobFrom ${JSON.stringify(continuation)}`, got, shared);
    }
    checked++;
  }

  for (const { prompt, label, confidence } of group.classify ?? []) {
    const got = await lm.classify(prompt);
    if (got.label !== label || !(Math.abs(got.confidence - confidence) < CONFIDENCE_TOLERANCE)) {
      fail(`${name} classify`, `${label} ${confidence.toFixed(4)}`,
        `${got.label} ${got.confidence.toFixed(4)}`);
    }
    checked++;
  }

  for (const { prompt, legal, action } of group.move ?? []) {
    const got = await lm.move(prompt, legal);
    if (got !== action) fail(`${name} move`, action, got);
    checked++;
  }
  process.stdout.write("done\n");
}

// Finally, the path the chat page actually takes: candidates, filtering and MMI
// ranking through `engine.js`. There is no reference to compare a sampled reply
// against, so this asserts the contract instead — a real line, a finite
// relevance score — and prints it so a human can see it is not gibberish.
for (const name of ["chat", "expert-text"]) {
  if (!manifest.models[name] || manifest.models[name].kind !== "chat") continue;
  const lm = await loadLocal(name);
  const engine = new ChatEngine(lm, { random: mulberry32(7) });
  const reply = await engine.reply("hey there, what did you do today?");
  const ok = reply.text.length > 0 && Number.isFinite(reply.score) && reply.source;
  if (!ok) fail(`${name} ChatEngine.reply`, "a scored reply", JSON.stringify(reply));
  checked++;
  console.log(
    `${name} reply: ${JSON.stringify(reply.text)} ` +
      `(rel ${reply.score.toFixed(2)}, ${reply.candidates?.length ?? 0} candidates)`,
  );
}

console.log(`${checked - failed}/${checked} checks pass`);
process.exit(failed ? 1 : 0);

/** A seeded RNG, so a sampled reply is at least reproducible run to run. */
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
