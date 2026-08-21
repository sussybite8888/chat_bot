// Running an exported graph: the attention cache, sampling, and scoring.
//
// This is the browser's answer to `MiniChatLM` / `ExpertLM` — same protocol
// (`generateLine`, `logprob`), same sampling knobs, but stepping an ONNX graph
// through onnxruntime-web instead of a torch module.
//
// The graphs (see `sodachat/export_onnx.py`) take a cache and return it grown:
//
//   idx (1,T) int64, pos (T) int64, past_k/past_v (L,1,H,P,D) float32
//     -> logits (1,T,V), [action_logits, head_logits], present_k/present_v
//
// So a reply is one prefill of the prompt followed by T single-token steps. The
// prompt is the same for every candidate in a reply, so `prefill` is run once
// and its cache *forked* per candidate — 12 candidates for the price of one
// prompt pass, which is most of the work.

import { Tokenizer } from "./tokenizer.js";

/** Sampling defaults, from `model.py` (_CHAT_TOP_K / _CHAT_TOP_P / ...). */
export const CHAT_SAMPLING = {
  top_k: 40,
  top_p: 0.95,
  repetition_penalty: 1.15,
};

const ort = () => globalThis.ort;

/** A cache pair, kept as plain typed arrays so forking is a memcpy. */
class Cache {
  constructor(shape, k, v) {
    this.shape = shape; // [n_layer, 1, n_head, past, head_dim]
    this.k = k;
    this.v = v;
  }

  static empty(config) {
    const shape = [config.n_layer, 1, config.n_head, 0, config.head_dim];
    return new Cache(shape, new Float32Array(0), new Float32Array(0));
  }

  get length() {
    return this.shape[3];
  }

  fork() {
    return new Cache([...this.shape], this.k.slice(), this.v.slice());
  }
}

export class OnnxLM {
  constructor(session, spec, tokenizer) {
    this.session = session;
    this.spec = spec;
    this.tok = tokenizer;
    this.config = spec.config;
    this.block = this.config.block_size;
    this.maxNew = spec.max_new_tokens ?? 48;
    this.newline = spec.specials?.newline ?? this.tok.encode("\n")[0];
    // Stop a reply at end-of-line or end-of-conversation, whichever comes first
    // — `MiniChatLM.__init__` and `ExpertLM._gen` build the same pair.
    this.stop = new Set([this.newline]);
    for (const key of ["end", "dialog_sep"]) {
      if (spec.specials?.[key] !== undefined) this.stop.add(spec.specials[key]);
    }
  }

  /**
   * Fetch and start a model named in the manifest.
   * `onProgress` receives (loadedBytes, totalBytes) while the graph downloads.
   */
  static async load(baseUrl, name, manifest, { onProgress, providers } = {}) {
    const spec = manifest.models[name];
    if (!spec) throw new Error(`${name} is not in the manifest`);
    const [graph, tokenizerJson] = await Promise.all([
      fetchModel(baseUrl, spec, onProgress),
      fetch(`${baseUrl}/${spec.tokenizer}`).then((r) => r.text()),
    ]);
    const session = await ort().InferenceSession.create(graph, {
      executionProviders: providers ?? ["wasm"],
      graphOptimizationLevel: "all",
      // Errors only. Otherwise every load reports, as a console *error*, that
      // some shape ops were placed on CPU — which is ORT working as intended.
      logSeverityLevel: 3,
    });
    return new OnnxLM(session, spec, Tokenizer.fromJSON(tokenizerJson));
  }

  /** One forward pass. `ids` are the new tokens; `cache` is grown in place. */
  async _forward(ids, cache) {
    const T = ids.length;
    const past = cache.length;
    const { Tensor } = ort();
    const pos = BigInt64Array.from({ length: T }, (_, i) => BigInt(past + i));
    const feeds = {
      idx: new Tensor("int64", BigInt64Array.from(ids, BigInt), [1, T]),
      pos: new Tensor("int64", pos, [T]),
      past_k: new Tensor("float32", cache.k, cache.shape),
      past_v: new Tensor("float32", cache.v, cache.shape),
    };
    const out = await this.session.run(feeds);
    const grown = [...cache.shape];
    grown[3] = past + T;
    return {
      out,
      cache: new Cache(grown, out.present_k.data, out.present_v.data),
    };
  }

  /** Logits at the last position, as a plain Float32Array of length V. */
  _lastRow(tensor) {
    const width = tensor.dims[tensor.dims.length - 1];
    return tensor.data.subarray(tensor.data.length - width);
  }

  /**
   * Run the prompt and return the cache to fork from. Trimmed the way
   * `MiniChatLM.generate_line` trims: oldest context first, never the tail.
   */
  async prefill(prompt, { reserve = this.maxNew } = {}) {
    let ids = this.tok.encode(prompt);
    if (!ids.length) ids = [this.newline];
    ids = ids.slice(-(this.block - reserve));
    const { out, cache } = await this._forward(ids, Cache.empty(this.config));
    return { cache, logits: this._lastRow(out.logits), promptIds: ids, text: prompt, out };
  }

  /**
   * `logprob`, reusing a prefill of the context instead of recomputing it.
   *
   * Every candidate in a reply is scored against the same context, and the
   * context is far longer than the candidate — so prefilling it once and
   * forking turns 12 full passes into 12 short ones. The arithmetic is the
   * same either way; only the work is shared.
   */
  async logprobFrom(prefilled, continuation) {
    const cont = this.tok.encode(" " + continuation.trim() + "\n");
    if (!cont.length) return -Infinity;
    // `logprob` trims the context to make room for the continuation, which a
    // shared prefill cannot do per candidate. When that would bite, fall back
    // to the exact path rather than scoring against a longer context than
    // PyTorch would have used.
    if (prefilled.promptIds.length + cont.length > this.block) {
      return this.logprob(prefilled.text, continuation);
    }
    // The prefill's last row already predicts the continuation's first token.
    let total = logSoftmaxAt(prefilled.logits, cont[0]);
    if (cont.length > 1) {
      const { out } = await this._forward(cont.slice(0, -1), prefilled.cache.fork());
      const V = out.logits.dims[2];
      for (let i = 1; i < cont.length; i++) {
        total += logSoftmaxAt(out.logits.data.subarray((i - 1) * V, i * V), cont[i]);
      }
    }
    return total / cont.length;
  }

  /**
   * Sample a continuation, stopping at a stop token or `maxNew` tokens.
   * `prefilled` is what `prefill` returned; its cache is forked, not consumed.
   */
  async generate(prefilled, options = {}) {
    const {
      temperature = 0.8,
      maxNew = this.maxNew,
      stop = this.stop,
      // Greedy decoding takes the argmax of the raw logits, skipping the warps
      // entirely — reproducible, and what the parity checks compare against.
      greedy = false,
      random,
      ...warp
    } = { ...CHAT_SAMPLING, ...options };
    let cache = prefilled.cache.fork();
    let logits = prefilled.logits;
    const generated = [];
    for (let i = 0; i < maxNew; i++) {
      // The repetition penalty sees only this call's own output, matching
      // `generate`'s `idx[:, start:]` — the prompt is not penalized.
      const next = greedy
        ? argmax(logits)
        : sample(warpLogits(logits, generated, temperature, warp), random);
      if (stop.has(next)) break;
      generated.push(next);
      if (prefilled.cache.length + generated.length >= this.block) break;
      const step = await this._forward([next], cache);
      cache = step.cache;
      logits = this._lastRow(step.out.logits);
    }
    return { ids: generated, text: this.tok.decode(generated) };
  }

  /** Prompt -> one line of text, the `generate_line` of the PyTorch wrappers. */
  async generateLine(prompt, options = {}) {
    const prefilled = await this.prefill(prompt);
    const { text } = await this.generate(prefilled, options);
    return text.trim();
  }

  /**
   * Mean per-token log-prob of `continuation` given `context` — the MMI score
   * in `engine.py`, computed exactly as `MiniChatLM.logprob` does: the
   * continuation is framed as a full chat line (" reply\n"), and when the pair
   * overflows the block it is the *context* that gets trimmed.
   */
  async logprob(context, continuation) {
    let ctx = this.tok.encode(context);
    if (!ctx.length) ctx = [this.newline];
    const cont = this.tok.encode(" " + continuation.trim() + "\n");
    if (!cont.length) return -Infinity;
    const overflow = ctx.length + cont.length - this.block;
    if (overflow > 0) ctx = ctx.slice(overflow);
    if (!ctx.length) ctx = [this.newline];
    const ids = [...ctx, ...cont];
    // Predict each continuation token from its predecessor: feed all but the
    // last, and read the row before each target.
    const { out } = await this._forward(ids.slice(0, -1), Cache.empty(this.config));
    const V = out.logits.dims[2];
    const data = out.logits.data;
    let total = 0;
    for (let i = 0; i < cont.length; i++) {
      const row = data.subarray((ctx.length - 1 + i) * V, (ctx.length + i) * V);
      total += logSoftmaxAt(row, cont[i]);
    }
    return total / cont.length;
  }

  /** A specialist classifier: label + confidence off its head at the last token. */
  async classify(prompt) {
    const { out } = await this.prefill(prompt, { reserve: 0 });
    if (!out.head_logits) throw new Error(`${this.spec.file} has no classifier head`);
    const row = this._lastRow(out.head_logits);
    const probs = softmax(row);
    const best = argmax(probs);
    return { label: this.spec.labels[best] ?? String(best), confidence: probs[best] };
  }

  /**
   * A move off the action head, masked to the legal actions — `ExpertLM.move`.
   * The whole board is routed to the game expert simply by being run through
   * the game graph.
   */
  async move(prompt, legal) {
    const { out } = await this.prefill(prompt, { reserve: 0 });
    if (!out.action_logits) throw new Error(`${this.spec.file} has no action head`);
    const row = this._lastRow(out.action_logits);
    const actions = this.spec.actions;
    const allowed = legal?.length
      ? legal.map((a) => actions.indexOf(a)).filter((i) => i >= 0)
      : actions.map((_, i) => i);
    let best = allowed[0];
    for (const i of allowed) if (row[i] > row[best]) best = i;
    return actions[best];
  }
}

// ------------------------------------------------------------------- sampling
//
// A port of `blocks.warp_logits`, applied in the same order (repetition
// penalty, temperature, top-k, then nucleus) so that a browser reply is drawn
// from the same distribution the terminal one is.

export function warpLogits(logits, seq, temperature, { top_k, top_p, repetition_penalty } = {}) {
  const out = Float32Array.from(logits);
  if (repetition_penalty && repetition_penalty !== 1 && seq.length) {
    for (const id of new Set(seq)) {
      out[id] = out[id] > 0 ? out[id] / repetition_penalty : out[id] * repetition_penalty;
    }
  }
  const t = Math.max(temperature, 1e-5);
  for (let i = 0; i < out.length; i++) out[i] /= t;

  if (top_k) {
    const k = Math.min(top_k, out.length);
    // The k-th largest value is the cut; everything strictly below it goes.
    const kth = nthLargest(out, k);
    for (let i = 0; i < out.length; i++) if (out[i] < kth) out[i] = -Infinity;
  }
  if (top_p !== undefined && top_p > 0 && top_p < 1) {
    const order = [...out.keys()].sort((a, b) => out[b] - out[a]);
    const probs = softmax(out);
    let cumulative = 0;
    for (let rank = 0; rank < order.length; rank++) {
      const wasOver = cumulative > top_p;
      cumulative += probs[order[rank]];
      // Shifted by one on purpose: the token that *crosses* the threshold is
      // kept, so at least one candidate always survives (see warp_logits).
      if (wasOver) out[order[rank]] = -Infinity;
    }
  }
  return out;
}

/** Draw one id from warped logits (torch.multinomial over a softmax). */
export function sample(logits, random = Math.random) {
  const probs = softmax(logits);
  let roll = random();
  for (let i = 0; i < probs.length; i++) {
    roll -= probs[i];
    if (roll <= 0) return i;
  }
  return argmax(probs); // only reachable through floating-point drift
}

export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function softmax(logits) {
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  const out = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    total += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

function logSoftmaxAt(row, index) {
  let max = -Infinity;
  for (const x of row) if (x > max) max = x;
  let total = 0;
  for (const x of row) total += Math.exp(x - max);
  return row[index] - max - Math.log(total);
}

/** The k-th largest value, without sorting the whole vocabulary each step. */
function nthLargest(values, k) {
  const heap = Float32Array.from(values.subarray(0, k)).sort();
  for (let i = k; i < values.length; i++) {
    if (values[i] <= heap[0]) continue;
    // Keep the k best seen so far, smallest at the front.
    let j = 0;
    while (j + 1 < k && heap[j + 1] < values[i]) {
      heap[j] = heap[j + 1];
      j++;
    }
    heap[j] = values[i];
  }
  return heap[0];
}

/**
 * Download a model's graph as one `Uint8Array`.
 *
 * A model is either a single `.onnx` or, when `tools/build_dist.mjs` has split
 * it, an ordered list of `chunks` that concatenate back into exactly that file
 * — static hosts cap individual files (Cloudflare Pages at 25 MiB), and a 124MB
 * graph has to arrive in pieces. Either way the bytes land in one buffer sized
 * up front from the manifest, so nothing is copied twice on the way in.
 */
async function fetchModel(baseUrl, spec, onProgress) {
  const parts = spec.chunks?.length ? spec.chunks : [spec.file];
  const total = spec.bytes ?? 0;
  const buffer = new Uint8Array(total);
  let loaded = 0;
  const write = (bytes) => {
    // Overrunning would throw a bare RangeError out of `set`; say what it means.
    if (loaded + bytes.length > total) {
      throw new Error(`${spec.file}: parts are longer than the manifest's ${total} bytes`);
    }
    buffer.set(bytes, loaded);
    loaded += bytes.length;
    onProgress?.(loaded, total);
  };
  for (const part of parts) {
    const url = `${baseUrl}/${part}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
    if (!response.body) {
      write(new Uint8Array(await response.arrayBuffer())); // no streaming available
      continue;
    }
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      write(value);
    }
  }
  // A missing or truncated chunk otherwise reaches onnxruntime as a corrupt
  // protobuf, which it reports as a parse error a long way from the cause.
  if (loaded !== total) {
    throw new Error(
      `${spec.file}: expected ${total} bytes across ${parts.length} part(s), got ${loaded}`,
    );
  }
  return buffer;
}
