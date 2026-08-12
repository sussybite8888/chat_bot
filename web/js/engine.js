// The chat engine, ported from `sodachat/engine.py`.
//
// Same shape as the Python one, and deliberately so: a reply is several sampled
// candidates, filtered for basic acceptability, then ranked by MMI —
//
//     score = logP(reply | context) - LAMBDA * logP(reply)
//
// — so a fluent continuation that ignores what you said scores low and loses to
// a rougher one that answers it. Everything here is the browser's copy of a
// decision made in engine.py; where a constant appears below, it has the same
// value there, and changing it in one place alone will make the two frontends
// disagree about what the same model says.

const BLOCKLIST =
  /\b(?:fuck\w*|shit\w*|bitch\w*|cunt\w*|nigg\w*|fag\w*|cock\w*|dick\w*|pussy|horn(?:y|ie\w*)|sexy?|nude\w*|naked|porn\w*|slut\w*|whore\w*|rape\w*|penis|vagina|boob\w*|tit(?:s|ties)?)\b/i;

const HAS_CONTENT = /[A-Za-z0-9]/; // did the user type anything real
const HAS_LETTER = /[A-Za-z]/; // is a candidate reply actual text
const SENTENCE = /[^.!?]+[.!?]+(?:['")\]]+)?/g; // reply trimming

const NUDGES = [
  "you there? say something :)",
  "hmm? type something and i'll bite",
  "ok... use your words",
];

const MAX_REPLY_CHARS = 200;
const HISTORY_LINES = 8;
const NUM_CANDIDATES = 12;
const GEN_TEMPERATURE = 0.75;
const MMI_LAMBDA = 0.7;
const TRIM_MAX_SENTENCES = 2;
const TRIM_TARGET_CHARS = 100;

const clean = (text) => (text || "").replace(/\s+/g, " ").trim();

/** Keep the first sentence or two — sampled tails tend to wander. */
export function trimReply(text) {
  const sentences = text.match(SENTENCE);
  if (!sentences) return text; // no complete sentence ("lol", "why") — keep it
  const kept = [];
  for (const sentence of sentences) {
    if (
      kept.length &&
      (kept.length >= TRIM_MAX_SENTENCES ||
        kept.reduce((n, s) => n + s.length, 0) > TRIM_TARGET_CHARS)
    ) {
      break;
    }
    kept.push(sentence.trim());
  }
  return kept.join(" ");
}

/**
 * Render a conversation as tagged turns, ending with the bot's turn open.
 *
 * Ends with "B:" and no trailing space on purpose: byte-level BPE folds a
 * leading space into the following word, so a trailing space would be its own
 * token — a sequence never seen after "B:" in training. See `model.build_prompt`.
 */
export function buildPrompt(history, message, { user = "A", bot = "B", prefix = "" } = {}) {
  const turns = [...history, message];
  const speakers = [user, bot];
  const start = (turns.length - 1) % 2; // the user always speaks last
  const lines = turns.map((t, i) => `${speakers[(start + i) % 2]}: ${t}`);
  return prefix + lines.join("\n") + `\n${bot}:`;
}

/** A bot turn with no conversation — the MMI baseline for P(reply). */
export function nullPrompt({ bot = "B", prefix = "" } = {}) {
  return `${prefix}${bot}:`;
}

export class ChatEngine {
  /** `lm` is an OnnxLM; `filtered` keeps the word filter on, as in Python. */
  constructor(lm, { filtered = true, random = Math.random } = {}) {
    this.lm = lm;
    this.filtered = filtered;
    this.random = random;
    this.recent = [];
    this.backend = lm.spec.label ?? "onnx";
  }

  _acceptable(candidate, userText) {
    if (!candidate || !HAS_LETTER.test(candidate)) return false;
    if (candidate.length > MAX_REPLY_CHARS) return false;
    if (this.filtered && BLOCKLIST.test(candidate)) return false;
    const lowered = candidate.toLowerCase();
    return lowered !== userText.toLowerCase() && !this.recent.includes(lowered);
  }

  _nudge() {
    return {
      text: NUDGES[Math.floor(this.random() * NUDGES.length)],
      source: "canned",
      score: 0,
    };
  }

  /**
   * Generate a reply. `history` is recent conversation lines (both sides,
   * oldest first).
   *
   * `onEvent` reports progress, and is awaited: a reply is a few hundred
   * forward passes, and the caller needs somewhere to hand the browser a turn
   * to repaint — otherwise the page sits frozen and then blinks to an answer.
   */
  async reply(message, { history = [], onEvent = () => {} } = {}) {
    const text = clean(message);
    if (!HAS_CONTENT.test(text)) return this._nudge();

    const lines = history.map(clean).filter(Boolean);
    // Keep whole user/bot pairs so speaker tags stay aligned.
    const kept = lines.length ? lines.slice(-(HISTORY_LINES - (HISTORY_LINES % 2))) : [];
    const prompt = buildPrompt(kept, text, this.lm.spec.prompt);

    await onEvent({ stage: "prefill" });
    const prefilled = await this.lm.prefill(prompt);

    const candidates = [];
    for (let i = 0; i < NUM_CANDIDATES; i++) {
      const { text: raw } = await this.lm.generate(prefilled, {
        temperature: GEN_TEMPERATURE + 0.05 * (i % 3),
      });
      const candidate = trimReply(raw.trim());
      if (this._acceptable(candidate, text) && !candidates.includes(candidate)) {
        candidates.push(candidate);
      }
      await onEvent({
        stage: "sampling",
        done: i + 1,
        total: NUM_CANDIDATES,
        kept: candidates.length,
      });
    }
    if (!candidates.length) return this._nudge();

    // MMI: prefer candidates the conversation makes likely over ones that are
    // simply likely to be said at all. Both sides are scored in the same
    // bot-turn frame, so only the context differs.
    await onEvent({ stage: "ranking", total: candidates.length });
    const scoring = await this.lm.prefill(prompt, { reserve: 0 });
    const baseline = await this.lm.prefill(nullPrompt(this.lm.spec.prompt), { reserve: 0 });
    let best = null;
    for (const candidate of candidates) {
      const score =
        (await this.lm.logprobFrom(scoring, candidate)) -
        MMI_LAMBDA * (await this.lm.logprobFrom(baseline, candidate));
      if (!best || score > best.score) best = { text: candidate, score };
    }

    this.recent.push(best.text.toLowerCase());
    if (this.recent.length > 8) this.recent.shift();
    return { text: best.text, source: this.backend, score: best.score, candidates };
  }
}
