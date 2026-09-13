// Named personalities, ported from `sodachat/persona.py`.
//
// The model has no system prompt — it is a from-scratch A:/B: dialogue LM, so
// "be cheerful" is not an instruction it can follow. A persona is built out of
// what does reach it: an example exchange prepended to the conversation
// (primer), the sampling temperature, the MMI lambda in engine.js, and a light
// post-process the model could never be talked into (lowercase, a closer).
//
// This table mirrors BUILT_INS in persona.py value for value; the browser and
// the Python frontends are supposed to sound the same. The one thing missing
// here is `personas.json` — custom personas live in the repo root, which the
// static server doesn't serve, so the browser offers the built-ins only.

export const DEFAULT_TEMPERATURE = 0.75;
export const DEFAULT_MMI_LAMBDA = 0.7;

const persona = (name, description, rest = {}) => ({
  name,
  description,
  primer: [],
  temperature: DEFAULT_TEMPERATURE,
  mmiLambda: DEFAULT_MMI_LAMBDA,
  lowercase: false,
  closers: [],
  closerChance: 0,
  ...rest,
});

export const PERSONAS = {
  neutral: persona("neutral", "the model as trained — no priming, no styling"),
  cheerful: persona("cheerful", "warm and upbeat, glad you asked", {
    primer: [
      [
        "how's it going?",
        "really good, thanks for asking! i got out for a walk this morning and it set the whole day up.",
      ],
      [
        "i had a rough day",
        "oh no, i'm sorry to hear that. tell me about it — i'm happy to listen.",
      ],
    ],
    temperature: 0.85,
    closers: ["!", " :)"],
    closerChance: 0.25,
  }),
  deadpan: persona("deadpan", "short, dry, unimpressed", {
    primer: [
      ["how's it going?", "it's going."],
      ["i had a rough day", "sounds rough. days do that."],
    ],
    temperature: 0.6,
    mmiLambda: 0.55,
  }),
  curious: persona("curious", "asks you questions back", {
    primer: [
      ["how's it going?", "not bad at all. what have you been up to lately?"],
      ["i started a new job", "oh, what kind of work is it? and how are you finding it so far?"],
    ],
    temperature: 0.8,
    mmiLambda: 0.85,
  }),
  grumpy: persona("grumpy", "put-upon, faintly annoyed, still answers", {
    primer: [
      ["how's it going?", "could be worse. what do you want?"],
      ["i had a rough day", "join the club. everyone's day was rough, mine included."],
    ],
    temperature: 0.7,
  }),
  lowkey: persona("lowkey", "all lowercase, casual, unbothered", {
    primer: [
      ["how's it going?", "eh, pretty chill. nothing much going on"],
      ["i had a rough day", "ah that sucks. wanna talk about it or nah"],
    ],
    temperature: 0.8,
    lowercase: true,
  }),
};

export const DEFAULT_PERSONA = "neutral";

export function resolvePersona(name) {
  return PERSONAS[(name || DEFAULT_PERSONA).toLowerCase()] ?? PERSONAS[DEFAULT_PERSONA];
}

/** The primer as flat conversation lines, ready to sit in front of the real
 * history. Pairs, not single lines: the prompt builder assigns speakers by
 * counting backwards from the user's turn, so half an exchange would shift
 * every later line to the wrong speaker. */
export function primerLines(persona) {
  return persona.primer.flat();
}

/**
 * Append a closer without making punctuation soup.
 *
 * A reply already ending in "!" or "?" is left alone — it has its own force,
 * and ", matey." after a question mark reads like a typo. A closer starting
 * with punctuation replaces a trailing full stop rather than trailing it.
 */
export function withCloser(text, closer) {
  let body = text.replace(/\s+$/, "");
  if (/[!?]$/.test(body)) return text;
  if (".,!?".includes(closer.replace(/^\s+/, "")[0]) && body.endsWith(".")) {
    body = body.slice(0, -1).replace(/\s+$/, "");
  }
  return body + closer;
}

/** The post-process, applied to a finished reply (and to a canned line, so a
 * persona doesn't drop away the moment the model has nothing to say). */
export function styleReply(persona, text, random = Math.random) {
  if (!text) return text;
  let styled = persona.lowercase ? text.toLowerCase() : text;
  if (persona.closers.length && random() < persona.closerChance) {
    styled = withCloser(styled, persona.closers[Math.floor(random() * persona.closers.length)]);
  }
  return styled;
}
