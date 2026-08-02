"""Routing — which capability should answer a message — as a specialist itself.

The agent has grown one specialist per capability: vision (say what an image
shows), code (name a source file's language), codegen (write code), reason (work
a question out step by step). Deciding *which* of them a plain-text message
wants was, until now, a cascade of hand-written triggers in `agent.py`: a regex
per capability, tried in a fixed order, where the order itself did real work —
the reasoning check had to sit ahead of the code-Q&A check, whose trigger list
contains the word "in" and would otherwise swallow every word problem.

That cascade is a routing model with hand-tuned weights. This module trains the
real thing: a **routing specialist**, the same frozen-trunk add-on shape as the
others, that reads a message and names its destination.

    <|route|>
    what language was that file again?
    <|dest|>                              -> code (0.94)

Six destinations — exactly the ones the agent can dispatch a bare message to:

    chat      ordinary conversation: the default, and the one worth protecting
    reason    a question that wants working out          -> reason.think
    codegen   a request to write code                    -> codegen.generate
    code      a question about the source file last read -> the code specialist
    vision    a question about the image last seen       -> the vision specialist
    game      a question about the live game state       -> the reader

The label reads off a 6-way class head at the trailing `<|dest|>` token in a
single forward pass — the same mechanism as the vision label, the code language
and the game move. The whole document routes to the routing expert, seeded from
the *text* expert (unlike vision and code, whose glyph-grid inputs are closer to
the game expert's diet, a user message is ordinary dialogue).

`<|dest|>` is private rather than the shared `<|cls|>` marker on purpose: when
two specialists share a marker token, whichever attaches first owns its
embedding row and the later one's trained row is discarded
(`attach_specialist`). A classifier reads its head at exactly that position, so
inheriting a row it never trained against is the one place that quirk really
bites.

Corpus
------
There is no public dataset of "messages people send a chatbot, labelled by which
subsystem should answer". Three of the classes have a natural stand-in and three
are synthesized:

    chat      DailyDialog + SODA + NPS Chat — the same dialogue corpora the chat
              model is trained on, one utterance per example
    reason    the question side of GSM8K, Orca-Math, MetaMathQA and AQuA-RAT,
              plus short arithmetic asks ("what's 15% of 240"), which the word-
              problem corpora don't contain but users type constantly
    codegen   MBPP task descriptions and CodeSearchNet docstrings, each wrapped
              in a request phrasing ("write a {lang} function that {desc}")
    code      templated questions about a file just read
    vision    templated questions about an image just seen
    game      templated questions about a live game, plus `reader._QUESTIONS` —
              the phrasings the reader is itself trained to answer

Synthesized classes are the honest weak point: a few dozen skeletons crossed
with slot values and roughened by `_perturb` (fillers, punctuation, casing) are
much easier than real messages. Two things keep the reported numbers from
flattering that:

  * The held-out split holds out whole **skeletons**, not phrasings — one in
    six, by digest order, so a validation message's *shape* was never trained
    on and the proportion holds even for a class written from fifteen of them.
    (For the corpus-backed classes the key is the text itself; for codegen it is
    the task description, so one task can't appear on both sides in two
    different wrappers.)
  * `PROBE` is ~50 hand-written messages that came from no template at all.
    That accuracy, not the held-out number, is the one to believe.

Because chat is the default, the cost of the two error directions is not
symmetric: missing a specialist just means a chat reply, while a false positive
answers small talk with generated code. Inference therefore thresholds at
`MIN_CONF` and falls back to chat below it, and `evaluate` reports the chat
leakage rate at that threshold as the headline safety number.

    python -m sodachat.route train    # needs models/expert.pt; trunk stays frozen
    python -m sodachat.route eval     # held-out + hand-written probe accuracy
    python -m sodachat.route demo     # the probe set, message by message
    python -m sodachat.route ask "what language was that?"
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import make_amp, pick_device
from .expert import (
    _MODELS,
    DEFAULT_PATH as EXPERT_PATH,
    TEXT,
    ExpertLM,
    _interleave,
    save_specialist,
    scaffold_specialist,
    specialist_param_groups,
)

NAME = "route"
ROUTE, DEST = "<|route|>", "<|dest|>"
# Order is the head's label order; "chat" is index 0 and the fallback everywhere.
LABELS = ["chat", "reason", "codegen", "code", "vision", "game"]
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
DEFAULT_PATH = _MODELS / "specialist-route.pt"

MIN_CONF = 0.6   # below this the agent stays in chat (see `pick`)
MAX_CHARS = 400  # a chat message longer than this is truncated before routing
HOLDOUT = 6      # 1-in-6 keys held out for validation

# How many distinct examples to keep per class. They differ by an order of
# magnitude — the corpus-backed classes have far more phrasing to draw on than
# the templated ones — and training samples the classes evenly regardless.
CAPS = {"chat": 15_000, "reason": 10_000, "codegen": 8_000,
        "code": 2_500, "vision": 2_500, "game": 2_500}


# ------------------------------------------------------------------ phrasings


_FILLERS = ["hey ", "hi ", "so ", "ok ", "um ", "quick question, ", "wait ",
            "btw ", "hey, ", "also ", "one more thing, ", "sorry, ", "and "]
_TAILS = [" please", " thanks", " again", " for me", " if you can", " btw"]


def _perturb(phrase: str, rng: random.Random) -> str:
    """Roughen a template phrasing the way people actually type: a leading
    filler, a trailing politeness, punctuation and capitalization that come and
    go. Without this a synthesized class is identifiable by its punctuation
    alone, and the router learns that instead of the words."""
    s = phrase
    if rng.random() < 0.30:
        s = rng.choice(_FILLERS) + s
    if rng.random() < 0.20:
        s = s + rng.choice(_TAILS)
    if not s.endswith(("?", "!", ".")):
        s = s + rng.choice(["?", "?", "?", ".", "", "", "!"])
    roll = rng.random()
    if roll < 0.40:
        s = s[:1].upper() + s[1:]
    elif roll < 0.44:
        s = s.upper()
    return " ".join(s.split())


def _expand(templates: list[str], slots: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Every template crossed with every value of the slots it names, each paired
    with the template it came from. The split holds out whole skeletons, so the
    template is the key and the filled phrasing is the example."""
    out: list[tuple[str, str]] = []
    for template in templates:
        keys = [k for k in slots if "{" + k + "}" in template]
        combos = itertools.product(*(slots[k] for k in keys)) if keys else [()]
        for combo in combos:
            out.append((template, template.format(**dict(zip(keys, combo)))))
    return out


# Questions about the image the vision specialist last looked at.
_VISION = _expand([
    "what's in the {pic}",
    "what is in the {pic}",
    "what did you see in the {pic}",
    "what was that {pic}",
    "what does the {pic} show",
    "can you tell what the {pic} is",
    "describe the {pic}",
    "what was the {pic} again",
    "did you recognize the {pic}",
    "tell me what the {pic} was",
    "what do you think that {pic} is",
    "how sure are you about the {pic}",
    "what did the {pic} look like to you",
    "was the {pic} a dog",
    "what number was in the {pic}",
    "what did you make of the {pic}",
    "you saw the {pic} right, what was it",
    "identify the {pic}",
], {"pic": ["image", "picture", "photo", "pic", "screenshot", "photograph",
            "png", "jpg", "snapshot", "drawing"]})

# Questions about the source file the code specialist last read.
_CODE = _expand([
    "what language is {it}",
    "what language was {it} written in",
    "what programming language is {it}",
    "can you tell what language {it} is",
    "which language is {it}",
    "do you know what language {it} is in",
    "what kind of code is {it}",
    "was {it} python",
    "what language was {it} again",
    "tell me what language {it} is",
    "how sure are you about {it}",
    "what did you think {it} was written in",
    "is {it} javascript",
    "what language did you say {it} was",
    "name the language of {it}",
], {"it": ["that", "this", "that file", "the file", "that snippet", "the snippet",
           "that code", "the code", "it", "the source file", "my file"]})

# Questions about a game that is currently running — what `reader` answers.
_GAME = (
    _expand([
        "what's my {stat}",
        "what is the {stat} right now",
        "how's the {stat}",
        "tell me the {stat}",
        "what's the {stat} now",
        "can you tell me the {stat}",
        "any idea what the {stat} is",
        "give me the {stat}",
    ], {"stat": ["score", "length", "high score", "snake length", "current score",
                 "board", "state"]})
    + _expand([
        "where is the {thing}",
        "which way to the {thing}",
        "how far is the {thing}",
        "where do i find the {thing}",
        "can you see the {thing}",
        "how close am i to the {thing}",
    ], {"thing": ["food", "apple", "ball", "target", "block", "wall"]})
    + _expand([
        "is the {noun} over",
        "are we still {verb}ing",
        "did the {noun} end",
        "how's the {noun} going",
        "are you still {verb}ing",
        "did you lose the {noun}",
    ], {"noun": ["game", "round", "match"], "verb": ["play", "go"]})
)

# Short arithmetic asks. The word-problem corpora are all several sentences
# long; this is the other half of what people actually ask a bot to work out.
_ARITH = [
    "what's {a} times {b}",
    "what is {a} plus {b}",
    "how much is {a} minus {b}",
    "{a} divided by {b} is what",
    "what's {p}% of {a}",
    "how many is {a} times {b}",
    "what's the total of {a} and {b}",
    "add up {a}, {b} and {c}",
    "what do i get if i take {b} away from {a}",
    "if i split {a} between {b} people how much does each get",
    "how many {b}s go into {a}",
    "{a} plus {b} equals",
    "work out {a} times {b} minus {c}",
    "whats {p} percent of {a}",
]

_LANGS = ["python", "py", "javascript", "js", "java", "go", "ruby", "php",
          "typescript", "c++", "node", "bash"]

# Wrappers that turn a task description into a request to write code. A
# description arrives in one of two grammatical forms — MBPP's imperative
# ("find the shared elements") and a docstring's third person ("returns the
# sum") — and a wrapper fits one or the other, so they are kept apart.
_ASK_TO = [       # ... to <imperative>
    "write me a function to {desc}",
    "generate a {lang} function to {desc}",
    "make a {lang} function to {desc}",
    "write a script to {desc}",
    "show me {lang} code to {desc}",
    "how do i {desc} in {lang}",
    "what's the best way to {desc} in {lang}",
    "i need {lang} code to {desc}",
    "write the {lang} to {desc}",
]
_ASK_THAT = [     # ... that <third person>
    "write a {lang} function that {desc}",
    "can you write {lang} code that {desc}",
    "i need a {lang} function that {desc}",
    "give me a {lang} snippet that {desc}",
    "code up something that {desc}",
    "a {lang} function that {desc}",
    "could you write a function that {desc}",
    "write a script that {desc}",
]

_NOT_A_VERB = frozenset("this its his hers less class pass cross press access "
                        "address gas news series status always plus".split())


# ------------------------------------------------------------------- the split


def _is_holdout(key: str) -> bool:
    """Deterministic 1-in-HOLDOUT split on a stable hash of an example's *key* —
    its text or its task description, whichever carries the meaning (see the
    module docstring). Python's `hash` is per-process salted, so md5 it is."""
    digest = hashlib.md5((key or "").strip().lower().encode("utf-8", "ignore")).digest()
    return digest[0] % HOLDOUT == 0


def _every_nth(keys, n: int = HOLDOUT) -> set[str]:
    """A deterministic 1-in-n subset of a *small* key set. The hash split above
    is fine across tens of thousands of corpus texts, where the proportion comes
    out in the wash; a class written from fifteen skeletons can easily hash one
    of them (or fourteen) into validation. Ordering by digest and taking every
    nth keeps the proportion exact whatever the phrasings happen to be."""
    ordered = sorted(set(keys),
                     key=lambda k: hashlib.md5(k.encode("utf-8", "ignore")).hexdigest())
    return {k for i, k in enumerate(ordered) if i % n == 0}


# --------------------------------------------------------------- class sources
#
# Each source yields (held_out, text) pairs — the message the router sees, and
# which side of the split it belongs on. Sources decide that themselves because
# they know what carries the meaning: a corpus source hashes its text, a
# templated one holds out whole skeletons. Network or schema trouble on one
# dataset is logged and skipped — the rest still train, and `_corpus` refuses to
# build a corpus with an empty class.


def _stream(repo: str, config: str | None, split: str, log, columns=None, **kw):
    """Rows from a streamed dataset, pruned to the columns actually read. The
    pruning is not cosmetic: a CodeSearchNet row carries the whole function body
    and a SODA row a narrative and a dozen other fields, and streaming pulls
    remote parquet through memory a row group at a time — asking for one column
    is the difference between hundreds of megabytes of buffer and a few."""
    try:
        from datasets import load_dataset

        ds = load_dataset(repo, config, split=split, streaming=True, **kw)
        if columns:
            try:
                ds = ds.select_columns(columns)
            except Exception:  # unsupported builder or a renamed field
                pass
        yield from ds
    except Exception as e:  # offline, gated, renamed, schema change, ...
        tag = repo + (f"[{config}]" if config else "")
        log(f"  ! skipping {tag}: {type(e).__name__}: {str(e)[:120]}")


def _template_source(pairs: list[tuple[str, str]], rng: random.Random, n: int):
    """Sample template phrasings, roughened, until `n` attempts are spent —
    bounded rather than infinite so a caller that can never fill its cap stops
    anyway. Validation gets whole skeletons, so a held-out message is one whose
    *shape* was never trained on, not merely one whose slot value differs."""
    held_out = _every_nth(s for s, _ in pairs)
    for _ in range(n):
        skeleton, phrase = pairs[rng.randrange(len(pairs))]
        yield skeleton in held_out, _perturb(phrase, rng)


# Utterances that look like another class out of context. A dialogue corpus is
# full of "how much is it?" and "what language do they speak?"; keeping them as
# chat teaches the router the opposite of what the other classes teach. Dropping
# them is not a claim they are misrouted — with no context they are genuinely
# ambiguous, and the router only ever sees one message.
_CONFLICT = re.compile(
    r"\bwhat (programming )?language\b"
    r"|\b(write|generate|code) (me )?(a|an|some)?\s?\w*\s?(function|script|program|code|snippet)\b"
    r"|\bin the (image|picture|photo|screenshot)\b"
    r"|\bwhat'?s? the score\b|\bhow long is the snake\b|\bwhere'?s? the (food|apple)\b",
    re.IGNORECASE)
_QUANTITY = re.compile(
    r"\bhow (many|much)\b|\bwhat'?s? the (total|sum|average|difference|product)\b",
    re.IGNORECASE)


def _plain_chat(utterance: str) -> bool:
    if not 1 <= len(utterance) <= 200:
        return False
    if _CONFLICT.search(utterance):
        return False
    # A quantity question with real numbers in it is a `reason` example, whoever
    # said it. Bare "how much do you like me?" stays chat.
    return not (_QUANTITY.search(utterance) and len(re.findall(r"\d", utterance)) >= 2)


def _utterances(repo: str, field: str, scan: int, log, **kw):
    from .data import clean_utterance

    for row in itertools.islice(
            _stream(repo, None, "train", log, columns=[field], **kw), scan):
        for utterance in row.get(field) or ():
            text = clean_utterance(utterance)
            if _plain_chat(text):
                yield _is_holdout(text), text


def _nps_utterances(log):
    """NPS Chat: short, lowercase, typo-ridden — the closest thing in the project
    to how people actually type at a bot."""
    try:
        from .corpus import load_sessions

        sessions = load_sessions()
    except Exception as e:
        log(f"  ! skipping nps_chat: {type(e).__name__}: {str(e)[:120]}")
        return
    for session in sessions:
        for post in session:
            if post.act != "System" and _plain_chat(post.text):
                yield _is_holdout(post.text), post.text


def _chat_source(rng: random.Random, log):
    """One conversational turn per example, from the dialogue corpora the chat
    model itself is trained on — so the class the router must protect is drawn
    from exactly the traffic chat actually gets. Interleaved rather than
    concatenated: DailyDialog alone would fill the cap several times over, and
    the class would quietly become one corpus's house style."""
    from .data import DAILYDIALOG_REPO, SODA_REPO

    return _interleave(
        [_utterances(DAILYDIALOG_REPO, "dialog", 12_000, log,
                     revision="refs/convert/parquet"),
         _utterances(SODA_REPO, "dialogue", 12_000, log),
         _nps_utterances(log)],
        [1, 1, 1])


_REASON_SOURCES = [  # (repo, config, question field, rows to scan)
    ("openai/gsm8k", "main", "question", 7_000),
    ("microsoft/orca-math-word-problems-200k", None, "question", 6_000),
    ("meta-math/MetaMathQA", None, "query", 6_000),
    ("deepmind/aqua_rat", "raw", "question", 4_000),
]


def _questions(repo: str, config: str | None, field: str, scan: int, log):
    for row in itertools.islice(
            _stream(repo, config, "train", log, columns=[field]), scan):
        question = " ".join((row.get(field) or "").split())
        if 12 <= len(question) <= MAX_CHARS:
            yield _is_holdout(question), question


def _arithmetic(rng: random.Random, n: int):
    """Short calculations, filled with fresh numbers. Formatted first and
    roughened second — `_perturb` may upper-case the whole string, which would
    turn `{a}` into a `{A}` that no longer names a slot."""
    held_out = _every_nth(_ARITH)
    for _ in range(n):
        template = _ARITH[rng.randrange(len(_ARITH))]
        filled = template.format(
            a=rng.randint(2, 999), b=rng.randint(2, 99), c=rng.randint(2, 99),
            p=rng.choice([5, 10, 15, 20, 25, 30, 40, 60, 75, 90]))
        yield template in held_out, _perturb(filled, rng)


def _reason_source(rng: random.Random, log):
    """The question side of the same word-problem corpora the reasoning
    specialist trains on (its answers are irrelevant here — routing only ever
    sees the question), plus short arithmetic asks. The arithmetic stream is
    weighted into the mix rather than appended: GSM8K alone would fill the cap,
    and every reasoning example would then be several sentences long."""
    streams = [_questions(repo, config, field, scan, log)
               for repo, config, field, scan in _REASON_SOURCES]
    return _interleave(streams + [_arithmetic(rng, 6_000)],
                       [2, 2, 2, 1, 3])


_SENTENCE = re.compile(r"(?<=[.!?])\s")
_MBPP_LEAD = re.compile(
    r"^\s*write a (python )?(function|program|script)\s+(to|that|which)\s+", re.IGNORECASE)
# Docstring machinery and markup: a description is meant to read like something
# a person would type, and "+const_name+" (RDoc) or "`x`" (Markdown) doesn't.
_DOC_NOISE = re.compile(r":param|@param|>>>|--|\.\.\.|http|\bTODO\b"
                        r"|[`<>|\\]|\+\S+\+", re.IGNORECASE)


def _clean_desc(text: str) -> str | None:
    """A one-line task description a request can be built around, or None. Takes
    the first sentence, drops docstring machinery, and lowercases the leading
    word so "Returns the sum" reads as "... that returns the sum"."""
    first = _SENTENCE.split(" ".join((text or "").split()), 1)[0].strip().rstrip(".")
    if not 20 <= len(first) <= 120 or " " not in first:
        return None
    if _DOC_NOISE.search(first) or not first.isascii():
        return None
    return first[:1].lower() + first[1:]


def _third_person(desc: str) -> bool:
    """Whether a description reads "returns the sum" rather than "return the
    sum" — a docstring habit that MBPP's imperative task list doesn't share.
    First word only: wrong occasionally ("process the queue") and cheap."""
    head = desc.split(" ", 1)[0]
    return head.endswith("s") and len(head) > 3 and head not in _NOT_A_VERB


def _requests(descs, rng: random.Random):
    """Task descriptions -> requests to write code, in a wrapper that agrees
    with how the description is worded. The split is keyed on the description,
    so one task cannot appear on both sides wearing two different wrappers; the
    wrappers themselves are shared across the split, because what is held out
    here is the task, not the phrasing."""
    for desc in descs:
        asks = _ASK_THAT if _third_person(desc) else _ASK_TO
        template = asks[rng.randrange(len(asks))]
        yield (_is_holdout(desc),
               _perturb(template.format(lang=rng.choice(_LANGS), desc=desc), rng))


def _mbpp_descs(log):
    for row in itertools.islice(
            _stream("google-research-datasets/mbpp", "full", "train", log,
                    columns=["text"]), 1_000):
        if (desc := _clean_desc(_MBPP_LEAD.sub("", row.get("text") or ""))):
            yield desc


def _csn_descs(lang: str, log):
    for row in itertools.islice(
            _stream("code-search-net/code_search_net", lang, "train", log,
                    columns=["func_documentation_string"]), 4_000):
        if (desc := _clean_desc(row.get("func_documentation_string") or "")):
            yield desc


def _codegen_source(rng: random.Random, log):
    """Task descriptions wrapped in a request. MBPP is already phrased as a task
    ("write a function to find the shared elements from two lists"); a
    CodeSearchNet docstring is the same thing written the other way round, and
    there are far more of them — so the languages are interleaved and MBPP,
    which has under a thousand usable rows, gets a heavy weight to survive."""
    langs = ("python", "java", "javascript", "go", "php", "ruby")
    streams = [_mbpp_descs(log)] + [_csn_descs(lang, log) for lang in langs]
    return _requests(_interleave(streams, [3] + [1] * len(langs)), rng)


def _game_source(rng: random.Random, log):
    """Templated live-game questions plus `reader._QUESTIONS` — the phrasings the
    reader is trained to answer, so the class is defined by what the destination
    can actually do with it."""
    from .reader import _QUESTIONS

    asked = _GAME + [(q, q) for phrasings in _QUESTIONS.values() for q in phrasings]
    yield from _template_source(asked, rng, 40_000)


_SOURCES = {
    "chat": _chat_source,
    "reason": _reason_source,
    "codegen": _codegen_source,
    "code": lambda rng, log: _template_source(_CODE, rng, 40_000),
    "vision": lambda rng, log: _template_source(_VISION, rng, 40_000),
    "game": _game_source,
}


def _corpus(seed: int, caps: dict | None = None, sides=("train", "val"), log=print):
    """{side: (docs, labels)} — every class, deduplicated and capped, both sides
    of the split taken from a **single** pass over each source.

    One pass rather than one per side because these sources are streamed: a
    second traversal costs the same minutes again, and for the ones read as
    remote parquet (SODA, CodeSearchNet) it costs the same gigabytes of buffered
    file again too. The split is per-example anyway, so both sides fall out of
    the same walk.

    Classes come out at wildly different sizes — chat has far more phrasing to
    draw on than vision does — and training samples them evenly regardless."""
    caps = caps or CAPS
    out = {side: ([], []) for side in sides}
    for label, source in _SOURCES.items():
        want = {"train": caps[label], "val": max(caps[label] // HOLDOUT, 300)}
        kept = {side: [] for side in sides}
        # One `seen` across both sides: two skeletons can perturb into the same
        # string, and that string must not land on both sides of the split.
        seen: set[str] = set()
        for held_out, text in source(random.Random(seed + LABEL_ID[label]), log):
            side = "val" if held_out else "train"
            if side not in kept or len(kept[side]) >= want[side]:
                if all(len(k) >= want[s] for s, k in kept.items()):
                    break
                continue
            text = " ".join((text or "").split())[:MAX_CHARS]
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            kept[side].append(text)
        for side in sides:
            if not kept[side]:
                raise RuntimeError(f"no {side} examples for the {label!r} route — "
                                   f"its sources are all unreachable")
            docs, labels = out[side]
            docs.extend(route_doc(t) for t in kept[side])
            labels.extend([LABEL_ID[label]] * len(kept[side]))
        log("  " + f"{label:8} " + " | ".join(
            f"{len(kept[s]):>6,} {s}" for s in sides))
    return {side: (docs, np.asarray(labels, dtype=np.int64))
            for side, (docs, labels) in out.items()}


# --------------------------------------------------------------- doc rendering


def route_doc(message: str) -> str:
    """One message as a classification document. No `<|end|>`: docs are never
    packed into a stream, and the head fires at the trailing `<|dest|>`."""
    return f"{ROUTE}\n{' '.join((message or '').split())[:MAX_CHARS]}\n{DEST}"


def _tokenize(tok, docs) -> list[np.ndarray]:
    return [np.asarray(ids, dtype=np.int32) for ids in tok.encode_batch(docs)]


def _batch(ids_list, labels, pick, device):
    """Right-pad a batch; the head reads each doc at its own final (`<|dest|>`)
    position, so padding after it is causally invisible."""
    seqs = [ids_list[i] for i in pick]
    x = np.zeros((len(seqs), max(len(s) for s in seqs)), dtype=np.int64)
    last = np.empty(len(seqs), dtype=np.int64)
    for j, s in enumerate(seqs):
        x[j, : len(s)] = s
        last[j] = len(s) - 1
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(last).to(device),
            torch.from_numpy(labels[np.asarray(pick)]).to(device))


def _logits(model, task: int, x, last):
    feats = model._features(x, torch.full_like(x, task))
    return model.heads[NAME](feats[torch.arange(x.shape[0], device=x.device), last])


@torch.no_grad()
def _predict(model, task: int, ids_list, labels, device, bs: int = 128) -> np.ndarray:
    """Class probabilities for every doc — (n, len(LABELS))."""
    was_training = model.training
    model.eval()
    probs = np.zeros((len(ids_list), len(LABELS)), dtype=np.float32)
    for start in range(0, len(ids_list), bs):
        idx = np.arange(start, min(start + bs, len(ids_list)))
        x, last, _ = _batch(ids_list, labels, idx, device)
        probs[idx] = F.softmax(_logits(model, task, x, last).float(), -1).cpu().numpy()
    if was_training:
        model.train()
    return probs


# -------------------------------------------------------------------- scoring


def _macro(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean per-class accuracy. The classes are deliberately unbalanced, so plain
    accuracy would mostly report how well chat is doing."""
    pred = probs.argmax(1)
    per = [float((pred[y == i] == i).mean()) for i in range(len(LABELS)) if (y == i).any()]
    return sum(per) / max(len(per), 1)


def _report(probs: np.ndarray, y: np.ndarray, log=print, min_conf: float = MIN_CONF,
            title: str = "held-out") -> float:
    """Per-class accuracy, what the confidence threshold does to it, and the
    confusions that actually happened."""
    pred, conf = probs.argmax(1), probs.max(1)
    log(f"{title}: {len(y):,} messages, threshold {min_conf:.2f}")
    for i, label in enumerate(LABELS):
        mask = y == i
        if not mask.any():
            continue
        acc = float((pred[mask] == i).mean())
        firing = float(((pred[mask] == i) & (conf[mask] >= min_conf)).mean())
        note = ("stays in chat" if label == "chat" else "reaches its specialist")
        log(f"  {label:8} acc {acc:6.1%} | {firing:6.1%} {note} at the threshold "
            f"({int(mask.sum()):,} messages)")
    chat = y == LABEL_ID["chat"]
    if chat.any():
        leak = float(((pred[chat] != LABEL_ID["chat"]) & (conf[chat] >= min_conf)).mean())
        log(f"  chat leakage: {leak:.2%} of ordinary messages are confidently "
            f"routed away from chat")
    confusions = [(int(((y == i) & (pred == j)).sum()), LABELS[i], LABELS[j])
                  for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j]
    worst = [c for c in sorted(confusions, reverse=True) if c[0]][:5]
    if worst:
        log("  top confusions: "
            + ", ".join(f"{a}->{b} {n}" for n, a, b in worst))
    macro = _macro(probs, y)
    log(f"  macro accuracy {macro:.1%}")
    return macro


# ------------------------------------------------------------------- training


def train(base=EXPERT_PATH, out=DEFAULT_PATH, steps=1500, batch_size=36, lr=3e-4,
          device=None, seed=0, eval_every=250, caps=None, log=print) -> Path:
    """Train the routing specialist on a *frozen* expert: gradients reach only
    the new per-block FFN expert, the 6-way head, and the two new tokens'
    embedding rows. Every existing capability — including the specialists this
    one routes to — is untouched by construction.

    Batches are class-balanced (`batch_size // len(LABELS)` messages per route)
    rather than drawn from the corpus mix: the corpus has six times more chat
    than vision because chat has six times more phrasing to sample, which says
    nothing about how often either should win.

    The schedule is short on purpose. This task saturates in a few hundred steps
    — a 4,000-step run peaked on held-out macro accuracy at step 500 and then
    spent 20 minutes fitting the synthesized classes harder, which held the
    held-out number flat near 90% while probe accuracy *fell* from ~81% to ~69%.
    Only the best held-out checkpoint is kept, so a longer run is wasted time
    rather than a worse model, but there is nothing to buy past ~1.5k steps."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, tok, task = scaffold_specialist(base, name=NAME, special_tokens=[ROUTE, DEST],
                                           n_labels=len(LABELS), seed_from=TEXT,
                                           device=device)
    per_block = sum(p.numel() for p in model.blocks[0].ffn.experts[task].parameters())
    trainable = (per_block * model.cfg.n_layer
                 + sum(p.numel() for p in model.heads[NAME].parameters())
                 + 2 * model.cfg.n_embd)  # the <|route|>/<|dest|> embedding rows
    log(f"specialist '{NAME}' on {Path(base).name}: expert slot {task} | "
        f"{trainable / 1e6:.1f}M trainable of {model.num_params() / 1e6:.1f}M "
        f"(shared trunk frozen) | device {device}")

    log("building the routing corpus...")
    corpus = _corpus(seed, caps, log=log)
    train_docs, train_y = corpus["train"]
    val_docs, val_y = corpus["val"]
    log(f"tokenizing {len(train_docs):,} train + {len(val_docs):,} val messages...")
    train_ids, val_ids = _tokenize(tok, train_docs), _tokenize(tok, val_docs)
    lens = [len(s) for s in train_ids]
    assert max(lens) <= model.cfg.block_size, "routing docs overflow the block"
    assert train_ids[0][-1] == tok.token_id(DEST)
    probe_ids = _tokenize(tok, [route_doc(t) for t, _ in PROBE])
    probe_y = np.asarray([LABEL_ID[label] for _, label in PROBE], dtype=np.int64)

    by_class = [np.flatnonzero(train_y == i) for i in range(len(LABELS))]
    per_class = max(batch_size // len(LABELS), 1)
    seen = steps * per_class * len(LABELS)
    log(f"docs: {min(lens)}-{max(lens)} tokens (block {model.cfg.block_size}) | "
        f"schedule: {steps:,} steps x {per_class} per route x {len(LABELS)} routes "
        f"= {seen / 1e3:.0f}k messages")

    opt = torch.optim.AdamW(specialist_param_groups(model), lr=lr, betas=(0.9, 0.95))
    warmup = min(100, max(steps // 10, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1),
                                                    eta_min=lr * 0.1)],
        milestones=[warmup])

    amp = make_amp(device)
    model.train()
    started, best = time.time(), 0.0
    for step in range(1, steps + 1):
        pick = np.concatenate([rng.choice(idx, size=per_class) for idx in by_class])
        x, last, y = _batch(train_ids, train_y, pick, device)
        with amp.autocast():
            loss = F.cross_entropy(_logits(model, task, x, last), y)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % eval_every == 0 or step == steps:
            macro = _macro(_predict(model, task, val_ids, val_y, device), val_y)
            probe = _macro(_predict(model, task, probe_ids, probe_y, device), probe_y)
            mark = ""
            if macro > best:
                best, mark = macro, " <- saved"
                save_specialist(out, model, tok, name=NAME, task=task, labels=LABELS,
                                special_tokens=[ROUTE, DEST], steps=step, val_acc=macro,
                                meta={"min_conf": MIN_CONF, "probe_acc": probe,
                                      "caps": CAPS, "holdout": HOLDOUT})
            elapsed = time.time() - started
            log(f"step {step:>5}/{steps} | loss {loss.item():.3f} | val {macro:.1%} "
                f"| probe {probe:.1%} | {elapsed / 60:.0f}m, "
                f"eta {elapsed / step * (steps - step) / 60:.0f}m{mark}")

    lm = ExpertLM(base, device, specialists=[out])
    task = lm.specialists[NAME]["task"]
    _report(_predict(lm.model, task, val_ids, val_y, device), val_y, log)
    _report(_predict(lm.model, task, probe_ids, probe_y, device), probe_y, log,
            title="hand-written probe")
    log(f"done — best held-out macro accuracy {best:.1%}, saved to {out}")
    return out


# ------------------------------------------------------------------ inference


@torch.no_grad()
def scores(lm: ExpertLM, message: str) -> dict[str, float]:
    """The full destination distribution for one message."""
    logits = lm.model.specialist_logits(lm._ids(route_doc(message)), NAME)[0]
    probs = F.softmax(logits.float(), dim=-1).tolist()
    return dict(zip(lm.specialists[NAME]["labels"], probs))


def route(lm: ExpertLM, message: str) -> tuple[str, float]:
    """The most likely destination for one message, as (label, confidence)."""
    return lm.classify(NAME, route_doc(message))


def pick(lm: ExpertLM, message: str, min_conf: float = MIN_CONF) -> tuple[str, float]:
    """The destination the agent should actually use: `route`, but falling back
    to chat below `min_conf`. The confidence is returned either way, so the
    caller can tell "chat, certainly" from "chat, for want of anything better"."""
    label, conf = route(lm, message)
    return (label if conf >= min_conf else "chat"), conf


# -------------------------------------------------------------------- the probe
#
# Hand-written messages, none of them from a template above, each labelled with
# the destination that should answer it. This is the honest measure of the
# router: the synthesized classes score far higher on their own held-out
# skeletons than they do here.

PROBE: list[tuple[str, str]] = [
    ("hey, how's your day going", "chat"),
    ("i just got back from the gym and i'm wrecked", "chat"),
    ("do you like pineapple on pizza", "chat"),
    ("that's hilarious lol", "chat"),
    ("i'm so tired today", "chat"),
    ("what's your name again", "chat"),
    ("tell me something interesting", "chat"),
    ("good morning!", "chat"),
    ("ok cool, talk to you later", "chat"),
    ("my sister is getting married in june", "chat"),
    ("do you ever get bored", "chat"),
    ("nah i don't think so", "chat"),
    ("what kind of music do you like", "chat"),
    ("it's been raining all week here", "chat"),

    ("if a train leaves at 3pm doing 60 mph, how far has it gone by 5:30", "reason"),
    ("i bought 3 shirts at $12 each and paid with a fifty, what's my change", "reason"),
    ("there are 24 kids and 4 buses, how many kids per bus", "reason"),
    ("work out how many minutes there are in a fortnight", "reason"),
    ("a pizza is cut into 8 slices and i ate 3, what fraction is left", "reason"),
    ("solve this: 4x + 7 = 31", "reason"),
    ("if it takes 5 machines 5 minutes to make 5 widgets, how long for 100", "reason"),
    ("figure out the average of 12, 19 and 44", "reason"),

    ("write me a python function to reverse a linked list", "codegen"),
    ("can you code up a debounce helper in js", "codegen"),
    ("i need a script that renames files by their date", "codegen"),
    ("show me a go function that reads a csv", "codegen"),
    ("quick python snippet to flatten a nested list", "codegen"),
    ("build me a class that wraps an http client with retries", "codegen"),
    ("write something in ruby to strip html tags", "codegen"),
    ("how would you implement binary search in java", "codegen"),

    ("what language was that file", "code"),
    ("so what is it written in", "code"),
    ("you sure that's ruby", "code"),
    ("which language did you say it was", "code"),
    ("was the thing i just sent you go or rust", "code"),
    ("remind me what you decided about my script", "code"),

    ("what was in that picture again", "vision"),
    ("so what did you see", "vision"),
    ("how confident were you about that photo", "vision"),
    ("was it a cat or a dog in the pic", "vision"),
    ("did you get the digit right", "vision"),
    ("tell me again what the image showed", "vision"),

    ("what's the score at", "game"),
    ("how long is my snake now", "game"),
    ("which way should i go for the food", "game"),
    ("are we done yet", "game"),
    ("am i winning", "game"),
    ("is the round finished", "game"),
]


# ---------------------------------------------------------------------- CLI


def evaluate(path=DEFAULT_PATH, base=EXPERT_PATH, device=None, seed=0,
             min_conf: float = MIN_CONF, log=print) -> float:
    """Held-out accuracy (unseen phrasing skeletons) and probe accuracy (messages
    from no template at all). Returns the probe macro accuracy — the honest one."""
    device = device or pick_device()
    lm = ExpertLM(base, device, specialists=[path])
    task = lm.specialists[NAME]["task"]
    docs, y = _corpus(seed, sides=("val",), log=log)["val"]
    ids = _tokenize(lm.tok, docs)
    _report(_predict(lm.model, task, ids, y, device), y, log, min_conf)
    probe_ids = _tokenize(lm.tok, [route_doc(t) for t, _ in PROBE])
    probe_y = np.asarray([LABEL_ID[label] for _, label in PROBE], dtype=np.int64)
    return _report(_predict(lm.model, task, probe_ids, probe_y, device), probe_y,
                   log, min_conf, title="hand-written probe")


def demo(path=DEFAULT_PATH, base=EXPERT_PATH, device="cpu") -> None:
    lm = ExpertLM(base, device, specialists=[path])
    for message, gold in PROBE:
        label, conf = pick(lm, message)
        verdict = "ok  " if label == gold else f"WRONG (want {gold})"
        print(f"{verdict} {label:8} {conf:5.1%}  {message}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Routing specialist for the expert model: which capability "
                    "should answer a message.")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=3000)
    tr.add_argument("--batch-size", type=int, default=36)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--device", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--device", default=None)
    ev.add_argument("--min-conf", type=float, default=MIN_CONF)
    de = sub.add_parser("demo")
    de.add_argument("--device", default="cpu")
    ask = sub.add_parser("ask", help="route one message and show the distribution")
    ask.add_argument("message", nargs="+")
    ask.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, lr=a.lr, device=a.device)
    elif a.cmd == "eval":
        evaluate(device=a.device, min_conf=a.min_conf)
    elif a.cmd == "demo":
        demo(device=a.device)
    elif a.cmd == "ask":
        lm = ExpertLM(EXPERT_PATH, a.device, specialists=[DEFAULT_PATH])
        message = " ".join(a.message)
        label, conf = pick(lm, message)
        print(f"{message!r} -> {label} ({conf:.1%})")
        for name, prob in sorted(scores(lm, message).items(), key=lambda kv: -kv[1]):
            print(f"  {name:8} {prob:6.1%}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
