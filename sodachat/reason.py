"""Thinking and reasoning, added to the expert model as a fourth specialist.

The other specialists answer in one shot: vision reads a label off a head, code
names a language, codegen continues a snippet. This one is trained to **show its
work** — given a question it writes a chain of intermediate steps and only then
commits to an answer:

    <|reason|> Natalia sold clips to 48 friends in April, and half as many in
    May. How many clips did she sell altogether?
    <|think|> Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether.
    <|answer|> 72 <|end|>

Like `codegen` it is a **generator**: no class head, seeded from the TEXT expert
(which already decodes fluent text through the tied LM head), decoding through
the shared frozen LM head. Three things make it different from every specialist
before it:

  * **Prompt-masked loss.** The question is context, not a prediction target, so
    loss is taken only on the `<|think|>`/`<|answer|>` span. The corpus therefore
    ships as *two* parallel binaries — token ids and a supervision mask — the same
    trick `expert.py` uses to carry per-token task ids and action targets
    alongside the token stream. Training on the question too (what a plain packed
    LM stream does) spends most of the gradient learning to write questions.
  * **A separate answer tag.** `<|answer|>` is a token, not a phrase, so the final
    answer is recovered by *splitting on token ids* rather than by parsing prose —
    reliable even when the reasoning wanders, which at this scale it does.
  * **A much larger corpus.** Every earlier specialist trained on one dataset of a
    few thousand examples (MNIST, CIFAR-10, ~24k CodeSearchNet snippets). Reasoning
    does not survive that: step-by-step derivation has to be seen in bulk and in
    many phrasings. So the corpus is streamed and interleaved from **eight public
    datasets** (nine splits — `SOURCES` below) into a ~400M-token stream of ~1.6M
    reasoning traces: three orders of magnitude more examples than any specialist
    before it. `nvidia/OpenMathInstruct-2` supplies the bulk (22M rows available,
    of which the token target takes a slice); the rest are there for phrasing
    diversity, since one generator's habits are easy to overfit.

    The size is chosen to match the *training schedule*, not to be as big as
    possible — see `TARGET_TOKENS`. Every source is permissively licensed (MIT /
    Apache-2.0 / CC-BY-4.0). Left out on purpose, all of it measured rather than
    assumed:
      - non-commercial sets (camel-ai/math, facebook/natural_reasoning);
      - `nvidia/OpenMathInstruct-1` (6.9M rows) and other coding subsets — its
        solutions are `<llm-code>` Python blocks, which is `codegen`'s job and
        would teach this expert the wrong output shape;
      - `nvidia/OpenMathReasoning`, whose R1 traces average ~20k characters
        against a ~1.2k-character window;
      - `AI-MO/NuminaMath-1.5` (only 22% survives the filters — olympiad proofs
        whose "answer" field is the word "proof");
      - `microsoft/orca-agentinstruct-1M-v1`, the best hope for non-math
        diversity, which keeps 0% of `analytical_reasoning` and 2-12% elsewhere,
        and whose survivors are quantitative anyway;
      - `kaist-ai/CoT-Collection` (1.8M non-math rationales) — script-only, so
        unloadable under `datasets` 5.x, with no parquet mirror on the Hub.

Datasets are read with `streaming=True` and tokenized straight into the cached
`.bin` stream, so a corpus far larger than the training box's free disk never
lands on it whole.

Be honest about the ceiling: the trunk is a ~14M-parameter model frozen on a
chat/game diet with a dialogue BPE vocabulary, so this learns the *shape* of
step-by-step reasoning — the format, the moves, the arithmetic patois — far
better than it learns to be right. Expect fluent-looking derivations with wrong
totals, and read `eval`'s GSM8K exact-match number as the honest score.

**This was measured, and the corpus size is not the bottleneck.** Quadrupling the
corpus (100M -> 400M tokens, 412k -> 1.42M traces) and the schedule (5.5k -> 20k
steps) moved held-out perplexity 5.0 -> 3.5 and visibly improved how the chains are
structured — it now picks the right operations and often computes them correctly.
GSM8K test exact-match did **not** follow: 2.0% -> 1.5%, which on the same 200
problems is a one-problem difference, i.e. noise in both directions. What is left is
arithmetic slips and the occasional spurious extra operation after `<|answer|>`.
More tokens buy fluency of reasoning here, not correctness; real accuracy needs a
bigger or unfrozen trunk, or a calculator the model can call.

Be honest about the scope too: the mix is ~99.9% mathematical. StrategyQA is the
only non-arithmetic source that is both permissively licensed and loadable, and it
has all of 1,603 usable rows. This is a *quantitative* reasoner; general-knowledge
questions get confident nonsense. Fixing that needs a large non-math CoT corpus
that does not currently exist in loadable, permissive form (see the exclusions
above) — not a reweighting of this one.

Saved as `specialist-reason.pt`; `ExpertLM` grafts it on at startup alongside
vision, code, and codegen.

    python -m sodachat.reason data      # build/inspect the corpus cache only
    python -m sodachat.reason train     # needs models/expert.pt (train it first)
    python -m sodachat.reason eval      # held-out perplexity + GSM8K exact match
    python -m sodachat.reason think --question "..."
    python -m sodachat.reason demo      # a few questions + chains of thought
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import make_amp, pick_device
from .expert import (
    _MODELS,
    DEFAULT_PATH as EXPERT_PATH,
    END,
    ExpertLM,
    TEXT,
    _interleave,
    save_specialist,
    scaffold_specialist,
    specialist_param_groups,
)

NAME = "reason"
# The question is prefixed, the chain of thought follows <|think|>, and the
# committed answer follows <|answer|>. Three tokens, so each boundary is a single
# id the trainer can mask on and inference can split on.
REASON, THINK, ANSWER = "<|reason|>", "<|think|>", "<|answer|>"
DEFAULT_PATH = _MODELS / "specialist-reason.pt"
BLOCK = 512            # a reasoning trace is a few hundred tokens; 512 spans one
HOLDOUT = 40           # 1-in-40 examples (by question hash) are held out for val
# How many tokens to pack into the train stream. Matched to the training schedule
# rather than to what the sources could supply (they could supply billions): the
# default run sees 20k x 24 x 512 = 246M tokens, so a 400M-token corpus means each
# example is seen ~0.6 times and the model never gets to memorize one.
TARGET_TOKENS = 400_000_000
MAX_DOC_CHARS = 1200   # cap a trace so it fits a window with room for the question
MIN_THINK_CHARS = 40   # below this there is no reasoning to learn from, only a
                       # bare equation and an option letter (AQuA-RAT has these)
MAX_Q_CHARS = 600
MAX_ANSWER_CHARS = 120  # a "final answer" longer than this is really more steps


# =====================================================================
# Corpus: seven public reasoning datasets (eight splits), streamed and
# interleaved. Each source renders a row into (question, chain-of-thought,
# final answer); returning None drops the row.
# =====================================================================


# The trailing "so the answer is X" marker these datasets share, in its many
# spellings. Used to cut a solution into (steps, final answer) — we split on the
# *last* match, since "answer:" turns up mid-derivation too. The match covers
# only the marker itself, so cutting on it leaves the steps' own punctuation
# intact.
_ANS_MARK = re.compile(
    r"(?:the\s+(?:final\s+)?answer\s+is\s*:?|final\s+answer\s*:?|answer\s+is\s*:?"
    r"|answer\s*:|####)", re.IGNORECASE)
_CALC = re.compile(r"<<[^>]*>>")          # GSM8K's calculator annotations
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BLANKS = re.compile(r"\n{3,}")           # NuminaMath spaces its steps out a lot
# Byte-level BPE lets the model sample byte sequences that decode to raw control
# characters (a stray \x1a in the middle of a word). The corpus has none of these
# — it is the model inventing invalid bytes — so they are stripped on the way out
# rather than being printed into someone's terminal. Tab and newline stay.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A value a prose conclusion lands on: a number (with $ , . % signs) or a box.
# The decimal part requires digits after the point, so the sentence's own final
# period stays outside the captured value ("... is 710." -> "710", not "710.").
_TRAILING_VALUE = re.compile(r"([-+]?\$?\d[\d,]*(?:\.\d+)?\s*%?|\\boxed\{[^{}]*\})"
                             r"\s*[.!]?$")


def _strip_trailing_markers(steps: str) -> str:
    """Drop any closing answer clauses left dangling at the end of the steps.

    A solution can carry more than one: MetaMathQA's GSM8K-derived rows end with
    *both* "#### 752" and "The answer is: 752", so cutting once at the last marker
    leaves the other in the reasoning — 62% of that source, ~9% of the whole
    corpus, and the model duly learns to emit a stray "#### 752" mid-thought. So
    keep cutting while the tail is a short single-line closing clause; a marker
    with more reasoning after it (a mid-derivation "Answer: 9\\nNow verify…") is
    left alone."""
    while True:
        last = None
        for last in _ANS_MARK.finditer(steps):
            pass
        if last is None:
            return steps.strip()
        tail = steps[last.end():].strip()
        if len(tail) > MAX_ANSWER_CHARS or "\n" in tail:
            return steps.strip()
        steps = steps[: last.start()]


def _tighten_answer(answer: str) -> str:
    """Reduce a prose conclusion to the value it lands on ("Therefore, the largest
    number Haneul can make is 710." -> "710").

    Sources disagree about the answer slot: GSM8K gives a bare number, while 43%
    of orca-math's rows end on a full sentence. For a model this small a
    *consistent* short answer is far more learnable than a mixture, and it keeps
    `<|answer|>` meaning the same thing everywhere."""
    if len(answer.split()) <= 4:  # already terse ("72", "A", "yes")
        return answer
    m = _TRAILING_VALUE.search(answer)
    return m.group(1).strip() if m else answer


def _last_boxed(text: str) -> tuple[int, str] | None:
    """Locate the last `\\boxed{...}` — the canonical final answer in every
    MATH-derived set — and return (where it starts, what is inside). Brace
    matching is explicit because the contents nest routinely
    (`\\boxed{\\frac{1}{2}}`), which a regex would get wrong."""
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None
    i, depth = start + len(r"\boxed{"), 1
    while i < len(text) and depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    if depth:  # unbalanced — not something to trust
        return None
    return start, text[start + len(r"\boxed{"):i - 1]


def _split_answer(text: str) -> tuple[str, str]:
    """Cut a worked solution into (steps, final answer).

    Three strategies, strongest first: a `\\boxed{}` result, an explicit "the
    answer is X" marker, or the last short line — so a solution that simply ends
    on its result still yields an answer. ("", "") if there is nothing usable,
    which drops the example."""
    text = (text or "").strip()
    if not text:
        return "", ""
    boxed = _last_boxed(text)
    if boxed is not None and len(boxed[1]) <= MAX_ANSWER_CHARS:
        # Drop the whole line holding the box: cutting at the box itself would
        # leave a dangling "Thus, the answer is $" on the end of the steps.
        head = text[: boxed[0]]
        cut = head.rfind("\n")
        return _strip_trailing_markers(head[:cut] if cut != -1 else head), boxed[1]
    last = None
    for last in _ANS_MARK.finditer(text):
        pass
    if last is not None:
        steps, answer = text[: last.start()], text[last.end():].strip()
        # A long tail after the marker is continued reasoning, not an answer.
        if answer and len(answer) <= MAX_ANSWER_CHARS:
            return _strip_trailing_markers(steps), answer
        return text.strip(), ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 2 and len(lines[-1]) <= MAX_ANSWER_CHARS:
        return _strip_trailing_markers("\n".join(lines[:-1])), lines[-1]
    return text.strip(), ""


def _gsm8k(row: dict):
    """GSM8K: grade-school word problems whose solutions are already one step
    per line, ending in '#### <number>'. The cleanest CoT format there is."""
    solution = row.get("answer") or ""
    if "####" not in solution:
        return None
    steps, _, answer = solution.partition("####")
    return row.get("question"), _CALC.sub("", steps), answer


def _qa(question_key: str, solution_key: str):
    """Renderer for the common shape: one question field, one worked-solution
    field, the final answer left to `_split_answer`."""

    def render(row: dict):
        steps, answer = _split_answer(row.get(solution_key) or "")
        return row.get(question_key), steps, answer

    return render


def _metamath(row: dict):
    """MetaMathQA augments each source problem into many rephrasings (one problem
    can appear dozens of times). The fourth element is the *split identity*: keyed
    on the original question, every variant of a problem lands on the same side of
    the train/val split instead of leaking across it."""
    steps, answer = _split_answer(row.get("response") or "")
    query = row.get("query")
    return query, steps, answer, row.get("original_question") or query


def _mathinstruct(row: dict):
    # Program-of-thought rows answer by emitting Python; that is codegen's job,
    # and the code would teach this expert the wrong output shape.
    if "/CoT/" not in (row.get("source") or ""):
        return None
    steps, answer = _split_answer(row.get("output") or "")
    return row.get("instruction"), steps, answer


def _aqua_rat(row: dict):
    """AQuA-RAT: multiple-choice quantitative questions with a written rationale.
    The choices go in the question (the rationale refers to them) and the answer
    is the gold letter, so the target is exact rather than parsed from prose."""
    options = row.get("options") or []
    question = (row.get("question") or "").strip()
    if options:
        question = f"{question}\nOptions: " + " ".join(str(o) for o in options)
    rationale = (row.get("rationale") or "").strip()
    # Drop the rationale's own "The answer is E." tail so it isn't stated twice,
    # but only when there is really such a tail: `_split_answer`'s last-line
    # fallback would otherwise cut a legitimate final step off a rationale that
    # simply ends on its result, and here we keep only the steps.
    if _ANS_MARK.search(rationale) or _last_boxed(rationale) is not None:
        rationale = _split_answer(rationale)[0]
    return question, rationale, row.get("correct")


def _openmathinstruct(row: dict):
    """OpenMathInstruct-2: 22M synthesized GSM8K/MATH problems solved in plain
    step-by-step prose, with the final answer in its own `expected_answer` field —
    so the answer is read directly instead of being parsed back out of the prose,
    which is the most reliable shape any source here offers. Measured on 4k rows:
    69% survive the shared filters, steps run ~540 characters at the median, and
    only 0.1% carry a code block (dropped below — writing code is `codegen`'s
    job). This is the bulk of the corpus."""
    solution = row.get("generated_solution") or ""
    if "<llm-code>" in solution or "```" in solution:
        return None
    return row.get("problem"), solution, row.get("expected_answer")


def _strategyqa(row: dict):
    """StrategyQA: implicit multi-hop yes/no questions. The supporting facts are
    the reasoning; they keep the corpus from being all arithmetic."""
    answer = row.get("answer")
    if answer is None:
        return None
    return row.get("question"), row.get("facts"), "yes" if answer else "no"


@dataclass
class Source:
    """One public dataset in the mix. `cap` bounds how many examples are kept,
    `scan` how many rows are read off the stream (so a huge set with a low keep
    rate can't stream forever), and `weight` how many of its docs are pulled per
    interleave round — the mixing ratio."""

    repo: str
    split: str
    render: Callable[[dict], tuple | None]
    license: str
    cap: int
    scan: int
    weight: int = 1
    config: str | None = None
    kept: int = field(default=0, compare=False)
    seen: int = field(default=0, compare=False)

    @property
    def tag(self) -> str:
        return self.repo + (f"[{self.config}]" if self.config else "")

    def signature(self) -> tuple:
        return (self.repo, self.config, self.split, self.cap, self.scan,
                self.weight, _render_sig(self.render))


def _render_sig(fn) -> str:
    """A fingerprint of one renderer: its code, plus any values it closes over so
    that repointing `_qa` at different fields counts as a change too.

    This is in the corpus cache key on purpose. Without it, editing how rows are
    rendered would silently reuse a stream built by the *old* code — the trained
    checkpoint would then be unreproducible from the shipped source, which is a
    nasty thing to debug months later."""
    import inspect

    parts = [inspect.getsource(fn)]
    parts += [repr(cell.cell_contents) for cell in (fn.__closure__ or ())]
    return hashlib.md5("".join(parts).encode()).hexdigest()[:8]


def _shared_sig(tok) -> str:
    """The same idea for everything else that decides what lands in the stream:
    the helpers every renderer runs through, the module-level patterns and tag
    strings those helpers close over by *name* (so `inspect.getsource` can't see
    a change to their values), and the tokenizer itself.

    The tokenizer matters most: the `.bin` files hold raw token ids, so a base
    model retrained with a different vocabulary makes a cached stream mean
    something else entirely — which would train quietly on garbage, or crash deep
    in the embedding if the new vocabulary is smaller."""
    import inspect

    parts = [inspect.getsource(f) for f in
             (_split_answer, _last_boxed, _clean, reason_doc, _holdout_key,
              _is_holdout, _strip_trailing_markers, _tighten_answer)]
    parts += [_ANS_MARK.pattern, _CALC.pattern, _CHOICES.pattern, _BLANKS.pattern,
              _TRAILING_VALUE.pattern, REASON, THINK, ANSWER, str(len(tok))]
    payload = tok.to_payload()
    parts.append(payload.get("json") or str(payload))
    return hashlib.md5("".join(parts).encode()).hexdigest()[:8]


SOURCES = [
    # The bulk of the corpus: 22M rows, the cleanest shape of any source here, and
    # deep enough that the token target binds long before its cap does.
    Source("nvidia/OpenMathInstruct-2", "train", _openmathinstruct, "CC-BY-4.0",
           900_000, 1_500_000, 8),
    # Small, gold-standard, and the benchmark we score on (its *test* split is
    # never trained on — see `gsm8k_accuracy`). The `socratic` config is the same
    # problems solved as a chain of self-asked sub-questions, which is a genuinely
    # different reasoning *style* for the price of one more small download.
    Source("openai/gsm8k", "train", _gsm8k, "MIT", 8_000, 10_000, 1, config="main"),
    Source("openai/gsm8k", "train", _gsm8k, "MIT", 8_000, 10_000, 1, config="socratic"),
    # The other bulk suppliers of short step-by-step math, each with its own
    # phrasing habits — caps raised to take essentially all of what they offer.
    Source("meta-math/MetaMathQA", "train", _metamath, "MIT", 395_000, 400_000, 3),
    Source("TIGER-Lab/MathInstruct", "train", _mathinstruct, "MIT", 262_000, 300_000, 2),
    Source("microsoft/orca-math-word-problems-200k", "train",
           _qa("question", "answer"), "MIT", 200_000, 200_000, 3),
    Source("AI-MO/NuminaMath-CoT", "train", _qa("problem", "solution"),
           "Apache-2.0", 300_000, 500_000, 3),
    # Quantitative multiple-choice with written rationales.
    Source("deepmind/aqua_rat", "train", _aqua_rat, "Apache-2.0", 97_000, 100_000, 2,
           config="raw"),
    # Non-arithmetic reasoning: implicit multi-hop yes/no. Tiny, and it is the only
    # one of its kind — see the scope note in the module docstring.
    Source("ChilleD/StrategyQA", "train", _strategyqa, "MIT", 3_000, 5_000, 1),
]


def reason_doc(question: str, think: str, answer: str) -> tuple[str, str]:
    """Render one example as (prompt, target). The split is where the loss mask
    falls: the prompt is context the model reads, the target is what it must
    learn to produce. `<|answer|>` is always present, so the inference format
    matches training exactly."""
    prompt = f"{REASON} {question.strip()}\n{THINK}"
    return prompt, f" {think.strip()}\n{ANSWER} {answer.strip()}"


# The multiple-choice list appended to a question, which AQuA-RAT and MathInstruct
# spell differently ("Options: A)21" vs "Answer Choices: (A) 21") for what is
# frequently the *same* problem — MathInstruct's CoT mix contains ~89k AQuA-RAT
# rows. Cutting it off before hashing makes both spellings one identity.
_CHOICES = re.compile(r"\b(?:answer\s+choices|options)\b\s*:?", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _holdout_key(text: str) -> str:
    """Reduce a question to a problem *identity* for splitting: drop the
    multiple-choice list, then strip case and punctuation. Two sources that word
    the same problem differently then collapse to one key."""
    return _NON_ALNUM.sub("", _CHOICES.split(text or "")[0].lower())


def _is_holdout(identity: str) -> bool:
    """Deterministic 1-in-HOLDOUT split, keyed by a stable hash of the problem's
    identity (see `_holdout_key`). Python's `hash` is per-process salted, so md5
    it is; this keeps train and val disjoint without needing per-source test
    splits, and — because the key is an identity rather than the rendered
    question — keeps paraphrases and cross-source duplicates of one problem on
    the same side, so held-out perplexity isn't quietly scoring memorization."""
    digest = hashlib.md5(_holdout_key(identity).encode("utf-8", "ignore")).digest()
    return digest[0] % HOLDOUT == 0


def _clean(rendered) -> tuple[str, str, str, str] | None:
    """Apply the filters every source shares: a real question, real reasoning, a
    short final answer, and a doc that fits a window. Returns
    (question, think, answer, split identity) — a renderer may supply the identity
    as a fourth element, else the question stands in for it."""
    if not rendered:
        return None
    question, think, answer = rendered[:3]
    identity = rendered[3] if len(rendered) > 3 else None
    question = " ".join((question or "").split())
    think = "\n".join(ln.rstrip() for ln in (think or "").strip().split("\n")).strip()
    think = _BLANKS.sub("\n\n", _strip_trailing_markers(think))
    answer = _tighten_answer(" ".join((answer or "").split()))
    if not question or len(question) > MAX_Q_CHARS:
        return None
    if len(think) < MIN_THINK_CHARS or not answer or len(answer) > MAX_ANSWER_CHARS:
        return None
    if len(question) + len(think) + len(answer) > MAX_DOC_CHARS:
        return None
    return question, think, answer, identity or question


def _stream_source(src: Source, split: str, log=print):
    """Cleaned (prompt, target) pairs from one streamed dataset, restricted to
    the requested side of the hash split. Network or schema trouble on one source
    is logged and skipped — the remaining sources still train (and `prepare`
    refuses to cache a corpus if *every* source failed)."""
    try:
        from datasets import load_dataset

        ds = load_dataset(src.repo, src.config, split=src.split, streaming=True)
    except Exception as e:  # offline, gated, renamed, script-only, ...
        log(f"  ! skipping {src.tag}: {type(e).__name__}: {str(e)[:120]}")
        return
    want_val = split == "val"
    # Val needs 1/HOLDOUT of the rows, so it must scan proportionally further to
    # fill its (much smaller) cap.
    cap = max(src.cap // HOLDOUT, 50) if want_val else src.cap
    for row in ds:
        src.seen += 1
        if src.seen > src.scan or src.kept >= cap:
            break
        try:
            example = _clean(src.render(row))
        except Exception:  # a single malformed row is not worth failing over
            continue
        if example is None:
            continue
        if _is_holdout(example[3]) is not want_val:
            continue
        src.kept += 1
        yield reason_doc(*example[:3])


def _corpus(split: str, log=print):
    """All sources, round-robin by weight, so any window of the packed stream
    holds a mixture rather than a run of one dataset."""
    for src in SOURCES:
        src.kept = src.seen = 0
    streams = [_stream_source(s, split, log) for s in SOURCES]
    yield from _interleave(streams, [s.weight for s in SOURCES])


# ------------------------------------------------------- corpus -> token stream


def _batched(iterable, n: int):
    out = []
    for item in iterable:
        out.append(item)
        if len(out) == n:
            yield out
            out = []
    if out:
        yield out


def _write_stream(tok, docs, tok_path: Path, mask_path: Path, target_tokens: int,
                  log=print) -> tuple[int, int]:
    """Pack (prompt, target) pairs into one token stream plus a parallel
    supervision mask.

    mask[i] == 1 means "the token at i is a target the loss is taken on", which
    is exactly the alignment the trainer needs: a window predicts stream[i+1:]
    from stream[i:], so it reads the mask off the same shifted slice. The prompt
    is zeros, the chain of thought and answer are ones, and the closing `<|end|>`
    is a one — the model has to learn where a trace *stops*, or it runs on into
    the next question."""
    end = tok.token_id(END)
    docs_written = total = 0
    started = time.time()
    with open(tok_path, "wb") as ft, open(mask_path, "wb") as fm:
        for chunk in _batched(docs, 1000):
            # One encode_batch over prompts and targets interleaved: the fast
            # tokenizer batches better than 2000 single calls.
            flat = [side for pair in chunk for side in pair]
            enc = tok.encode_batch(flat)
            for i in range(0, len(enc), 2):
                prompt, target = enc[i], enc[i + 1]
                ids = [*prompt, *target, end]
                mask = np.zeros(len(ids), dtype=np.uint8)
                mask[len(prompt):] = 1
                np.asarray(ids, dtype=np.uint16).tofile(ft)
                mask.tofile(fm)
                docs_written += 1
                total += len(ids)
            if docs_written % 50_000 < 1000:
                log(f"  {docs_written:,} traces, {total / 1e6:.1f}M tokens "
                    f"({time.time() - started:.0f}s)")
            if total >= target_tokens:
                log(f"  reached the {target_tokens / 1e6:.0f}M-token target")
                break
    return docs_written, total


def prepare(cache: Path, tok, target_tokens: int, log=print, rebuild: bool = True):
    """Build (or reuse) the cached token stream + supervision mask for both
    splits. The cache is keyed by the corpus configuration, so changing a source,
    a cap, or the token target rebuilds it instead of silently reusing a stream
    that no longer matches.

    `rebuild=False` means "use the cache or nothing" — it returns None instead of
    re-streaming the whole corpus, so a stale key during `eval` costs a skipped
    metric rather than an hour of downloads."""
    cache.mkdir(parents=True, exist_ok=True)
    sig = hashlib.md5(
        json.dumps([[*s.signature()] for s in SOURCES]
                   + [target_tokens, HOLDOUT, MAX_DOC_CHARS, MIN_THINK_CHARS,
                      MAX_Q_CHARS, MAX_ANSWER_CHARS, _shared_sig(tok)]).encode()
    ).hexdigest()[:12]
    kinds = ("tok", "mask")
    # The signature is in the *filename*, not just inside the metadata, so two
    # corpus configurations coexist instead of overwriting each other. That matters
    # more than it sounds: a training run holds these files mmap'd for hours, and
    # building a differently-configured corpus under the old fixed names would
    # rewrite them underneath it.
    bins = {s: {k: cache / f"reason-{sig}-{s}-{k}.bin" for k in kinds}
            for s in ("train", "val")}
    meta_path = cache / f"reason-{sig}-meta.json"
    fresh = (meta_path.exists()
             and all(p.exists() for s in bins.values() for p in s.values()))
    if fresh:
        meta = json.loads(meta_path.read_text())
        log(f"reusing tokenized cache ({meta['train']['tokens']:,} train tokens "
            f"from {meta['train']['docs']:,} traces)")
    elif not rebuild:
        log(f"no corpus cache matching this configuration in {cache}")
        return None
    else:
        meta = {"sig": sig}
        for split, budget in (("train", target_tokens),
                              ("val", max(target_tokens // 50, 200_000))):
            log(f"streaming the {split} corpus from "
                f"{len(SOURCES)} public datasets...")
            docs, total = _write_stream(tok, _corpus(split, log),
                                        bins[split]["tok"], bins[split]["mask"],
                                        budget, log)
            meta[split] = {"docs": docs, "tokens": total,
                           "sources": {s.tag: s.kept for s in SOURCES}}
            log(f"{split}: {docs:,} traces, {total / 1e6:.2f}M tokens — " +
                ", ".join(f"{s.tag.split('/')[-1]} {s.kept:,}" for s in SOURCES))
            # Loudly, because a source that contributes nothing (offline, gated,
            # renamed field) is otherwise invisible against seven that worked.
            for src in SOURCES:
                if not src.kept:
                    log(f"  ! {src.tag} contributed nothing to {split}")
            if not docs:
                raise RuntimeError(
                    f"the {split} corpus came out empty — every source failed to "
                    f"stream (no network? gated repo?). Nothing was cached; fix "
                    f"the cause and re-run.")
        # Only stamp the signature once both splits are real. Writing it
        # unconditionally would let one offline run cache an empty corpus as
        # valid, and every later run would "reuse" it and die in _batch.
        meta_path.write_text(json.dumps(meta, indent=1))

    def load(split):
        return (np.memmap(bins[split]["tok"], dtype=np.uint16, mode="r"),
                np.memmap(bins[split]["mask"], dtype=np.uint8, mode="r"))

    return load("train"), load("val"), meta


# ------------------------------------------------------------------- training


def _batch(data, block: int, bs: int, task: int, device):
    """Sample `bs` random windows, their next-token targets, and the mask saying
    which of those targets the loss counts (see `_write_stream`). Everything is
    routed to the reasoning expert."""
    stream, mask = data
    ix = np.random.randint(0, len(stream) - block - 1, size=bs)
    x = np.stack([stream[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([stream[i + 1:i + block + 1] for i in ix]).astype(np.int64)
    m = np.stack([mask[i + 1:i + block + 1] for i in ix]).astype(bool)
    xs, ys, ms = (torch.from_numpy(a) for a in (x, y, m))
    t = torch.full_like(xs, task)
    if device == "cuda":
        return (xs.pin_memory().to(device, non_blocking=True),
                ys.pin_memory().to(device, non_blocking=True),
                ms.pin_memory().to(device, non_blocking=True), t.to(device))
    return xs.to(device), ys.to(device), ms.to(device), t.to(device)


def _loss(model, x, y, m, t):
    """Next-token loss over the supervised positions only — the chain of thought
    and the answer, never the question."""
    logits = model(x, t)[0]
    flat = m.reshape(-1)
    if not bool(flat.any()):  # a window of pure question text; nothing to learn
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.size(-1))[flat], y.reshape(-1)[flat])


@torch.no_grad()
def _validate(model, data, block, task, device, iters=40, bs=16) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        losses.append(_loss(model, *_batch(data, block, bs, task, device)).item())
    if was_training:
        model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


def train(base=EXPERT_PATH, out=DEFAULT_PATH, steps=20_000, batch_size=24, lr=3e-4,
          block_size=BLOCK, target_tokens=TARGET_TOKENS, device=None, seed=0,
          eval_every=250, log=print) -> Path:
    """Train the reasoning specialist on top of a *frozen* expert model.

    Gradients reach only the new per-block FFN expert (seeded from TEXT) and the
    three new tags' embedding rows. Chat, reading, play, vision, and both code
    specialists are untouched by construction — their weights never receive a
    gradient. The objective is next-token loss through the shared, frozen LM
    head, masked to the `<|think|>`/`<|answer|>` span so the gradient is spent on
    producing reasoning rather than on modelling questions."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, tok, task = scaffold_specialist(
        base, name=NAME, special_tokens=[REASON, THINK, ANSWER], n_labels=None,
        seed_from=TEXT, device=device)
    assert len(tok) <= np.iinfo(np.uint16).max, "vocab outgrew the uint16 stream"
    block = min(block_size, model.cfg.block_size)
    per_block = sum(p.numel() for p in model.blocks[0].ffn.experts[task].parameters())
    trainable = per_block * model.cfg.n_layer + 3 * model.cfg.n_embd
    log(f"specialist '{NAME}' (generator) on {Path(base).name}: expert slot {task} | "
        f"{trainable / 1e6:.1f}M trainable of {model.num_params() / 1e6:.1f}M "
        f"(shared trunk + LM head frozen) | seeded from TEXT | device {device}")

    train_data, val_data, meta = prepare(_MODELS, tok, target_tokens, log)
    seen = steps * batch_size * block
    log(f"stream: {len(train_data[0]) / 1e6:.2f}M train / "
        f"{len(val_data[0]) / 1e6:.2f}M val tokens (block {block}) | "
        f"supervised {float(np.mean(train_data[1][:2_000_000])):.0%} of positions | "
        f"schedule: {steps:,} steps x {batch_size} x {block} = {seen / 1e6:.0f}M "
        f"tokens (~{seen / max(len(train_data[0]), 1):.2f} epochs)")

    opt = torch.optim.AdamW(specialist_param_groups(model), lr=lr, betas=(0.9, 0.95))
    warmup = min(200, max(steps // 10, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1),
                                                    eta_min=lr * 0.1)],
        milestones=[warmup])

    amp = make_amp(device)
    model.train()
    started, best = time.time(), float("inf")
    for step in range(1, steps + 1):
        x, y, m, t = _batch(train_data, block, batch_size, task, device)
        with amp.autocast():
            loss = _loss(model, x, y, m, t)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % eval_every == 0 or step == steps:
            vloss = _validate(model, val_data, block, task, device)
            mark = ""
            if vloss < best:
                best, mark = vloss, " <- saved"
                save_specialist(out, model, tok, name=NAME, task=task, labels=[],
                                special_tokens=[REASON, THINK, ANSWER],
                                kind="generate", steps=step, val_acc=vloss,
                                meta={"block": block, "val_loss": vloss,
                                      "sources": [s.tag for s in SOURCES],
                                      "train_tokens": meta["train"]["tokens"],
                                      "train_traces": meta["train"]["docs"]})
            el = time.time() - started
            log(f"step {step:>5}/{steps} | train loss {loss.item():.3f} | val loss "
                f"{vloss:.3f} (ppl {np.exp(vloss):.1f}) | {el / 60:.0f}m, "
                f"eta {el / step * (steps - step) / 60:.0f}m{mark}")

    log(f"done — best val loss {best:.3f} (ppl {np.exp(best):.1f}), saved to {out}")
    return out


# ------------------------------------------------------------------ inference


@torch.no_grad()
def think(lm: ExpertLM, question: str, max_new_tokens: int = 220,
          temperature: float = 0.7, top_p: float = 0.95) -> tuple[str, str]:
    """Reason about a question and commit to an answer. Returns
    (chain_of_thought, answer).

    The prompt is the exact training frame up to `<|think|>`, every token routed
    through the reasoning expert and read off the shared LM head. The split is
    done on **token ids**, before decoding: `<|answer|>` is a special token and
    the tokenizer drops special tokens when it decodes, so splitting the decoded
    string would find nothing to split on."""
    task = lm.specialists[NAME]["task"]
    end = lm.tok.token_id(END)
    answer_id = lm.tok.token_id(ANSWER)
    idx = lm._ids(f"{REASON} {' '.join(question.split())}\n{THINK}")
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature,
                                 top_k=40, top_p=top_p, repetition_penalty=1.15,
                                 stop_tokens=[end])
    ids = out[0][idx.shape[1]:].tolist()
    if end in ids:
        ids = ids[:ids.index(end)]
    if answer_id in ids:
        cut = ids.index(answer_id)
        steps, answer = ids[:cut], ids[cut + 1:]
    else:  # ran out of budget mid-derivation — caller reports steps with no answer
        steps, answer = ids, []
    return (_CTRL.sub("", lm.tok.decode(steps)).strip(),
            _CTRL.sub("", lm.tok.decode(answer)).strip())


def solve(lm: ExpertLM, question: str, **kw) -> str:
    """Just the committed answer, for callers that don't want the work shown."""
    return think(lm, question, **kw)[1]


def evaluate(path=DEFAULT_PATH, base=EXPERT_PATH, device=None, iters=100,
             n_gsm8k=200, log=print) -> dict:
    """Two held-out numbers: masked next-token perplexity on the val stream (how
    well it models reasoning) and GSM8K exact-match accuracy (whether it is
    actually right). The second is the honest one."""
    device = device or pick_device()
    lm = ExpertLM(base, device, specialists=[path])
    task = lm.specialists[NAME]["task"]
    block = lm.specialists[NAME].get("block", BLOCK)
    out = {"val_loss": float("nan"), "perplexity": float("nan")}
    cached = prepare(_MODELS, lm.tok, TARGET_TOKENS, log, rebuild=False)
    if cached is not None:
        val_data = cached[1]
        vloss = _validate(lm.model, val_data, block, task, device, iters=iters)
        out = {"val_loss": vloss, "perplexity": float(np.exp(vloss))}
        log(f"{NAME}: val loss {vloss:.3f} | perplexity {np.exp(vloss):.1f} "
            f"over {len(val_data[0]) / 1e3:.0f}k held-out tokens")
    else:
        log("skipping perplexity (no matching corpus cache); scoring GSM8K only")
    out["gsm8k"] = gsm8k_accuracy(lm, n=n_gsm8k, log=log)
    return out


def gsm8k_accuracy(lm: ExpertLM, n: int = 200, log=print) -> float:
    """Exact-match accuracy on GSM8K's **test** split — never streamed into
    training (only `train` is in SOURCES), so this is a clean benchmark. The
    final number in the model's answer is compared to the gold number; sampling
    is near-greedy so the score reflects the model, not the dice."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    except Exception as e:
        log(f"GSM8K test split unavailable ({type(e).__name__}); skipping")
        return float("nan")
    hit = total = 0
    for row in ds:
        if total >= n:
            break
        # Require the '####' marker rather than trusting rpartition, which returns
        # the *whole* string when the separator is missing — that would silently
        # score against the last number of the full solution instead of skipping.
        marker, sep, tail = row["answer"].partition("####")
        if not sep:
            continue
        gold = _NUMBER.findall(tail.replace(",", ""))
        if not gold:
            continue
        _, answer = think(lm, row["question"], max_new_tokens=200, temperature=0.1)
        # The first number of the committed answer span is the answer ("72 clips"
        # -> 72); comparing numerically, so 72.0 counts as 72.
        got = _NUMBER.findall(answer.replace(",", ""))
        total += 1
        hit += bool(got) and abs(float(got[0]) - float(gold[0])) < 1e-6
    log(f"{NAME}: GSM8K test exact match {hit}/{total} = {hit / max(total, 1):.1%} "
        f"(a ~14M frozen trunk; the format is learned far better than the arithmetic)")
    return hit / max(total, 1)


_QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether?",
    "A shirt costs $15 and a pair of jeans costs twice as much. How much do both cost?",
    "If a train travels 60 miles in 2 hours, what is its average speed?",
    "Tom has 3 boxes with 7 pencils each. He gives away 5 pencils. How many are left?",
]


def demo(path=DEFAULT_PATH, base=EXPERT_PATH, device="cpu",
         max_new_tokens=200) -> None:
    lm = ExpertLM(base, device, specialists=[path])
    for question in _QUESTIONS:
        steps, answer = think(lm, question, max_new_tokens=max_new_tokens)
        print(f"Q: {question}\nthinking: {steps}\nanswer: {answer}\n{'-' * 60}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Reasoning specialist for the expert model: think step by "
                    "step, then answer (trained on seven public CoT datasets).")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=20_000)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--block-size", type=int, default=BLOCK)
    tr.add_argument("--target-tokens", type=int, default=TARGET_TOKENS,
                    help="stop packing the train stream at this many tokens")
    tr.add_argument("--device", default=None)
    da = sub.add_parser("data", help="build/inspect the corpus cache, no training")
    da.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    ev = sub.add_parser("eval")
    ev.add_argument("--iters", type=int, default=100)
    ev.add_argument("--n-gsm8k", type=int, default=200)
    ev.add_argument("--device", default=None)
    de = sub.add_parser("demo")
    de.add_argument("--max-new-tokens", type=int, default=200)
    de.add_argument("--device", default="cpu")
    th = sub.add_parser("think", help="reason about one question")
    th.add_argument("--question", required=True)
    th.add_argument("--max-new-tokens", type=int, default=220)
    th.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, lr=a.lr,
              block_size=a.block_size, target_tokens=a.target_tokens,
              device=a.device)
    elif a.cmd == "data":
        from .blocks import tokenizer_from_payload

        # Only the tokenizer is needed to build the corpus — skip constructing
        # (and holding in memory) the whole expert model.
        ck = torch.load(EXPERT_PATH, map_location="cpu", weights_only=True)
        tok = tokenizer_from_payload(ck["tokenizer"])
        tok.add_special([REASON, THINK, ANSWER])
        print(json.dumps(prepare(_MODELS, tok, a.target_tokens)[2], indent=1))
    elif a.cmd == "eval":
        evaluate(iters=a.iters, n_gsm8k=a.n_gsm8k, device=a.device)
    elif a.cmd == "demo":
        demo(max_new_tokens=a.max_new_tokens, device=a.device)
    elif a.cmd == "think":
        lm = ExpertLM(EXPERT_PATH, a.device, specialists=[DEFAULT_PATH])
        steps, answer = think(lm, a.question, max_new_tokens=a.max_new_tokens)
        print(f"thinking: {steps}\nanswer: {answer}")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
