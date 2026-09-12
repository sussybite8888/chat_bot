"""Neural chatbot engine.

Replies come from a language model that continues the chat stream: recent
conversation lines go in, several candidate next lines are sampled out, and
the most relevant candidate is returned.

Backends (ChatEngine(backend=...), CLI --backend, or SODACHAT_BACKEND):
- "mini" (default): a small GPT (~14M params, BPE subword vocabulary)
  trained from scratch on SODA (model.py / train.py) — ~200M tokens of
  narrative-grounded dialogue, roughly Chinchilla-optimal for this size.
- "gpt2": GPT-2 fine-tuned on DailyDialog (hf_model.py / finetune.py).
  Opt-in only: fine-tuning needs ~16GB RAM or a GPU and is never started
  automatically.

Reply length (ChatEngine(reply_length=...), CLI --reply-length, or
SODACHAT_REPLY_LENGTH): "short", "medium" (default) or "long" — see
`ReplyLength`, which carries the generation budget and the trim caps together
because setting one without the others does nothing.

Relevance comes from MMI reranking: each candidate is scored by how much the
conversation context raises its likelihood versus no context at all,
    score = logP(reply | context) - LAMBDA * logP(reply)
(mean per-token, computed with the same model). Fluent-but-generic
continuations that ignore the prompt score low and are discarded.
"""

from __future__ import annotations

import os
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .model import build_prompt, null_prompt

# Word filter: the training data (and anything a model can dream up from it)
# can be crude; keep the default experience safe for public channels.
# Disable with ChatEngine(filtered=False).
_BLOCKLIST_RE = re.compile(
    r"\b(?:fuck\w*|shit\w*|bitch\w*|cunt\w*|nigg\w*|fag\w*|cock\w*|dick\w*"
    r"|pussy|horn(?:y|ie\w*)|sexy?|nude\w*|naked|porn\w*|slut\w*|whore\w*"
    r"|rape\w*|penis|vagina|boob\w*|tit(?:s|ties)?)\b",
    re.IGNORECASE,
)

# Input/output hygiene, unrelated to the word filter above.
_WS_RE = re.compile(r"\s+")  # normalize whitespace
_HAS_CONTENT_RE = re.compile(r"[A-Za-z0-9]")  # did the user type anything real
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")  # is a candidate reply actual text
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+(?:['\")\]]+)?")  # reply trimming

_NUDGE_LINES = [
    "you there? say something :)",
    "hmm? type something and i'll bite",
    "ok... use your words",
]

_HISTORY_LINES = 8
_NUM_CANDIDATES = 12
_GEN_TEMPERATURE = 0.75
_MMI_LAMBDA = 0.7

BACKENDS = ("mini", "gpt2")


# ------------------------------------------------------------- reply length


@dataclass(frozen=True)
class ReplyLength:
    """How long a reply is allowed to be, end to end.

    Four numbers, because reply length is decided in three places and setting
    only one of them does nothing useful:

    * `max_new_tokens` — the generation budget, passed to the model. Raising
      the trim caps without this just lets through a reply the model was
      already cut off from finishing.
    * `max_sentences` / `target_chars` — `_trim_reply` keeps whole sentences
      until either is reached, so a reply ends on a sentence rather than
      mid-clause. `target_chars` is a floor to stop at, not a hard cut: the
      sentence that crosses it is kept whole.
    * `max_chars` — a candidate longer than this is discarded outright rather
      than trimmed, which is the backstop against a run-on that trimming would
      turn into something that reads fine but answers nothing.

    Tokens are quoted in **subword** tokens; a char-tokenizer model scales them
    up itself (see `MiniChatLM._budget`).
    """

    max_new_tokens: int
    max_sentences: int
    target_chars: int
    max_chars: int


# Named lengths, so a frontend or an env var can ask for one without spelling
# out four numbers. "medium" is what this shipped with and stays the default.
REPLY_LENGTHS: dict[str, ReplyLength] = {
    "short": ReplyLength(max_new_tokens=32, max_sentences=1,
                         target_chars=60, max_chars=140),
    "medium": ReplyLength(max_new_tokens=48, max_sentences=2,
                          target_chars=100, max_chars=200),
    "long": ReplyLength(max_new_tokens=96, max_sentences=4,
                        target_chars=220, max_chars=400),
}
DEFAULT_REPLY_LENGTH = "medium"


def resolve_reply_length(value: "str | ReplyLength | None" = None) -> ReplyLength:
    """Turn a name, a `ReplyLength`, or nothing into a `ReplyLength`.

    Nothing falls back to `SODACHAT_REPLY_LENGTH` and then to the default, so
    the two bot frontends — which take their configuration from the
    environment rather than from flags — pick this up without any wiring."""
    if isinstance(value, ReplyLength):
        return value
    name = (value or os.environ.get("SODACHAT_REPLY_LENGTH")
            or DEFAULT_REPLY_LENGTH).strip().lower()
    try:
        return REPLY_LENGTHS[name]
    except KeyError:
        raise ValueError(
            f"unknown reply length {name!r} "
            f"(expected one of {', '.join(REPLY_LENGTHS)})"
        ) from None


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _trim_reply(text: str, length: ReplyLength) -> str:
    """Keep whole sentences up to `length`'s caps — sampled tails wander."""
    sentences = _SENTENCE_RE.findall(text)
    if not sentences:  # no complete sentence (e.g. "lol", "why") — keep as is
        return text
    kept: list[str] = []
    for sentence in sentences:
        if kept and (
            len(kept) >= length.max_sentences
            or sum(len(s) for s in kept) > length.target_chars
        ):
            break
        kept.append(sentence.strip())
    return " ".join(kept)


@dataclass(frozen=True)
class Reply:
    text: str
    source: str  # backend name, or "canned"
    score: float  # MMI relevance score (0.0 for canned lines)


def _load_backend(backend: str, model_path: Path | None):
    if backend == "mini":
        from .model import DEFAULT_MODEL_PATH, MiniChatLM

        target = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not target.exists():
            print(
                "[sodachat] no trained model found — training the mini-GPT on "
                "SODA now (one-time, several hours; see README)..."
            )
            from .train import train_model

            train_model(dataset="soda", out_path=target)
        return MiniChatLM(target)
    if backend == "gpt2":
        from .hf_model import DEFAULT_HF_MODEL_DIR, HFChatLM, hf_model_exists

        target = Path(model_path) if model_path else DEFAULT_HF_MODEL_DIR
        if not hf_model_exists(target):
            # Fine-tuning GPT-2 needs ~16GB RAM or a GPU; never start it as a
            # side effect of launching a chat frontend.
            raise FileNotFoundError(
                f"no fine-tuned GPT-2 model at {target}. Run "
                "`python -m sodachat.finetune` on a machine with >=16GB "
                "RAM or a GPU, or use the default mini backend."
            )
        return HFChatLM(target)
    raise ValueError(f"unknown backend {backend!r} (expected one of {BACKENDS})")


class ChatEngine:
    def __init__(
        self,
        filtered: bool = True,
        seed: int | None = None,
        backend: str | None = None,
        model_path: Path | None = None,
        lm=None,
        reply_length: "str | ReplyLength | None" = None,
    ):
        """`lm` injects a pre-built inference model (anything exposing
        `generate_line(prompt, temperature, max_new_tokens=None)` and
        `logprob(context, continuation)`)
        — e.g. an ExpertLM or UnifiedLM. When given, `backend`/`model_path` are
        ignored and the MMI reranking in reply() runs over that model, so the
        same relevance filtering applies to every mode, not just the mini/specialist
        path. Frontends keep using backend=/model_path= as before.

        `reply_length` is a name from `REPLY_LENGTHS` ("short"/"medium"/"long")
        or a `ReplyLength` of your own; omitting it reads `SODACHAT_REPLY_LENGTH`
        and falls back to the default. It applies whichever way the model got
        here, injected or loaded, because the budget rides on each
        `generate_line` call rather than on the model."""
        self.reply_length = resolve_reply_length(reply_length)
        self._filtered = filtered
        self._rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self._recent: deque[str] = deque(maxlen=8)

        if lm is not None:
            # Name it after the class so the Reply.source field stays informative.
            self.backend = type(lm).__name__.removesuffix("LM").lower() or "lm"
            self._lm = lm
        else:
            self.backend = (backend or os.environ.get("SODACHAT_BACKEND", "mini")).lower()
            self._lm = _load_backend(self.backend, model_path)

    def _acceptable(self, candidate: str, user_text: str) -> bool:
        if not candidate or not _HAS_LETTER_RE.search(candidate):
            return False
        if len(candidate) > self.reply_length.max_chars:
            return False
        if self._filtered and _BLOCKLIST_RE.search(candidate):
            return False
        lowered = candidate.lower()
        return lowered != user_text.lower() and lowered not in self._recent

    def reply(self, message: str, history: Sequence[str] = ()) -> Reply:
        """Generate a reply. `history` is recent conversation lines (both
        sides, oldest first) used to condition the model."""
        text = _clean(message)
        if not _HAS_CONTENT_RE.search(text):
            return Reply(self._rng.choice(_NUDGE_LINES), "canned", 0.0)

        lines = [_clean(h) for h in history if _clean(h)]
        # Keep whole user/bot pairs so speaker tags stay aligned.
        kept = lines[-(_HISTORY_LINES - _HISTORY_LINES % 2) :] if lines else []
        prompt = build_prompt(kept, text)

        candidates: list[str] = []
        for i in range(_NUM_CANDIDATES):
            candidate = _trim_reply(
                self._lm.generate_line(
                    prompt,
                    temperature=_GEN_TEMPERATURE + 0.05 * (i % 3),
                    max_new_tokens=self.reply_length.max_new_tokens,
                ),
                self.reply_length,
            )
            if self._acceptable(candidate, text) and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            return Reply(self._rng.choice(_NUDGE_LINES), "canned", 0.0)

        # MMI: prefer candidates the conversation makes likely over ones that
        # are simply likely to be said at all. Both sides are scored in the
        # same bot-turn frame, so only the context differs.
        null = null_prompt()
        scored = [
            (
                candidate,
                self._lm.logprob(prompt, candidate)
                - _MMI_LAMBDA * self._lm.logprob(null, candidate),
            )
            for candidate in candidates
        ]
        best, score = max(scored, key=lambda pair: pair[1])
        self._recent.append(best.lower())
        return Reply(best, self.backend, score)
