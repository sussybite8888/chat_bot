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

_MAX_REPLY_CHARS = 200
_HISTORY_LINES = 8
_NUM_CANDIDATES = 12
_GEN_TEMPERATURE = 0.75
_MMI_LAMBDA = 0.7
_TRIM_MAX_SENTENCES = 2
_TRIM_TARGET_CHARS = 100

BACKENDS = ("mini", "gpt2")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _trim_reply(text: str) -> str:
    """Keep the first sentence or two — sampled tails tend to wander."""
    sentences = _SENTENCE_RE.findall(text)
    if not sentences:  # no complete sentence (e.g. "lol", "why") — keep as is
        return text
    kept: list[str] = []
    for sentence in sentences:
        if kept and (
            len(kept) >= _TRIM_MAX_SENTENCES
            or sum(len(s) for s in kept) > _TRIM_TARGET_CHARS
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
    ):
        """`lm` injects a pre-built inference model (anything exposing
        `generate_line(prompt, temperature)` and `logprob(context, continuation)`)
        — e.g. an ExpertLM or UnifiedLM. When given, `backend`/`model_path` are
        ignored and the MMI reranking in reply() runs over that model, so the
        same relevance filtering applies to every mode, not just the mini/specialist
        path. Frontends keep using backend=/model_path= as before."""
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
        if len(candidate) > _MAX_REPLY_CHARS:
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
                    prompt, temperature=_GEN_TEMPERATURE + 0.05 * (i % 3)
                )
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
