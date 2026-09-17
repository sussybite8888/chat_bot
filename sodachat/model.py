"""The base decoder LM (`MiniGPT`) and the chat model built on it.

`MiniGPT` is a nanoGPT-style decoder-only transformer, assembled from the shared
building blocks in `blocks.py`. It is the **base model reused across the
package** — chat, the reader, the game specialists, the unified model, and the
instruct post-train are all a `MiniGPT` trained on different data. Two other
architectures extend the same trunk in their own files: `MultiHeadGPT`
(narrate.py) and `ExpertGPT` (expert.py). See the MODEL MAP in
`sodachat/__init__.py`.

This file owns:
    MiniGPT                         the base LM
    save_checkpoint/load_checkpoint MiniGPT checkpoint I/O
    MiniChatLM                      the chat model's inference wrapper
    build_prompt/null_prompt/...    the A:/B: chat protocol

The shared building blocks it uses (GPTConfig, tokenizers, pick_device,
pad_load, …) live in `blocks.py` and are re-exported here for compatibility.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared toolkit from blocks.py. Block/RMSNorm/_rope_cache build MiniGPT below;
# the rest are re-exported so `from .model import GPTConfig` (etc.) keeps working
# for callers that predate blocks.py. New code should import them from .blocks.
from .blocks import (  # noqa: F401
    Block,
    configure_cpu,
    BPETokenizer,
    CharTokenizer,
    GPTConfig,
    KVCache,
    RMSNorm,
    _rope_cache,
    config_from_payload,
    pad_load,
    pick_device,
    tokenizer_from_payload,
    warp_logits,
)

# The chat protocol shared by training data (data.py) and inference. Turns are
# tagged by speaker and conversations end with a separator token, so the model
# learns who it is answering and never treats a topic switch as a valid reply.
DIALOG_SEP = "<|endofdialog|>"
SPEAKERS = ("A", "B")
USER, BOT = SPEAKERS


def build_prompt(history: list[str], message: str) -> str:
    """Render a conversation as tagged turns, ending with the bot's turn open
    so that generation continues as the bot's reply.

    Deliberately ends with "B:" and NO trailing space. Byte-level BPE folds
    the space into the following word (" Electronic" is one token), so a
    trailing space would be its own token — a sequence never seen after "B:"
    in training, leaving the model to emit word-continuation fragments
    ("lect", "ronic") as if the word had already started.
    """
    turns = [*history, message]
    # The user always speaks last, so assign speakers backwards from them.
    start = (len(turns) - 1) % 2
    lines = [f"{SPEAKERS[(start + i) % 2]}: {t}" for i, t in enumerate(turns)]
    return "\n".join(lines) + f"\n{BOT}:"


def null_prompt() -> str:
    """A bot turn with no conversation — the MMI baseline for `P(reply)`."""
    return f"{BOT}:"


def default_model_path(dataset: str = "soda") -> Path:
    env = os.environ.get("SODACHAT_MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models" / f"minigpt-{dataset}.pt"


DEFAULT_MODEL_PATH = default_model_path()


# --------------------------------------------------------- base decoder LM


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        # Zero the residual-path output projections, so every block starts as
        # exactly the identity and the residual stream carries the embedding
        # untouched at step 0. This replaces GPT-2's depth-scaled init, which
        # aimed at the same problem — a residual stream that grows with depth —
        # by shrinking the branches rather than switching them off. Nothing
        # stays stuck at zero: a zeroed projection still receives gradient on
        # its first step, and the layers feeding it start learning right after.
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.zeros_(p)
        # RoPE angles are fixed, not learned — cache them per (device, dtype).
        self._rope: dict = {}

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_for(self, T: int, device, dtype, offset: int = 0
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotation factors for `T` positions starting at absolute `offset`.

        `offset` is what makes a KV cache correct: token number `n` must be
        rotated by its own absolute angle whether it arrived in a full prompt
        or alone as the next step, or the relative distances attention reads
        out would not line up with the cached keys."""
        key = (device, dtype)
        cached = self._rope.get(key)
        need = offset + T
        if cached is None or cached[0].shape[0] < need:
            cached = _rope_cache(
                max(need, self.cfg.block_size),
                self.cfg.n_embd // self.cfg.n_head,
                self.cfg.rope_theta,
                device,
                dtype,
            )
            self._rope[key] = cached
        return cached[0][offset:need], cached[1][offset:need]

    @staticmethod
    def _doc_mask(doc_ids: torch.Tensor, T: int) -> torch.Tensor:
        """A (B, 1, T, T) attention mask that is causal *and* confined to one
        document.

        Training windows are random crops of a stream of packed dialogues, so a
        window nearly always straddles a boundary. Plain causal attention lets
        the tail of one conversation attend back into an unrelated one, and the
        model spends capacity learning to ignore it. Masking to the current
        document removes the distraction outright. Every row keeps at least its
        own position, so no row is fully masked.
        """
        same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
        causal = torch.ones(T, T, dtype=torch.bool, device=doc_ids.device).tril()
        return (same_doc & causal).unsqueeze(1)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`last_only` returns logits for the final position alone.

        Sampling reads exactly one row of the logits — the last — but the head
        is the widest matmul in the model (n_embd x vocab, and tied, so it is
        also the largest weight). Projecting a whole prompt through it during
        the prefill computes a full vocabulary distribution for every token of
        the context and then throws all but one away. Scoring (`logprob`) and
        training do read every position, so this stays opt-in.
        """
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        n_past = cache.n_past if cache is not None else 0
        cos, sin = self._rope_for(T, x.device, x.dtype, offset=n_past)
        attn_mask = None if doc_ids is None else self._doc_mask(doc_ids, T)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, attn_mask, cache=cache, layer=i)
        if last_only and targets is None:
            x = x[:, -1:, :]
        logits = self.head(self.ln_f(x))
        if self.cfg.logit_softcap:
            # Squash logits into (-cap, cap). A from-scratch model will happily
            # drive a few logits far out to win the cross-entropy on easy
            # tokens; capping keeps the softmax in a range where gradients stay
            # informative, and takes the lid off the learning rate (Gemma 2).
            cap = self.cfg.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if targets is not None:
            # Targets of -100 are ignored — see `_batch` in train.py, which
            # drops the one position per document whose next token belongs to
            # the following document.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        stop_tokens: list[int] | None = None,
        use_cache: bool = True,
        shared_prompt: bool = False,
        return_lengths: bool = False,
    ):
        """Sample a continuation of `idx` — one row, or a whole batch of them.

        With `use_cache` the prompt is encoded once and each later token costs
        a single-position forward rather than another pass over the whole
        window — the same arithmetic, minus the part that was already done.
        `use_cache=False` keeps the plain re-encode loop, which is what the
        cached path is checked against.

        **Batching.** `idx` may carry B rows, each sampled independently: rows
        get their own `stop_tokens` handling and, if `temperature` is a (B,)
        tensor, their own temperature. This is the difference between a GEMV
        and a GEMM. Decoding one row reads all ~300MB of weights to produce a
        single token, so it runs at memory bandwidth with the vector units
        mostly idle; B rows read those same weights once and do B times the
        arithmetic with them, so the SIMD units have something to chew on.
        Measured on the chat model: 7.7 ms/token at B=1 against 1.28 ms/token
        at B=12 — the same work, six times cheaper per token.

        `shared_prompt` says every row starts from the same prompt (candidate
        replies to one message), so the prefill runs once on one row and the
        cache is broadcast, instead of encoding B identical copies.

        `return_lengths` gives back `(idx, lengths)`, where `lengths[b]` is how
        many tokens row `b` produced before it hit a stop token — rows finish
        at different steps, so the padding past that point is not the row's
        output and callers must slice it off.
        """
        self.eval()
        B = idx.shape[0]
        start = idx.shape[1]
        window = self.cfg.block_size
        device = idx.device
        stop = (torch.tensor(sorted(set(stop_tokens)), device=device)
                if stop_tokens else None)

        cache = KVCache(len(self.blocks), max_len=window) if use_cache else None
        if shared_prompt and cache is not None and B > 1:
            logits, _ = self(idx[:1, -window:], cache=cache, last_only=True)
            cache.expand_to(B)
            # Materialized, not left as a stride-0 view: warp_logits writes
            # into its logits in place.
            logits = logits.expand(B, -1, -1).contiguous()
        else:
            logits, _ = self(idx[:, -window:], cache=cache, last_only=True)

        # A row that has stopped keeps being stepped — the batch moves as one —
        # but its length is pinned at the step it stopped, so whatever it emits
        # afterwards is never read back.
        done = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.full((B,), max_new_tokens, dtype=torch.long, device=device)
        for step in range(max_new_tokens):
            warped = warp_logits(
                logits[:, -1, :],
                idx[:, start:],  # penalize only what this call generated
                temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
            )
            next_id = torch.multinomial(F.softmax(warped, dim=-1), num_samples=1)
            if stop is not None:
                hit = (next_id == stop).any(dim=-1)
                lengths = torch.where(hit & ~done, step, lengths)
                done |= hit
                if bool(done.all()):
                    break  # the stop token itself is never part of the output
            idx = torch.cat([idx, next_id], dim=1)
            if step == max_new_tokens - 1:
                break  # nothing would read the next logits
            if cache is None:
                logits, _ = self(idx[:, -window:], last_only=True)
            elif cache.n_past < window:
                logits, _ = self(next_id, cache=cache, last_only=True)
            else:
                # Window full. Re-encode the trailing `window` tokens from
                # scratch — exactly what the uncached loop does every step —
                # so positions restart at 0 and RoPE is never asked to
                # extrapolate past the context the model was trained on.
                cache = KVCache(len(self.blocks), max_len=window)
                logits, _ = self(idx[:, -window:], cache=cache, last_only=True)
        return (idx, lengths) if return_lengths else idx


# ---------------------------------------------------- MiniGPT checkpoint I/O


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    tokenizer: CharTokenizer | BPETokenizer,
    steps: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.cfg),
            "tokenizer": tokenizer.to_payload(),
            "state_dict": model.state_dict(),
            "steps": steps,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: str | None = None
) -> tuple[MiniGPT, CharTokenizer | BPETokenizer]:
    device = device or pick_device()
    if device == "cpu":
        configure_cpu()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MiniGPT(config_from_payload(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    if "tokenizer" in ckpt:
        tokenizer = tokenizer_from_payload(ckpt["tokenizer"])
    else:  # legacy char-only checkpoints
        tokenizer = CharTokenizer(ckpt["chars"])
    return model, tokenizer


# --------------------------------------------------------- the chat model


# Chat-reply sampling. A nucleus cut plus a mild repetition penalty steer the
# from-scratch model off the generic, self-looping continuations a bare top-k
# leaves in; the same knobs the GPT-2 backend already uses (hf_model.py). Kept
# off the reader/game paths, which want exact, low-temperature reads.
_CHAT_TOP_K = 40
_CHAT_TOP_P = 0.95
_CHAT_REPETITION_PENALTY = 1.15

# How many tokens a reply may generate before it is cut off, quoted in **subword
# tokens** — the unit `engine.ReplyLength` speaks, and the one every chat model
# here shares (unified.py and expert.py import it rather than repeating the
# number). Callers override it per call via `generate_line(max_new_tokens=...)`.
CHAT_MAX_NEW_TOKENS = 48
# A character tokenizer spends ~2.5x as many tokens on the same words, so a
# budget quoted in subword tokens is scaled by this before a char model uses it.
# 48 -> 120, the pair this shipped with.
_CHAR_TOKEN_RATIO = 2.5


class MiniChatLM:
    """Inference wrapper around the from-scratch GPT."""

    def __init__(self, path: Path = DEFAULT_MODEL_PATH, device: str | None = None,
                 max_new_tokens: int | None = None):
        """`max_new_tokens` sets how long a reply may run, in subword tokens
        (None = `CHAT_MAX_NEW_TOKENS`). It is the generation half of reply
        length; the trimming half lives in `engine.ReplyLength`, which drives
        this through `generate_line`."""
        self.model, self.tokenizer = load_checkpoint(path, device)
        self.device = next(self.model.parameters()).device
        self._newline_id = self.tokenizer.encode("\n")[0]
        self.max_new_tokens = self._budget(max_new_tokens)
        # Stop a reply at end-of-line or end-of-conversation, whichever comes
        # first, so the model never runs on into the next speaker's turn.
        self._stop_ids = [self._newline_id]
        try:
            self._stop_ids.append(self.tokenizer.token_id(DIALOG_SEP))
        except (AttributeError, KeyError):  # char/legacy checkpoints
            pass

    def _budget(self, tokens: int | None) -> int:
        """This model's reply budget for a length quoted in subword tokens.

        Two adjustments: char tokenizers are scaled up by `_CHAR_TOKEN_RATIO`,
        and the result is capped at half the block so that a long reply cannot
        squeeze the conversation out of the context window — at which point the
        model would be answering a prompt it can no longer see."""
        n = CHAT_MAX_NEW_TOKENS if tokens is None else max(1, int(tokens))
        if self.tokenizer.kind == "char":
            n = round(n * _CHAR_TOKEN_RATIO)
        return min(n, self.model.cfg.block_size // 2)

    def _prompt_ids(self, prompt: str, max_new: int) -> list[int]:
        ids = self.tokenizer.encode(prompt) or [self._newline_id]
        # Trim old context so prompt + reply fit in the block.
        return ids[-(self.model.cfg.block_size - max_new):]

    def generate_line(self, prompt: str, temperature: float = 0.8,
                      max_new_tokens: int | None = None) -> str:
        return self.generate_lines(prompt, [temperature], max_new_tokens)[0]

    def generate_lines(self, prompt: str, temperatures: "Sequence[float]",
                       max_new_tokens: int | None = None) -> list[str]:
        """Sample one continuation of `prompt` per entry in `temperatures`, in a
        single batched pass.

        This is the batched form of `generate_line`, and the reason the chat
        engine asks for its candidates all at once: decoding them one after
        another reads the model's weights once per candidate for a single
        token's worth of arithmetic each time, which is memory-bandwidth-bound
        with the vector units idle. Sampled together they share one read of the
        weights and the per-token matmuls become wide enough to keep SIMD busy
        (see `MiniGPT.generate`). The candidates stay independent — same
        distribution, same per-candidate temperature, just sampled side by side.
        """
        max_new = (self.max_new_tokens if max_new_tokens is None
                   else self._budget(max_new_tokens))
        ids = self._prompt_ids(prompt, max_new)
        n = len(temperatures)
        idx = torch.tensor([ids], dtype=torch.long,
                           device=self.device).expand(n, -1).contiguous()
        out, lengths = self.model.generate(
            idx,
            max_new_tokens=max_new,
            temperature=torch.tensor(list(temperatures), dtype=torch.float32,
                                     device=self.device),
            top_k=_CHAT_TOP_K,
            top_p=_CHAT_TOP_P,
            repetition_penalty=_CHAT_REPETITION_PENALTY,
            stop_tokens=self._stop_ids,
            shared_prompt=True,
            return_lengths=True,
        )
        start = len(ids)
        return [
            self.tokenizer.decode(out[b, start:start + int(lengths[b])].tolist()).strip()
            for b in range(n)
        ]

    @torch.inference_mode()
    def logprob(self, context: str, continuation: str) -> float:
        """Mean per-token log-prob of `continuation` (as a full chat line)
        given `context`. Used for MMI relevance reranking."""
        return self.logprob_batch(context, [continuation])[0]

    @torch.inference_mode()
    def logprob_batch(self, context: str, continuations: "Sequence[str]"
                      ) -> list[float]:
        """`logprob` for several continuations of one context, in one pass.

        MMI scores every candidate twice — once against the conversation, once
        against nothing — so a reply costs two of these calls per candidate.
        They all share a context, which makes them a single padded batch: one
        read of the weights instead of one per candidate.

        Right-padding is safe because attention is causal, so a row's real
        tokens cannot see the padding that follows them; the padded positions
        are simply dropped from each row's mean.
        """
        ctx = self.tokenizer.encode(context) or [self._newline_id]
        # Score the reply exactly as it appears in training: "B:" + " reply\n".
        # The leading space belongs to the first word's token (see build_prompt).
        conts = [self.tokenizer.encode(" " + c.strip() + "\n") for c in continuations]
        width = max((len(c) for c in conts), default=0)
        if not width:
            return [float("-inf")] * len(conts)
        # Trim to fit the longest candidate, so every candidate is scored
        # against an identical context — which is what makes the scores
        # comparable, and is the point of the exercise.
        overflow = len(ctx) + width - self.model.cfg.block_size
        if overflow > 0:  # trim old context, never the continuation
            ctx = ctx[overflow:] or [self._newline_id]
        pad = self._newline_id
        rows = [ctx + c + [pad] * (width - len(c)) for c in conts]
        ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        logits, _ = self.model(ids[:, :-1])
        # logP(target) = logit[target] - logsumexp(logits), which avoids
        # materializing a second (B, T, vocab) tensor for the log-softmax.
        targets = ids[:, 1:]
        picked = (logits.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                  - logits.logsumexp(-1))
        n = len(ctx)
        scored = picked[:, n - 1:]  # the continuation's own positions
        keep = (torch.arange(width, device=self.device)[None, :]
                < torch.tensor([len(c) for c in conts], device=self.device)[:, None])
        totals = (scored * keep).sum(-1) / keep.sum(-1).clamp_min(1)
        return [float(t) if len(c) else float("-inf")
                for t, c in zip(totals, conts)]
