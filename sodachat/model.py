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

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared toolkit from blocks.py. Block/RMSNorm/_rope_cache build MiniGPT below;
# the rest are re-exported so `from .model import GPTConfig` (etc.) keeps working
# for callers that predate blocks.py. New code should import them from .blocks.
from .blocks import (  # noqa: F401
    Block,
    BPETokenizer,
    CharTokenizer,
    GPTConfig,
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

    def _rope_for(self, T: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._rope.get(key)
        if cached is None or cached[0].shape[0] < T:
            cached = _rope_cache(
                max(T, self.cfg.block_size),
                self.cfg.n_embd // self.cfg.n_head,
                self.cfg.rope_theta,
                device,
                dtype,
            )
            self._rope[key] = cached
        return cached[0][:T], cached[1][:T]

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_for(T, x.device, x.dtype)
        attn_mask = None if doc_ids is None else self._doc_mask(doc_ids, T)
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        stop_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        self.eval()
        stop = set(stop_tokens or ())
        start = idx.shape[1]
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size :]
            logits, _ = self(ctx)
            logits = warp_logits(
                logits[:, -1, :],
                idx[:, start:],  # penalize only what this call generated
                temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if next_id.item() in stop:
                break
            idx = torch.cat([idx, next_id], dim=1)
        return idx


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


class MiniChatLM:
    """Inference wrapper around the from-scratch GPT."""

    def __init__(self, path: Path = DEFAULT_MODEL_PATH, device: str | None = None):
        self.model, self.tokenizer = load_checkpoint(path, device)
        self.device = next(self.model.parameters()).device
        self._newline_id = self.tokenizer.encode("\n")[0]
        # A reply line is ~one sentence: chars need a longer budget than words.
        self._max_new = 120 if self.tokenizer.kind == "char" else 48
        # Stop a reply at end-of-line or end-of-conversation, whichever comes
        # first, so the model never runs on into the next speaker's turn.
        self._stop_ids = [self._newline_id]
        try:
            self._stop_ids.append(self.tokenizer.token_id(DIALOG_SEP))
        except (AttributeError, KeyError):  # char/legacy checkpoints
            pass

    def generate_line(self, prompt: str, temperature: float = 0.8) -> str:
        ids = self.tokenizer.encode(prompt) or [self._newline_id]
        # Trim old context so prompt + reply fit in the block.
        budget = self.model.cfg.block_size - self._max_new
        ids = ids[-budget:]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=self._max_new,
            temperature=temperature,
            top_k=_CHAT_TOP_K,
            top_p=_CHAT_TOP_P,
            repetition_penalty=_CHAT_REPETITION_PENALTY,
            stop_tokens=self._stop_ids,
        )
        return self.tokenizer.decode(out[0][len(ids) :].tolist()).strip()

    @torch.no_grad()
    def logprob(self, context: str, continuation: str) -> float:
        """Mean per-token log-prob of `continuation` (as a full chat line)
        given `context`. Used for MMI relevance reranking."""
        ctx = self.tokenizer.encode(context) or [self._newline_id]
        # Score the reply exactly as it appears in training: "B:" + " reply\n".
        # The leading space belongs to the first word's token (see build_prompt).
        cont = self.tokenizer.encode(" " + continuation.strip() + "\n")
        if not cont:
            return float("-inf")
        overflow = len(ctx) + len(cont) - self.model.cfg.block_size
        if overflow > 0:  # trim old context, never the continuation
            ctx = ctx[overflow:] or [self._newline_id]
        ids = torch.tensor([ctx + cont], dtype=torch.long, device=self.device)
        logits, _ = self.model(ids[:, :-1])
        logprobs = F.log_softmax(logits[0], dim=-1)
        targets = ids[0, len(ctx) :]
        picked = logprobs[len(ctx) - 1 :].gather(1, targets.unsqueeze(1))
        return float(picked.mean())
