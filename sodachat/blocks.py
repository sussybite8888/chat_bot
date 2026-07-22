"""Shared building blocks — the neural-net toolkit every model is built from.

This module is **model-agnostic**: it holds the transformer primitives, the
tokenizers, the device helper, and the vocab-tolerant weight loader that all of
sodachat's models reuse. Nothing here is specific to chatting, reading, or
playing — those live in the per-model files (see the MODEL MAP in
`sodachat/__init__.py`).

    building blocks   RMSNorm, RoPE, CausalSelfAttention, SwiGLU, Block
    config            GPTConfig
    tokenizers        CharTokenizer, BPETokenizer, tokenizer_from_payload
    utilities         pick_device, pad_load

The models assembled from these:
    MiniGPT      (model.py)    — base decoder LM; reused by chat, reader,
                                 game specialists, unified, instruct
    MultiHeadGPT (narrate.py)  — MiniGPT + an action head
    ExpertGPT    (expert.py)   — task-routed FFN experts + action head
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _Amp:
    """Mixed-precision training context. Use via `make_amp(device)`:

        amp = make_amp(device)
        with amp.autocast():
            loss = ...                 # forward + loss in bf16/fp16
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)             # replaces loss.backward()
        amp.step(opt, model)           # unscale + clip_grad_norm_ + opt.step()

    It engages only on CUDA (bf16 on Ampere+, else fp16 with a GradScaler); on
    CPU/MPS it is a transparent no-op that preserves the exact fp32 path, so
    behaviour there is unchanged."""

    def __init__(self, device: str, enabled: bool, dtype):
        self.device = device
        self.enabled = enabled
        self.dtype = dtype
        # A loss scaler is only needed for fp16 on CUDA; bf16 and the fp32
        # fallback run without one (None => plain backward/step below).
        self.scaler = None
        if enabled and device == "cuda" and dtype == torch.float16:
            try:  # prefer the non-deprecated torch.amp API when present
                self.scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                self.scaler = torch.cuda.amp.GradScaler()

    def autocast(self):
        return torch.autocast(self.device, dtype=self.dtype, enabled=self.enabled)

    def backward(self, loss):
        (self.scaler.scale(loss) if self.scaler else loss).backward()

    def step(self, optimizer, model=None, grad_clip: float = 1.0):
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        if grad_clip and model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    @property
    def tag(self) -> str:
        return f"{str(self.dtype).rsplit('.', 1)[-1]} AMP" if self.enabled else "fp32"


def make_amp(device: str, enabled: bool | None = None) -> _Amp:
    """Build the mixed-precision helper for a training loop (see `_Amp`).

    Auto-enables on CUDA only, choosing bf16 where the GPU supports it (no loss
    scaling needed) and fp16 otherwise. `SODACHAT_NO_AMP=1` forces full precision;
    pass `enabled=` to override explicitly."""
    import os

    if enabled is None:
        enabled = device == "cuda" and os.environ.get("SODACHAT_NO_AMP") != "1"
    dtype = torch.float16
    if enabled and device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    return _Amp(device, bool(enabled), dtype)


def warp_logits(
    logits: torch.Tensor,
    seq: torch.Tensor | None,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Turn raw last-position logits into a distribution ready for sampling.

    `logits` is (B, V) — the logits at the position being generated. `seq` is
    (B, T), the tokens generated so far, used only by the repetition penalty.
    Applied in the conventional order:

    1. **Repetition penalty** (CTRL, Keskar et al. 2019): each logit for a token
       already in `seq` is divided by `repetition_penalty` if positive, else
       multiplied by it, so >1 discourages repeats. This is what tames the
       word-looping small models fall into; 1.0 is a no-op.
    2. **Temperature.**
    3. **Top-k**, then **nucleus (top-p)** — top-p keeps the smallest set of
       most-likely tokens whose mass reaches `top_p`, a softer tail cut than a
       fixed k. Filtered entries are set to -inf; the caller softmaxes.

    Defaults are all no-ops so callers that pass only temperature/top_k behave
    exactly as before.
    """
    if repetition_penalty != 1.0 and seq is not None and seq.numel():
        for b in range(seq.shape[0]):
            ids = torch.unique(seq[b])
            picked = logits[b, ids]
            logits[b, ids] = torch.where(
                picked > 0, picked / repetition_penalty, picked * repetition_penalty
            )
    logits = logits / max(temperature, 1e-5)
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        # Keep the first token that crosses the threshold, so at least one
        # candidate always survives.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        logits[remove.scatter(1, sorted_idx, remove)] = -float("inf")
    return logits


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1
    rope_theta: float = 10000.0


# ------------------------------------------------------------- tokenizers


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = chars
        self._stoi = {c: i for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        # Characters unseen at training time are silently dropped.
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def to_payload(self) -> dict:
        return {"type": "char", "chars": self.chars}


class BPETokenizer:
    """Small byte-level BPE vocabulary trained on the training dialogues."""

    kind = "bpe"

    def __init__(self, tok):
        self._tok = tok

    @classmethod
    def train(
        cls, texts, vocab_size: int, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token=None))
        # Special tokens must survive pre-tokenization as single units.
        if special_tokens:
            from tokenizers import AddedToken

            tok.add_special_tokens(
                [AddedToken(t, normalized=False, special=True) for t in special_tokens]
            )
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special_tokens or [],
        )
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    def token_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise KeyError(f"token {token!r} not in vocabulary")
        return tid

    def add_special(self, tokens: list[str]) -> int:
        """Append new special tokens (kept as single units). Returns the new
        vocab size. Existing ids are unchanged; new tokens get ids at the end,
        so a model's embedding can be padded to match (see pad_load)."""
        from tokenizers import AddedToken

        self._tok.add_special_tokens(
            [AddedToken(t, normalized=False, special=True) for t in tokens]
        )
        return self._tok.get_vocab_size()

    def __len__(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch_fast(texts)]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def to_payload(self) -> dict:
        return {"type": "bpe", "json": self._tok.to_str()}


def tokenizer_from_payload(payload: dict) -> CharTokenizer | BPETokenizer:
    if payload["type"] == "char":
        return CharTokenizer(payload["chars"])
    if payload["type"] == "bpe":
        from tokenizers import Tokenizer

        return BPETokenizer(Tokenizer.from_str(payload["json"]))
    raise ValueError(f"unknown tokenizer type {payload['type']!r}")


# ----------------------------------------------------- transformer blocks


class RMSNorm(nn.Module):
    """Llama-style normalization: like LayerNorm without the mean-centering."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    angles = torch.outer(torch.arange(seq_len, device=device).float(), freqs)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim) — rotate each (even, odd) dimension pair by a
    # position-dependent angle, so attention sees *relative* distance.
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class SwiGLU(nn.Module):
    """Gated feed-forward (Llama-style). Hidden width is 2/3 of the usual 4x
    so the gate's extra matrix keeps the parameter count the same."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(2 / 3 * 4 * cfg.n_embd)
        hidden += (-hidden) % 64  # round up for efficient matmuls
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


# ------------------------------------------------- vocab-tolerant loading


@torch.no_grad()
def pad_load(model: nn.Module, state_dict: dict) -> dict:
    """Load `state_dict` into `model`, tolerating a grown vocabulary.

    A post-trained model may add tokens (a new task marker, instruction
    vocabulary), so its embedding — and the tied LM head — are taller than the
    source checkpoint's. For any parameter whose shape only grew, the
    overlapping region is copied and the new rows keep their fresh init (the
    source weights are "padded" up); everything else is copied outright. This
    lets instruction post-training warm-start from the base model's weights.
    """
    target = model.state_dict()
    stats = {"copied": 0, "padded": 0, "skipped": 0}
    for key, src in state_dict.items():
        dst = target.get(key)
        if dst is None:
            stats["skipped"] += 1
            continue
        if dst.shape == src.shape:
            dst.copy_(src)
            stats["copied"] += 1
        elif dst.dim() == src.dim() and all(d >= s for d, s in zip(dst.shape, src.shape)):
            region = tuple(slice(0, s) for s in src.shape)
            dst[region].copy_(src)
            stats["padded"] += 1
        else:
            stats["skipped"] += 1
    model.load_state_dict(target)
    return stats
