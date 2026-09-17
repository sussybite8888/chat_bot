"""Shared building blocks — the neural-net toolkit every model is built from.

This module is **model-agnostic**: it holds the transformer primitives, the
tokenizers, the device helper, and the vocab-tolerant weight loader that all of
sodachat's models reuse. Nothing here is specific to chatting, reading, or
playing — those live in the per-model files (see the MODEL MAP in
`sodachat/__init__.py`).

    building blocks   RMSNorm, RoPE, QK-norm, CausalSelfAttention,
                      SwiGLU / ReLU2MLP (make_mlp), Block
    config            GPTConfig, config_from_payload
    tokenizers        CharTokenizer, BPETokenizer, tokenizer_from_payload
    utilities         pick_device, pad_load

The models assembled from these:
    MiniGPT      (model.py)    — base decoder LM; reused by chat, reader,
                                 game specialists, unified, instruct
    MultiHeadGPT (narrate.py)  — MiniGPT + an action head
    ExpertGPT    (expert.py)   — task-routed FFN experts + action head
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Torch's own default thread count, which is one per *physical* core. Captured
# at import, before any model's load path can change it, so `configure_cpu`
# still knows the machine's real width after something has pinned the process.
_PHYSICAL_THREADS = torch.get_num_threads()
_cpu_configured = False


def pick_device() -> str:
    """The device models load onto.

    `SODACHAT_DEVICE` wins when set — it is the deployment knob (see
    .env.example), and a box with a GPU it does not want the bot using is the
    normal reason to set it. Otherwise prefer an accelerator."""
    env = os.environ.get("SODACHAT_DEVICE", "").strip().lower()
    if env:
        return env
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def configure_cpu(threads: int | None = None) -> int:
    """Tune this process for CPU inference. Idempotent; returns the thread count.

    Every knob here is process-global, which is why this is one function called
    from the model load paths rather than a setting carried on each model — two
    models in one process cannot disagree about it.

    * **Threads.** Torch already defaults to one per physical core, which is the
      right answer; both extremes are expensive. Pinning to 1 gives up every
      core but one (~2.5x on the measured chat model). Oversubscribing is far
      worse: on a hybrid Intel part, 20 threads across 14 physical cores
      measured **~30x slower** than 14, because the OpenMP barrier closing every
      GEMM ends up waiting on threads the scheduler has parked. So the default
      is left alone and an explicit request is clamped to the physical count.
    * **Denormals.** Attention tails and softcapped logits drift into denormal
      range, and a denormal operand drops the vector units onto a microcode
      path costing ~100x a normal FMA. Flushing to zero keeps every kernel on
      the fast SIMD path; the values involved are already numerically nothing.
    * **oneDNN.** The fused AVX2/AVX-512 (and NEON) kernels behind `nn.Linear`
      and SDPA — where the SIMD actually happens. On by default; asserted here
      so a stray disable elsewhere cannot quietly cost a factor of two.

    `SODACHAT_THREADS` overrides, for sharing a box with something else.
    """
    global _cpu_configured
    if _cpu_configured and threads is None:
        return torch.get_num_threads()
    if threads is None:
        env = os.environ.get("SODACHAT_THREADS", "").strip()
        threads = int(env) if env.isdigit() and int(env) > 0 else _PHYSICAL_THREADS
    torch.set_num_threads(max(1, min(int(threads), _PHYSICAL_THREADS)))
    # Inter-op parallelism is a second thread pool over *independent* ops. The
    # decode loop is a chain of dependent matmuls, so there is nothing for it to
    # overlap; leaving it wide only lets it compete with the intra-op pool.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already started — only settable once per process
    torch.set_flush_denormal(True)
    torch.backends.mkldnn.enabled = True
    _cpu_configured = True
    return torch.get_num_threads()


@contextlib.contextmanager
def cpu_threads(n: int):
    """Run a block at a different intra-op thread count, then restore it.

    The thread count is process-global, but the right value is not: a ~1M-param
    game model wants one thread (a thread pool costs more than the work it
    splits), while the ~77M chat model wants every core. Before this was scoped,
    whichever loaded last decided for both — so playing one round of a game left
    every later chat reply pinned to a single core."""
    prev = torch.get_num_threads()
    if n != prev:
        torch.set_num_threads(n)
    try:
        yield
    finally:
        if n != prev:
            torch.set_num_threads(prev)


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
        # `optimizer` may be one optimizer or several stepped together (the
        # Muon/AdamW split in optim.py). Clipping happens once, between
        # unscaling and the steps, so every optimizer sees the same gradients.
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        if self.scaler is not None:
            for opt in optimizers:
                self.scaler.unscale_(opt)
        if grad_clip and model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            if self.scaler is not None:
                self.scaler.step(opt)
            else:
                opt.step()
        if self.scaler is not None:
            self.scaler.update()

    @property
    def tag(self) -> str:
        return f"{str(self.dtype).rsplit('.', 1)[-1]} AMP" if self.enabled else "fp32"


def make_amp(device: str, enabled: bool | None = None) -> _Amp:
    """Build the mixed-precision helper for a training loop (see `_Amp`).

    Auto-enables on CUDA only, choosing bf16 where the GPU supports it (no loss
    scaling needed) and fp16 otherwise. `SODACHAT_NO_AMP=1` forces full precision;
    pass `enabled=` to override explicitly."""
    if enabled is None:
        enabled = device == "cuda" and os.environ.get("SODACHAT_NO_AMP") != "1"
    dtype = torch.float16
    if enabled and device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    return _Amp(device, bool(enabled), dtype)


# How far back the frequency penalty counts (see `warp_logits`). A loop lives in
# the tail of what has been generated, so counting there makes a cycle stand out
# sharply against ordinary repetition; counting over the whole output instead
# would let a long, healthy chain of thought accumulate enough "the"s to be
# penalized like a loop.
_FREQ_WINDOW = 64


def warp_logits(
    logits: torch.Tensor,
    seq: torch.Tensor | None,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
) -> torch.Tensor:
    """Turn raw last-position logits into a distribution ready for sampling.

    `logits` is (B, V) — the logits at the position being generated. `seq` is
    (B, T), the tokens generated so far, read by the two repetition controls.
    Applied in the conventional order:

    1. **Frequency penalty**: `frequency_penalty * count` is *subtracted* from
       each token's logit, counting occurrences over the last `_FREQ_WINDOW`
       generated tokens. 0.0 is a no-op.
    2. **Repetition penalty** (CTRL, Keskar et al. 2019): each logit for a token
       already in `seq` is divided by `repetition_penalty` if positive, else
       multiplied by it, so >1 discourages repeats. This is what tames the
       word-looping small models fall into; 1.0 is a no-op.
    3. **Temperature** — a scalar, or a (B,) tensor for a per-row temperature.
    4. **Top-k**, then **nucleus (top-p)** — top-p keeps the smallest set of
       most-likely tokens whose mass reaches `top_p`, a softer tail cut than a
       fixed k. Filtered entries are set to -inf; the caller softmaxes.

    Why (2) is not enough on its own, and what (1) adds. The repetition penalty
    is *presence*-based: it shaves a distinct token id once however many times
    that id has already appeared, and `logit_softcap` bounds logits to
    (-cap, cap), so dividing by 1.15 moves a winning logit by at most ~2 —
    nowhere near enough to dislodge a confident cycle. A model that has fallen
    into "...97989798..." keeps emitting it until the token budget runs out.

    The frequency penalty grows with the count instead, so a cycle digs its own
    grave: every time round, each of its ids is pushed further down. Keeping
    the count to a window rather than the whole output is what makes that safe
    on prose — a loop fills the window and is crushed, while an ordinary
    frequent word appears a handful of times and is barely moved.

    A no-repeat-n-gram ban was tried here first and removed: it is the usual
    answer for this failure, but it cannot see these loops. Byte-level BPE has
    tokens for "0", "00", "000" and "0000", so one character-level cycle is
    spelled differently each time round and the token n-grams never match. It
    left the digit loops untouched (8/60 -> 9/60 on the arithmetic prompts that
    provoke them) while costing answers elsewhere.

    Defaults are all no-ops so callers that pass only temperature/top_k behave
    exactly as before.
    """
    if frequency_penalty and seq is not None and seq.numel():
        window = seq[:, -_FREQ_WINDOW:]
        counts = torch.zeros_like(logits)
        counts.scatter_add_(1, window, torch.ones_like(window, dtype=logits.dtype))
        logits -= frequency_penalty * counts
    if repetition_penalty != 1.0 and seq is not None and seq.numel():
        # Presence of each id in the row's own history, as a (B, V) mask. This
        # says exactly what a per-row `torch.unique` said, without the Python
        # loop over the batch — which mattered once candidates are sampled as a
        # batch, since that loop ran per row per generated token.
        present = torch.zeros_like(logits, dtype=torch.bool)
        present.scatter_(1, seq, True)
        logits = torch.where(
            present,
            torch.where(logits > 0, logits / repetition_penalty,
                        logits * repetition_penalty),
            logits,
        )
    if torch.is_tensor(temperature):
        # Per-row temperature: the candidates in a batch are deliberately
        # sampled at slightly different temperatures (see ChatEngine.reply).
        logits = logits / temperature.to(logits.dtype).clamp_min(1e-5).unsqueeze(-1)
    else:
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
    # Architecture options. The defaults are the current ones; a checkpoint
    # trained before an option existed gets the old behaviour back through
    # `config_from_payload`, so old weights keep loading and running exactly
    # as they were trained.
    qk_norm: bool = True
    mlp: str = "relu2"  # "relu2" | "swiglu"
    logit_softcap: float = 15.0  # 0 disables


# What each option meant before it existed. Every checkpoint written by an
# older version of this package describes the architecture below.
LEGACY_CONFIG = {"qk_norm": False, "mlp": "swiglu", "logit_softcap": 0.0}


def config_from_payload(payload: dict, **overrides) -> GPTConfig:
    """Rebuild the config a checkpoint was trained with.

    Always use this instead of `GPTConfig(**ckpt["config"])`: a saved config
    only carries the fields that existed when it was written, and the dataclass
    defaults are *today's* architecture. Missing fields therefore have to fall
    back to `LEGACY_CONFIG`, not to the defaults, or an old checkpoint would be
    loaded into a model shaped differently from the one that produced it.
    """
    return GPTConfig(**{**LEGACY_CONFIG, **payload, **overrides})


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
    # Built with inference mode explicitly off. These angles are cached on the
    # model and outlive the call that first asked for them, so if the first
    # caller happened to be a sampling loop (which runs under
    # `torch.inference_mode`) they would be *inference tensors* — and a later
    # autograd-tracked forward on the same model instance would then fail with
    # "Inference tensors cannot be saved for backward". Nothing here is
    # differentiable, so opting out costs nothing.
    with torch.inference_mode(False):
        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        angles = torch.outer(torch.arange(seq_len, device=device).float(), freqs)
        return angles.cos().to(dtype), angles.sin().to(dtype)


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization with no learned gain — the QK-norm of nanochat and
    modded-nanogpt.

    Deliberately parameter-free. A learned per-dimension gain applied to a
    query or key would scale RoPE's dimension pairs unevenly, and attention
    scores would stop depending only on *relative* position — the one property
    RoPE exists to provide. With no gain, normalizing and rotating commute and
    the question of which order to apply them in goes away.
    """
    scale = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * scale).type_as(x)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim) — rotate each (even, odd) dimension pair by a
    # position-dependent angle, so attention sees *relative* distance.
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Without one, sampling N tokens re-runs the whole forward pass over the
    whole context N times; with one, the prompt is encoded once and each new
    token only ever computes its own row. What is stored is *post*-RoPE and
    post-QK-norm keys — exactly the tensors that go into the dot product — so
    a cached key never has to be re-rotated. That works because RoPE encodes
    absolute position at write time and attention reads out the difference:
    as long as every new token is rotated at its own absolute index (see
    `MiniGPT._rope_for`'s `offset`), relative distances stay correct.

    Storage is a **preallocated** buffer per layer, written in place, rather
    than a `torch.cat` per step. Concatenating reallocates and recopies the
    whole run every token, so the bytes moved over a reply grow with the square
    of its length — at the chat model's size that was a few hundred MB of pure
    memcpy per step, competing for the same memory bandwidth the weights need.
    `max_len` (the block size, which generation can never exceed) sizes the
    buffer once up front; without it the buffer doubles as it grows.
    """

    def __init__(self, n_layer: int, max_len: int | None = None):
        self.k: list[torch.Tensor | None] = [None] * n_layer
        self.v: list[torch.Tensor | None] = [None] * n_layer
        # Positions written per layer. Kept explicitly because the buffer's own
        # length is now its capacity, not its contents.
        self._fill = [0] * n_layer
        self.max_len = max_len

    @property
    def n_past(self) -> int:
        """How many positions are already cached (0 before the prefill)."""
        return self._fill[0]

    def _reserve(self, layer: int, k: torch.Tensor, need: int) -> None:
        buf = self.k[layer]
        if buf is not None and buf.shape[-2] >= need:
            return
        B, H, _, D = k.shape
        cap = max(need, self.max_len or 0, 2 * (buf.shape[-2] if buf is not None else 0))
        new_k, new_v = k.new_empty((B, H, cap, D)), k.new_empty((B, H, cap, D))
        if buf is not None:
            n = self._fill[layer]
            new_k[..., :n, :] = buf[..., :n, :]
            new_v[..., :n, :] = self.v[layer][..., :n, :]
        self.k[layer], self.v[layer] = new_k, new_v

    def update(self, layer: int, k: torch.Tensor,
               v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's keys/values for `layer` and return the full run."""
        n, T = self._fill[layer], k.shape[-2]
        self._reserve(layer, k, n + T)
        self.k[layer][..., n:n + T, :] = k
        self.v[layer][..., n:n + T, :] = v
        self._fill[layer] = n + T
        # A narrow on the position axis, so nothing is copied out. The result
        # is still a batch of matrices with a regular leading stride, which is
        # what the BLAS/oneDNN attention kernels want.
        return self.k[layer][..., :n + T, :], self.v[layer][..., :n + T, :]

    def expand_to(self, batch: int) -> None:
        """Broadcast a batch-1 cache across `batch` rows.

        Sampling several candidate replies means running one prompt as a batch,
        and every row's prompt is identical — so the prefill is done once and
        its cache copied out here, rather than paying for the same prompt B
        times. Only the filled region is copied, not the reserved capacity."""
        for i, k in enumerate(self.k):
            if k is None or k.shape[0] == batch:
                continue
            n, (_, H, cap, D) = self._fill[i], k.shape
            new_k, new_v = k.new_empty((batch, H, cap, D)), k.new_empty((batch, H, cap, D))
            new_k[..., :n, :] = k[..., :n, :]
            new_v[..., :n, :] = self.v[i][..., :n, :]
            self.k[i], self.v[i] = new_k, new_v


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
        # QK-norm: normalize each query and key vector to unit RMS before the
        # dot product, so attention logits can no longer be driven off by a
        # query or key that has simply grown large. Without it the softmax
        # saturates onto one position early in training and the head stops
        # exploring; with it the learning rate can go higher safely.
        self.qk_norm = cfg.qk_norm

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache: "KVCache | None" = None,
        layer: int = 0,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if self.qk_norm:
            q, k = rms_normalize(q), rms_normalize(k)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if cache is not None:
            # Store the rotated, normalized keys: what the dot product consumes.
            k, v = cache.update(layer, k, v)
        S = k.shape[-2]
        if attn_mask is None and S != T:
            # Keys reach further back than queries. SDPA's `is_causal` aligns
            # its triangle top-left, which is only correct when the two lengths
            # match — with a warm cache it would mask off the very history the
            # cache exists to keep. One query against a full cache needs no mask
            # (every cached key precedes it); a longer window needs the triangle
            # aligned bottom-right, built here.
            if T > 1:
                q_pos = torch.arange(S - T, S, device=x.device)
                k_pos = torch.arange(S, device=x.device)
                attn_mask = (k_pos[None, :] <= q_pos[:, None])[None, None]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            # A mask already carries the causal structure (see MiniGPT._doc_mask);
            # SDPA rejects being given both.
            is_causal=attn_mask is None and S == T,
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


class ReLU2MLP(nn.Module):
    """Ungated feed-forward with a squared-ReLU nonlinearity — what nanochat
    and the nanoGPT speedrun use in place of SwiGLU.

    Two matrices at 4x width rather than SwiGLU's three at 8/3x: at any n_embd
    the parameter count is identical (3 * 8/3 == 2 * 4), so this trades one of
    the three matmuls for a cheaper elementwise nonlinearity at equal capacity.
    Squaring keeps the gate-like behaviour that makes SwiGLU work — small
    activations are suppressed, large ones amplified — without the gate matrix.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.n_embd
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.relu(self.up(x)).square()))


def make_mlp(cfg: GPTConfig) -> nn.Module:
    """The feed-forward this config asks for. Both options name their output
    projection `down`, which is what the zero-init rule in MiniGPT keys on."""
    if cfg.mlp == "relu2":
        return ReLU2MLP(cfg)
    if cfg.mlp == "swiglu":
        return SwiGLU(cfg)
    raise ValueError(f"unknown mlp {cfg.mlp!r} (expected relu2 or swiglu)")


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = make_mlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache: "KVCache | None" = None,
        layer: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, attn_mask, cache=cache, layer=layer)
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
