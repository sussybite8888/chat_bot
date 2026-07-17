"""A small GPT, trained from scratch on a dialogue dataset (DailyDialog by
default — see train.py).

nanoGPT-style decoder-only transformer. Every weight comes from the training
dialogues — no pretrained parts. Tokenization is either a small BPE subword
vocabulary trained on the same dialogues (default — the model sees whole
words and many turns of context) or plain characters.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    env = os.environ.get("NPSCHAT_MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models" / f"minigpt-{dataset}.pt"


DEFAULT_MODEL_PATH = default_model_path()


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.1
    rope_theta: float = 10000.0


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
        # Scale down the residual-path output projections by depth, so the
        # residual stream does not blow up in deeper stacks (GPT-2 init).
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
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

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_for(T, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
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
        stop_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        self.eval()
        stop = set(stop_tokens or ())
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if next_id.item() in stop:
                break
            idx = torch.cat([idx, next_id], dim=1)
        return idx


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
    model = MiniGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    if "tokenizer" in ckpt:
        tokenizer = tokenizer_from_payload(ckpt["tokenizer"])
    else:  # legacy char-only checkpoints
        tokenizer = CharTokenizer(ckpt["chars"])
    return model, tokenizer


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
            top_k=40,
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
