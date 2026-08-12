"""One model, but the game part and the chat part barely share weights.

The problem with the unified model was that chat and game-playing competed for
the *same* feed-forward weights, so training one dislodged the other. Here the
attention layers stay shared (they route information and are task-general), but
each block's feed-forward network is split into **task experts** — a "text"
expert and a "game" expert — and every token is routed to its task's expert.
Chat tokens and game tokens therefore flow through different FFN weights and
don't compete, while the model is still a single network (shared embeddings,
attention, norms, and output heads).

Two output heads, as before: the LM head (text) and the action head (moves).

    task ids per token:  TEXT = 0  (chat, read, commentary)
                         GAME = 1  (board -> move, instruction-conditioned)

Warm-starts from a plain MiniGPT checkpoint (e.g. unified.pt): shared params
copy over, and the single pretrained FFN is duplicated into both experts so
each starts from a competent initialization (see `from_minigpt`).

The expert slots are also **pluggable**: a new capability can be trained as a
*specialist* — a fresh FFN expert per block plus its own output head and any
new special tokens, trained with the whole shared trunk frozen — and saved as a
small standalone checkpoint (`models/specialist-<name>.pt`). `ExpertLM`
auto-loads every such file at startup, grafting each one onto the model as a
new routed expert (`attach_specialist`), so capabilities can be added after the
fact without touching the base weights. `vision.py` (image recognition) is the
first one; `scaffold_specialist` / `save_specialist` are the training-side API.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    CausalSelfAttention,
    GPTConfig,
    RMSNorm,
    _rope_cache,
    config_from_payload,
    make_amp,
    make_mlp,
    pick_device,
    tokenizer_from_payload,
    warp_logits,
)
from .model import _CHAT_REPETITION_PENALTY, _CHAT_TOP_P

TEXT, GAME = 0, 1
N_EXPERTS = 2

# Task tags in the tagged-text corpus.
CHAT, READ, GAME_TAG, ACT, END = "<|chat|>", "<|read|>", "<|game|>", "<|act|>", "<|end|>"

# One action head over the union of every game's moves; each game masks to its
# legal subset at inference. Directions + tic-tac-toe's nine cells.
ACTION_VOCAB = ["up", "down", "left", "right", "stay",
                "0", "1", "2", "3", "4", "5", "6", "7", "8"]
ACTION_ID = {a: i for i, a in enumerate(ACTION_VOCAB)}

_MODELS = Path(__file__).resolve().parent.parent / "models"
DEFAULT_PATH = _MODELS / "expert.pt"
BASE_PATH = _MODELS / "unified.pt"

# A goal string on every play doc so the action head is always instruction-
# conditioned (the VLA property). Snake also gets richer goals from instruct._goals.
_DEFAULT_GOALS = {
    "snake": ["eat the food", "get the apple", "go for the food"],
    "pong": ["hit the ball", "block the ball", "keep the ball in play"],
    "dodge": ["dodge the blocks", "avoid the blocks", "don't get hit"],
    "tictactoe": ["win the game", "make the best move", "beat your opponent"],
    "sandbox": ["go to the target", "reach the target", "eat the food"],
}


def default_goal(game_name: str) -> str:
    return _DEFAULT_GOALS.get(game_name, ["play well"])[0]


class RoutedFFN(nn.Module):
    """One feed-forward per task; each token uses only its task's expert (so compute
    per token is a single FFN — same FLOPs as a dense model, more capacity)."""

    def __init__(self, cfg: GPTConfig, n_experts: int = N_EXPERTS):
        super().__init__()
        self.experts = nn.ModuleList(make_mlp(cfg) for _ in range(n_experts))

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = task == e  # (B, T)
            if mask.any():
                out[mask] = expert(x[mask]).to(out.dtype)
        return out


class ExpertBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, n_experts: int = N_EXPERTS):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)  # shared across tasks
        self.ln2 = RMSNorm(cfg.n_embd)
        self.ffn = RoutedFFN(cfg, n_experts)  # per-task

    def forward(self, x, cos, sin, task) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.ffn(self.ln2(x), task)


class ExpertGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, n_actions: int, n_experts: int = N_EXPERTS):
        super().__init__()
        self.cfg = cfg
        self.n_experts = n_experts
        self.n_actions = n_actions
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ExpertBlock(cfg, n_experts) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.action_head = nn.Linear(cfg.n_embd, n_actions)
        self.heads = nn.ModuleDict()  # specialist heads, keyed by specialist name
        self.specialists: dict[str, dict] = {}  # name -> {task, labels, ...}
        self.apply(self._init)
        import math

        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._rope: dict = {}

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_for(self, T, device, dtype):
        key = (device, dtype)
        c = self._rope.get(key)
        if c is None or c[0].shape[0] < T:
            c = _rope_cache(max(T, self.cfg.block_size),
                            self.cfg.n_embd // self.cfg.n_head,
                            self.cfg.rope_theta, device, dtype)
            self._rope[key] = c
        return c[0][:T], c[1][:T]

    def _features(self, idx: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """The shared trunk: embeddings -> routed blocks -> final norm. Every
        head (LM, action, specialist) reads from this."""
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_for(idx.shape[1], x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin, task)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor, task: torch.Tensor):
        """idx, task: (B, T). Returns (lm_logits [B,T,V], action_logits [B,T,A])."""
        x = self._features(idx, task)
        return self.lm_head(x), self.action_head(x)

    @torch.no_grad()
    def generate_text(self, idx, task_id, max_new_tokens, temperature=0.8,
                      top_k=40, top_p=None, repetition_penalty=1.0, stop_tokens=()):
        """Autoregressive text from the LM head, every token routed to `task_id`'s
        expert (TEXT for chat/read). Games don't use this — see `move_logits`."""
        stop = set(stop_tokens)
        start = idx.shape[1]
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            task = torch.full_like(cond, task_id)
            logits = warp_logits(
                self(cond, task)[0][:, -1, :],
                idx[:, start:],  # penalize only what this call generated
                temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() in stop:
                break
        return idx

    @torch.no_grad()
    def move_logits(self, idx):
        """Action-head logits at the final position (an `<|act|>` token), with the
        whole board routed to the GAME expert. One forward pass, no sampling."""
        task = torch.full_like(idx, GAME)
        return self(idx, task)[1][:, -1, :]

    @torch.no_grad()
    def specialist_logits(self, idx, name: str):
        """A specialist head's logits at the final position (its `<|cls|>`-style
        token), the whole prompt routed to that specialist's expert."""
        task = torch.full_like(idx, self.specialists[name]["task"])
        return self.heads[name](self._features(idx, task)[:, -1, :])

    # ------------------------------------------------- pluggable specialists

    def grow_vocab(self, new_size: int) -> None:
        """Enlarge the embedding (and the tied LM head) for newly added special
        tokens. Old rows are preserved; new rows get a fresh init."""
        old = self.tok_emb.weight.data
        emb = nn.Embedding(new_size, self.cfg.n_embd).to(old.device)
        nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            emb.weight[: old.shape[0]] = old
        self.tok_emb = emb
        self.lm_head = nn.Linear(self.cfg.n_embd, new_size, bias=False).to(old.device)
        self.lm_head.weight = self.tok_emb.weight
        self.cfg.vocab_size = new_size

    def add_expert(self, seed_from: int | None = None) -> int:
        """Append one fresh FFN expert to every block (a new routed slot),
        optionally seeded from an existing expert's weights. Returns its task id."""
        device = self.tok_emb.weight.device
        for block in self.blocks:
            expert = make_mlp(self.cfg).to(device)
            if seed_from is not None:
                expert.load_state_dict(block.ffn.experts[seed_from].state_dict())
            block.ffn.experts.append(expert)
        self.n_experts += 1
        return self.n_experts - 1

    def _trunk_sig(self) -> float:
        # A cheap fingerprint of the frozen shared trunk. A specialist is only
        # valid against the exact trunk it was trained on (its expert learned
        # to read *these* attention features), so attach checks this.
        return float(self.blocks[0].attn.qkv.weight.detach().float().sum())

    def attach_specialist(self, ck: dict, tok) -> int:
        """Graft a trained specialist checkpoint (see `save_specialist`) onto
        this model: add its special tokens (with their trained embedding rows),
        append its FFN expert to every block, and register its head. Returns
        the task id the specialist was routed to."""
        name = ck["specialist"]
        if name in self.specialists:
            raise ValueError(f"specialist {name!r} is already attached")
        for k in ("n_layer", "n_embd", "n_head"):
            if ck["config"][k] != getattr(self.cfg, k):
                raise ValueError(f"specialist {name!r} was trained for a "
                                 f"{ck['config']['n_layer']}L/{ck['config']['n_embd']}d "
                                 f"trunk, this model is "
                                 f"{self.cfg.n_layer}L/{self.cfg.n_embd}d")
        if abs(self._trunk_sig() - ck["trunk_sig"]) > 1e-2:
            raise ValueError(f"specialist {name!r} was trained against a different "
                             f"expert checkpoint — retrain it against this one")
        # 1) vocabulary: re-add the specialist's special tokens and restore their
        # trained embedding rows. A token's id need not match training time: two
        # specialists trained from the same base each grabbed the first free slot
        # for their private tag (e.g. both <|img|> and <|code|> were id 10001),
        # so we key by the token string and write each trained row at wherever
        # the token resolves *now*. The embedding follows its token, so attach
        # order no longer has to match training order. A marker token shared with
        # an already-attached specialist (e.g. <|cls|>) keeps the embedding that
        # specialist installed rather than being overwritten by this one's.
        token_ids = ck["token_ids"]
        before = len(tok)
        tok.add_special(list(token_ids))
        if len(tok) > self.cfg.vocab_size:
            self.grow_vocab(len(tok))
        with torch.no_grad():
            for t in token_ids:
                i = tok.token_id(t)
                if i < before:  # a shared marker an earlier specialist owns
                    continue
                self.tok_emb.weight[i] = ck["emb_rows"][t].to(self.tok_emb.weight.device)
        # 2) a fresh expert slot, plus a head for a classifier (a generator has
        # none — it decodes through the shared, frozen LM head), then load the
        # trained weights, remapping the expert onto its new slot.
        task = self.add_expert()
        head_key = f"heads.{name}.weight"
        if head_key in ck["state_dict"]:
            head_w = ck["state_dict"][head_key]
            self.heads[name] = nn.Linear(self.cfg.n_embd, head_w.shape[0]).to(
                self.tok_emb.weight.device)
        src = {k.replace(f".experts.{ck['task_trained']}.", f".experts.{task}."): v
               for k, v in ck["state_dict"].items()}
        result = self.load_state_dict(src, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        self.specialists[name] = {"task": task, "labels": ck.get("labels", []),
                                  "kind": ck.get("kind", "classify"),
                                  **ck.get("meta", {})}
        return task

    @classmethod
    def from_minigpt(cls, state_dict: dict, cfg: GPTConfig, n_actions: int,
                     n_experts: int = N_EXPERTS) -> "ExpertGPT":
        """Warm-start: copy shared params from a dense MiniGPT, and seed every
        FFN expert with the pretrained block's single feed-forward weights.

        Handles a grown vocabulary (e.g. adding `<|act|>`) by region-copying the
        overlapping rows of tok_emb (and the tied lm_head), like model.pad_load —
        the extra rows stay freshly initialized."""
        model = cls(cfg, n_actions, n_experts)
        tgt = model.state_dict()

        def region_copy(dst, src):
            if dst.shape == src.shape:
                dst.copy_(src)
                return True
            if dst.dim() == src.dim() and all(d >= s for d, s in zip(dst.shape, src.shape)):
                dst[tuple(slice(0, s) for s in src.shape)].copy_(src)
                return True
            return False

        copied = padded = 0
        with torch.no_grad():
            for k, v in state_dict.items():
                if ".mlp." in k:  # dense FFN -> seed every expert
                    for e in range(n_experts):
                        ek = k.replace(".mlp.", f".ffn.experts.{e}.")
                        if ek in tgt and region_copy(tgt[ek], v):
                            copied += 1
                elif k in tgt:  # emb, attn, norms, (tied) lm_head
                    if tgt[k].shape == v.shape:
                        tgt[k].copy_(v); copied += 1
                    elif region_copy(tgt[k], v):
                        padded += 1
            model.load_state_dict(tgt)
        model._warmstart = {"copied": copied, "padded": padded}
        return model


# =====================================================================
# Tagged corpus: text docs (LM loss, text expert) + game docs (LM loss +
# action loss, game expert). Each doc carries a per-token task id, and game
# docs carry an action-class target at their trailing <|act|> position.
# =====================================================================

import random
import time
from itertools import islice

import numpy as np


def _fmt_chat(utterances: list[str]) -> str:
    lines = [f"{'AB'[i % 2]}: {u}" for i, u in enumerate(utterances)]
    return f"{CHAT}\n" + "\n".join(lines) + f"\n{END}\n"


def _text_docs(split: str, seed: int, n_chat: int, n_read: int):
    """Chat + reader docs -> (doc, TEXT, None). LM loss only. Capped, because
    the text expert starts warm (from unified) and only needs preservation —
    the bulk of fresh learning goes to the game expert."""
    from .data import dailydialog_dialogues, nps_dialogues, soda_dialogues
    from .reader import _example

    def chat_stream():
        for utts in soda_dialogues("train" if split == "train" else "validation"):
            yield _fmt_chat(utts)
        if split == "train":
            for utts in dailydialog_dialogues("train"):
                yield _fmt_chat(utts)
            for utts in nps_dialogues():
                yield _fmt_chat(utts)

    for doc in islice(chat_stream(), n_chat):
        yield doc, TEXT, None
    rng = random.Random(seed)
    for _ in range(n_read):
        text, _ = _example(rng, allow_holdout=(split == "train"))
        yield f"{READ} {text.strip()} {END}\n", TEXT, None


def _play_doc(name: str, board: str, goal: str) -> str:
    # The action head fires at the trailing <|act|>; <|end|> gives a clean
    # LM boundary so windows spanning docs don't learn cross-doc targets.
    return f"{GAME_TAG} {name} | goal: {goal}\n{board}\n{ACT}\n{END}\n"


def _game_docs(n_per_game: int, n_snake_instruct: int, seed: int):
    """Play docs for every game + goal-conditioned snake docs.
    -> (doc, GAME, action_id). LM loss + action loss."""
    from .games import GAMES

    rng = random.Random(seed)
    for name, cls in GAMES.items():
        if not cls.TRAINABLE:  # held-out probe games (sandbox) stay unseen
            continue
        made = 0
        while made < n_per_game:
            g = cls(seed=rng.randrange(1 << 30))
            while not g.done and made < n_per_game:
                action = g.expert()
                goal = rng.choice(_DEFAULT_GOALS.get(name, ["play well"]))
                yield _play_doc(name, g.model_board(), goal), GAME, ACTION_ID[action]
                made += 1
                nxt = (rng.choice(g.safe_actions() or [action])
                       if rng.random() < 0.15 else action)
                g.step(nxt)

    # Instruction variety (VLA): same board, different goal -> different move.
    from .games.snake import SnakeGame
    from .instruct import _goals

    made = 0
    while made < n_snake_instruct:
        g = SnakeGame(seed=rng.randrange(1 << 30))
        while not g.done and made < n_snake_instruct:
            goals = _goals(g)
            fn, phrasings = goals[rng.choice(list(goals))]
            action = fn()
            if action in ACTION_ID:
                yield (_play_doc("snake", g.model_board(), rng.choice(phrasings)),
                       GAME, ACTION_ID[action])
                made += 1
            g.step(g.expert() if rng.random() > 0.15
                   else rng.choice(g.safe_actions() or [action]))


def _interleave(streams, weights):
    """Round-robin the streams (weight = docs pulled per round) so text and
    game docs interleave — the shared attention sees both, and most training
    windows contain some <|act|> positions for the action head."""
    its = [iter(s) for s in streams]
    alive = [True] * len(streams)
    while any(alive):
        for i, it in enumerate(its):
            if not alive[i]:
                continue
            for _ in range(weights[i]):
                try:
                    yield next(it)
                except StopIteration:
                    alive[i] = False
                    break


def _corpus(split: str, seed: int, counts: dict):
    text = _text_docs(split, seed, counts["chat"], counts["read"])
    game = _game_docs(counts["per_game"], counts["snake_instruct"], seed + 1)
    yield from _interleave([text, game], [1, 2])  # game-heavy: dedicated expert


def _write_dual(tok, tagged_docs, paths: dict, log) -> int:
    """Write three parallel binaries: token ids (uint16), task ids (uint8),
    and action targets (int16, -1 except at <|act|> positions)."""
    act_id = tok.token_id(ACT)
    total, n, started = 0, 0, time.time()
    files = {k: open(v, "wb") for k, v in paths.items()}
    try:
        while True:
            chunk = list(islice(tagged_docs, 2000))
            if not chunk:
                break
            for (doc, task, action), ids in zip(chunk, tok.encode_batch([d[0] for d in chunk])):
                toks = np.asarray(ids, dtype=np.uint16)
                tasks = np.full(len(ids), task, dtype=np.uint8)
                acts = np.full(len(ids), -1, dtype=np.int16)
                if action is not None:
                    pos = next((i for i in range(len(ids) - 1, -1, -1)
                                if ids[i] == act_id), None)
                    if pos is not None:
                        acts[pos] = action
                toks.tofile(files["tok"]); tasks.tofile(files["task"]); acts.tofile(files["act"])
                total += len(ids)
            n += len(chunk)
            if n % 100_000 < 2000:
                log(f"  {n:,} docs, {total:,} tokens ({time.time() - started:.0f}s)")
    finally:
        for f in files.values():
            f.close()
    return total


def prepare(cache: Path, tok, seed: int, counts: dict, log):
    cache.mkdir(parents=True, exist_ok=True)
    import json

    val_counts = {"per_game": 300, "snake_instruct": 300, "chat": 2000, "read": 500}
    meta_path = cache / "expert-meta.json"
    kinds = ("tok", "task", "act")
    bins = {s: {k: cache / f"expert-{s}-{k}.bin" for k in kinds} for s in ("train", "val")}
    if meta_path.exists() and all(p.exists() for s in bins.values() for p in s.values()):
        meta = json.loads(meta_path.read_text())
        log(f"reusing tokenized cache ({meta['train']:,} train tokens)")
    else:
        meta = {}
        for split, c in (("train", counts), ("val", val_counts)):
            log(f"tokenizing {split} corpus...")
            meta[split] = _write_dual(tok, _corpus(split, seed, c), bins[split], log)
        meta_path.write_text(json.dumps(meta))

    def load(split):
        return {k: np.memmap(bins[split][k], dtype=dt, mode="r")
                for k, dt in zip(kinds, (np.uint16, np.uint8, np.int16))}

    return load("train"), load("val")


# ------------------------------------------------------------------- trainer


def _batch(data, block, bs, device):
    tok, task, act = data["tok"], data["task"], data["act"]
    ix = np.random.randint(0, len(tok) - block - 1, size=bs)
    x = np.stack([tok[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([tok[i + 1:i + block + 1] for i in ix]).astype(np.int64)
    tk = np.stack([task[i:i + block] for i in ix]).astype(np.int64)
    ac = np.stack([act[i:i + block] for i in ix]).astype(np.int64)
    ts = [torch.from_numpy(a) for a in (x, y, tk, ac)]
    if device == "cuda":
        return [t.pin_memory().to(device, non_blocking=True) for t in ts]
    return [t.to(device) for t in ts]


def _losses(model, batch, lam):
    x, y, task, act = batch
    lm_logits, action_logits = model(x, task)
    lm_loss = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)), y.reshape(-1))
    mask = act >= 0
    if mask.any():
        action_loss = F.cross_entropy(action_logits[mask], act[mask])
        acc = (action_logits[mask].argmax(-1) == act[mask]).float().mean()
    else:
        action_loss = torch.zeros((), device=x.device)
        acc = torch.zeros((), device=x.device)
    return lm_loss + lam * action_loss, lm_loss, action_loss, acc


@torch.no_grad()
def _validate(model, data, block, device, lam, iters=40):
    model.eval()
    lm, act_l, accs = [], [], []
    for _ in range(iters):
        _, l, a, ac = _losses(model, _batch(data, block, 16, device), lam)
        lm.append(l.item()); act_l.append(a.item()); accs.append(ac.item())
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(lm) / len(lm), sum(act_l) / len(act_l), sum(accs) / len(accs)


def train(base=BASE_PATH, out=DEFAULT_PATH, steps=8000, batch_size=24, lr=2e-4,
          lam=1.0, block_size=1024, n_per_game=50_000, n_snake_instruct=100_000,
          n_chat=250_000, n_read=100_000, device=None, seed=1337, log=print) -> Path:
    """Warm-start ExpertGPT from the dense unified model and train with two
    objectives: LM loss on every token (routed to its task's expert) and action
    loss on the game expert's <|act|> positions. Chat/read tokens flow through
    the text expert, game tokens through the game expert — so teaching it to
    play no longer dislodges its chat ability.

    block_size is raised well above unified's window (RoPE has no learned position
    params, so warm-start is unaffected) to give chat/read long context and leave
    generous headroom for game docs. Boards are now the compact `model_board()`
    ASCII (~100 tokens for 20x20, vs ~800 for the human `render()` — which
    overflowed even this window and truncated the goal header off the front)."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ck = torch.load(base, map_location=device, weights_only=True)
    tok = tokenizer_from_payload(ck["tokenizer"])
    old_v = len(tok)
    new_v = tok.add_special([ACT])
    cfg = config_from_payload(ck["config"], vocab_size=new_v,
                              block_size=block_size, dropout=0.0)
    model = ExpertGPT.from_minigpt(ck["state_dict"], cfg, n_actions=len(ACTION_VOCAB)).to(device)
    log(f"warm-start from {base.name}: {model._warmstart} | vocab {old_v} -> {new_v} | "
        f"block {ck['config']['block_size']} -> {block_size} | "
        f"{model.num_params():,} params ({model.n_experts} experts) | device {device}")

    counts = {"per_game": n_per_game, "snake_instruct": n_snake_instruct,
              "chat": n_chat, "read": n_read}
    train_data, val_data = prepare(_MODELS, tok, seed, counts, log)
    block = cfg.block_size
    seen = steps * batch_size * block
    log(f"data: {len(train_data['tok']):,} train tokens | schedule: {steps:,} steps "
        f"x {batch_size} x {block} = {seen / 1e6:.0f}M tokens "
        f"(~{seen / max(len(train_data['tok']), 1):.1f} epochs) | action weight {lam}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    warmup = min(600, steps // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, max(warmup, 1)),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1), eta_min=lr * 0.1)],
        milestones=[max(warmup, 1)])

    amp = make_amp(device)
    model.train()
    started, best = time.time(), float("inf")
    for step in range(1, steps + 1):
        with amp.autocast():
            loss, lm_l, act_l, acc = _losses(model, _batch(train_data, block, batch_size, device), lam)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % 500 == 0 or step == steps:
            vlm, vact, vacc = _validate(model, val_data, block, device, lam)
            score = vlm + lam * vact
            mark = ""
            if score < best:
                best, mark = score, " <- saved"
                save(out, model, tok, step, vlm, vact, vacc)
            el = time.time() - started
            log(f"step {step:>5}/{steps} | train lm {lm_l.item():.3f} act {act_l.item():.3f} "
                f"acc {acc.item():.2f} | val lm {vlm:.3f} act {vact:.3f} acc {vacc:.2f} | "
                f"{el / 60:.0f}m, eta {el / step * (steps - step) / 60:.0f}m{mark}")
    log(f"done — best val score {best:.3f}, saved to {out}")
    return out


# ---------------------------------------------------------------- persistence


def save(path, model: ExpertGPT, tok, steps, val_lm, val_act, val_acc) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(model.cfg),
        "tokenizer": tok.to_payload(),
        "state_dict": model.state_dict(),
        "n_actions": model.n_actions,
        "n_experts": model.n_experts,
        "action_vocab": ACTION_VOCAB,
        "steps": steps,
        "val_lm": val_lm, "val_act": val_act, "val_acc": val_acc,
    }, path)


def load(path=DEFAULT_PATH, device=None):
    device = device or "cpu"
    ck = torch.load(path, map_location=device, weights_only=True)
    model = ExpertGPT(config_from_payload(ck["config"]), ck["n_actions"],
                      ck.get("n_experts", N_EXPERTS))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tok = tokenizer_from_payload(ck["tokenizer"])
    return model, tok, ck.get("action_vocab", ACTION_VOCAB)


# ------------------------------------------------------- specialist training
#
# A specialist = one fresh FFN expert per block + its own head + any new
# special tokens, trained with the whole shared trunk frozen. It ships as a
# small standalone checkpoint that `attach_specialist` grafts back on at load
# time, so new capabilities never touch (or risk) the base model's weights.


def scaffold_specialist(base=DEFAULT_PATH, *, name: str, special_tokens: list[str],
                        n_labels: int | None = None, seed_from: int = GAME,
                        device=None):
    """Load the trained expert model and bolt on an empty specialist: any new
    special tokens (fresh embedding rows) and a new FFN expert per block (seeded
    from an existing expert — GAME by default, since its glyph-grid diet is the
    closest thing to most new modalities; pass TEXT for a generator that decodes
    natural language / code through the LM head).

    A specialist is one of two shapes:
      * classifier — pass `n_labels`: an `n_labels`-way head reads a label off
        the final token (vision, code language-ID).
      * generator — leave `n_labels` None: no head. The expert is trained on the
        shared, frozen LM head (next-token loss) and produces text with
        `generate_text`. Give it no special tokens and it needs no vocabulary
        growth at all (codegen).

    Everything shared is frozen, so specialist training cannot disturb chat,
    reading, or play: only the new expert, the head (if any), and any new tokens'
    embedding rows receive gradients (old embedding rows are pinned by a gradient
    mask — the embedding is one tensor, tied to the LM head).

    Returns (model, tok, task_id)."""
    device = device or pick_device()
    model, tok, _ = load(base, device)
    old_v = len(tok)
    new_v = tok.add_special(special_tokens)
    model.grow_vocab(new_v)
    task = model.add_expert(seed_from)
    if n_labels:
        model.heads[name] = nn.Linear(model.cfg.n_embd, n_labels).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        for p in block.ffn.experts[task].parameters():
            p.requires_grad_(True)
    if n_labels:
        for p in model.heads[name].parameters():
            p.requires_grad_(True)
    if new_v > old_v:  # only the new tokens' rows train; nothing to unfreeze else
        w = model.tok_emb.weight
        w.requires_grad_(True)
        mask = torch.zeros(new_v, 1, device=device)
        mask[old_v:] = 1.0
        w.register_hook(lambda g: g * mask)
    return model, tok, task


def specialist_param_groups(model: ExpertGPT, weight_decay: float = 0.05) -> list[dict]:
    """Optimizer param groups for a scaffolded specialist. The embedding must
    sit in a no-decay group: the gradient mask pins the old rows' *gradients*,
    but AdamW's decoupled weight decay shrinks every row of any decayed
    parameter regardless of gradient — which would quietly erode the frozen
    vocabulary the specialist trains against."""
    emb = model.tok_emb.weight
    rest = [p for p in model.parameters() if p.requires_grad and p is not emb]
    groups = [{"params": rest, "weight_decay": weight_decay}]
    if emb.requires_grad:  # a generator with no new tokens leaves it frozen
        groups.append({"params": [emb], "weight_decay": 0.0})
    return groups


def save_specialist(path, model: ExpertGPT, tok, *, name: str, task: int,
                    labels: list[str], special_tokens: list[str],
                    steps: int, val_acc: float, kind: str = "classify",
                    meta: dict | None = None) -> None:
    """Persist just the specialist's parameters: its per-block FFN expert, its
    head (classifiers only), and the embedding rows of any special tokens —
    keyed exactly as they will live in the grafted model, so `attach_specialist`
    is a partial state-dict load. `kind` is "classify" or "generate"; a
    generator has no head and typically no special tokens."""
    sd = model.state_dict()
    keep = {k: v.detach().cpu().clone() for k, v in sd.items()
            if f".experts.{task}." in k or k.startswith(f"heads.{name}.")}
    token_ids = {t: tok.token_id(t) for t in special_tokens}
    emb_rows = {t: model.tok_emb.weight[i].detach().cpu().clone()
                for t, i in token_ids.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "specialist": name,
        "kind": kind,
        "config": asdict(model.cfg),
        "state_dict": keep,
        "task_trained": task,
        "token_ids": token_ids,
        "emb_rows": emb_rows,
        "labels": labels,
        "trunk_sig": model._trunk_sig(),
        "meta": meta or {},
        "steps": steps,
        "val_acc": val_acc,
    }, path)


# ------------------------------------------------------------------- inference


class ExpertLM:
    """One model, routed by task: chat/read use the text expert + LM head;
    game moves use the game expert + action head (a single forward pass).

    Trained specialists (`models/specialist-*.pt`) are grafted on at startup —
    each adds a routed expert and a head (e.g. `vision`: image -> digit via
    `classify`). Pass `specialists=[]` to load the base model alone, or a list
    of paths to pick exactly which ones."""

    def __init__(self, path=DEFAULT_PATH, device=None, specialists="auto"):
        device = device or "cpu"
        if device == "cpu":
            torch.set_num_threads(1)
        self.model, self.tok, self.actions = load(path, device)
        self.device = device
        if specialists == "auto":
            specialists = sorted(_MODELS.glob("specialist-*.pt"))
        for p in specialists:
            try:
                self.load_specialist(p)
            except Exception as e:  # a bad specialist never takes down the model
                print(f"[expert] skipping specialist {Path(p).name}: {e}")
        self._nl = self.tok.encode("\n")[0]
        self._end = self.tok.token_id(END)
        self.block = self.model.cfg.block_size

    @property
    def specialists(self) -> dict:
        return self.model.specialists

    def load_specialist(self, path) -> str:
        """Attach one specialist checkpoint; returns its name."""
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.model.attach_specialist(ck, self.tok)
        return ck["specialist"]

    @torch.no_grad()
    def classify(self, name: str, prompt: str) -> tuple[str, float]:
        """Run a specialist classifier: one forward pass through its expert,
        label read off its head at the prompt's final (`<|cls|>`) token.
        Returns (label, confidence)."""
        logits = self.model.specialist_logits(self._ids(prompt), name)[0]
        probs = F.softmax(logits.float(), dim=-1)
        i = int(probs.argmax())
        return self.model.specialists[name]["labels"][i], float(probs[i])

    def _ids(self, prompt):
        ids = self.tok.encode(prompt)[-self.block:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _gen(self, prompt, task_id, max_new, temperature, top_p=None,
             repetition_penalty=1.0):
        idx = self._ids(prompt)
        out = self.model.generate_text(idx, task_id, max_new, temperature,
                                       top_k=40, top_p=top_p,
                                       repetition_penalty=repetition_penalty,
                                       stop_tokens=[self._nl, self._end])
        return self.tok.decode(out[0][idx.shape[1]:].tolist())

    def chat(self, history: list[str], message: str, temperature=0.8) -> str:
        turns = [*history, message]
        lines = [f"{'AB'[(len(turns) - 1 - i) % 2]}: {t}" for i, t in enumerate(turns)]
        prompt = f"{CHAT}\n" + "\n".join(lines) + "\nB:"
        return self._gen(prompt, TEXT, 48, temperature, top_p=_CHAT_TOP_P,
                         repetition_penalty=_CHAT_REPETITION_PENALTY).strip()

    # The ChatEngine protocol (generate_line + logprob), so the expert model can
    # be wrapped by ChatEngine and get the same MMI relevance reranking the mini
    # model does. `prompt` is a tag-less "A:…\nB:" chat frame (from build_prompt);
    # we re-tag it and route to the TEXT expert to match training data exactly.
    def generate_line(self, prompt: str, temperature: float = 0.8) -> str:
        return self._gen(f"{CHAT}\n" + prompt, TEXT, 48, temperature,
                         top_p=_CHAT_TOP_P,
                         repetition_penalty=_CHAT_REPETITION_PENALTY).strip()

    @torch.no_grad()
    def logprob(self, context: str, continuation: str) -> float:
        """Mean per-token log-prob of `continuation` as a bot chat line, given
        `context` — for MMI reranking. Routed to the TEXT expert."""
        ctx = self.tok.encode(f"{CHAT}\n" + context) or [self._nl]
        cont = self.tok.encode(" " + continuation.strip() + "\n")
        if not cont:
            return float("-inf")
        overflow = len(ctx) + len(cont) - self.block
        if overflow > 0:  # trim old context, never the continuation
            ctx = ctx[overflow:] or [self._nl]
        ids = torch.tensor([ctx + cont], dtype=torch.long, device=self.device)
        inp = ids[:, :-1]  # predict each continuation token from its predecessor
        task = torch.full_like(inp, TEXT)  # whole doc is a chat doc
        logits = self.model(inp, task)[0]  # lm_head logits only
        logprobs = F.log_softmax(logits[0], dim=-1)
        targets = ids[0, len(ctx):]
        picked = logprobs[len(ctx) - 1:].gather(1, targets.unsqueeze(1))
        return float(picked.mean())

    def read(self, fields: dict, question: str) -> str:
        from .reader import state_block

        prompt = f"{READ} {state_block(fields)} | q: {question} | a:"
        return self._gen(prompt, TEXT, 40, 0.3).strip()

    def move(self, game_name: str, board: str, legal: list[str], goal: str | None = None) -> str:
        """Board -> move in one forward pass, read straight off the action head
        and masked to the legal moves. `goal` conditions the move (VLA)."""
        goal = goal or default_goal(game_name)
        idx = self._ids(f"{GAME_TAG} {game_name} | goal: {goal}\n{board}\n{ACT}")
        logits = self.model.move_logits(idx)[0]
        legal_ids = [ACTION_ID[a] for a in legal if a in ACTION_ID] or list(range(len(self.actions)))
        pick = max(legal_ids, key=lambda i: logits[i].item())
        return self.actions[pick]

    def instruct_move(self, game_name: str, board: str, instruction: str,
                      legal: list[str]) -> str:
        """Instruction-conditioned move (same signature the agent uses for the
        unified/instruct movers)."""
        return self.move(game_name, board, legal, goal=instruction)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Task-routed expert model: train / play / chat.")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=8000)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--lam", type=float, default=1.0)
    tr.add_argument("--device", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--game", default="snake")
    ev.add_argument("--episodes", type=int, default=20)
    vp = sub.add_parser("vla", help="zero-shot instruction-following probe on the sandbox game")
    vp.add_argument("--trials", type=int, default=15)
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, lam=a.lam, device=a.device)
    elif a.cmd == "eval":
        _eval_cli(a.game, a.episodes)
    elif a.cmd == "vla":
        vla_probe(trials=a.trials)


def vla_probe(path=DEFAULT_PATH, trials=15, device="cpu", seed=0, log=print) -> dict:
    """Zero-shot VLA test on the `sandbox` game — which the model never trained
    on. For directional goals the move must match exactly; for corner and
    target goals it must move the agent *closer* to the goal. Because the moves
    and goals are ones the model learned on Snake, a good result here is pure
    instruction-following *transfer*, no training involved."""
    from .games.sandbox import SandboxGame

    D = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "stay": (0, 0)}
    R = C = 10
    corners = {"the top left corner": (0, 0), "the top right corner": (0, C - 1),
               "the bottom left corner": (R - 1, 0), "the bottom right corner": (R - 1, C - 1)}

    def move_to(pos, mv):
        dr, dc = D.get(mv, (0, 0))
        return (min(max(pos[0] + dr, 0), R - 1), min(max(pos[1] + dc, 0), C - 1))

    def dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    lm = ExpertLM(path, device)
    dir_hit = dir_tot = cor_hit = cor_tot = food_hit = food_tot = 0
    for t in range(trials):
        g = SandboxGame(seed=1000 + t)
        g.ar, g.ac = 3 + (t % 4), 3 + ((t * 3) % 4)  # interior, deterministic
        g._respawn_target()
        pos, tgt, legal, board = (g.ar, g.ac), (g.tr, g.tc), list(g.ACTIONS), g.model_board()

        for d in ("up", "down", "left", "right"):
            dir_hit += lm.move("sandbox", board, legal, goal=f"go {d}") == d
            dir_tot += 1
        for name, cell in corners.items():
            mv = lm.move("sandbox", board, legal, goal=f"go to {name}")
            cor_hit += dist(move_to(pos, mv), cell) < dist(pos, cell)
            cor_tot += 1
        for goal in ("eat the food", "go to the target"):
            mv = lm.move("sandbox", board, legal, goal=goal)
            food_hit += dist(move_to(pos, mv), tgt) < dist(pos, tgt)
            food_tot += 1

    res = {"directional": dir_hit / dir_tot, "corner": cor_hit / cor_tot,
           "target": food_hit / food_tot}
    log(f"VLA probe on 'sandbox' (never trained on it), {trials} boards:")
    log(f"  directional  'go up/down/left/right'    {dir_hit:>3}/{dir_tot}  = {res['directional']:.0%}")
    log(f"  corner       'go to the ... corner'     {cor_hit:>3}/{cor_tot}  = {res['corner']:.0%}")
    log(f"  target/food  'eat the food'             {food_hit:>3}/{food_tot}  = {res['target']:.0%}")
    return res


def _eval_cli(game_name: str, episodes: int) -> None:
    from .games import GAMES

    lm = ExpertLM()
    cls = GAMES[game_name]
    scores = []
    for ep in range(episodes):
        g = cls(seed=1000 + ep)
        while not g.done:
            legal = [a for a in g.ACTIONS if g.legal(a)]
            g.step(lm.move(game_name, g.model_board(), legal))
        scores.append(getattr(g, "score", 0))
    print(f"{game_name}: mean score {sum(scores) / len(scores):.2f} over {episodes} episodes")


if __name__ == "__main__":
    main()
