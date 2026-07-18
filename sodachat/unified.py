"""One bigger model trained on everything: all the chat datasets, the game
state-reader task, and playing the games.

Instead of a separate small model per job, this trains a single larger GPT
(~30M params) on a mixture of tasks, each written as tagged text so one model
learns them all:

    <|chat|>\\nA: hi there\\nB: hey, how are you?\\n<|end|>
    <|read|> s: score 7 length 4 food up | q: whats the score | a: you have 7 points <|end|>
    <|game|> snake\\n<board>\\nmove: up <|end|>

At inference the same model chats, reads game state, and picks game moves —
just prompt it with the matching tag (see UnifiedLM). One BPE tokenizer covers
all of it.

    python -m sodachat.unified train --device cuda    # hours on a GPU
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import islice
from pathlib import Path
from typing import Iterator

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import torch

from .model import BPETokenizer, GPTConfig, MiniGPT, pick_device, save_checkpoint

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_PATH = MODELS_DIR / "unified.pt"

CHAT, READ, GAME, END = "<|chat|>", "<|read|>", "<|game|>", "<|end|>"
_SPECIALS = [CHAT, READ, GAME, END]

# ~30M params: bigger than the 14M chat model (6L/384d).
PRESET = dict(vocab_size=10000, block_size=256, n_layer=8, n_head=8, n_embd=512)

# How many task docs to mint. Chat (SODA) dominates; oversample the smaller
# tasks so the model actually learns them.
_READER_DOCS = 250_000
_GAME_FRAMES_PER = 40_000
_BPE_SAMPLE = 60_000


def _fmt_chat(utterances: list[str]) -> str:
    lines = [f"{'AB'[i % 2]}: {u}" for i, u in enumerate(utterances)]
    return f"{CHAT}\n" + "\n".join(lines) + f"\n{END}\n"


def _chat_docs(split: str) -> Iterator[str]:
    from .data import dailydialog_dialogues, nps_dialogues, soda_dialogues

    for utts in soda_dialogues("train" if split == "train" else "validation"):
        yield _fmt_chat(utts)
    if split == "train":
        for utts in dailydialog_dialogues("train"):
            yield _fmt_chat(utts)
        for utts in nps_dialogues():
            yield _fmt_chat(utts)


def _reader_docs(n: int, seed: int) -> Iterator[str]:
    from .reader import _example

    rng = random.Random(seed)
    for _ in range(n):
        text, _ = _example(rng, allow_holdout=True)
        yield f"{READ} {text.strip()} {END}\n"


def _game_docs(n_per_game: int, seed: int) -> Iterator[str]:
    from .games import GAMES

    rng = random.Random(seed)
    for name, cls in GAMES.items():
        made = 0
        while made < n_per_game:
            g = cls(seed=rng.randrange(1 << 30))
            while not g.done and made < n_per_game:
                action = g.expert()
                yield f"{GAME} {name}\n{g.render()}\nmove: {action} {END}\n"
                nxt = (rng.choice(g.safe_actions() or [action])
                       if rng.random() < 0.15 else action)
                g.step(nxt)
                made += 1


def _documents(split: str, seed: int) -> Iterator[str]:
    yield from _chat_docs(split)
    if split == "train":
        yield from _reader_docs(_READER_DOCS, seed)
        yield from _game_docs(_GAME_FRAMES_PER, seed + 1)
    else:  # small val set for the extra tasks
        yield from _reader_docs(2000, seed + 7)
        yield from _game_docs(500, seed + 8)


def _write_tokens(tokenizer, docs: Iterator[str], path: Path, log) -> int:
    total, n, started = 0, 0, time.time()
    with open(path, "wb") as f:
        while True:
            chunk = list(islice(docs, 2000))
            if not chunk:
                break
            flat = [i for ids in tokenizer.encode_batch(chunk) for i in ids]
            np.asarray(flat, dtype=np.uint16).tofile(f)
            total += len(flat)
            n += len(chunk)
            if n % 100_000 < 2000:
                log(f"  {n:,} docs, {total:,} tokens ({time.time() - started:.0f}s)")
    return total


def prepare(cache: Path, seed: int, log) -> tuple[np.ndarray, np.ndarray, BPETokenizer]:
    cache.mkdir(parents=True, exist_ok=True)
    tok_path, meta_path = cache / "unified-tok.json", cache / "unified-meta.json"
    bins = {s: cache / f"unified-{s}.bin" for s in ("train", "val")}
    if all(p.exists() for p in (tok_path, meta_path, *bins.values())):
        from .model import tokenizer_from_payload

        tok = tokenizer_from_payload(json.loads(tok_path.read_text()))
        meta = json.loads(meta_path.read_text())
        log(f"reusing tokenized cache ({meta['train']:,} train tokens)")
    else:
        log("training BPE tokenizer on a sample of the mixture...")
        sample = list(islice(_documents("train", seed), _BPE_SAMPLE))
        random.Random(seed).shuffle(sample)
        tok = BPETokenizer.train(sample, PRESET["vocab_size"], special_tokens=_SPECIALS)
        tok_path.write_text(json.dumps(tok.to_payload()))
        meta = {}
        for split, path in bins.items():
            log(f"tokenizing {split} -> {path.name}")
            meta[split] = _write_tokens(tok, _documents(split, seed), path, log)
        meta_path.write_text(json.dumps(meta))
    train = np.memmap(bins["train"], dtype=np.uint16, mode="r")
    val = np.memmap(bins["val"], dtype=np.uint16, mode="r")
    return train, val, tok


def _batch(data, block, bs, device):
    ix = np.random.randint(0, len(data) - block - 1, size=bs)
    x = np.stack([data[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + block + 1] for i in ix]).astype(np.int64)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":
        return xt.pin_memory().to(device, non_blocking=True), yt.pin_memory().to(device, non_blocking=True)
    return xt.to(device), yt.to(device)


@torch.no_grad()
def _val_loss(model, data, block, device, iters=40):
    model.eval()
    losses = [model(*_batch(data, block, 16, device))[1].item() for _ in range(iters)]
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


def train(out=DEFAULT_PATH, steps=24000, batch_size=24, lr=3e-4, device=None,
          seed=1337, log=print) -> Path:
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data, val_data, tok = prepare(MODELS_DIR, seed, log)

    cfg = GPTConfig(vocab_size=len(tok), block_size=PRESET["block_size"],
                    n_layer=PRESET["n_layer"], n_head=PRESET["n_head"],
                    n_embd=PRESET["n_embd"], dropout=0.0)
    model = MiniGPT(cfg).to(device)
    seen = steps * batch_size * cfg.block_size
    log(f"data: {len(train_data):,} train tokens | model: {model.num_params():,} "
        f"params | device: {device}")
    log(f"schedule: {steps:,} steps x {batch_size} x {cfg.block_size} = "
        f"{seen / 1e6:.0f}M tokens (~{seen / max(len(train_data), 1):.1f} epochs)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    warmup = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, max(warmup, 1)),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1), eta_min=lr * 0.1)],
        milestones=[max(warmup, 1)])

    model.train()
    started, best = time.time(), float("inf")
    for step in range(1, steps + 1):
        x, y = _batch(train_data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % 500 == 0 or step == steps:
            vl = _val_loss(model, val_data, cfg.block_size, device)
            mark = ""
            if vl < best:
                best, mark = vl, " <- saved"
                save_checkpoint(out, model, tok, step, vl)
            el = time.time() - started
            log(f"step {step:>6}/{steps} | train {loss.item():.3f} | val {vl:.3f} | "
                f"{el / 60:.0f}m, eta {el / step * (steps - step) / 60:.0f}m{mark}")
    log(f"done — best val {best:.3f}, saved to {out}")
    return out


# ------------------------------------------------------------------ inference


class UnifiedLM:
    """One model, three jobs, selected by the task prefix."""

    def __init__(self, path=DEFAULT_PATH, device=None):
        from .model import load_checkpoint

        device = device or "cpu"
        if device == "cpu":
            torch.set_num_threads(1)
        self.model, self.tok = load_checkpoint(path, device)
        self.device = device
        self._nl = self.tok.encode("\n")[0]
        self._end = self.tok.token_id(END)

    def _gen(self, prompt, max_new, temperature, stop):
        ids = self.tok.encode(prompt)
        idx = torch.tensor([ids[-self.model.cfg.block_size:]], dtype=torch.long,
                           device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new, temperature=temperature,
                                  top_k=40, stop_tokens=stop)
        return self.tok.decode(out[0][idx.shape[1]:].tolist())

    def chat(self, history: list[str], message: str, temperature=0.8) -> str:
        turns = [*history, message]
        lines = [f"{'AB'[(len(turns) - 1 - i) % 2]}: {t}"
                 for i, t in enumerate(turns)]
        prompt = f"{CHAT}\n" + "\n".join(lines) + "\nB:"
        return self._gen(prompt, 48, temperature, [self._nl, self._end]).strip()

    def read(self, fields: dict, question: str) -> str:
        from .reader import state_block

        prompt = f"{READ} {state_block(fields)} | q: {question} | a:"
        return self._gen(prompt, 40, 0.3, [self._nl, self._end]).strip()

    def move(self, game_name: str, board: str, legal: list[str]) -> str:
        prompt = f"{GAME} {game_name}\n{board}\nmove:"
        out = self._gen(prompt, 4, 0.4, [self._nl, self._end]).strip()
        return next((a for a in legal if out.startswith(a)), out.split()[0] if out else legal[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Train/inspect the unified model.")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=24000)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--device", default=None)
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, device=a.device)


if __name__ == "__main__":
    main()
