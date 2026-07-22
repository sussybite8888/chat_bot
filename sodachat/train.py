"""Train the from-scratch GPT on a dialogue dataset.

    python -m sodachat.train [--dataset soda|dailydialog|nps] [--steps N]

Dialogues are rendered as tagged turns (see data.py), tokenized once into a
flat uint16 file under models/, and trained on as random crops of that token
stream. Keeping tokens on disk rather than in RAM is what makes the ~200M-token
SODA corpus trainable on modest hardware.

The default (soda) is ~200M tokens for a ~14M-param model — roughly the
Chinchilla-optimal ratio. DailyDialog alone is ~1.5M tokens, which leaves a
model this size badly under-fed: fluent, but with nothing to say.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator

# Cap the MPS allocator (read when it first initializes) so a memory-hungry
# run raises an OOM error instead of swap-freezing the whole machine. The low
# watermark must stay <= the high one (its default is 1.4). Harmless on CUDA.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import torch

from .data import (
    DIALOG_SEP,
    dailydialog_dialogues,
    format_dialogue,
    nps_dialogues,
    soda_dialogues,
)
from .blocks import BPETokenizer, CharTokenizer, GPTConfig, make_amp, pick_device
from .model import MiniGPT, default_model_path, save_checkpoint

# Model size and schedule per dataset. Dropout only earns its keep when the
# model sees the data many times; at ~1 epoch over SODA there is nothing to
# memorize, so it is off there.
_DATASET_PRESETS: dict[str, dict] = {
    # ~205M tokens seen ≈ 1 epoch ≈ 15 tokens/param — near Chinchilla-optimal.
    # Wants a real GPU (~6h, ~3GB VRAM at bs=32); pass --batch-size 16 on a
    # memory-tight machine.
    "soda": {
        "tokenizer": "bpe", "vocab_size": 8000,
        "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.0,
        "steps": 25000, "batch_size": 32, "bpe_sample": 60000,
    },
    "dailydialog": {
        "tokenizer": "bpe", "vocab_size": 8000,
        "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.2,
        "steps": 4000, "batch_size": 16, "bpe_sample": None,
    },
    "nps": {
        "tokenizer": "char",
        "n_layer": 4, "n_head": 4, "n_embd": 192, "dropout": 0.1,
        "steps": 2500, "batch_size": 64, "bpe_sample": None,
    },
}

_ENCODE_CHUNK = 2000


def _dialogues(dataset: str, split: str) -> Iterator[list[str]]:
    """Yield dialogues (lists of alternating utterances) for a split."""
    if dataset == "soda":
        return soda_dialogues("train" if split == "train" else "validation")
    if dataset == "dailydialog":
        return iter(dailydialog_dialogues("train" if split == "train" else "validation"))
    if dataset == "nps":
        all_d = nps_dialogues()
        cut = max(1, int(0.9 * len(all_d)))
        return iter(all_d[:cut] if split == "train" else all_d[cut:])
    raise ValueError(f"unknown dataset {dataset!r} (expected soda, dailydialog or nps)")


def _chunks(it: Iterable, size: int) -> Iterator[list]:
    it = iter(it)
    while chunk := list(islice(it, size)):
        yield chunk


def _write_tokens(tokenizer, dialogues: Iterator[list[str]], path: Path, log) -> int:
    """Tokenize dialogues into a flat uint16 file. Returns the token count."""
    total, dialogue_count, started = 0, 0, time.time()
    with open(path, "wb") as f:
        for batch in _chunks(dialogues, _ENCODE_CHUNK):
            texts = [format_dialogue(d) for d in batch]
            if hasattr(tokenizer, "encode_batch"):
                encoded = tokenizer.encode_batch(texts)
            else:
                encoded = [tokenizer.encode(t) for t in texts]
            flat = [i for ids in encoded for i in ids]
            np.asarray(flat, dtype=np.uint16).tofile(f)
            total += len(flat)
            dialogue_count += len(batch)
            if dialogue_count % 100_000 < _ENCODE_CHUNK:
                log(
                    f"  tokenized {dialogue_count:,} dialogues "
                    f"({total:,} tokens, {time.time() - started:.0f}s)"
                )
    return total


def prepare_data(
    dataset: str, preset: dict, cache_dir: Path, log: Callable[[str], None] = print
) -> tuple[np.ndarray, np.ndarray, CharTokenizer | BPETokenizer]:
    """Tokenize the dataset to disk once, then memory-map it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok_path = cache_dir / f"{dataset}-tokenizer.json"
    meta_path = cache_dir / f"{dataset}-meta.json"
    bins = {s: cache_dir / f"{dataset}-{s}.bin" for s in ("train", "val")}

    if tok_path.exists() and meta_path.exists() and all(p.exists() for p in bins.values()):
        meta = json.loads(meta_path.read_text())
        tokenizer = _load_tokenizer(tok_path, meta)
        log(f"reusing tokenized cache ({meta['train_tokens']:,} train tokens)")
    else:
        if preset["tokenizer"] == "bpe":
            sample_n = preset.get("bpe_sample")
            log(
                "training BPE vocabulary"
                + (f" on {sample_n:,} sampled dialogues..." if sample_n else "...")
            )
            sample = (
                format_dialogue(d)
                for d in islice(_dialogues(dataset, "train"), sample_n)
            )
            tokenizer = BPETokenizer.train(
                sample, preset["vocab_size"], special_tokens=[DIALOG_SEP]
            )
        else:
            text = "".join(format_dialogue(d) for d in _dialogues(dataset, "train"))
            tokenizer = CharTokenizer(sorted(set(text)))

        meta = {"tokenizer": preset["tokenizer"]}
        for split, path in bins.items():
            log(f"tokenizing {split} split -> {path.name}")
            meta[f"{split}_tokens"] = _write_tokens(
                tokenizer, _dialogues(dataset, split), path, log
            )
        _save_tokenizer(tok_path, tokenizer, meta)
        meta_path.write_text(json.dumps(meta))

    train = np.memmap(bins["train"], dtype=np.uint16, mode="r")
    val = np.memmap(bins["val"], dtype=np.uint16, mode="r")
    return train, val, tokenizer


def _save_tokenizer(path: Path, tokenizer, meta: dict) -> None:
    path.write_text(json.dumps(tokenizer.to_payload()))


def _load_tokenizer(path: Path, meta: dict):
    from .blocks import tokenizer_from_payload

    return tokenizer_from_payload(json.loads(path.read_text()))


def _batch(
    data: np.ndarray, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":  # overlap the host->device copy with compute
        return xt.pin_memory().to(device, non_blocking=True), yt.pin_memory().to(
            device, non_blocking=True
        )
    return xt.to(device), yt.to(device)


@torch.no_grad()
def _eval_loss(
    model: MiniGPT, data: np.ndarray, block_size: int, device: str, iters: int = 40
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = _batch(data, block_size, 16, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


def train_model(
    dataset: str = "soda",
    out_path: Path | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    lr: float = 3e-4,
    device: str | None = None,
    seed: int = 1337,
    log: Callable[[str], None] = print,
) -> Path:
    preset = _DATASET_PRESETS.get(dataset)
    if preset is None:
        raise ValueError(f"unknown dataset {dataset!r}")
    steps = steps or preset["steps"]
    batch_size = batch_size or preset["batch_size"]
    out_path = Path(out_path) if out_path else default_model_path(dataset)

    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, val_data, tokenizer = prepare_data(
        dataset, preset, out_path.parent, log
    )

    cfg = GPTConfig(
        vocab_size=len(tokenizer),
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        n_embd=preset["n_embd"],
        dropout=preset["dropout"],
    )
    model = MiniGPT(cfg).to(device)
    seen = steps * batch_size * cfg.block_size
    log(
        f"data: {len(train_data):,} train / {len(val_data):,} val tokens "
        f"(vocab {len(tokenizer)}) | model: {model.num_params():,} params | "
        f"device: {device}"
    )
    log(
        f"schedule: {steps:,} steps x {batch_size} x {cfg.block_size} = "
        f"{seen/1e6:.0f}M tokens seen (~{seen/max(len(train_data),1):.1f} epochs) | "
        f"{len(train_data)/model.num_params():.1f} tokens/param"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    warmup = min(500, steps // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=max(warmup, 1)
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(steps - warmup, 1), eta_min=lr * 0.1
            ),
        ],
        milestones=[max(warmup, 1)],
    )

    eval_every = max(250, steps // 40)
    amp = make_amp(device)
    model.train()
    started = time.time()
    best_val = float("inf")
    for step in range(1, steps + 1):
        x, y = _batch(train_data, cfg.block_size, batch_size, device)
        with amp.autocast():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(optimizer, model)
        sched.step()

        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

        if step % eval_every == 0 or step == steps:
            val_loss = _eval_loss(model, val_data, cfg.block_size, device)
            marker = ""
            if val_loss < best_val:  # keep only the best-generalizing weights
                best_val = val_loss
                save_checkpoint(out_path, model, tokenizer, step, val_loss)
                marker = " <- saved"
            elapsed = time.time() - started
            eta = elapsed / step * (steps - step)
            log(
                f"step {step:>6}/{steps} | train {loss.item():.3f} | "
                f"val {val_loss:.3f} | {elapsed/60:.0f}m elapsed, {eta/60:.0f}m left"
                f"{marker}"
            )

    log(f"done — best checkpoint at {out_path} (val loss {best_val:.3f})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mini chat GPT.")
    parser.add_argument(
        "--dataset", choices=list(_DATASET_PRESETS), default="soda"
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    train_model(
        dataset=args.dataset,
        out_path=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
