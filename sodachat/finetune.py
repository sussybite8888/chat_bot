"""Fine-tune a pretrained GPT-2-family model on a dialogue dataset.

    python -m sodachat.finetune [--dataset dailydialog|nps] [--epochs 3]

Dialogues are flattened to one utterance per line (an <|endoftext|> token
between dialogues) and the model learns to continue the conversation. The
checkpoint with the best validation loss is kept.

Datasets:
- dailydialog (default): ~13k clean two-party dialogues (~5.4M chars) from
  Hugging Face — teaches actual turn-taking.
- nps: the NPS Chat corpus (~170K chars of 2006 chat-room posts) — small and
  noisy; kept for flavor experiments.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Callable

# Cap the MPS allocator (read when it first initializes) so a memory-hungry
# run raises an OOM error instead of swap-freezing the whole machine. The low
# watermark must stay <= the high one (its default is 1.4).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import torch

from .blocks import make_amp
from .data import dailydialog_texts, nps_texts
from .hf_model import DEFAULT_HF_MODEL_DIR
from .model import pick_device

_MODELS_DIR = DEFAULT_HF_MODEL_DIR.parent


def _build_blocks(tokenizer, texts: list[str], block_size: int) -> list[list[int]]:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        ids.append(tokenizer.eos_token_id)
    return [
        ids[i : i + block_size]
        for i in range(0, len(ids) - block_size + 1, block_size)
    ]


@torch.no_grad()
def _eval_loss(model, blocks: list[list[int]], batch_size: int, device: str) -> float:
    model.eval()
    losses = []
    for i in range(0, len(blocks), batch_size):
        x = torch.tensor(blocks[i : i + batch_size], device=device)
        losses.append(model(input_ids=x, labels=x).loss.item())
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


def finetune(
    dataset: str = "dailydialog",
    base_model: str = "gpt2",
    out_dir: Path | None = None,
    epochs: int = 3,
    block_size: int = 256,
    batch_size: int = 1,
    grad_accum: int = 8,
    gradient_checkpointing: bool = True,
    lr: float = 5e-5,
    device: str | None = None,
    seed: int = 1337,
    log: Callable[[str], None] = print,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out_dir = Path(out_dir) if out_dir else _MODELS_DIR / f"{base_model}-{dataset}"

    log(f"loading base model {base_model!r} (downloads on first use)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)

    log(f"loading dataset {dataset!r}...")
    if dataset == "dailydialog":
        # DailyDialog ships with held-out dialogues; use them for validation.
        train_blocks = _build_blocks(tokenizer, dailydialog_texts("train"), block_size)
        val_blocks = _build_blocks(tokenizer, dailydialog_texts("validation"), block_size)
    elif dataset == "nps":
        blocks = _build_blocks(tokenizer, nps_texts(), block_size)
        rng.shuffle(blocks)
        split = max(1, int(0.1 * len(blocks)))
        val_blocks, train_blocks = blocks[:split], blocks[split:]
    else:
        raise ValueError(f"unknown dataset {dataset!r} (expected dailydialog or nps)")
    # A fixed subset is plenty to pick the best epoch, and keeps evals quick.
    if len(val_blocks) > 128:
        val_blocks = val_blocks[:128]
    log(
        f"{len(train_blocks)} train / {len(val_blocks)} val blocks of "
        f"{block_size} tokens | device: {device} | "
        f"batch {batch_size} x accum {grad_accum}"
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    amp = make_amp(device)

    best_val = float("inf")
    started = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_blocks)
        total, batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(train_blocks) - batch_size + 1, batch_size):
            x = torch.tensor(train_blocks[i : i + batch_size], device=device)
            with amp.autocast():
                loss = model(input_ids=x, labels=x).loss
            amp.backward(loss / grad_accum)
            total += loss.item()
            batches += 1
            if batches % grad_accum == 0:
                amp.step(optimizer, model)
                optimizer.zero_grad(set_to_none=True)
            if device == "mps" and batches % 100 == 0:
                torch.mps.empty_cache()
            if batches % 400 == 0:
                log(
                    f"  epoch {epoch} step {batches} | "
                    f"train {total / batches:.3f} | {time.time() - started:.0f}s"
                )

        val_loss = _eval_loss(model, val_blocks, batch_size, device)
        marker = ""
        # Small datasets overfit within a few epochs; keep only the best.
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            marker = " <- saved"
        log(
            f"epoch {epoch}/{epochs} | train {total / max(batches, 1):.3f} | "
            f"val {val_loss:.3f} | {time.time() - started:.0f}s{marker}"
        )

    log(f"done — best checkpoint at {out_dir} (val loss {best_val:.3f})")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained GPT-2-family model on a dialogue dataset."
    )
    parser.add_argument("--dataset", choices=["dailydialog", "nps"], default="dailydialog")
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="faster, but uses much more memory",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default=None, help="mps, cuda, or cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    finetune(
        dataset=args.dataset,
        base_model=args.base_model,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
