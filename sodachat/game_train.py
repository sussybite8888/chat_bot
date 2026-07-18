"""Train the tiny GPT to play a game by behaviour cloning.

    python -m sodachat.game_train --game snake|pong|dodge [--steps 4000]

Each training example is a serialized board (see games/core.py); the model
predicts the single action token that follows. This is supervised action
prediction — we read the logits at the final position and cross-entropy them
against the expert's move — so no capacity is wasted reconstructing the board.
The model is ~1M params and trains in a few minutes on any device.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Cap the MPS allocator so an oversized run fails cleanly instead of freezing
# the machine (harmless on CUDA/CPU). See train.py for the full story.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import time

import numpy as np
import torch
import torch.nn.functional as F

from .games import GAMES, GamePlayer, evaluate_headless, generate_dataset
from .model import GPTConfig, MiniGPT, pick_device, save_checkpoint

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def game_model_path(game: str) -> Path:
    return MODELS_DIR / f"{game}-gpt.pt"


def _accuracy(model, X, y, device, batch=2048) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch].astype(np.int64)).to(device)
            pred = model(xb)[0][:, -1, :].argmax(-1).cpu().numpy()
            correct += int((pred == y[i : i + batch]).sum())
    model.train()
    return correct / len(X)


def train_game_model(
    game: str = "snake",
    n_games: int = 8000,
    steps: int = 4000,
    batch_size: int = 256,
    lr: float = 3e-4,
    epsilon: float = 0.15,
    max_frames: int = 200_000,
    out_path: Path | None = None,
    device: str | None = None,
    seed: int = 0,
    log=print,
) -> Path:
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r} (have: {', '.join(GAMES)})")
    game_cls = GAMES[game]
    out_path = Path(out_path) if out_path else game_model_path(game)
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    log(f"[{game}] generating expert games...")
    X, y, tok = generate_dataset(
        game_cls, n_games, epsilon=epsilon, seed=seed, max_frames=max_frames, log=log
    )

    n_val = max(2000, len(X) // 20)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    Xtr, ytr, Xval, yval = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]

    cfg = GPTConfig(
        vocab_size=len(tok),
        block_size=tok.state_len,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
    )
    model = MiniGPT(cfg).to(device)
    log(f"[{game}] model {model.num_params():,} params | "
        f"{len(Xtr):,} train / {len(Xval):,} val frames | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    started = time.time()
    best_acc = 0.0
    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(Xtr), size=batch_size)
        xb = torch.from_numpy(Xtr[idx].astype(np.int64)).to(device)
        target = torch.from_numpy(ytr[idx].astype(np.int64)).to(device)
        loss = F.cross_entropy(model(xb)[0][:, -1, :], target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

        if step % 500 == 0 or step == steps:
            acc = _accuracy(model, Xval, yval, device)
            marker = ""
            if acc >= best_acc:
                best_acc = acc
                save_checkpoint(out_path, model, tok, step, 1.0 - acc)
                marker = " <- saved"
            log(f"[{game}] step {step:>5}/{steps} | loss {loss.item():.3f} | "
                f"val acc {acc:.3f} | {time.time() - started:.0f}s{marker}")

    # Real metric: play the game and compare to the expert.
    model.eval()
    stats = evaluate_headless(game_cls, GamePlayer(model, tok, device), n_games=50)
    log(f"[{game}] done — action acc {best_acc:.3f} | played avg score "
        f"{stats['avg_score']:.1f} (max {stats['max_score']}) | "
        f"{stats['avg_ms']:.2f} ms/move ({stats['moves_per_sec']:,.0f} moves/s)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Train a game-playing GPT.")
    p.add_argument("--game", choices=list(GAMES), default="snake")
    p.add_argument("--games", type=int, default=8000, dest="n_games")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--max-frames", type=int, default=200_000, dest="max_frames")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train_game_model(
        game=a.game, n_games=a.n_games, steps=a.steps, batch_size=a.batch_size,
        lr=a.lr, epsilon=a.epsilon, max_frames=a.max_frames, out_path=a.out,
        device=a.device, seed=a.seed,
    )


if __name__ == "__main__":
    main()
