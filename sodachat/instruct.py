"""Instruction-conditioned game control — a VLA-style post-training.

A vision-language-action model reads an observation *and* a language
instruction, then acts. Here the observation is the game board (our "vision",
rendered as text), the instruction is a natural-language goal, and the action
is a move token. The point is language conditioning: the *same* board produces
*different* moves depending on the instruction —

    <|act|> snake | goal: eat the food
    <board>
    move: up

    <|act|> snake | goal: go to the top left corner
    <board>            (same board)
    move: left

The supervision comes from **goal-conditioned experts** — several scripted
policies per game, one per goal — so "eat the food", "survive", "go up", and
"reach a corner" each have a correct action from the same state.

This trains as a *post-training* pass: it pad-loads the unified model's weights
(model.pad_load) into a model with one extra token (`<|act|>`), then continues
training on instruction data mixed with a replay of the base tasks. See
`post_train`.
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from itertools import islice
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

from .games.snake import SnakeGame

ACT = "<|act|>"
_MODELS = Path(__file__).resolve().parent.parent / "models"
BASE_PATH = _MODELS / "unified.pt"
OUT_PATH = _MODELS / "unified-instruct.pt"


# ------------------------------------------------------------- goal experts


def _safe(game: SnakeGame):
    """(action, new_head, reachable_free_cells) for each non-fatal move."""
    out = []
    for a in game.ACTIONS:
        head, eating = game._result(a)
        if head is None:
            continue
        body = deque(game.snake)
        body.appendleft(head)
        if not eating:
            body.pop()
        out.append((a, head, game._reachable(head, set(body))))
    return out


def _bfs_dist(game: SnakeGame, start, target) -> int:
    block = set(game.snake)
    if len(game.snake) > 1:
        block.discard(game.snake[-1])
    seen = {start}
    dq = deque([(start, 0)])
    while dq:
        (r, c), d = dq.popleft()
        if (r, c) == target:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if (0 <= nb[0] < game.rows and 0 <= nb[1] < game.cols
                    and nb not in block and nb not in seen):
                seen.add(nb)
                dq.append((nb, d + 1))
    return 1 << 20


def _toward(game: SnakeGame, target) -> str:
    moves = _safe(game)
    if not moves:
        return game.heading
    roomy = [m for m in moves if m[2] >= len(game.snake)] or moves
    return min(roomy, key=lambda m: (_bfs_dist(game, m[1], target), -m[2]))[0]


def _go_dir(game: SnakeGame, direction: str) -> str:
    if game.legal(direction) and game._result(direction)[0] is not None:
        return direction
    moves = _safe(game)
    return moves[0][0] if moves else game.heading


def _survive(game: SnakeGame) -> str:
    moves = _safe(game)
    return max(moves, key=lambda m: m[2])[0] if moves else game.heading


def _corners(game: SnakeGame):
    h, w = game.rows - 1, game.cols - 1
    return {"the top left corner": (0, 0), "the top right corner": (0, w),
            "the bottom left corner": (h, 0), "the bottom right corner": (h, w),
            "the center": (game.rows // 2, game.cols // 2)}


# goal-key -> (action function, [instruction phrasings])
def _goals(game: SnakeGame) -> dict:
    goals = {
        "food": (game.expert,
                 ["eat the food", "get the apple", "go for the food",
                  "chase the food", "find the food", "grab the apple"]),
        "survive": (lambda: _survive(game),
                    ["survive", "stay alive", "don't die", "keep yourself alive",
                     "avoid trapping yourself", "last as long as you can"]),
    }
    for d in ("up", "down", "left", "right"):
        goals[f"go {d}"] = (lambda d=d: _go_dir(game, d),
                            [f"go {d}", f"move {d}", f"head {d}"])
    for name, cell in _corners(game).items():
        goals[name] = (lambda cell=cell: _toward(game, cell),
                       [f"go to {name}", f"reach {name}", f"head to {name}"])
    return goals


# ------------------------------------------------------------------- data


def instruction_docs(n: int, seed: int = 0):
    """Yield `<|act|> snake | goal: <instruction>\\n<board>\\nmove: <action>` docs.

    States come from playing the base expert (realistic boards); at each state a
    random goal is drawn and labelled with that goal's expert action."""
    rng = random.Random(seed)
    made = 0
    while made < n:
        game = SnakeGame(seed=rng.randrange(1 << 30))
        while not game.done and made < n:
            goals = _goals(game)
            key = rng.choice(list(goals))
            fn, phrasings = goals[key]
            action = fn()
            instr = rng.choice(phrasings)
            yield (f"{ACT} snake | goal: {instr}\n{game.render()}\n"
                   f"move: {action} <|end|>\n")
            made += 1
            # advance with the base objective so states stay realistic
            game.step(game.expert() if rng.random() > 0.15
                      else rng.choice(game.safe_actions() or [action]))


def demo_goal_conditioning(seed: int = 3) -> list[tuple[str, str]]:
    """Same board, several goals -> the expert action for each. Shows that the
    instruction changes the move (the property the model must learn)."""
    game = SnakeGame(seed=seed)
    for _ in range(4):  # a few steps in so the board is interesting
        game.step(game.expert())
    goals = _goals(game)
    keys = ["food", "go up", "go down", "go left", "go right",
            "the top left corner", "the bottom right corner", "survive"]
    return [(k, goals[k][0]()) for k in keys if k in goals]


# ------------------------------------------------------- post-training


def _game_play_docs(n_total: int, seed: int):
    """Base `<|game|>` play data (all games) — teaches the model to actually
    read a board and move, which the instruction task builds on."""
    from .unified import _game_docs
    from .games import GAMES

    yield from _game_docs(max(n_total // len(GAMES), 1), seed)


def _chat_replay(n: int):
    from .unified import _chat_docs

    yield from islice(_chat_docs("train"), n)


def _reader_replay(n: int, seed: int):
    from .unified import _reader_docs

    yield from _reader_docs(n, seed)


def _tokenize(tok, docs, log) -> "np.ndarray":
    import numpy as np

    ids: list[int] = []
    started, n = time.time(), 0
    while True:
        batch = list(islice(docs, 2000))
        if not batch:
            break
        for e in tok.encode_batch(batch):
            ids.extend(e)
        n += len(batch)
        if n % 100_000 < 2000:
            log(f"  tokenized {n:,} docs ({len(ids):,} tokens, {time.time()-started:.0f}s)")
    return np.asarray(ids, dtype=np.uint16)


def post_train(base=BASE_PATH, out=OUT_PATH, n_instruct=150_000, n_game=160_000,
               n_reader=60_000, n_chat=90_000, steps=6000, batch_size=24, lr=1e-4,
               replay=True, device=None, seed=0, log=print) -> Path:
    """Warm-start from the unified model (pad_load) and continue-train on a
    game-heavy corpus: base gameplay (teaches board->move) + instruction-
    conditioned data (the VLA task), plus a chat/reader replay so those aren't
    forgotten. Games dominate here (~2/3 of the docs) precisely because the base
    model can barely play — this is the "games training" the instructions need."""
    import numpy as np
    import torch

    from .blocks import GPTConfig, make_amp, pad_load, pick_device, tokenizer_from_payload
    from .model import MiniGPT

    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ck = torch.load(base, map_location=device, weights_only=True)
    tok = tokenizer_from_payload(ck["tokenizer"])
    old_v = ck["config"]["vocab_size"]
    new_v = tok.add_special([ACT])
    cfg = GPTConfig(**{**ck["config"], "vocab_size": new_v})
    model = MiniGPT(cfg).to(device)
    stats = pad_load(model, ck["state_dict"])
    log(f"pad-loaded base: {stats} | vocab {old_v} -> {new_v} | "
        f"{model.num_params():,} params | device {device}")

    def corpus():
        yield from _game_play_docs(n_game, seed)        # play (all games) — heavy
        yield from instruction_docs(n_instruct, seed)   # instruction-conditioned
        if replay:                                      # keep chat + reading
            yield from _reader_replay(n_reader, seed + 2)
            yield from _chat_replay(n_chat)

    log(f"corpus mix: {n_game:,} play + {n_instruct:,} instruction"
        + (f" + {n_reader:,} reader + {n_chat:,} chat replay" if replay else ""))
    log("tokenizing corpus...")
    data = _tokenize(tok, corpus(), log)
    block = cfg.block_size
    log(f"corpus: {len(data):,} tokens")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    amp = make_amp(device)
    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        ix = np.random.randint(0, len(data) - block - 1, size=batch_size)
        xb = torch.from_numpy(np.stack([data[i:i+block] for i in ix]).astype(np.int64)).to(device)
        yb = torch.from_numpy(np.stack([data[i+1:i+block+1] for i in ix]).astype(np.int64)).to(device)
        with amp.autocast():
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % 500 == 0 or step == steps:
            log(f"step {step:>5}/{steps} | loss {loss.item():.3f} | "
                f"{time.time()-started:.0f}s")

    from .model import save_checkpoint

    save_checkpoint(out, model, tok, steps, loss.item())
    log(f"saved instruction model to {out}")
    return out


class InstructPlayer:
    """Instruction-conditioned control: (game, instruction) -> move."""

    def __init__(self, path=OUT_PATH, device=None):
        import torch

        from .model import load_checkpoint

        device = device or "cpu"
        if device == "cpu":
            torch.set_num_threads(1)
        self.model, self.tok = load_checkpoint(path, device)
        self.device = device
        self._nl = self.tok.encode("\n")[0]
        self._end = self.tok.token_id("<|end|>")

    def act(self, game, instruction: str) -> str:
        import torch

        legal = [a for a in game.ACTIONS if game.legal(a)]
        prompt = f"{ACT} {game.NAME} | goal: {instruction}\n{game.render()}\nmove:"
        ids = self.tok.encode(prompt)
        idx = torch.tensor([ids[-self.model.cfg.block_size:]], device=self.device)
        out = self.model.generate(idx, max_new_tokens=4, temperature=0.3, top_k=20,
                                  stop_tokens=[self._nl, self._end])
        text = self.tok.decode(out[0][idx.shape[1]:].tolist()).strip()
        return next((a for a in legal if text.startswith(a)),
                    text.split()[0] if text else legal[0])


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Instruction-conditioned game post-training.")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("post-train")
    tr.add_argument("--steps", type=int, default=5000)
    tr.add_argument("--no-replay", action="store_true")
    tr.add_argument("--device", default=None)
    a = p.parse_args()
    if a.cmd == "post-train":
        post_train(steps=a.steps, replay=not a.no_replay, device=a.device)


if __name__ == "__main__":
    main()
