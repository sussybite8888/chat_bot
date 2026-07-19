"""Play a game and talk about it at the same time, from one model.

A `MultiHeadGPT` has a shared trunk and two heads. Given a board rendered as
text, a single forward pass produces:

- an **action** (from the action head) — the move to play;
- a **commentary** (from the LM head) — words about what it's doing.

Both outputs come from the same model and the same encoding of the state, so
the bot controls the game and narrates it simultaneously on two different
outputs. Train and watch:

    python -m sodachat.narrate train         # ~a few minutes on CPU
    python -m sodachat.narrate play          # watch it play + narrate

The commentary is trained from templated narration derived from the game
state; the action head is trained from the same scripted expert used
everywhere else. Losses are summed over one sequence, so the trunk learns a
representation that serves both.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import GPTConfig, pick_device
from .games.snake import BODY, EMPTY, FOOD, HEAD, SnakeGame
from .model import MiniGPT

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "snake-narrator.pt"
_GLYPH = {EMPTY: ".", BODY: "o", HEAD: "@", FOOD: "*"}


# ---------------------------------------------------- the narrator model


class MultiHeadGPT(MiniGPT):
    """One shared transformer trunk, two output heads producing different things
    from the same forward pass:

    - the **LM head** (inherited, tied to the embeddings) predicts text tokens —
      used to generate a running commentary or a chat reply;
    - the **action head** predicts a game action from the trunk's hidden state.

    So a single pass over a board-as-text observation yields *both* the move to
    play (action head) and the words to say about it (LM head). The trunk is
    shared, so the two tasks inform one representation.
    """

    def __init__(self, cfg: GPTConfig, n_actions: int):
        super().__init__(cfg)
        self.n_actions = n_actions
        self.action_head = nn.Linear(cfg.n_embd, n_actions)
        nn.init.normal_(self.action_head.weight, std=0.02)
        nn.init.zeros_(self.action_head.bias)

    def _trunk(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_for(idx.shape[1], x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ln_f(x)

    def forward_both(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lm_logits [B,T,V], action_logits [B,T,A]) — one trunk pass."""
        x = self._trunk(idx)
        return self.head(x), self.action_head(x)


# ---------------------------------------------------------------- board + words


def board_text(game: SnakeGame) -> str:
    grid, _ = game.observe()
    return "\n".join("".join(_GLYPH[int(v)] for v in row) for row in grid)


def snake_commentary(game: SnakeGame, action: str, rng) -> str:
    """Templated narration of the current state and chosen move (training
    target for the LM head). Lowercase, short, varied."""
    hr, hc = game.head
    fr, fc = game.food
    vy = "up" if fr < hr else ("down" if fr > hr else "")
    vx = "left" if fc < hc else ("right" if fc > hc else "")
    where = " and ".join(w for w in (vy, vx) if w) or "right here"
    n = len(game.snake)
    options = [
        f"apple's {where}, going {action}",
        f"heading {action} for the food",
        f"chasing it {where}",
        f"food is {where}, turning {action}",
        f"length {n} now, moving {action}",
        f"going {action}",
    ]
    if n >= 12:
        options.append(f"getting long, careful, {action}")
    return rng.choice(options)


# ------------------------------------------------------------------- tokenizer


class NarrateTokenizer:
    _CHARS = "\n .,!'?-:0123456789abcdefghijklmnopqrstuvwxyz@o*"

    def __init__(self, actions, max_len=176, charset=None):
        seen: list[str] = []
        for c in charset or self._CHARS:
            if c not in seen:
                seen.append(c)
        self.charset = "".join(seen)
        self.actions = tuple(actions)
        self.max_len = int(max_len)
        self._stoi = {c: i for i, c in enumerate(self.charset)}
        self._space = self._stoi.get(" ", 0)
        C = len(self.charset)
        self.PAD, self.SEP, self.EOS = C, C + 1, C + 2
        self._act_base = C + 3
        self.vocab_size = self._act_base + len(self.actions)

    def encode_chars(self, s: str) -> list[int]:
        return [self._stoi.get(c, self._space) for c in s]

    def decode_chars(self, ids) -> str:
        return "".join(self.charset[i] for i in ids if i < len(self.charset))

    def text_token_ids(self) -> list[int]:
        """Ids the LM head may emit as commentary: characters + EOS."""
        return list(range(len(self.charset))) + [self.EOS]

    def action_index(self, action: str) -> int:
        return self.actions.index(action)

    def to_payload(self) -> dict:
        return {"charset": self.charset, "actions": list(self.actions),
                "max_len": self.max_len}

    @classmethod
    def from_payload(cls, p: dict) -> "NarrateTokenizer":
        return cls(p["actions"], p["max_len"], p["charset"])


# --------------------------------------------------------------------- data


def build_sample(tok: NarrateTokenizer, board: str, action: str, comment: str):
    """One training row: input ids, action label + position, LM targets (-100
    where no text should be predicted)."""
    board_ids = tok.encode_chars(board)
    comm_ids = tok.encode_chars(comment)
    ids = board_ids + [tok.SEP] + comm_ids + [tok.EOS]
    ids = ids[: tok.max_len]
    sep_pos = min(len(board_ids), tok.max_len - 1)

    x = np.full(tok.max_len, tok.PAD, dtype=np.int16)
    x[: len(ids)] = ids
    lm = np.full(tok.max_len, -100, dtype=np.int64)
    # From SEP onward, predict the next token (commentary chars, then EOS).
    for p in range(sep_pos, min(sep_pos + len(comm_ids) + 1, len(ids) - 1) + 1):
        if p + 1 < len(ids):
            lm[p] = ids[p + 1]
    return x, tok.action_index(action), sep_pos, lm


def generate_dataset(n_frames: int, seed: int = 0, log=print):
    import random

    rng = random.Random(seed)
    tok = NarrateTokenizer(SnakeGame.ACTIONS)
    X, A, P, Y = [], [], [], []
    while len(X) < n_frames:
        game = SnakeGame(seed=rng.randrange(1 << 30))
        while not game.done and len(X) < n_frames:
            action = game.expert()
            comment = snake_commentary(game, action, rng)
            x, a, p, lm = build_sample(tok, board_text(game), action, comment)
            X.append(x); A.append(a); P.append(p); Y.append(lm)
            game.step(action)
        if len(X) % 20000 < 400:
            log(f"  {len(X):,} frames")
    return (np.stack(X), np.asarray(A, np.int64),
            np.asarray(P, np.int64), np.stack(Y), tok)


# --------------------------------------------------------------------- train


def train(path: Path = DEFAULT_PATH, n_frames=80000, steps=3500,
          batch_size=128, lr=3e-4, device=None, seed=0, log=print) -> Path:
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    log("generating expert games + commentary...")
    X, A, P, Y, tok = generate_dataset(n_frames, seed=seed, log=log)
    log(f"{len(X):,} frames | vocab {tok.vocab_size} | device {device}")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=tok.max_len,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = MultiHeadGPT(cfg, len(tok.actions)).to(device)
    log(f"MultiHeadGPT {model.num_params():,} params (shared trunk + 2 heads)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(X), size=batch_size)
        xb = torch.from_numpy(X[idx].astype(np.int64)).to(device)
        ab = torch.from_numpy(A[idx]).to(device)
        pb = torch.from_numpy(P[idx]).to(device)
        yb = torch.from_numpy(Y[idx]).to(device)

        lm_logits, act_logits = model.forward_both(xb)
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)), yb.view(-1), ignore_index=-100
        )
        act_at_sep = act_logits[torch.arange(batch_size, device=device), pb]
        act_loss = F.cross_entropy(act_at_sep, ab)
        loss = act_loss + lm_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

        if step % 500 == 0 or step == steps:
            acc = (act_at_sep.argmax(-1) == ab).float().mean().item()
            log(f"step {step:>5}/{steps} | act_loss {act_loss.item():.3f} "
                f"(acc {acc:.3f}) | lm_loss {lm_loss.item():.3f} | "
                f"{time.time() - started:.0f}s")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(cfg), "n_actions": len(tok.actions),
                "tokenizer": tok.to_payload(), "state_dict": model.state_dict()}, path)
    log(f"saved player+narrator to {path}")
    return path


# --------------------------------------------------------------------- play


class NarratingPlayer:
    """Both outputs from one model: the action head chooses the move, the LM
    head speaks — from a single trunk pass over the board."""

    def __init__(self, path: Path = DEFAULT_PATH, device=None):
        device = device or "cpu"
        if device == "cpu":
            torch.set_num_threads(1)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.tok = NarrateTokenizer.from_payload(ckpt["tokenizer"])
        self.model = MultiHeadGPT(GPTConfig(**ckpt["config"]), ckpt["n_actions"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self._text_ids = torch.tensor(self.tok.text_token_ids(), device=device)
        self.last_ms = 0.0

    @torch.inference_mode()
    def act_and_say(self, game: SnakeGame, temperature=0.7, max_words=40):
        t0 = time.perf_counter()
        ids = self.tok.encode_chars(board_text(game)) + [self.tok.SEP]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)

        lm_logits, act_logits = self.model.forward_both(x)
        # action head — highest-scoring legal move
        legal = [i for i, a in enumerate(self.tok.actions) if game.legal(a)]
        legal = legal or list(range(len(self.tok.actions)))
        a_logits = act_logits[0, -1]
        action = self.tok.actions[legal[int(a_logits[legal].argmax())]]

        # LM head — continue the same sequence into commentary
        comment_ids: list[int] = []
        cur = x
        step_lm = lm_logits
        for _ in range(max_words):
            logits = step_lm[0, -1] / temperature
            mask = torch.full_like(logits, float("-inf"))
            mask[self._text_ids] = logits[self._text_ids]
            nxt = int(torch.multinomial(F.softmax(mask, dim=-1), 1))
            if nxt == self.tok.EOS:
                break
            comment_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], device=self.device)], 1)
            step_lm, _ = self.model.forward_both(cur)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return action, self.tok.decode_chars(comment_ids).strip()


def play(path: Path = DEFAULT_PATH, fps=6.0, n_games=3, device="cpu", seed=0):
    import gc

    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    if not Path(path).exists():
        raise SystemExit(f"no narrator model at {path}. Train it:\n"
                         f"  python -m sodachat.narrate train")
    player = NarratingPlayer(path, device)
    period = 1.0 / fps if fps > 0 else 0.0

    def render(game, action, words):
        grid, _ = game.observe()
        body = Text()
        styles = {EMPTY: "grey30", BODY: "green", HEAD: "bright_green", FOOD: "red"}
        for r, row in enumerate(grid):
            for v in row:
                body.append(_GLYPH[int(v)] + " ", style=styles[int(v)])
            body.append("\n" if r < len(grid) - 1 else "")
        bubble = Text(f'\n💬 "{words}"', style="italic cyan")
        hud = Text(f"\nscore {game.score}   move: {action}   "
                   f"[both heads: {player.last_ms:.0f} ms]", style="dim")
        return Panel(Group(body, bubble, hud), title="sodachat plays & narrates",
                     border_style="cyan", width=40)

    best = 0
    gc.disable()
    try:
        with Live(console=console, refresh_per_second=max(fps, 4)) as live:
            for gi in range(1, n_games + 1):
                g = SnakeGame(seed=seed + gi)
                deadline = time.perf_counter()
                action, words = "", ""
                while not g.done:
                    action, words = player.act_and_say(g)
                    g.step(action)
                    live.update(render(g, action, words))
                    if period:
                        deadline += period
                        slack = deadline - time.perf_counter()
                        time.sleep(slack) if slack > 0 else None
                best = max(best, g.score)
                live.update(render(g, action, words))
                time.sleep(1.0)
    finally:
        gc.enable()
    console.print(f"[bold]best score: [green]{best}[/]")


def main() -> None:
    p = argparse.ArgumentParser(description="Snake player + narrator (multi-head).")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--frames", type=int, default=80000)
    tr.add_argument("--steps", type=int, default=3500)
    tr.add_argument("--device", default=None)
    pl = sub.add_parser("play")
    pl.add_argument("--fps", type=float, default=6.0)
    pl.add_argument("--games", type=int, default=3)
    pl.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.cmd == "train":
        train(n_frames=a.frames, steps=a.steps, device=a.device)
    else:
        play(fps=getattr(a, "fps", 6.0), n_games=getattr(a, "games", 3),
             device=getattr(a, "device", "cpu"))


if __name__ == "__main__":
    main()
