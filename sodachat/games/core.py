"""A general framework for making the tiny GPT control games of any kind.

The unifying idea is that everything is a sequence of tokens. A game exposes an
*observation* and the model predicts an *action* token. Two observation
modalities are supported, so games need not be bitmap/grid based:

- "grid"  — a 2D array of cell symbols (Snake, Pong, Dodge). Encoded one token
            per cell by `GridTokenizer`.
- "text"  — a plain-text description of the state (Tic-Tac-Toe, and anything
            symbolic: card games, text adventures, dialogue games...). Encoded
            character by character by `TextTokenizer`.

Because the text modality is just characters, the same representation the chat
model uses, one agent can both converse and play (see ../agent.py). Add a game
by subclassing `Game`, setting MODALITY and the action list, and providing a
scripted `expert` for the training data — the pipeline does the rest.
"""

from __future__ import annotations

import random
import string
import time
from pathlib import Path

import numpy as np

GAMES: dict[str, type["Game"]] = {}


def register(cls: type["Game"]) -> type["Game"]:
    GAMES[cls.NAME] = cls
    return cls


class Game:
    """Interface every controllable game implements."""

    NAME: str = "base"
    MODALITY: str = "grid"  # "grid" or "text"
    ACTIONS: tuple[str, ...] = ()
    GLYPHS: dict[int, tuple[str, str]] = {}
    # Model-facing glyphs: one ASCII *byte* per cell value, for the text board an
    # LM reads (see `model_board`). GLYPHS above are for humans — box-drawing and
    # block characters that look good in a terminal but are 2-3 bytes each, so a
    # 20x20 board tokenizes to ~800 subwords and blows past the block size; worse,
    # snake's head and body share one glyph (they differ only by colour, which the
    # text board drops), so the LM can't even see where its head is. ASCII bytes
    # fix both: ~1 token/cell and a distinct char per value. Convention across
    # games so instruction-following transfers: '.' empty, '@' the controlled
    # entity, '#' obstacle, '*' the objective.
    MODEL_GLYPHS: dict[int, str] = {}
    # Whether the unified/expert training corpora include this game. Set False
    # for held-out probe games (see sandbox.py) that must stay unseen so they
    # measure zero-shot transfer, not memorization.
    TRAINABLE: bool = True

    # grid modality
    SIZE: tuple[int, int] = (10, 10)
    N_CELLS: int = 1
    AUX_DIMS: tuple[int, ...] = ()
    # text modality
    MAX_LEN: int = 96

    score: int
    done: bool

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    @property
    def rows(self) -> int:
        return self.SIZE[0]

    @property
    def cols(self) -> int:
        return self.SIZE[1]

    def reset(self) -> "Game":
        raise NotImplementedError

    def observe(self):
        """(grid, aux) for grid games, or a str for text games."""
        raise NotImplementedError

    def legal(self, action: str) -> bool:
        return True

    def safe_actions(self) -> list[str]:
        return [a for a in self.ACTIONS if self.legal(a)]

    def step(self, action: str) -> None:
        raise NotImplementedError

    def expert(self) -> str:
        raise NotImplementedError

    def render(self) -> str:
        """Human-readable board for the terminal. Grid games draw their glyphs;
        text games override with something nicer."""
        if self.MODALITY == "grid":
            grid, _ = self.observe()
            return "\n".join(
                "".join(self.GLYPHS.get(int(v), ("?", ""))[0] for v in row)
                for row in grid
            )
        return str(self.observe())

    def model_board(self) -> str:
        """The board an LM reads (expert/unified movers): compact single-byte
        ASCII per cell (MODEL_GLYPHS), ~8x fewer tokens than `render()` and with
        every cell value distinct. Falls back to `render()` for text games (whose
        board is already a compact character stream) or if MODEL_GLYPHS is unset."""
        if self.MODALITY == "grid" and self.MODEL_GLYPHS:
            grid, _ = self.observe()
            return "\n".join(
                "".join(self.MODEL_GLYPHS.get(int(v), "?") for v in row)
                for row in grid
            )
        return self.render()

    def facts(self) -> dict[str, str]:
        """Queryable facts about the current state, for grounded Q&A. Games
        override to add specifics (food location, whose turn, ...)."""
        return {"score": str(self.score), "over": "yes" if self.done else "no"}

    def describe(self) -> str:
        """One-line factual summary of the state."""
        f = self.facts()
        bits = [f"score {f['score']}"]
        for k, v in f.items():
            if k not in ("score", "over"):
                bits.append(f"{k.replace('_', ' ')} {v}")
        return ", ".join(bits) + (" — game over" if self.done else "")


# --------------------------------------------------------------- half-block draw

# Upper/lower half-block glyphs. A single character cell can show two vertical
# "pixels": the upper half is painted in the foreground colour, the lower half
# in the background colour. Drawing a board this way packs two grid rows into
# one terminal line, so a 20-row board fits a 24-row terminal — the plain
# one-glyph-per-cell view is twice as tall and overflows. Only the live
# human-facing displays use this; the model still reads Game.render()'s
# full-height text, which must stay unchanged.
HALF_UPPER, HALF_LOWER = "▀", "▄"  # ▀ ▄


def half_block_board(rows: int, cols: int, colour):
    """Render a rows×cols board into a `rich.text.Text` at half the height.

    `colour(r, c)` returns the rich colour of cell (r, c), or None for an empty
    cell (drawn blank, so the board keeps its dark backdrop). Rows are taken two
    at a time: the upper cell becomes the half-block's foreground, the lower its
    background. Colour alone distinguishes pieces here, since the per-cell shape
    glyphs (●, ▲, …) collapse to solid half-cells."""
    from rich.text import Text

    out = Text()
    for r in range(0, rows, 2):
        for c in range(cols):
            top = colour(r, c)
            bot = colour(r + 1, c) if r + 1 < rows else None
            if top and bot:
                out.append(HALF_UPPER, style=f"{top} on {bot}")
            elif top:
                out.append(HALF_UPPER, style=top)
            elif bot:
                out.append(HALF_LOWER, style=bot)
            else:
                out.append(" ")
        if r + 2 < rows:
            out.append("\n")
    return out


def half_block_render(game):
    """Half-block board for a grid `game`, coloured straight from its GLYPHS
    (empty cell value 0 -> blank). A drop-in, half-height replacement for the
    live human display; the model keeps reading full-height `game.render()`."""
    grid, _ = game.observe()

    def colour(r, c):
        v = int(grid[r][c])
        return None if v == 0 else game.GLYPHS.get(v, ("?", "white"))[1]

    return half_block_board(game.rows, game.cols, colour)


# Wall colour for the play-field frame — bright enough to read as a boundary
# without competing with the pieces inside it.
BOUNDARY_STYLE = "grey58"


def framed_board(board, cols: int, style: str = BOUNDARY_STYLE):
    """Wrap a half-block `board` (a `rich.text.Text`, `cols` characters wide) in
    a tight box that marks the play-field boundary — the walls a piece dies
    against. Once empty cells draw blank, the field's edges are otherwise
    invisible; this puts them back and highlights them."""
    from rich import box
    from rich.panel import Panel

    return Panel(board, box=box.SQUARE, border_style=style, padding=0,
                 width=cols + 2, expand=False)


# ------------------------------------------------------------------ tokenizers


class GridTokenizer:
    """One token per cell, plus structure/aux/action tokens."""

    kind = "grid"

    def __init__(self, n_cells, actions, aux_dims=(), size=(0, 0), game=None):
        self.n_cells = int(n_cells)
        self.actions = tuple(actions)
        self.aux_dims = tuple(int(d) for d in aux_dims)
        self.size = (int(size[0]), int(size[1]))
        self.game = game
        self.ROW_SEP = self.n_cells
        self.SEP = self.n_cells + 1
        self._aux_base = self.n_cells + 2
        self._act_base = self._aux_base + sum(self.aux_dims)
        self.vocab_size = self._act_base + len(self.actions)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def state_len(self) -> int:
        rows, cols = self.size
        return rows * (cols + 1) + len(self.aux_dims) + 1

    def encode(self, obs) -> list[int]:
        grid, aux = obs
        ids: list[int] = []
        for row in grid:
            ids.extend(int(v) for v in row)
            ids.append(self.ROW_SEP)
        off = self._aux_base
        for dim, val in zip(self.aux_dims, aux):
            ids.append(off + int(val))
            off += dim
        ids.append(self.SEP)
        return ids

    def action_id(self, action: str) -> int:
        return self._act_base + self.actions.index(action)

    def action_token_ids(self) -> list[int]:
        return [self._act_base + i for i in range(len(self.actions))]

    def id_to_action(self, token_id: int) -> str:
        return self.actions[token_id - self._act_base]

    def to_payload(self) -> dict:
        return {
            "type": "grid",
            "n_cells": self.n_cells,
            "actions": list(self.actions),
            "aux_dims": list(self.aux_dims),
            "size": list(self.size),
            "game": self.game,
        }

    @classmethod
    def from_payload(cls, p: dict) -> "GridTokenizer":
        return cls(p["n_cells"], p["actions"], p["aux_dims"], p["size"], p.get("game"))


class TextTokenizer:
    """Character-level encoding of a text observation, padded to a fixed length
    so the model input (and thus per-move latency) stays constant."""

    kind = "text"
    DEFAULT_CHARSET = string.digits + string.ascii_letters + " .,:;|=/+-_#?!*()\n"

    def __init__(self, actions, max_len=96, charset=None, game=None):
        self.actions = tuple(actions)
        self.max_len = int(max_len)
        self.charset = charset or self.DEFAULT_CHARSET
        self._stoi = {c: i for i, c in enumerate(self.charset)}
        self.PAD = len(self.charset)  # also stands in for unknown characters
        self._act_base = self.PAD + 1
        self.vocab_size = self._act_base + len(self.actions)
        self.game = game

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def state_len(self) -> int:
        return self.max_len

    def encode(self, obs) -> list[int]:
        text = obs if isinstance(obs, str) else str(obs)
        ids = [self._stoi.get(c, self.PAD) for c in text[: self.max_len]]
        ids += [self.PAD] * (self.max_len - len(ids))
        return ids

    def action_id(self, action: str) -> int:
        return self._act_base + self.actions.index(action)

    def action_token_ids(self) -> list[int]:
        return [self._act_base + i for i in range(len(self.actions))]

    def id_to_action(self, token_id: int) -> str:
        return self.actions[token_id - self._act_base]

    def to_payload(self) -> dict:
        return {
            "type": "text",
            "actions": list(self.actions),
            "max_len": self.max_len,
            "charset": self.charset,
            "game": self.game,
        }

    @classmethod
    def from_payload(cls, p: dict) -> "TextTokenizer":
        return cls(p["actions"], p["max_len"], p["charset"], p.get("game"))


def tokenizer_for(game_cls: type[Game]):
    if game_cls.MODALITY == "text":
        return TextTokenizer(game_cls.ACTIONS, game_cls.MAX_LEN, game=game_cls.NAME)
    return GridTokenizer(
        game_cls.N_CELLS, game_cls.ACTIONS, game_cls.AUX_DIMS,
        game_cls.SIZE, game=game_cls.NAME,
    )


def tokenizer_from_payload(p: dict):
    return (TextTokenizer if p["type"] == "text" else GridTokenizer).from_payload(p)


# --------------------------------------------------------------- data / models


def generate_dataset(
    game_cls: type[Game],
    n_games: int,
    epsilon: float = 0.15,
    seed: int = 0,
    max_frames: int | None = None,
    log=print,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Behaviour-cloning data: record the expert's action at each state, but
    sometimes execute a random safe move (DAgger-style) for off-policy coverage."""
    tok = tokenizer_for(game_cls)
    rng = random.Random(seed)
    states: list[list[int]] = []
    actions: list[int] = []
    scores: list[int] = []

    for gi in range(n_games):
        game = game_cls(seed=rng.randrange(1 << 30))
        while not game.done:
            states.append(tok.encode(game.observe()))
            label = game.expert()
            actions.append(tok.action_id(label))
            if epsilon and rng.random() < epsilon:
                game.step(rng.choice(game.safe_actions() or [label]))
            else:
                game.step(label)
            if max_frames and len(states) >= max_frames:
                break
        scores.append(game.score)
        if max_frames and len(states) >= max_frames:
            break
        if (gi + 1) % 1000 == 0:
            log(f"  {gi + 1} games, {len(states):,} frames, "
                f"expert avg {np.mean(scores[-1000:]):.1f}")

    X = np.asarray(states, dtype=np.int16)
    y = np.asarray(actions, dtype=np.int16)
    log(f"generated {len(X):,} frames from {len(scores)} games "
        f"(expert avg score {np.mean(scores):.1f})")
    return X, y, tok


def load_model(path: Path, device: str | None = None):
    """Load a game-controller checkpoint -> (model, tokenizer, game_name)."""
    import torch

    from ..model import MiniGPT, config_from_payload, pick_device

    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MiniGPT(config_from_payload(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tok = tokenizer_from_payload(ckpt["tokenizer"])
    return model, tok, tok.game


class GamePlayer:
    """Runs a trained model as a real-time controller, tuned for *consistent*
    per-move latency (see the timing notes below).

    - **Warmup** moves kernel-compilation / allocator priming out of the loop.
    - **A reused input buffer** avoids per-move allocation.
    - **Single-threaded CPU** by default: for a ~1M-param model that is faster
      and far steadier than a thread pool or a GPU's async dispatch.
    - **inference_mode + a device sync** so a timed move reflects real
      completion, not deferred work.
    - Illegal actions are **masked**, so the model only ever emits a legal move.
    """

    def __init__(self, model, tok, device, warmup=32, single_thread=True):
        import torch

        if device == "cpu" and single_thread:
            torch.set_num_threads(1)
        self.model = model
        self.tok = tok
        self.device = device
        self._action_ids = tok.action_token_ids()
        self._buf = torch.zeros((1, tok.state_len), dtype=torch.long, device=device)
        self.last_ms = 0.0
        self._warmup(warmup)

    def _sync(self) -> None:
        import torch

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def _warmup(self, n: int) -> None:
        import torch

        for _ in range(max(n, 1)):
            with torch.inference_mode():
                self.model(self._buf)
        self._sync()

    def act(self, game: Game) -> str:
        import torch

        t0 = time.perf_counter()
        ids = self.tok.encode(game.observe())
        self._buf[0].copy_(torch.as_tensor(ids, dtype=torch.long))
        with torch.inference_mode():
            logits = self.model(self._buf)[0][0, -1]
        # Choose the highest-scoring *legal* action.
        legal = [
            i for i, a in enumerate(self.tok.actions) if game.legal(a)
        ] or list(range(len(self.tok.actions)))
        act_logits = logits[[self._action_ids[i] for i in legal]]
        choice = legal[int(act_logits.argmax())]
        self._sync()
        self.last_ms = (time.perf_counter() - t0) * 1000
        return self.tok.actions[choice]


def benchmark_latency(game_cls, player: GamePlayer, moves: int = 4000) -> dict:
    """Per-move latency distribution — the real jitter test. Consistency is
    p99/max staying close to the mean, the headroom a real-time loop needs.

    Garbage collection is paused, matching the play loop — the cyclic
    collector is the main source of rare multi-ms spikes, and refcounting
    still frees the (acyclic) game objects, so it is safe to disable."""
    import gc

    lat: list[float] = []
    game = game_cls(seed=0)
    gc.disable()
    try:
        while len(lat) < moves:
            if game.done:
                game = game_cls(seed=len(lat))
            game.step(player.act(game))
            lat.append(player.last_ms)
    finally:
        gc.enable()
    a = np.asarray(lat)
    return {
        "moves": len(a), "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
        "std": float(a.std()),
    }


def evaluate_headless(game_cls, player: GamePlayer, n_games=50, seed=999) -> dict:
    scores, latencies = [], []
    for i in range(n_games):
        game = game_cls(seed=seed + i)
        while not game.done:
            game.step(player.act(game))
            latencies.append(player.last_ms)
        scores.append(game.score)
    return {
        "avg_score": float(np.mean(scores)), "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)), "avg_ms": float(np.mean(latencies)),
        "moves_per_sec": 1000.0 / max(np.mean(latencies), 1e-6),
    }
