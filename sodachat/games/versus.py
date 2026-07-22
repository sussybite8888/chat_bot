"""Snake, for two — you versus the bot, on one shared board.

Multiplayer snake. Two snakes share a 20x20 grid and race for the *same* food.
You steer one with WASD / arrow keys; the bot steers the other with the very
same trained snake model it uses to play solo. The trick is that no new model is
needed: each tick the multiplayer board is drawn into the *single-snake* view
the model already understands — the bot's own snake as head+body, and your snake
as ordinary body cells, i.e. one more wall to avoid. So the model transfers
straight from solo play to a live opponent it has never seen.

    python -m sodachat.games.versus            # play in your terminal
    # or, inside the agent:  /duel

Rules: run into a wall, into yourself, or into either snake's body and you're
out; head-on, the longer snake survives (a tie takes both). Last snake alive
wins — or, if you both fall on the same tick, whoever ate more. Food grows you
by one and respawns elsewhere.
"""

from __future__ import annotations

import random
import time
from collections import deque

import numpy as np

from .snake import BODY, FOOD, HEAD, SnakeGame, _DELTA, _OPPOSITE

SIZE = SnakeGame.SIZE
ACTIONS = SnakeGame.ACTIONS

# Per-snake (body_style, head_style): snake 0 is you (cyan), snake 1 the bot.
_SNAKE_STYLES = (("cyan", "bright_cyan bold"), ("green", "bright_green bold"))
_FOOD_GLYPH = ("●", "red")
_EMPTY_GLYPH = ("·", "grey30")


class _Snake:
    """One player's snake: a deque of (row, col) with the head at the left."""

    def __init__(self, cells, heading: str):
        self.body: deque[tuple[int, int]] = deque(cells)
        self.heading = heading
        self.alive = True
        self.score = 0
        self.since_food = 0

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    def __len__(self) -> int:
        return len(self.body)


class SnakeDuel:
    """Two snakes, one board. Both move simultaneously each `step`; deaths are
    resolved together so a real head-on can take both. Reuses Snake's constants
    and geometry so the trained solo model drops straight in (see `ego_view`)."""

    SIZE = SIZE
    ACTIONS = ACTIONS

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    @property
    def rows(self) -> int:
        return self.SIZE[0]

    @property
    def cols(self) -> int:
        return self.SIZE[1]

    def reset(self) -> "SnakeDuel":
        mid = self.rows // 2
        # Facing away from each other so the opening move is never a forced clash.
        self.snakes = [
            _Snake([(mid - 2, 2)], "right"),
            _Snake([(mid + 2, self.cols - 3)], "left"),
        ]
        self.food: tuple[int, int] | None = None
        self.done = False
        self.winner: int | None = None  # snake index, or None for a draw / ongoing
        self.ticks = 0
        self._place_food()
        return self

    # ---------------------------------------------------------------- geometry

    def _occupied(self) -> set[tuple[int, int]]:
        occ: set[tuple[int, int]] = set()
        for s in self.snakes:
            if s.alive:
                occ |= set(s.body)
        return occ

    def _place_food(self) -> None:
        free = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self._occupied()
        ]
        if free:
            self.food = self.rng.choice(free)
        else:
            self.food = None
            self.done = True

    # -------------------------------------------------------------------- step

    def step(self, actions: dict[int, str]) -> None:
        """Advance one tick. `actions` maps snake index -> chosen action; a
        missing (or illegal-reversal) action means "keep going straight"."""
        if self.done:
            return
        self.ticks += 1
        alive = [i for i, s in enumerate(self.snakes) if s.alive]

        # 1. Resolve each snake's heading and its new head cell.
        new_head: dict[int, tuple[int, int]] = {}
        for i in alive:
            s = self.snakes[i]
            a = actions.get(i) or s.heading
            if a not in _DELTA or (len(s) > 1 and a == _OPPOSITE[s.heading]):
                a = s.heading
            s.heading = a
            dr, dc = _DELTA[a]
            new_head[i] = (s.head[0] + dr, s.head[1] + dc)

        eats = {i: new_head[i] == self.food for i in alive}

        # 2. Cells that stay occupied: every body, minus the tail each non-eating
        #    snake vacates as it slides forward (mirrors Snake._result).
        blockers: set[tuple[int, int]] = set()
        for i in alive:
            s = self.snakes[i]
            cells = set(s.body)
            if not eats[i] and len(s) > 0:
                cells.discard(s.body[-1])
            blockers |= cells

        # 3. Wall and body collisions.
        dead: set[int] = set()
        for i in alive:
            r, c = new_head[i]
            if not (0 <= r < self.rows and 0 <= c < self.cols) or new_head[i] in blockers:
                dead.add(i)

        # 4. Head-on: two live heads to the same cell — longer wins, tie kills both.
        contenders = [i for i in alive if i not in dead]
        for a_i in contenders:
            for b_i in contenders:
                if a_i < b_i and new_head[a_i] == new_head[b_i]:
                    la, lb = len(self.snakes[a_i]), len(self.snakes[b_i])
                    if la >= lb:
                        dead.add(b_i)
                    if lb >= la:
                        dead.add(a_i)

        # 5. Apply movement for the survivors.
        ate = False
        for i in alive:
            s = self.snakes[i]
            if i in dead:
                s.alive = False
                continue
            s.body.appendleft(new_head[i])
            if eats[i]:
                s.score += 1
                s.since_food = 0
                ate = True
            else:
                s.body.pop()
                s.since_food += 1
        if ate:
            self._place_food()

        # 6. End conditions: at most one snake left, or a long stalemate.
        still = [s for s in self.snakes if s.alive]
        if len(still) <= 1 or all(
            s.since_food > 3 * self.rows * self.cols for s in still
        ):
            self.done = True
            self._decide_winner()

    def _decide_winner(self) -> None:
        alive = [i for i, s in enumerate(self.snakes) if s.alive]
        if len(alive) == 1:
            self.winner = alive[0]
            return
        best = max(s.score for s in self.snakes)
        leaders = [i for i, s in enumerate(self.snakes) if s.score == best]
        self.winner = leaders[0] if len(leaders) == 1 else None

    # ------------------------------------------------- the bot's single-snake view

    def ego_observation(self, idx: int) -> tuple[np.ndarray, tuple[int, ...]]:
        """The board as snake `idx` sees it, in the exact format the solo model
        was trained on: own head is HEAD, every other body cell (yours included)
        is BODY, food is FOOD, aux is the heading. That reuse is the whole point —
        the opponent simply looks like more wall."""
        g = np.zeros(self.SIZE, dtype=np.int8)
        for s in self.snakes:
            if not s.alive:
                continue
            for cell in s.body:
                g[cell] = BODY
        me = self.snakes[idx]
        g[me.head] = HEAD  # drawn last, so my head wins its cell
        if self.food is not None:
            g[self.food] = FOOD
        return g, (ACTIONS.index(me.heading),)

    def ego_legal(self, idx: int, action: str) -> bool:
        s = self.snakes[idx]
        return not (len(s) > 1 and action == _OPPOSITE[s.heading])

    def ego_view(self, idx: int) -> "_EgoView":
        return _EgoView(self, idx)


class _EgoView:
    """Adapts one snake's egocentric board to the `Game`-shaped surface every
    mover expects (GamePlayer reads `observe`/`legal`; the unified/expert/instruct
    movers read `render`/`NAME`/`ACTIONS`/`legal`). So `agent._player_for("snake")`
    drives the bot's snake unchanged, whatever model the agent is currently in."""

    NAME = "snake"
    MODALITY = "grid"
    ACTIONS = ACTIONS
    GLYPHS = SnakeGame.GLYPHS
    SIZE = SIZE

    def __init__(self, duel: SnakeDuel, idx: int):
        self._duel = duel
        self._idx = idx

    @property
    def rows(self) -> int:
        return self.SIZE[0]

    @property
    def cols(self) -> int:
        return self.SIZE[1]

    def observe(self):
        return self._duel.ego_observation(self._idx)

    def legal(self, action: str) -> bool:
        return self._duel.ego_legal(self._idx, action)

    def render(self) -> str:
        grid, _ = self.observe()
        return "\n".join(
            "".join(self.GLYPHS.get(int(v), ("?", ""))[0] for v in row) for row in grid
        )


# ------------------------------------------------------------------- the bot


class DuelBot:
    """Steers the bot's snake. Prefers a trained mover (the agent's current snake
    model, or the solo specialist checkpoint); falls back to a scripted greedy so
    a duel still works before anything is trained.

    A second snake is out of the solo model's training distribution, so it will
    sometimes drive straight into the opponent — a move it would never make on an
    empty solo board. The codebase already masks *illegal* moves so "the model
    only ever emits a legal move"; here we extend that to *suicidal* ones: keep
    the model's strategy, but veto a move that dies this tick when a survivable
    move exists (falling back to the room-aware greedy to choose the escape)."""

    def __init__(self, mover=None):
        self.mover = mover

    def move(self, duel: SnakeDuel, idx: int) -> str:
        if self.mover is None:
            return _greedy(duel, idx)
        action = self.mover.act(duel.ego_view(idx))
        if _survivable(duel, idx, action):
            return action
        return _greedy(duel, idx)  # model would crash; steer clear if we can


def build_bot(agent=None, device: str = "cpu") -> DuelBot:
    mover = None
    if agent is not None:
        try:
            mover = agent._player_for("snake")
        except Exception:
            mover = None
    if mover is None:
        from ..game_train import game_model_path
        from .core import GamePlayer, load_model

        path = game_model_path("snake")
        if path.exists():
            model, tok, _ = load_model(path, device)
            mover = GamePlayer(model, tok, device)
    return DuelBot(mover)


def _survivable(duel: SnakeDuel, idx: int, action: str) -> bool:
    """Would `idx` still be alive after taking `action` this tick? Checks walls
    and bodies the way `SnakeDuel.step` does (each non-eating snake vacates its
    tail); the opponent is treated as static — good enough for a one-ply veto."""
    s = duel.snakes[idx]
    if not duel.ego_legal(idx, action):
        return False
    dr, dc = _DELTA[action]
    nh = (s.head[0] + dr, s.head[1] + dc)
    if not (0 <= nh[0] < duel.rows and 0 <= nh[1] < duel.cols):
        return False
    eating = nh == duel.food
    blockers: set[tuple[int, int]] = set()
    for j, o in enumerate(duel.snakes):
        if not o.alive:
            continue
        cells = set(o.body)
        if j == idx and not eating:
            cells.discard(o.body[-1])
        blockers |= cells
    return nh not in blockers


def _greedy(duel: SnakeDuel, idx: int) -> str:
    """Room-aware greedy on the egocentric board — the same idea as Snake.expert,
    but every occupied cell (both snakes) is treated as a static wall."""
    grid, _ = duel.ego_observation(idx)
    me = duel.snakes[idx]
    rows, cols, food = duel.rows, duel.cols, duel.food
    blocked = {(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == BODY}

    def inb(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols

    def room(start):
        seen, dq, n = {start}, deque([start]), 0
        while dq:
            r, c = dq.popleft()
            for dr, dc in _DELTA.values():
                nb = (r + dr, c + dc)
                if inb(nb) and nb not in blocked and nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
                    n += 1
        return n

    def food_dist(start):
        if food is None:
            return 1 << 20
        seen, dq = {start}, deque([(start, 0)])
        while dq:
            (r, c), d = dq.popleft()
            if (r, c) == food:
                return d
            for dr, dc in _DELTA.values():
                nb = (r + dr, c + dc)
                if inb(nb) and nb not in blocked and nb not in seen:
                    seen.add(nb)
                    dq.append((nb, d + 1))
        return 1 << 20

    cand = []
    for a in ACTIONS:
        if not duel.ego_legal(idx, a):
            continue
        dr, dc = _DELTA[a]
        nb = (me.head[0] + dr, me.head[1] + dc)
        if inb(nb) and nb not in blocked:
            cand.append((a, nb, room(nb)))
    if not cand:
        return me.heading
    roomy = [c for c in cand if c[2] >= len(me)] or cand
    return min(roomy, key=lambda c: (food_dist(c[1]), -c[2]))[0]


# --------------------------------------------------------- interactive terminal


class _KeyReader:
    """Non-blocking single-key reader for a POSIX terminal (cbreak mode), so
    steering never waits on Enter. Maps WASD, hjkl and the arrow keys to
    directions; 'q' or a lone Esc means quit. A no-op where there is no tty."""

    def __init__(self):
        import sys

        self.fd = sys.stdin.fileno()
        self._old = None

    def __enter__(self):
        import termios
        import tty

        self._old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        import termios

        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)

    def poll(self) -> str | None:
        import os
        import select

        if not select.select([self.fd], [], [], 0)[0]:
            return None
        return self._decode(os.read(self.fd, 32))

    @staticmethod
    def _decode(data: bytes) -> str | None:
        last = None
        i = 0
        _wasd = {"w": "up", "k": "up", "s": "down", "j": "down",
                 "a": "left", "h": "left", "d": "right", "l": "right"}
        _arrow = {65: "up", 66: "down", 67: "right", 68: "left"}
        while i < len(data):
            b = data[i]
            if b == 0x1B:  # escape — an arrow sequence, or a bare Esc to quit
                if data[i + 1 : i + 2] == b"[" and i + 2 < len(data):
                    last = _arrow.get(data[i + 2], last)
                    i += 3
                    continue
                return "quit"
            ch = chr(b).lower()
            if ch == "q":
                return "quit"
            if ch in _wasd:
                last = _wasd[ch]
            i += 1
        return last


def _panel(duel: SnakeDuel, note: str = ""):
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from .core import framed_board, half_block_board

    heads = [s.head if s.alive else None for s in duel.snakes]
    bodies = [set(s.body) if s.alive else set() for s in duel.snakes]

    def colour(r, c):
        cell = (r, c)
        if duel.food == cell:
            return _FOOD_GLYPH[1]
        for i in range(len(duel.snakes)):
            if heads[i] == cell:
                return _SNAKE_STYLES[i][1]  # head
            if cell in bodies[i]:
                return _SNAKE_STYLES[i][0]  # body
        return None  # empty — blank backdrop

    # Half-block board: two rows per line, so the 20-row duel fits the terminal.
    # Framed so the walls (which end a snake) are visible around the empty field.
    board = framed_board(half_block_board(duel.rows, duel.cols, colour), duel.cols)

    hud = Table.grid(padding=(0, 2))
    hud.add_column(justify="left")
    hud.add_column(justify="right")
    you, bot = duel.snakes[0], duel.snakes[1]
    you_tag = "you" if you.alive else "you (out)"
    bot_tag = "bot" if bot.alive else "bot (out)"
    hud.add_row(
        Text(f"{you_tag}  {you.score}", style=_SNAKE_STYLES[0][0]),
        Text(f"{bot_tag}  {bot.score}", style=_SNAKE_STYLES[1][0]),
    )
    footer = note or "[dim]WASD / arrows to steer · q to quit[/]"
    return Panel(
        Group(Align.center(board), Text(""), hud, Text.from_markup(footer)),
        title="snake · you vs. the bot",
        border_style="cyan",
        width=max(duel.cols + 4, 34),
    )


def play_duel(agent=None, device: str = "cpu", fps: float = 10.0,
              seed: int | None = None, console=None) -> None:
    """Run one interactive duel to completion in the terminal."""
    import sys

    from rich.console import Console

    console = console or Console()
    if not sys.stdin.isatty():
        console.print("[yellow]Multiplayer snake needs an interactive terminal.[/]")
        return

    bot = build_bot(agent, device)
    duel = SnakeDuel(seed=seed if seed is not None else int(time.time()))
    period = 1.0 / fps if fps > 0 else 0.0
    console.print("[bold cyan]snake · you vs. the bot[/] — "
                  "[cyan]you[/] chase the [red]●[/]; dodge the walls, yourself, "
                  "and the [green]bot[/].")

    from rich.live import Live

    aborted = False
    try:
        with _KeyReader() as keys, Live(console=console, screen=False,
                                        refresh_per_second=max(fps, 6)) as live:
            for n in (3, 2, 1):  # countdown; swallow any early keypresses
                live.update(_panel(duel, note=f"[bold]starting in {n}…[/]"))
                keys.poll()
                time.sleep(0.7)

            # Input is sampled every few ms and buffered in `pending`, but the
            # game only advances one step per `period`. Decoupling the two is what
            # keeps steering responsive: a keypress registers in ~8ms instead of
            # waiting out the rest of the (slow) tick before it's even read.
            pending: str | None = None
            next_tick = time.perf_counter()  # first step fires right away
            while not duel.done:
                k = keys.poll()
                while k is not None:
                    if k == "quit":
                        aborted = True
                        break
                    pending = k
                    k = keys.poll()
                if aborted:
                    break

                now = time.perf_counter()
                if not period or now >= next_tick:
                    actions = {1: bot.move(duel, 1)}
                    if pending is not None:
                        actions[0] = pending
                    duel.step(actions)
                    live.update(_panel(duel))
                    next_tick += period
                    if next_tick < time.perf_counter():
                        next_tick = time.perf_counter() + period  # fell behind; resync
                else:
                    time.sleep(min(next_tick - now, 0.008))
    except KeyboardInterrupt:
        aborted = True

    if aborted:
        console.print("[dim]Duel abandoned.[/]")
        return
    outcome = {0: "[bold cyan]You win![/] 🎉", 1: "[bold green]The bot wins.[/]"}
    console.print(_panel(duel, note=outcome.get(duel.winner, "[bold]A draw.[/]")))
    console.print(f"final — you {duel.snakes[0].score}, bot {duel.snakes[1].score}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Play snake against the bot.")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    play_duel(device=a.device, fps=a.fps, seed=a.seed)


if __name__ == "__main__":
    main()
