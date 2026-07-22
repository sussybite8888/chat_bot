"""Watch the tiny GPT play a game in real time, in your terminal.

    python -m sodachat.play --game snake|pong|dodge [--fps 12] [--games 3]
    python -m sodachat.play --game snake --bench      # latency distribution

The model picks an action every frame from the current board. Real-time
control cares about *consistent* latency, so:

- the controller is warmed up and single-threaded (see GamePlayer);
- the loop uses a fixed timestep with deadline scheduling, so the game runs at
  a steady rate and never spirals if one frame is slightly slow;
- garbage collection is paused during play (GC pauses are a jitter source).

`--bench` reports the per-move latency distribution (mean / p99 / max) so you
can confirm the worst case fits inside your frame budget with room to spare.
"""

from __future__ import annotations

import argparse
import gc
import time
from collections import deque
from pathlib import Path

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .games import (
    GAMES, GamePlayer, benchmark_latency, framed_board, half_block_render, load_model,
)
from .game_train import game_model_path

_MS_BUDGETS = {"30 fps": 33.33, "60 fps": 16.67, "120 fps": 8.33}


def _render(game, player, game_num, n_games, p99_ms) -> Panel:
    # Half-block board: two grid rows per line, so a 20-row board fits a normal
    # terminal instead of spilling past the bottom. Framed so the play-field
    # walls are visible even when the board is mostly empty.
    body = framed_board(half_block_render(game), game.cols)

    hud = Table.grid(padding=(0, 2))
    hud.add_column(justify="left")
    hud.add_column(justify="right")
    hud.add_row(
        f"[bold]{game.NAME}[/]  game {game_num}/{n_games}",
        f"score [bold green]{game.score}[/]",
    )
    hud.add_row(
        f"decide: [cyan]{player.last_ms:.2f} ms[/]",
        f"[dim]p99 {p99_ms:.2f} ms[/]",
    )
    return Panel(
        Group(Align.center(body), Text(""), hud),
        title="sodachat plays",
        border_style="cyan",
        width=max(game.cols + 4, 34),
    )


def _bench(game: str, path: Path, device: str, console: Console) -> None:
    model, tok, _ = load_model(path, device)
    player = GamePlayer(model, tok, device)
    console.print(f"[dim]measuring {game} latency on {device}...[/]")
    s = benchmark_latency(GAMES[game], player, moves=5000)

    t = Table(title=f"{game}: per-move latency ({s['moves']:,} moves, {device})")
    t.add_column("metric")
    t.add_column("ms", justify="right")
    for k in ("mean", "p50", "p95", "p99", "max", "std"):
        t.add_row(k, f"{s[k]:.3f}")
    console.print(t)

    console.print(f"sustained rate: [bold]{1000 / s['mean']:,.0f} moves/s[/] "
                  f"(worst-case {1000 / s['max']:,.0f}/s)")
    verdict = Table(title="fits in frame budget? (worst case vs period)")
    verdict.add_column("target")
    verdict.add_column("budget ms", justify="right")
    verdict.add_column("headroom", justify="right")
    verdict.add_column("ok?", justify="center")
    for name, budget in _MS_BUDGETS.items():
        ok = s["max"] < budget
        verdict.add_row(name, f"{budget:.1f}", f"{budget - s['max']:.2f} ms",
                        "[green]yes[/]" if ok else "[red]no[/]")
    console.print(verdict)


def play(game, model_path, fps, n_games, device, seed) -> None:
    console = Console()
    if game not in GAMES:
        raise SystemExit(f"unknown game {game!r} (have: {', '.join(GAMES)})")
    path = Path(model_path) if model_path else game_model_path(game)
    if not path.exists():
        raise SystemExit(
            f"no trained model at {path}. Train one first:\n"
            f"  python -m sodachat.game_train --game {game}"
        )

    game_cls = GAMES[game]
    if game_cls.MODALITY != "grid":
        raise SystemExit(
            f"'{game}' is a text game — play it interactively:\n"
            f"  python -m sodachat.agent   (then: play {game})"
        )
    model, tok, _ = load_model(path, device)
    player = GamePlayer(model, tok, device)
    period = 1.0 / fps if fps > 0 else 0.0
    recent = deque(maxlen=200)

    def p99() -> float:
        return sorted(recent)[int(len(recent) * 0.99)] if recent else 0.0

    best = 0
    gc.disable()  # avoid GC pauses mid-loop
    try:
        with Live(console=console, refresh_per_second=max(fps, 4), screen=False) as live:
            for gi in range(1, n_games + 1):
                g = game_cls(seed=seed + gi)
                deadline = time.perf_counter()
                while not g.done:
                    g.step(player.act(g))
                    recent.append(player.last_ms)
                    live.update(_render(g, player, gi, n_games, p99()))
                    if period:
                        deadline += period
                        slack = deadline - time.perf_counter()
                        if slack > 0:
                            time.sleep(slack)
                        else:
                            deadline = time.perf_counter()  # fell behind; resync
                best = max(best, g.score)
                live.update(_render(g, player, gi, n_games, p99()))
                time.sleep(0.8)
    finally:
        gc.enable()
    console.print(f"[bold]best score over {n_games} games: [green]{best}[/]  "
                  f"[dim](decide p99 {p99():.2f} ms)[/]")


def main() -> None:
    p = argparse.ArgumentParser(description="Watch the GPT play a game.")
    p.add_argument("--game", choices=list(GAMES), default="snake")
    p.add_argument("--fps", type=float, default=12.0,
                   help="frames/sec (0 = as fast as possible)")
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--bench", action="store_true",
                   help="measure per-move latency distribution and exit")
    # CPU is the default: for a ~1M model it is faster and far more consistent
    # than a GPU for single-step inference.
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.bench:
        path = Path(a.model) if a.model else game_model_path(a.game)
        if not path.exists():
            raise SystemExit(f"no trained model at {path}")
        _bench(a.game, path, a.device, Console())
        return
    play(a.game, a.model, a.fps, a.games, a.device, a.seed)


if __name__ == "__main__":
    main()
