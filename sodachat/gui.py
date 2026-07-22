"""Watch the tiny GPT play its games in a desktop window (pygame).

    python -m sodachat.gui [--game snake] [--fps 12] [--device cpu]

A GUI counterpart to `sodachat.play`: the same trained controllers, drawn in a
window instead of the terminal. Grid games (snake, pong, dodge) play
themselves — the model picks an action every step and a new game starts when
one ends. Tic-tac-toe is interactive, as in the agent: you're X (click a
cell), the model answers as O.

Controls: click the tabs (or keys 1-4) to switch games, space pauses,
n starts a new game, +/- changes the step rate, q quits.

Models load lazily in a background thread the first time a game is opened, so
switching tabs never freezes the window.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque

import pygame

from .game_train import game_model_path
from .games import GAMES, GamePlayer, load_model

# Pixel size of one grid cell. Shrunk so even the largest registered board fits
# a sensible window (a 20x20 board at 40px would be 800px wide); small boards
# keep the nominal 40px. Cell-drawing offsets below are relative to CELL, so the
# shapes scale with it.
_NOMINAL_CELL = 40
_BOARD_TARGET_PX = 560
_grid_sizes = [c.SIZE for c in GAMES.values() if getattr(c, "MODALITY", "grid") == "grid"]
_max_cells = max((max(s) for s in _grid_sizes), default=12)
CELL = max(20, min(_NOMINAL_CELL, _BOARD_TARGET_PX // _max_cells))
TTT_CELL = 116
TABS_H = 54
HUD_H = 104
W = 600            # minimum window width (grown to fit wider boards, see App.w)

BG = (15, 17, 21)
PANEL = (22, 26, 33)
BOARD_BG = (26, 30, 38)
GRID_DOT = (46, 52, 64)
FG = (230, 233, 239)
DIM = (136, 145, 162)
ACCENT = (57, 197, 207)
GOOD = (74, 222, 128)
BAD = (239, 88, 88)

# cell value -> (colour, shape) per grid game; 0 (empty) is drawn as a dot.
# Mirrors the terminal GLYPHS of each game.
STYLES = {
    "snake": {1: ((22, 163, 74), "square"),    # body
              2: ((74, 222, 128), "square"),   # head
              3: ((239, 88, 88), "circle")},   # food
    "pong":  {1: ((250, 204, 21), "circle"),   # ball
              2: ((34, 211, 238), "square")},  # paddle
    "dodge": {1: ((74, 222, 128), "triangle"), # agent
              2: ((239, 88, 88), "square")},   # wall
}

RESTART_DELAY = 1.2  # seconds a finished grid game stays on screen


class App:
    def __init__(self, device: str, fps: float, start_game: str):
        self.device = device
        self.step_fps = fps
        self.names = [n for n, c in GAMES.items() if c.TRAINABLE]

        self.players: dict[str, GamePlayer] = {}
        self.loading: set[str] = set()
        self.errors: dict[str, str] = {}

        self.current = start_game
        self.game = None
        self.paused = False
        self.game_num = 0
        self.best = 0
        self.tally = {"you": 0, "me": 0, "draw": 0}  # tic-tac-toe results
        self.recent = deque(maxlen=200)              # per-move latency, ms
        self.next_step = 0.0
        self.over_since: float | None = None

        board_h = max(GAMES[n].SIZE[0] * CELL if GAMES[n].MODALITY == "grid"
                      else 3 * TTT_CELL for n in self.names)
        board_w = max(GAMES[n].SIZE[1] * CELL if GAMES[n].MODALITY == "grid"
                      else 3 * TTT_CELL for n in self.names)
        self.h = TABS_H + board_h + 40 + HUD_H
        # Grow the window to fit the widest board (W is just a floor), so a board
        # is never drawn off-screen. Height already scales with board_h.
        self.w = max(board_w + 40, W)
        self.tab_rects: list[pygame.Rect] = []

    # ------------------------------------------------------------- models

    def _ensure_player(self, name: str) -> None:
        if name in self.players or name in self.loading or name in self.errors:
            return
        path = game_model_path(name)
        if not path.exists():
            self.errors[name] = (f"no trained model at {path.name}\n"
                                 f"python -m sodachat.game_train --game {name}")
            return
        self.loading.add(name)
        threading.Thread(target=self._load, args=(name,), daemon=True).start()

    def _load(self, name: str) -> None:
        try:
            model, tok, _ = load_model(game_model_path(name), self.device)
            # A checkpoint bakes in the board size it was trained on; if the game
            # has since been resized, its fixed input buffer can't hold the board
            # and GamePlayer.act would crash the loop. Show the fix instead.
            trained = tuple(getattr(tok, "size", ()) or ())
            want = tuple(GAMES[name].SIZE)
            if GAMES[name].MODALITY == "grid" and trained and trained != want:
                self.errors[name] = (
                    f"model is for a {trained[0]}x{trained[1]} board,\n"
                    f"but {name} is now {want[0]}x{want[1]} — retrain it:\n"
                    f"python -m sodachat.game_train --game {name}")
                return
            self.players[name] = GamePlayer(model, tok, self.device)
        except Exception as e:  # surface the error in the window, don't die
            self.errors[name] = f"failed to load model:\n{e}"
        finally:
            self.loading.discard(name)

    # ------------------------------------------------------------- session

    def switch(self, name: str) -> None:
        if name not in self.names:
            return
        self.current = name
        self.game = None
        self.paused = False
        self.game_num = 0
        self.best = 0
        self.recent.clear()
        self.over_since = None
        self._ensure_player(name)

    def new_game(self) -> None:
        self.game = GAMES[self.current]()
        self.game_num += 1
        self.over_since = None
        self.next_step = time.perf_counter()
        if self.is_ttt:
            # keep the model's opening consistent with the agent: human first
            pass

    @property
    def is_ttt(self) -> bool:
        return GAMES[self.current].MODALITY == "text"

    @property
    def player(self) -> GamePlayer | None:
        return self.players.get(self.current)

    def p99(self) -> float:
        return sorted(self.recent)[int(len(self.recent) * 0.99)] if self.recent else 0.0

    # ------------------------------------------------------------- update

    def update(self) -> None:
        self._ensure_player(self.current)
        if self.player is None:
            return
        if self.game is None:
            self.new_game()

        if self.is_ttt:
            return  # advanced from click events

        now = time.perf_counter()
        if self.game.done:
            if self.over_since is None:
                self.over_since = now
                self.best = max(self.best, self.game.score)
            elif not self.paused and now - self.over_since > RESTART_DELAY:
                self.new_game()
            return
        if self.paused or now < self.next_step:
            return
        self.game.step(self.player.act(self.game))
        self.recent.append(self.player.last_ms)
        self.next_step = max(self.next_step + 1.0 / self.step_fps, now)

    def click_ttt(self, cell: int) -> None:
        g, player = self.game, self.player
        if g is None or player is None:
            return
        if g.done:
            self.new_game()
            return
        if g.turn != "X" or not g.legal(str(cell)):
            return
        g.step(str(cell))
        if not g.done:
            g.step(player.act(g))  # model replies as O
            self.recent.append(player.last_ms)
        if g.done:
            key = {"X": "you", "O": "me"}.get(g.winner, "draw")
            self.tally[key] += 1

    # ------------------------------------------------------------- drawing

    def draw(self, screen: pygame.Surface, fonts) -> None:
        screen.fill(BG)
        self._draw_tabs(screen, fonts)
        if self.errors.get(self.current):
            self._draw_message(screen, fonts, self.errors[self.current], BAD)
        elif self.player is None:
            self._draw_message(screen, fonts, "loading model...", DIM)
        elif self.game is not None:
            if self.is_ttt:
                self._draw_ttt(screen, fonts)
            else:
                self._draw_grid(screen)
        self._draw_hud(screen, fonts)

    def _draw_tabs(self, screen, fonts) -> None:
        self.tab_rects = []
        x = 16
        for i, name in enumerate(self.names):
            label = fonts["tab"].render(f"{i + 1} {name}", True,
                                        FG if name == self.current else DIM)
            r = pygame.Rect(x, 10, label.get_width() + 24, TABS_H - 20)
            pygame.draw.rect(screen, PANEL if name == self.current else BG, r,
                             border_radius=8)
            if name == self.current:
                pygame.draw.rect(screen, ACCENT, r, width=1, border_radius=8)
            screen.blit(label, (r.x + 12, r.y + (r.h - label.get_height()) // 2))
            self.tab_rects.append(r)
            x = r.right + 8

    def _board_origin(self, bw: int, bh: int) -> tuple[int, int]:
        return (self.w - bw) // 2, TABS_H + 20

    def _draw_grid(self, screen) -> None:
        g = self.game
        grid, _ = g.observe()
        bw, bh = g.cols * CELL, g.rows * CELL
        x0, y0 = self._board_origin(bw, bh)
        pygame.draw.rect(screen, BOARD_BG, (x0 - 8, y0 - 8, bw + 16, bh + 16),
                         border_radius=10)
        style = STYLES[g.NAME]
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                cx, cy = x0 + c * CELL + CELL // 2, y0 + r * CELL + CELL // 2
                v = int(v)
                if v == 0:
                    pygame.draw.circle(screen, GRID_DOT, (cx, cy), 2)
                    continue
                colour, shape = style[v]
                if shape == "circle":
                    pygame.draw.circle(screen, colour, (cx, cy), CELL // 2 - 6)
                elif shape == "triangle":
                    s = CELL // 2 - 5
                    pygame.draw.polygon(screen, colour, [
                        (cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)])
                else:
                    pygame.draw.rect(screen, colour,
                                     (x0 + c * CELL + 3, y0 + r * CELL + 3,
                                      CELL - 6, CELL - 6), border_radius=6)
        if g.done:
            self._overlay(screen, f"game over — score {g.score}",
                          (x0, y0, bw, bh))

    def _draw_ttt(self, screen, fonts) -> None:
        g = self.game
        bw = bh = 3 * TTT_CELL
        x0, y0 = self._board_origin(bw, bh)
        pygame.draw.rect(screen, BOARD_BG, (x0 - 8, y0 - 8, bw + 16, bh + 16),
                         border_radius=10)
        for i in (1, 2):
            pygame.draw.line(screen, GRID_DOT, (x0 + i * TTT_CELL, y0 + 8),
                             (x0 + i * TTT_CELL, y0 + bh - 8), 3)
            pygame.draw.line(screen, GRID_DOT, (x0 + 8, y0 + i * TTT_CELL),
                             (x0 + bw - 8, y0 + i * TTT_CELL), 3)
        for i, ch in enumerate(g.board):
            cx = x0 + (i % 3) * TTT_CELL + TTT_CELL // 2
            cy = y0 + (i // 3) * TTT_CELL + TTT_CELL // 2
            s = TTT_CELL // 2 - 26
            if ch == "X":
                for d in (-1, 1):
                    pygame.draw.line(screen, GOOD, (cx - s, cy - d * s),
                                     (cx + s, cy + d * s), 7)
            elif ch == "O":
                pygame.draw.circle(screen, ACCENT, (cx, cy), s + 2, 7)
        if g.done:
            msg = {"X": "you win!", "O": "I win this one!"}.get(
                g.winner, "a draw.")
            self._overlay(screen, f"{msg}  (click for a new game)",
                          (x0, y0, bw, bh))

    def _overlay(self, screen, text: str, board: tuple[int, int, int, int]) -> None:
        x0, y0, bw, bh = board
        shade = pygame.Surface((bw, bh), pygame.SRCALPHA)
        shade.fill((10, 12, 16, 150))
        screen.blit(shade, (x0, y0))
        font = pygame.font.SysFont("menlo, monaco, courier", 22, bold=True)
        label = font.render(text, True, FG)
        screen.blit(label, (x0 + (bw - label.get_width()) // 2,
                            y0 + (bh - label.get_height()) // 2))

    def _draw_message(self, screen, fonts, text: str, colour) -> None:
        y = TABS_H + 60
        for line in text.split("\n"):
            label = fonts["hud"].render(line, True, colour)
            screen.blit(label, ((self.w - label.get_width()) // 2, y))
            y += 26

    def _draw_hud(self, screen, fonts) -> None:
        y0 = self.h - HUD_H
        pygame.draw.rect(screen, PANEL, (0, y0, self.w, HUD_H))
        g = self.game
        if self.is_ttt:
            t = self.tally
            left = (f"you {t['you']}   me {t['me']}   draws {t['draw']}")
            state = ("" if g is None else
                     "your move — click a cell" if not g.done and g.turn == "X"
                     else "")
        else:
            left = (f"score {g.score if g else 0}   best {self.best}   "
                    f"game #{self.game_num}")
            state = "paused" if self.paused else ""
        right = (f"decide {self.recent[-1]:.2f} ms   p99 {self.p99():.2f} ms"
                 if self.recent else "")
        screen.blit(fonts["hud"].render(left, True, FG), (16, y0 + 14))
        if right:
            label = fonts["hud"].render(right, True, DIM)
            screen.blit(label, (self.w - label.get_width() - 16, y0 + 14))
        if state:
            screen.blit(fonts["hud"].render(state, True, ACCENT), (16, y0 + 42))
        hint = ("click a cell to play X · n new game · 1-4 games · q quit"
                if self.is_ttt else
                f"space pause · n new · +/- speed ({self.step_fps:.0f}/s) · "
                f"1-4 games · q quit")
        screen.blit(fonts["hint"].render(hint, True, DIM), (16, self.h - 30))

    # ------------------------------------------------------------- events

    def handle(self, ev) -> bool:
        """Process one event; returns False to quit."""
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                return False
            if ev.key == pygame.K_SPACE and not self.is_ttt:
                self.paused = not self.paused
                self.next_step = time.perf_counter()
            elif ev.key == pygame.K_n:
                self.new_game() if self.player else None
            elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.step_fps = min(60.0, self.step_fps + 2)
            elif ev.key == pygame.K_MINUS:
                self.step_fps = max(2.0, self.step_fps - 2)
            elif pygame.K_1 <= ev.key <= pygame.K_9:
                idx = ev.key - pygame.K_1
                if idx < len(self.names):
                    self.switch(self.names[idx])
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for name, r in zip(self.names, self.tab_rects):
                if r.collidepoint(ev.pos):
                    self.switch(name)
                    return True
            if self.is_ttt and self.game is not None:
                bw = 3 * TTT_CELL
                x0, y0 = self._board_origin(bw, bw)
                cx, cy = ev.pos[0] - x0, ev.pos[1] - y0
                if 0 <= cx < bw and 0 <= cy < bw:
                    self.click_ttt((cy // TTT_CELL) * 3 + (cx // TTT_CELL))
                elif self.game.done:
                    self.new_game()
        return True

    # ------------------------------------------------------------- main loop

    def run(self, max_seconds: float | None = None) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("sodachat plays")
        fonts = {
            "tab": pygame.font.SysFont("menlo, monaco, courier", 18, bold=True),
            "hud": pygame.font.SysFont("menlo, monaco, courier", 17),
            "hint": pygame.font.SysFont("menlo, monaco, courier", 14),
        }
        clock = pygame.time.Clock()
        self.switch(self.current)
        t0 = time.perf_counter()
        running = True
        while running:
            for ev in pygame.event.get():
                running = self.handle(ev) and running
            self.update()
            self.draw(screen, fonts)
            pygame.display.flip()
            clock.tick(60)
            if max_seconds and time.perf_counter() - t0 > max_seconds:
                running = False
        pygame.quit()


def main() -> None:
    trainable = [n for n, c in GAMES.items() if c.TRAINABLE]
    p = argparse.ArgumentParser(description="Watch the GPT play, in a window.")
    p.add_argument("--game", choices=trainable, default="snake")
    p.add_argument("--fps", type=float, default=12.0,
                   help="model steps per second for grid games")
    # CPU: for a ~1M model it is faster and steadier than a GPU per step.
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    App(a.device, a.fps, a.game).run()


if __name__ == "__main__":
    main()
