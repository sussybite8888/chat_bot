"""Pong — a tracking game. A ball bounces around walls; the agent moves a
paddle on the right edge to intercept and return it. Each return scores."""

from __future__ import annotations

import numpy as np

from .core import Game, register

EMPTY, BALL, PADDLE = 0, 1, 2
_PAD = 3  # paddle height
_MAX_STEPS = 400


def _reflect(x: int, n: int) -> int:
    """Position after bouncing within [0, n-1] (triangle wave)."""
    if n == 1:
        return 0
    period = 2 * (n - 1)
    x %= period
    return x if x < n else period - x


@register
class PongGame(Game):
    NAME = "pong"
    SIZE = (12, 12)
    ACTIONS = ("up", "down", "stay")
    N_CELLS = 3
    AUX_DIMS = (2, 2)  # sign of vertical, horizontal velocity
    GLYPHS = {
        EMPTY: ("·", "grey30"),
        BALL: ("●", "bright_yellow"),
        PADDLE: ("█", "cyan"),
    }

    def reset(self) -> "PongGame":
        self.pad = self.rows // 2 - _PAD // 2  # top row of paddle
        self.br = self.rng.randrange(self.rows)
        self.bc = 1
        self.vr = self.rng.choice((-1, 1))
        self.vc = 1
        self.score = 0
        self.steps = 0
        self.done = False
        return self

    def _move_paddle(self, action: str) -> None:
        if action == "up":
            self.pad -= 1
        elif action == "down":
            self.pad += 1
        self.pad = max(0, min(self.rows - _PAD, self.pad))

    def step(self, action: str) -> None:
        if self.done:
            return
        self._move_paddle(action)

        nr = self.br + self.vr
        if nr < 0 or nr >= self.rows:  # top/bottom bounce
            self.vr = -self.vr
            nr = self.br + self.vr
        nc = self.bc + self.vc
        if nc < 0:  # left wall bounce
            self.vc = -self.vc
            nc = self.bc + self.vc

        if nc >= self.cols - 1:  # reached the paddle column
            if self.pad <= nr < self.pad + _PAD:
                self.vc = -1
                nc = self.cols - 1
                self.score += 1
            else:
                self.done = True
                self.br, self.bc = nr, min(nc, self.cols - 1)
                return
        self.br, self.bc = nr, nc

        self.steps += 1
        if self.steps >= _MAX_STEPS:
            self.done = True

    def observe(self) -> tuple[np.ndarray, tuple[int, ...]]:
        g = np.zeros(self.SIZE, dtype=np.int8)
        for i in range(_PAD):
            g[self.pad + i, self.cols - 1] = PADDLE
        g[self.br, self.bc] = BALL
        return g, (int(self.vr > 0), int(self.vc > 0))

    def facts(self) -> dict[str, str]:
        return {
            "score": str(self.score),
            "rallies_returned": str(self.score),
            "ball": "coming toward me" if self.vc > 0 else "heading away",
            "ball_row": str(self.br),
            "over": "yes" if self.done else "no",
        }

    def expert(self) -> str:
        # Predict the row where the ball will reach the paddle column, then
        # move the paddle centre toward it.
        if self.vc > 0:
            dist = (self.cols - 1) - self.bc
            target = _reflect(self.br + self.vr * dist, self.rows)
        else:
            target = self.br
        center = self.pad + _PAD // 2
        if center < target:
            return "down"
        if center > target:
            return "up"
        return "stay"
