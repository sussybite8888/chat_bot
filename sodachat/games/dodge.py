"""Dodge — an avoidance game. Walls with a gap descend from the top; the agent
moves along the bottom row to line up with each gap. One point per wall passed."""

from __future__ import annotations

import numpy as np

from .core import Game, register

EMPTY, AGENT, WALL = 0, 1, 2
_GAP = 3      # width of the opening in each wall
_SPACING = 4  # ticks between wall spawns
_MAX_STEPS = 500


@register
class DodgeGame(Game):
    NAME = "dodge"
    SIZE = (12, 10)
    ACTIONS = ("left", "right", "stay")
    N_CELLS = 3
    AUX_DIMS = ()
    GLYPHS = {
        EMPTY: ("·", "grey30"),
        AGENT: ("▲", "bright_green"),
        WALL: ("█", "red"),
    }

    def reset(self) -> "DodgeGame":
        self.col = self.cols // 2
        # each wall: [row, gap_start]
        self.walls: list[list[int]] = []
        self.score = 0
        self.steps = 0
        self.done = False
        self._spawn()
        return self

    def _spawn(self) -> None:
        self.walls.append([0, self.rng.randrange(0, self.cols - _GAP + 1)])

    def _in_gap(self, col: int, gap_start: int) -> bool:
        return gap_start <= col < gap_start + _GAP

    def step(self, action: str) -> None:
        if self.done:
            return
        if action == "left":
            self.col = max(0, self.col - 1)
        elif action == "right":
            self.col = min(self.cols - 1, self.col + 1)

        for w in self.walls:
            w[0] += 1
        agent_row = self.rows - 1
        for row, gap in self.walls:
            if row == agent_row and not self._in_gap(self.col, gap):
                self.done = True
                return
        passed = [w for w in self.walls if w[0] >= self.rows]
        self.score += len(passed)
        self.walls = [w for w in self.walls if w[0] < self.rows]

        self.steps += 1
        if self.steps % _SPACING == 0:
            self._spawn()
        if self.steps >= _MAX_STEPS:
            self.done = True

    def observe(self) -> tuple[np.ndarray, tuple[int, ...]]:
        g = np.zeros(self.SIZE, dtype=np.int8)
        for row, gap in self.walls:
            if 0 <= row < self.rows:
                for c in range(self.cols):
                    if not self._in_gap(c, gap):
                        g[row, c] = WALL
        g[self.rows - 1, self.col] = AGENT
        return g, ()

    def facts(self) -> dict[str, str]:
        pending = [w for w in self.walls if w[0] < self.rows - 1]
        gap = "no wall coming"
        if pending:
            _, gs = max(pending, key=lambda w: w[0])
            gap = f"columns {gs}-{gs + _GAP - 1}"
        return {
            "score": str(self.score),
            "walls_passed": str(self.score),
            "position": f"column {self.col}",
            "next_gap": gap,
            "over": "yes" if self.done else "no",
        }

    def expert(self) -> str:
        # Aim for the gap of the lowest wall still above the agent.
        pending = [w for w in self.walls if w[0] < self.rows - 1]
        if not pending:
            return "stay"
        row, gap = max(pending, key=lambda w: w[0])
        target = gap + _GAP // 2
        if self.col < target:
            return "right"
        if self.col > target:
            return "left"
        return "stay"
