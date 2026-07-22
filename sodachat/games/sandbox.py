"""Sandbox — a free-movement grid for testing instruction-following (VLA),
*without training a thing*.

One agent (`█`) and one target (`●`) on an empty grid; the four directional
moves plus `stay`. There is no death and no growth — reaching the target scores
a point and respawns it, but the real purpose is to *steer the agent with a
goal* and watch whether the model obeys:

    /play sandbox
    /goal go to the top left corner      # then /watch or /board

Why it needs no training: the moves (`up/down/left/right`) and the goals
("go up", "go to a corner", "eat the food") are exactly what the expert model
already learned on Snake, and the board is drawn with Snake's glyphs so it is
in-distribution. A goal-conditioned model therefore transfers its instruction-
following to this game it has *never seen*. That transfer is the thing under
test — run `python -m sodachat.expert vla` for an automated probe.

The scripted `expert` (greedy toward the target) exists only to satisfy the
Game interface; this game is a probe, not a training target.
"""

from __future__ import annotations

import numpy as np

from .core import Game, register

EMPTY, AGENT, TARGET = 0, 1, 2
_MAX_STEPS = 600
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "stay": (0, 0)}


@register
class SandboxGame(Game):
    NAME = "sandbox"
    SIZE = (20, 20)
    ACTIONS = ("up", "down", "left", "right", "stay")
    N_CELLS = 3
    AUX_DIMS = ()
    TRAINABLE = False  # held out of training — it's a zero-shot VLA probe
    # Snake's glyphs on purpose, so the board is in-distribution for a model
    # trained on Snake (the agent looks like the snake head, the target like food).
    GLYPHS = {
        EMPTY: ("·", "grey30"),
        AGENT: ("█", "bright_green"),
        TARGET: ("●", "yellow"),
    }

    def reset(self) -> "SandboxGame":
        self.ar, self.ac = self.rng.randrange(self.rows), self.rng.randrange(self.cols)
        self.tr, self.tc = self.ar, self.ac
        self._respawn_target()
        self.score = 0
        self.steps = 0
        self.done = False
        return self

    def _respawn_target(self) -> None:
        while (self.tr, self.tc) == (self.ar, self.ac):
            self.tr = self.rng.randrange(self.rows)
            self.tc = self.rng.randrange(self.cols)

    def legal(self, action: str) -> bool:
        return action in self.ACTIONS  # movement clamps at walls; all moves allowed

    def step(self, action: str) -> None:
        if self.done:
            return
        dr, dc = _DELTA.get(action, (0, 0))
        self.ar = min(max(self.ar + dr, 0), self.rows - 1)
        self.ac = min(max(self.ac + dc, 0), self.cols - 1)
        if (self.ar, self.ac) == (self.tr, self.tc):
            self.score += 1
            self._respawn_target()
        self.steps += 1
        if self.steps >= _MAX_STEPS:
            self.done = True

    def observe(self) -> tuple[np.ndarray, tuple[int, ...]]:
        g = np.zeros(self.SIZE, dtype=np.int8)
        g[self.tr, self.tc] = TARGET
        g[self.ar, self.ac] = AGENT  # drawn last, so it wins if they overlap
        return g, ()

    def facts(self) -> dict[str, str]:
        return {
            "score": str(self.score),
            "position": f"row {self.ar} col {self.ac}",
            "target": f"row {self.tr} col {self.tc}",
            "over": "yes" if self.done else "no",
        }

    def expert(self) -> str:
        # Greedy toward the target — only to satisfy the interface (not trained).
        if self.ar > self.tr:
            return "up"
        if self.ar < self.tr:
            return "down"
        if self.ac > self.tc:
            return "left"
        if self.ac < self.tc:
            return "right"
        return "stay"
