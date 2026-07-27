"""Snake — a pathfinding game. The agent must reach food while not trapping
itself in its own growing body."""

from __future__ import annotations

from collections import deque

import numpy as np

from .core import Game, register

EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@register
class SnakeGame(Game):
    NAME = "snake"
    SIZE = (20, 20)
    ACTIONS = ("up", "down", "left", "right")
    N_CELLS = 4
    AUX_DIMS = (4,)  # heading
    GLYPHS = {
        EMPTY: ("·", "grey30"),
        BODY: ("█", "green"),
        HEAD: ("█", "bright_green"),
        FOOD: ("●", "red"),
    }
    # head '@' vs body '#' (render() shows both as █, so the LM couldn't tell
    # them apart); food '*'. Matches sandbox's agent/target so VLA transfers.
    MODEL_GLYPHS = {EMPTY: ".", BODY: "#", HEAD: "@", FOOD: "*"}

    def reset(self) -> "SnakeGame":
        c = self.SIZE[0] // 2
        self.snake: deque[tuple[int, int]] = deque([(c, c)])
        self.body = {(c, c)}
        self.heading = "right"
        self.score = 0
        self.since_food = 0
        self.done = False
        self._place_food()
        return self

    def _place_food(self) -> None:
        free = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.body
        ]
        self.food = self.rng.choice(free) if free else None

    @property
    def head(self) -> tuple[int, int]:
        return self.snake[0]

    def legal(self, action: str) -> bool:
        return not (len(self.snake) > 1 and action == _OPPOSITE[self.heading])

    def _result(self, action: str):
        if not self.legal(action):
            return None, False
        dr, dc = _DELTA[action]
        nr, nc = self.head[0] + dr, self.head[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return None, False
        eating = (nr, nc) == self.food
        occ = set(self.snake)
        if not eating:
            occ.discard(self.snake[-1])
        if (nr, nc) in occ:
            return None, False
        return (nr, nc), eating

    def safe_actions(self) -> list[str]:
        return [a for a in self.ACTIONS if self._result(a)[0] is not None]

    def step(self, action: str) -> None:
        if self.done:
            return
        if not self.legal(action):
            action = self.heading
        result, eating = self._result(action)
        self.heading = action
        if result is None:
            self.done = True
            return
        self.snake.appendleft(result)
        if eating:
            self.score += 1
            self.since_food = 0
            self._place_food()
            if self.food is None:
                self.done = True
        else:
            self.snake.pop()
            self.since_food += 1
        self.body = set(self.snake)
        if self.since_food > 2 * self.rows * self.cols:
            self.done = True

    def observe(self) -> tuple[np.ndarray, tuple[int, ...]]:
        g = np.zeros(self.SIZE, dtype=np.int8)
        for r, c in self.snake:
            g[r, c] = BODY
        g[self.head] = HEAD
        if self.food is not None:
            g[self.food] = FOOD
        return g, (self.ACTIONS.index(self.heading),)

    def facts(self) -> dict[str, str]:
        hr, hc = self.head
        fr, fc = self.food if self.food else (hr, hc)
        vy = "up" if fr < hr else ("down" if fr > hr else "")
        vx = "left" if fc < hc else ("right" if fc > hc else "")
        where = " and ".join(w for w in (vy, vx) if w) or "right next to me"
        return {
            "score": str(self.score),
            "length": str(len(self.snake)),
            "food": where,
            "heading": self.heading,
            "over": "yes" if self.done else "no",
        }

    # ---- expert: greedy to food, but only via moves that keep enough room ----

    def _reachable(self, start, block) -> int:
        seen = {start}
        dq = deque([start])
        n = 0
        while dq:
            r, c = dq.popleft()
            for dr, dc in _DELTA.values():
                nb = (r + dr, c + dc)
                if (
                    0 <= nb[0] < self.rows
                    and 0 <= nb[1] < self.cols
                    and nb not in block
                    and nb not in seen
                ):
                    seen.add(nb)
                    dq.append(nb)
                    n += 1
        return n

    def _food_distance(self, start) -> int:
        if self.food is None:
            return 1 << 20
        block = set(self.snake)
        if len(self.snake) > 1:
            block.discard(self.snake[-1])
        seen = {start}
        dq = deque([(start, 0)])
        while dq:
            (r, c), d = dq.popleft()
            if (r, c) == self.food:
                return d
            for dr, dc in _DELTA.values():
                nb = (r + dr, c + dc)
                if (
                    0 <= nb[0] < self.rows
                    and 0 <= nb[1] < self.cols
                    and nb not in block
                    and nb not in seen
                ):
                    seen.add(nb)
                    dq.append((nb, d + 1))
        return 1 << 20

    def expert(self) -> str:
        scored = []
        for a in self.ACTIONS:
            head, eating = self._result(a)
            if head is None:
                continue
            new_body = deque(self.snake)
            new_body.appendleft(head)
            if not eating:
                new_body.pop()
            room = self._reachable(head, set(new_body))
            scored.append((a, head, room))
        if not scored:
            return self.heading
        roomy = [s for s in scored if s[2] >= len(self.snake)] or scored
        return min(roomy, key=lambda s: (self._food_distance(s[1]), -s[2]))[0]
