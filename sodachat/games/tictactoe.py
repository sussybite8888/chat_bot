"""Tic-Tac-Toe — a *non-bitmap* game. The observation is text ("turn X board
X.O..O..."), not a grid of pixels, and the expert is minimax. It demonstrates
that the same pipeline plays symbolic/text games, and it is interactive: in the
unified agent (../agent.py) you play one side and the model plays the other."""

from __future__ import annotations

from functools import lru_cache

from .core import Game, register

_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def _winner(board: str) -> str | None:
    for a, b, c in _LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=None)
def _minimax(board: str, player: str) -> tuple[int, int]:
    """Best (value, move) for `player` to move, value from their perspective."""
    if _winner(board):  # the player who just moved won, so player-to-move lost
        return -1, -1
    if "." not in board:
        return 0, -1
    opp = "O" if player == "X" else "X"
    best = (-2, -1)
    for i, ch in enumerate(board):
        if ch == ".":
            nxt = board[:i] + player + board[i + 1 :]
            score = -_minimax(nxt, opp)[0]
            if score > best[0]:
                best = (score, i)
    return best


@register
class TicTacToe(Game):
    NAME = "tictactoe"
    MODALITY = "text"
    ACTIONS = tuple(str(i) for i in range(9))
    MAX_LEN = 32

    def reset(self) -> "TicTacToe":
        self.board = "." * 9
        self.turn = "X"
        self.winner: str | None = None
        self.done = False
        self.score = 0
        return self

    def observe(self) -> str:
        return f"turn {self.turn} board {self.board}"

    def legal(self, action: str) -> bool:
        return (not self.done) and self.board[int(action)] == "."

    def safe_actions(self) -> list[str]:
        return [a for a in self.ACTIONS if self.legal(a)]

    def step(self, action: str) -> None:
        if self.done:
            return
        i = int(action)
        if self.board[i] != ".":
            return  # illegal; masking should prevent this
        self.board = self.board[:i] + self.turn + self.board[i + 1 :]
        w = _winner(self.board)
        if w:
            self.done, self.winner = True, w
            self.score = 1 if w == "X" else -1
        elif "." not in self.board:
            self.done = True  # draw
        else:
            self.turn = "O" if self.turn == "X" else "X"

    def facts(self) -> dict[str, str]:
        moves = 9 - self.board.count(".")
        winner = self.winner if self.winner else ("draw" if self.done else "none yet")
        return {
            "score": winner if self.done else "in progress",
            "turn": self.turn,
            "moves_played": str(moves),
            "cells_left": str(self.board.count(".")),
            "winner": winner,
            "over": "yes" if self.done else "no",
        }

    def expert(self) -> str:
        _, move = _minimax(self.board, self.turn)
        return str(move) if move >= 0 else self.safe_actions()[0]

    def render(self) -> str:
        cells = [c if c != "." else str(i) for i, c in enumerate(self.board)]
        rows = [" ".join(cells[r * 3 : r * 3 + 3]) for r in range(3)]
        return "\n".join(rows)
