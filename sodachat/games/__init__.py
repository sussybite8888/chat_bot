"""Pluggable real-time game control for the tiny GPT.

Import a game module to register it; `GAMES` maps name -> class. Add a new
game by subclassing `Game` (see snake.py) and decorating with `@register` —
the training, evaluation, and play pipelines pick it up automatically.
"""

from .core import (
    GAMES,
    Game,
    GamePlayer,
    GridTokenizer,
    TextTokenizer,
    benchmark_latency,
    evaluate_headless,
    generate_dataset,
    load_model,
    register,
    tokenizer_for,
)
from . import dodge, pong, snake, tictactoe  # noqa: F401  (register built-ins)

__all__ = [
    "GAMES",
    "Game",
    "GamePlayer",
    "GridTokenizer",
    "TextTokenizer",
    "benchmark_latency",
    "evaluate_headless",
    "generate_dataset",
    "load_model",
    "register",
    "tokenizer_for",
]
