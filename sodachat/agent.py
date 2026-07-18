"""One interface that chats and plays games.

The rule is simple: **plain text always goes to the model** (chat, or a
tic-tac-toe move it replies to), and **everything deterministic is a
`/command`**. So starting or stopping a game, and reading the score/state from
the game object, are explicit slash commands — never guessed from natural
language. Type `/help` for the list.

Grid games (Snake, Pong, Dodge) run *continuously* on their own in a background
thread once started; ask about them with `/score` or `/state` at any time.
Tic-Tac-Toe is turn-based — type a cell number to move and I reply.

    python -m sodachat.agent
"""

from __future__ import annotations

import threading
import time

from .game_train import game_model_path
from .games import GAMES, GamePlayer, load_model


def _match_game(name: str) -> str | None:
    key = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    return next((g for g in GAMES if g.replace("_", "") == key), None)


def _ckpt_info(path) -> dict | None:
    """Read a checkpoint's architecture and training metadata without building
    the model. Params are counted from the state dict, de-duplicating tied
    weights (the LM head shares the embedding)."""
    import torch

    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    seen, params = set(), 0
    for t in ck["state_dict"].values():
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            params += t.numel()
    return {
        "params": params,
        "arch": f"{cfg.get('n_layer', '?')}L/{cfg.get('n_embd', '?')}d/"
                f"{cfg.get('n_head', '?')}h",
        "vocab": cfg.get("vocab_size", "?"),
        "block": cfg.get("block_size", "?"),
        "steps": ck.get("steps"),
        "val": ck.get("val_loss"),
        "mb": path.stat().st_size / 1e6,
    }


class ContinuousGame:
    """Steps a grid game in a background thread at a fixed tick rate, restarting
    a fresh game whenever one ends, so it runs forever until stopped. All access
    to the (not thread-safe) game object goes through a lock, so the main thread
    can safely read the live board while it plays."""

    def __init__(self, game_cls, player: GamePlayer, tps: float = 8.0):
        self.game_cls = game_cls
        self.player = player
        self.period = 1.0 / tps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.game = game_cls(seed=0)
        self.games_played = 0
        self.best = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "ContinuousGame":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self.game.done:
                    self.best = max(self.best, self.game.score)
                    self.games_played += 1
                    self.game = self.game_cls(seed=self.games_played)
                else:
                    self.game.step(self.player.act(self.game))
            time.sleep(self.period)

    def view(self) -> tuple[str, str, str]:
        """(board, one-line state, score) read atomically."""
        with self._lock:
            return (self.game.render(), self.game.describe(),
                    str(self.game.facts()["score"]))

    def facts(self) -> dict:
        with self._lock:
            return dict(self.game.facts())

    def best_score(self) -> int:
        with self._lock:
            return max(self.best, self.game.score)

    def hud(self) -> str:
        with self._lock:
            return (f"score {self.game.score}   best {max(self.best, self.game.score)}"
                    f"   game #{self.games_played + 1}")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class _UnifiedPlayer:
    """Adapts the unified model's move() to the .act(game) interface that
    ContinuousGame and tic-tac-toe expect, so the same game loop runs whether
    the mover is a specialist GamePlayer or the one unified model."""

    def __init__(self, lm):
        self.lm = lm
        self.last_ms = 0.0

    def act(self, game) -> str:
        legal = [a for a in game.ACTIONS if game.legal(a)]
        t0 = time.perf_counter()
        move = self.lm.move(game.NAME, game.render(), legal)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return move


class SodaAgent:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._chat = None
        self._reader = None
        self._unified = None
        self.mode = "specialist"  # "specialist" (separate models) or "unified"
        self.history: list[str] = []
        self.continuous: ContinuousGame | None = None  # running grid game
        self.game = None  # interactive text game (tic-tac-toe)
        self.player = None

    # -------------------------------------------------------------- routing

    def handle(self, text: str) -> str:
        s = text.strip()
        if s.startswith("/"):
            name, _, arg = s[1:].partition(" ")
            command = _COMMANDS.get(name.lower())
            if command is None:
                return f"Unknown command /{name}. Try /help."
            return command(self, arg.strip())
        # plain text → the model. During tic-tac-toe a bare cell is a move I
        # reply to; a legal move goes to the game, other text is read against
        # the state. With a grid game running, questions read the live state.
        if self.game is not None:
            legal = [a for a in self.game.ACTIONS if self.game.legal(a)]
            if s in legal:
                return self._text_move(s)
            if s.isdigit():
                return f"Cell {s} isn't open. Try one of: {', '.join(legal)}."
            return self._read(text)
        if self.continuous is not None:
            return self._read(text)
        return self._chat_reply(text)

    # ----------------------------------------------------------------- chat

    def _chat_engine(self):
        if self._chat is None:
            from .engine import ChatEngine

            self._chat = ChatEngine()
        return self._chat

    def _unified_lm(self):
        if self._unified is None:
            from .unified import DEFAULT_PATH, UnifiedLM

            if not DEFAULT_PATH.exists():
                return None
            self._unified = UnifiedLM(DEFAULT_PATH, self.device)
        return self._unified

    def _chat_reply(self, text: str) -> str:
        if self.mode == "unified" and (lm := self._unified_lm()) is not None:
            reply = lm.chat(self.history, text)
            self.history.extend([text, reply])
            return reply
        reply = self._chat_engine().reply(text, history=self.history)
        self.history.extend([text, reply.text])
        return reply.text

    # ---------------------------------------------------------- game control

    def _game_view(self) -> tuple[str, str, str] | None:
        if self.continuous is not None:
            return self.continuous.view()
        if self.game is not None:
            return (self.game.render(), self.game.describe(),
                    str(self.game.facts()["score"]))
        return None

    def _player_for(self, name: str):
        """The mover for a game, per the current mode: the unified model, or the
        game's specialist checkpoint."""
        if self.mode == "unified" and (lm := self._unified_lm()) is not None:
            return _UnifiedPlayer(lm)
        path = game_model_path(name)
        if not path.exists():
            return None
        model, tok, _ = load_model(path, self.device)
        return GamePlayer(model, tok, self.device)

    def _start_game(self, name: str) -> str:
        player = self._player_for(name)
        if player is None:
            return (f"I haven't learned {name} yet — train it first:\n"
                    f"  python -m sodachat.game_train --game {name}")
        if GAMES[name].MODALITY == "text":  # tic-tac-toe: you play me
            self.player, self.game = player, GAMES[name](seed=0)
            return (f"Tic-tac-toe — you're X, I'm O, you first. Type a cell 0-8 to "
                    f"move (/stop to quit):\n\n{self.game.render()}")
        self.continuous = ContinuousGame(GAMES[name], player).start()
        return (f"Playing {name} — I'll keep going on my own. "
                f"/score, /state, /board, /watch to follow along; /stop to end.")

    def _stop_game(self) -> str:
        if self.continuous is not None:
            best = self.continuous.best_score()
            self.continuous.stop()
            self.continuous = None
            return f"Stopped — best score was {best}."
        self.game = self.player = None
        return "Ok, good game!"

    def _text_move(self, move: str) -> str:  # move already validated as legal
        self.game.step(move)
        if not self.game.done:
            self.game.step(self.player.act(self.game))  # my reply move
        if self.game.done:
            outcome = {"X": "You win! Nicely played.", "O": "I win this one!"}
            msg, board = outcome.get(self.game.winner, "A draw."), self.game.render()
            self.game = self.player = None
            return f"{board}\n\n{msg}"
        return f"I'll take my turn.\n\n{self.game.render()}\n\nYour move:"

    def _read(self, text: str) -> str:
        """Answer a question by reading the live game state — with the unified
        model or the specialist reader, per mode. Falls back to chat on silence."""
        from .reader import _facts_to_fields

        facts = (self.continuous.facts() if self.continuous is not None
                 else self.game.facts())
        fields = _facts_to_fields(facts)
        if self.mode == "unified" and (lm := self._unified_lm()) is not None:
            return lm.read(fields, text) or self._chat_reply(text)
        if self._reader is None:
            from .reader import Reader

            self._reader = Reader()
        return self._reader.answer(fields, text) or self._chat_reply(text)

    # ------------------------------------------------------------- commands

    def _cmd_play(self, arg: str) -> str:
        if not arg:
            return f"Usage: /play <game>. {self._cmd_games('')}"
        game = _match_game(arg)
        if game is None:
            return f"I don't know '{arg}'. {self._cmd_games('')}"
        return self._start_game(game)

    def _cmd_stop(self, arg: str) -> str:
        if self.continuous is None and self.game is None:
            return "No game running."
        return self._stop_game()

    def _cmd_score(self, arg: str) -> str:
        v = self._game_view()
        return f"Score: {v[2]}." if v else "No game running — /play <game> to start."

    def _cmd_state(self, arg: str) -> str:
        v = self._game_view()
        return v[1] if v else "No game running."

    def _cmd_board(self, arg: str) -> str:
        v = self._game_view()
        return v[0] if v else "No game running."

    def _cmd_games(self, arg: str) -> str:
        return "Games: " + ", ".join(GAMES) + "."

    def _cmd_model(self, arg: str) -> str:
        a = arg.strip().lower()
        if a in ("unified", "one", "big"):
            return self._set_mode("unified")
        if a in ("specialist", "specialists", "separate", "default"):
            return self._set_mode("specialist")
        if a:
            return (f"Unknown model '{arg}'. Use /model unified or "
                    f"/model specialist, or /model with no argument for info.")
        return self._model_info()

    def _set_mode(self, mode: str) -> str:
        if mode == "unified":
            from .unified import DEFAULT_PATH

            if not DEFAULT_PATH.exists():
                return "No unified model trained yet (models/unified.pt not found)."
        self.mode = mode
        note = " (a running game keeps its current model until /stop + /play)" \
            if self.continuous is not None or self.game is not None else ""
        if mode == "unified":
            return ("Now using the unified model — one 30M model handles chat, "
                    "reading, and moves." + note)
        return "Now using the specialist models — a separate model per task." + note

    def _model_info(self) -> str:
        from .model import DEFAULT_MODEL_PATH
        from .reader import DEFAULT_PATH as READER_PATH
        from .unified import DEFAULT_PATH as UNIFIED_PATH

        lines = [f"models (mode: {self.mode}, device: {self.device})"]

        def row(label, path, loaded):
            info = _ckpt_info(path)
            if info is None:
                lines.append(f"  {label}: not trained ({path.name})")
                return
            val = f", val {info['val']:.2f}" if info["val"] is not None else ""
            step = f" @ {info['steps']:,} steps" if info["steps"] else ""
            tag = " [loaded]" if loaded else ""
            lines.append(
                f"  {label}: {path.name} — {info['params'] / 1e6:.1f}M params, "
                f"{info['arch']}, vocab {info['vocab']}{val}{step}{tag}"
            )

        row("chat", DEFAULT_MODEL_PATH, self._chat is not None)
        row("reader", READER_PATH, self._reader is not None)
        row("unified", UNIFIED_PATH, self._unified is not None)
        trained = [n for n in GAMES if game_model_path(n).exists()]
        lines.append(f"  games trained: {', '.join(trained) or 'none'}")
        active = (self.continuous.game_cls.NAME if self.continuous is not None
                  else self.game.NAME if self.game is not None else None)
        if active:
            lines.append(f"  active game: {active}")
        lines.append("switch with: /model unified | /model specialist")
        return "\n".join(lines)

    def _cmd_help(self, arg: str) -> str:
        return "commands:\n" + "\n".join(f"  {c:16} {d}" for c, d in _HELP)


# name -> handler. The dispatch is a table, not a branch chain.
_COMMANDS = {
    "play": SodaAgent._cmd_play,
    "stop": SodaAgent._cmd_stop,
    "score": SodaAgent._cmd_score,
    "state": SodaAgent._cmd_state,
    "status": SodaAgent._cmd_state,
    "board": SodaAgent._cmd_board,
    "games": SodaAgent._cmd_games,
    "model": SodaAgent._cmd_model,
    "info": SodaAgent._cmd_model,
    "help": SodaAgent._cmd_help,
}

# For /help. /watch and /exit are handled by the terminal loop, not the agent.
_HELP = [
    ("/play <game>", "start a game (snake, pong, dodge, tictactoe)"),
    ("/stop", "stop the current game"),
    ("/score", "current score"),
    ("/state", "one-line game state (food, length, whose turn, ...)"),
    ("/board", "print the board now"),
    ("/watch", "stream the live board for a few seconds"),
    ("/games", "list games"),
    ("/model", "show models, or switch: /model unified | specialist"),
    ("/help", "this list"),
    ("/exit", "leave"),
]


# ------------------------------------------------------------------- terminal


def _watch(agent: SodaAgent, console) -> None:
    """Stream the live board for a few seconds. Input is paused here, so the
    animation never fights with the prompt. Ctrl-C returns early."""
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    console.print("[dim](watching — Ctrl-C to stop)[/]")
    try:
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            start = time.perf_counter()
            while agent.continuous is not None and time.perf_counter() - start < 8:
                board, _, _ = agent.continuous.view()
                live.update(Panel(Text(board + f"\n\n{agent.continuous.hud()}",
                                       style="green"),
                                  title="playing live", border_style="cyan", width=40))
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    from rich.console import Console
    from rich.markup import escape

    console = Console()
    agent = SodaAgent()
    console.print("[bold cyan]sodachat agent[/] — just type to chat. "
                  "Commands start with '/'; type [bold]/help[/].")
    while True:
        try:
            text = console.input("[bold cyan]you ›[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        low = text.strip().lower()
        if low in {"/exit", "/quit"}:
            break
        if low == "/watch":
            _watch(agent, console) if agent.continuous else console.print(
                "[bold magenta]bot ›[/] No game running.")
            continue
        if not text.strip():
            continue
        console.print(f"[bold magenta]bot ›[/] {escape(agent.handle(text))}")
        cg = agent.continuous
        if cg is not None and agent.continuous is cg:  # one-line pulse of the game
            console.print(f"[dim]  {cg.game_cls.NAME}: {cg.hud()}[/]")

    if agent.continuous is not None:
        agent.continuous.stop()


if __name__ == "__main__":
    main()
