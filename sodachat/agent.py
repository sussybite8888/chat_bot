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
from pathlib import Path

from .game_train import game_model_path
from .games import GAMES, GamePlayer, load_model

_MODELS = Path(__file__).resolve().parent.parent / "models"


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
        self.error: Exception | None = None  # set if the play thread dies
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "ContinuousGame":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    if self.game.done:
                        self.best = max(self.best, self.game.score)
                        self.games_played += 1
                        self.game = self.game_cls(seed=self.games_played)
                    else:
                        self.game.step(self.player.act(self.game))
                time.sleep(self.period)
        except Exception as e:  # never die silently — record it and stop cleanly
            self.error = e
            self._stop.set()

    def view(self) -> tuple[str, str, str]:
        """(board, one-line state, score) read atomically."""
        with self._lock:
            return (self.game.render(), self.game.describe(),
                    str(self.game.facts()["score"]))

    def board_text(self):
        """A compact, coloured half-block board (a `rich.text.Text`) for the live
        display — half the height of `render()`, so it fits the terminal."""
        from .games.core import half_block_render

        with self._lock:
            return half_block_render(self.game)

    def facts(self) -> dict:
        with self._lock:
            return dict(self.game.facts())

    def best_score(self) -> int:
        with self._lock:
            return max(self.best, self.game.score)

    def hud(self) -> str:
        with self._lock:
            note = f"   [stopped: {self.error}]" if self.error else ""
            return (f"score {self.game.score}   best {max(self.best, self.game.score)}"
                    f"   game #{self.games_played + 1}{note}")

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


class _InstructPlayer:
    """Instruction-conditioned mover: reads the agent's current goal each move,
    so `/goal <instruction>` steers a running game (the VLA behaviour)."""

    def __init__(self, lm, agent):
        self.lm = lm
        self.agent = agent
        self.last_ms = 0.0

    def act(self, game) -> str:
        legal = [a for a in game.ACTIONS if game.legal(a)]
        t0 = time.perf_counter()
        move = self.lm.instruct_move(game.NAME, game.render(), self.agent.goal, legal)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return move


class _ExpertPlayer:
    """Mover for the task-routed expert model: one forward pass through the game
    expert + action head. Uses the game's natural goal unless the user set one
    with /goal (then that steers the moves)."""

    def __init__(self, lm, agent):
        self.lm = lm
        self.agent = agent
        self.last_ms = 0.0

    def act(self, game) -> str:
        legal = [a for a in game.ACTIONS if game.legal(a)]
        goal = self.agent.goal if self.agent.goal_set else None
        t0 = time.perf_counter()
        move = self.lm.move(game.NAME, game.model_board(), legal, goal=goal)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return move


class SodaAgent:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._chat = None
        self._reader = None
        self._lms: dict = {}  # UnifiedLM/ExpertLM cache, keyed by checkpoint path
        # ChatEngine wrapping an active LM, so MMI relevance reranking runs in
        # every mode (not just specialist). Keyed by checkpoint path so the
        # per-engine recent-reply deque persists across mode switches.
        self._engines: dict = {}
        # Default to the task-routed expert: one model that chats, reads, plays
        # every game (goal-conditioned), and hosts the vision specialist. Once it
        # was retrained on the 20x20 boards — reading the compact `model_board()`
        # ASCII, so a full board fits the context — it plays snake *better* than
        # the dedicated specialist (26 vs 12) and ties dodge, while keeping chat.
        # Switch to per-task specialists with /model specialist (still best at pong).
        self.mode = "expert"
        self.goal = "eat the food"  # instruction used for moves (instruct/expert)
        self.goal_set = False  # True once the user picks a goal with /goal
        self.history: list[str] = []
        self.continuous: ContinuousGame | None = None  # running grid game
        self.game = None  # interactive text game (tic-tac-toe)
        self.player = None
        self._counter = None  # tokenizer used only to count reply tokens
        self.seen_image: dict | None = None  # what /see (or a dropped image) last saw
        self.seen_code: dict | None = None   # what /code (or a dropped source file) last read
        self.stats = {"replies": 0, "tokens": 0, "seconds": 0.0}
        self.last = {"tokens": 0, "ms": 0.0, "tok_s": 0.0, "hz": 0.0}

    # -------------------------------------------------------------- routing

    def handle(self, text: str) -> str:
        s = text.strip()
        if s.startswith("/"):
            name, _, arg = s[1:].partition(" ")
            command = _COMMANDS.get(name.lower())
            if command is None:
                return f"Unknown command /{name}. Try /help."
            return command(self, arg.strip())
        # An image dropped into the chat (its path, quoted or bare) is looked
        # at right away; plain-text questions that name it ("what's in the
        # picture?") are answered from what was seen — the same pattern as the
        # reader answering questions about the live game state.
        if (img := self._find_image_path(s)) is not None:
            return self._look_at(img, s)
        if (answer := self._image_answer(s)) is not None:
            return answer
        # A source file dropped into the chat is read by the code specialist —
        # it names the language (and can continue it). Same pattern as /see.
        if (code := self._find_code_path(s)) is not None:
            return self._read_code(code, s)
        # "write a function that ..." → the code-generation specialist. Checked
        # before code Q&A so "can you write code that ...?" generates rather than
        # being read as a question about a previously-seen file.
        if (answer := self._code_gen_answer(s)) is not None:
            return answer
        # A question worth working through ("how many apples are left if ...") →
        # the reasoning specialist, which shows its steps before answering.
        # Deliberately ahead of the code Q&A check below, whose trigger list
        # includes the word "in" and so would otherwise swallow word problems.
        if (answer := self._reason_answer(s)) is not None:
            return answer
        if (answer := self._code_answer(s)) is not None:
            return answer
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

    def _engine_for(self, lm) -> "ChatEngine":
        """Wrap an active LM (ExpertLM/UnifiedLM) in a ChatEngine so its chat
        runs through MMI relevance reranking. Cached per checkpoint path."""
        from .engine import ChatEngine

        path = self._mode_path()
        key = str(path) if path is not None else id(lm)
        if key not in self._engines:
            self._engines[key] = ChatEngine(lm=lm)
        return self._engines[key]

    def _mode_path(self, mode: str | None = None):
        from .expert import DEFAULT_PATH as EXPERT_PATH
        from .instruct import OUT_PATH
        from .unified import DEFAULT_PATH

        return {"unified": DEFAULT_PATH, "instruct": OUT_PATH,
                "expert": EXPERT_PATH}.get(mode or self.mode)

    def _active_lm(self):
        """The single model for the current mode (unified/instruct/expert), or
        None for specialist mode. Cached per checkpoint path."""
        path = self._mode_path()
        if path is None or not path.exists():
            return None
        if path not in self._lms:
            if self.mode == "expert":
                from .expert import ExpertLM

                self._lms[path] = ExpertLM(path, self.device)
            else:
                from .unified import UnifiedLM

                self._lms[path] = UnifiedLM(path, self.device)
        return self._lms[path]

    def _vision_lm(self):
        """The ExpertLM that hosts the vision specialist, for /see — available
        in any mode. Reuses the expert model if it's already loaded (expert
        mode caches it under the same path), else loads it once to host the
        specialist. None if the expert or the vision specialist isn't trained."""
        from .expert import DEFAULT_PATH as EXPERT_PATH
        from .vision import NAME

        return self._specialist_lm(EXPERT_PATH, NAME)

    def _code_lm(self):
        """The ExpertLM that hosts the code specialist, for /code — available
        in any mode. None if the expert or the code specialist isn't trained."""
        from .expert import DEFAULT_PATH as EXPERT_PATH
        from .code import NAME

        return self._specialist_lm(EXPERT_PATH, NAME)

    def _specialist_lm(self, expert_path, name: str):
        """Shared backing for _vision_lm / _code_lm: ensure the expert model is
        loaded and return it iff the named specialist is attached, else None."""
        from .expert import ExpertLM

        if not expert_path.exists():
            return None
        if expert_path not in self._lms:
            self._lms[expert_path] = ExpertLM(expert_path, self.device)
        lm = self._lms[expert_path]
        return lm if name in lm.specialists else None

    def _chat_reply(self, text: str) -> str:
        if (lm := self._active_lm()) is not None:  # expert/unified/instruct mode
            # Route through ChatEngine so MMI relevance reranking runs — the
            # single-sample lm.chat() path ignores the prompt too often.
            reply = self._engine_for(lm).reply(text, history=self.history)
            self.history.extend([text, reply.text])
            return reply.text
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
        """The mover for a game, per the current mode: the task-routed expert
        (goal-conditioned game expert), the instruction model, the base unified
        model, or the game's specialist."""
        lm = self._active_lm()
        if self.mode == "expert" and lm is not None:
            return _ExpertPlayer(lm, self)
        if self.mode == "instruct" and lm is not None:
            return _InstructPlayer(lm, self)
        if self.mode == "unified" and lm is not None:
            return _UnifiedPlayer(lm)
        path = game_model_path(name)
        if not path.exists():
            return None
        model, tok, _ = load_model(path, self.device)
        return GamePlayer(model, tok, self.device)

    def _size_mismatch(self, name: str, player) -> str | None:
        """A specialist checkpoint bakes in the board size it was trained on. If
        the game has since been resized (e.g. the 10x10 -> 20x20 enlargement), a
        GamePlayer's fixed-size input buffer can't hold the new board and would
        crash mid-play. Detect that here and explain it instead. The LM movers
        (expert/unified/instruct) read a text board and aren't size-locked."""
        if not isinstance(player, GamePlayer):
            return None
        trained = tuple(getattr(player.tok, "size", ()) or ())
        want = tuple(GAMES[name].SIZE)
        if trained and trained != want:
            return (f"My {name} model was trained for a {trained[0]}x{trained[1]} "
                    f"board, but {name} is now {want[0]}x{want[1]}. Retrain it for "
                    f"the new size:\n  python -m sodachat.game_train --game {name}")
        return None

    def _start_game(self, name: str) -> str:
        player = self._player_for(name)
        if player is None:
            return (f"I haven't learned {name} yet — train it first:\n"
                    f"  python -m sodachat.game_train --game {name}")
        if (mismatch := self._size_mismatch(name, player)) is not None:
            return mismatch
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
        if (lm := self._active_lm()) is not None:  # unified or instruct mode
            return lm.read(fields, text) or self._chat_reply(text)
        if self._reader is None:
            from .reader import Reader

            self._reader = Reader()
        return self._reader.answer(fields, text) or self._chat_reply(text)

    # ------------------------------------------------------- image understanding

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    _CODE_EXTS = (".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".cc", ".hpp",
                  ".go", ".rb", ".php", ".rs", ".sh", ".swift", ".kt")

    def _find_image_path(self, text: str) -> Path | None:
        """An existing image file mentioned in the message — a drag-and-dropped
        path (the terminal pastes it quoted, spaces intact) or a bare token."""
        return self._find_file_path(text, self._IMAGE_EXTS)

    def _find_code_path(self, text: str) -> Path | None:
        """Same idea as `_find_image_path`, for source files dropped into the
        chat — routed to the code specialist (/code, or a bare .py file)."""
        return self._find_file_path(text, self._CODE_EXTS)

    @staticmethod
    def _find_file_path(text: str, exts) -> Path | None:
        """An existing file (by extension) mentioned in the message — a drag-
        and-dropped path (the terminal pastes it quoted, spaces intact) or a
        bare token."""
        import re

        quoted = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", text)]
        bare = [t.strip("'\"").rstrip("?!.,") for t in text.split()]
        for cand in quoted + bare:
            if cand.lower().endswith(exts):
                path = Path(cand).expanduser()
                if path.exists():
                    return path
        return None

    @staticmethod
    def _image_sentence(label: str, conf: float) -> str:
        article = "an" if label[0] in "aeiou" or label == "8" else "a"
        if conf >= 0.75:
            return f"That's {article} {label} — I'm {conf:.0%} sure."
        if conf >= 0.4:
            return f"That looks like {article} {label} ({conf:.0%} sure)."
        return f"I'm not certain — maybe {article} {label}? ({conf:.0%})"

    def _look_at(self, path: Path, text: str = "") -> str:
        """Classify an image with the vision specialist, remember the result
        for follow-up questions, and put a clean version of the exchange in the
        chat history so the conversation can continue about it."""
        lm = self._vision_lm()
        if lm is None:
            return ("I see an image, but my vision specialist isn't trained yet:\n"
                    "  python -m sodachat.vision train")
        try:
            import numpy as np
            from PIL import Image

            px = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        except Exception as e:
            return f"Couldn't read {path.name} as an image: {e}"
        from .vision import classify, render_image

        label, conf = classify(lm, px)
        self.seen_image = {"label": label, "conf": conf, "name": path.name}
        sentence = self._image_sentence(label, conf)
        # History gets the words, not the path or the glyph grid — the chat
        # model should see "user showed a picture, I said dog", nothing weirder.
        asked = " ".join(text.replace(f"'{path}'", "").replace(f'"{path}"', "")
                         .replace(str(path), "").split())
        self.history.extend([asked or "what is this image?", sentence])
        return f"{render_image(px)}\n\n{sentence}"

    def _image_answer(self, text: str) -> str | None:
        """A deterministic answer for plain-text questions about the last seen
        image ("what's in the picture?"), or None to fall through to chat."""
        import re

        low = text.lower()
        if not re.search(r"\b(images?|pictures?|photos?|pics?|screenshots?)\b", low):
            return None
        if "?" not in text and not re.search(
                r"\b(what|which|tell|guess|know|recognize|see|saw|seen|think)\b", low):
            return None
        if self.seen_image is None:
            return ("I haven't seen an image yet — drop one into the chat "
                    "(or /see <file>) and I'll take a look.")
        seen = self.seen_image
        answer = f"{self._image_sentence(seen['label'], seen['conf'])} ({seen['name']})"
        self.history.extend([text, answer])
        return answer

    # The code specialist mirrors the vision one: a dropped source file is
    # classified (and optionally continued), the result is remembered so a
    # plain-text follow-up can ask about it, and the exchange lands in the
    # chat history so the conversation can keep going.

    def _read_code(self, path: Path, text: str = "") -> str:
        """Classify a source file's language with the code specialist, remember
        the result for follow-ups, and put a clean version of the exchange in
        the chat history. If the message says "complete", also continue it."""
        lm = self._code_lm()
        if lm is None:
            return ("I see a source file, but my code specialist isn't trained yet:\n"
                    "  python -m sodachat.code train")
        try:
            snippet = path.read_text(errors="replace")
        except Exception as e:
            return f"Couldn't read {path.name}: {e}"
        from .code import classify, complete

        snippet = snippet.strip()
        if not snippet:
            return f"{path.name} looks empty."
        # A small, stable prefix is enough to identify a language and keeps the
        # log readable; full completion only when asked for it.
        want_complete = "complete" in text.lower() or "finish" in text.lower()
        label, conf = classify(lm, snippet)
        self.seen_code = {"label": label, "conf": conf, "name": path.name}
        sentence = self._code_sentence(label, conf)
        body = sentence
        if want_complete:
            cont = complete(lm, snippet)
            body = f"{sentence}\n\ncontinuation:\n{cont}" if cont else sentence
        asked = " ".join(text.replace(f"'{path}'", "").replace(f'"{path}"', "")
                         .replace(str(path), "").split())
        self.history.extend([asked or f"what language is {path.name}?", sentence])
        return body

    def _code_sentence(self, label: str, conf: float) -> str:
        if conf >= 0.75:
            return f"That's {label} — I'm {conf:.0%} sure."
        if conf >= 0.4:
            return f"That looks like {label} ({conf:.0%} sure)."
        return f"I'm not certain — maybe {label}? ({conf:.0%})"

    # A natural-language request to *write* code ("write a python function that
    # reverses a string") is routed by the main loop to the code-generation
    # specialist (codegen.py). The specialist is a completer, not an instruction
    # model, so the request is turned into a seed — a comment describing the task
    # plus a function header — that it continues into code.
    _SEED = {  # language -> (line-comment marker, function opener)
        "python": ("#", "def "), "javascript": ("//", "function "),
        "typescript": ("//", "function "), "go": ("//", "func "),
        "ruby": ("#", "def "), "php": ("//", "function "),
        "java": ("//", ""), "c": ("//", ""), "cpp": ("//", ""),
    }

    def _codegen_lm(self):
        """The ExpertLM hosting the code-*generation* specialist, for /gen and
        "write a function ..." requests. None if it isn't trained yet."""
        from .expert import DEFAULT_PATH as EXPERT_PATH

        return self._specialist_lm(EXPERT_PATH, "codegen")

    def _codegen_seed(self, text: str, force: bool = False):
        """Turn a request into a (language, seed) pair the codegen completer can
        continue, or None if `text` isn't a code-generation request (unless
        `force`, as from the explicit /gen command)."""
        import re

        low = f" {text.lower()} "
        verb = (r"(write|generate|create|implement|make|code up|build|give me|"
                r"show me|need|want)")
        noun = (r"(function|method|def|class|program|script|snippet|code|"
                r"algorithm|routine|one[- ]?liner)")
        langs = r"(python|py|javascript|js|node|typescript|ts|java|c\+\+|cpp|golang|go|ruby|php)"
        if not force and not (
                (re.search(rf"\b{verb}\b", low) and re.search(rf"\b{noun}\b", low))
                or re.search(rf"\b{langs}\b[^.]*\b{noun}\b", low)):
            return None
        lang = "python"
        for key, canon in (("python", "python"), (" py ", "python"),
                           ("typescript", "typescript"), (" ts ", "typescript"),
                           ("javascript", "javascript"), (" js ", "javascript"),
                           ("node", "javascript"), ("c++", "cpp"), (" cpp ", "cpp"),
                           ("java", "java"), ("golang", "go"), (" go ", "go"),
                           ("ruby", "ruby"), ("php", "php"), (" c ", "c")):
            if key in low:
                lang = canon
                break
        comment, opener = self._SEED[lang]
        m = re.search(r"\b(?:called|named)\s+([A-Za-z_]\w*)", text)
        name = m.group(1) if m else self._derive_name(text)
        desc = " ".join(text.split()).rstrip(".!?") or "a helper"
        # Seed the signature start (name + "(") so the model completes params and
        # body on-topic; langs with no simple opener (java/c/cpp) just get the
        # describing comment and generate from there.
        head = f"{opener}{name}(" if (opener and name) else opener
        return lang, f"{comment} {desc}\n{head}"

    _NAME_STOP = frozenset(
        "write writes generate create creates implement make makes build given "
        "give show need want a an the some this that these those function method "
        "def class program script snippet code algorithm routine oneliner one "
        "liner to which for in on of and or with using please could can you my me "
        "it is python py javascript js node typescript ts java golang go ruby php "
        "cpp c takes take returns return string".split())

    def _derive_name(self, text: str) -> str:
        """A snake_case function name from the request's content words, so the
        signature is on-topic ('reverse a string' -> reverse_string)."""
        import re

        words = [w for w in re.findall(r"[a-z]+", text.lower())
                 if w not in self._NAME_STOP]
        return "_".join(words[:3])

    def _run_codegen(self, echo: str, lang: str, seed: str) -> str:
        lm = self._codegen_lm()
        if lm is None:
            return ("I'd generate that, but my code-generation specialist isn't "
                    "trained yet:\n  python -m sodachat.codegen train")
        from .codegen import generate

        code = (seed + generate(lm, seed, max_new_tokens=160)).rstrip()
        # History keeps the words, not the code block, so chat stays coherent.
        self.history.extend([" ".join(echo.split()), f"(generated {lang} code)"])
        return (f"Here's some {lang} — generated by a small model, so read it "
                f"before trusting it:\n\n{code}")

    def _code_gen_answer(self, text: str) -> str | None:
        """Generate code for a natural-language request, or None to fall through
        to chat. This is the main loop 'activating' the codegen specialist."""
        seeded = self._codegen_seed(text)
        return None if seeded is None else self._run_codegen(text, *seeded)

    # ------------------------------------------------------------- reasoning
    #
    # A question that wants working-out rather than small talk ("if I have 12
    # apples and eat 3, how many are left?") goes to the reasoning specialist
    # (reason.py), which writes its steps and then commits to an answer. The
    # trigger has to be conservative: chat is the default, and hijacking ordinary
    # conversation to "reason" about it reads far worse than missing a word
    # problem.

    def _reason_lm(self):
        """The ExpertLM hosting the reasoning specialist, for /think and worked
        questions asked in chat. None if it isn't trained yet."""
        from .expert import DEFAULT_PATH as EXPERT_PATH
        from .reason import NAME

        return self._specialist_lm(EXPERT_PATH, NAME)

    # Asking to be shown the work, in so many words — enough on its own.
    _REASON_VERB = (r"(solve|calculate|compute|work out|figure out|reason|"
                    r"think through|think step|step by step|explain why|prove)")
    # Quantity questions: "how many/much", "what is the total/average/..." — the
    # shapes the corpus is full of.
    _REASON_ASK = (r"(how (many|much|long|old|far|fast)|what('s| is|s) the "
                   r"(total|sum|difference|product|average|mean|answer|result|"
                   r"percentage|remainder))")

    def _reason_request(self, text: str) -> bool:
        """Whether `text` is a question to be worked through. Requires either an
        explicit ask to solve/explain, or a quantity question that actually has
        numbers in it — a bare "how much do you like me?" is chat, not maths."""
        import re

        low = f" {text.lower()} "
        if re.search(rf"\b{self._REASON_VERB}\b", low):
            return True
        has_numbers = len(re.findall(r"\d", text)) >= 2
        return bool(has_numbers and re.search(rf"\b{self._REASON_ASK}", low))

    def _run_reason(self, question: str) -> str:
        lm = self._reason_lm()
        if lm is None:
            return ("I'd think that through, but my reasoning specialist isn't "
                    "trained yet:\n  python -m sodachat.reason train")
        from .reason import think

        steps, answer = think(lm, question)
        if not steps and not answer:
            return "I couldn't get anywhere with that one."
        # History keeps the question and the conclusion, not the whole
        # derivation, so the chat model isn't fed a wall of arithmetic.
        self.history.extend([" ".join(question.split()),
                             answer or "(no answer reached)"])
        body = f"thinking: {steps}" if steps else "(no steps)"
        if answer:
            body += f"\n\nanswer: {answer}"
        return (f"{body}\n\n(worked out by a ~14M-parameter model — it imitates "
                f"reasoning far better than it does arithmetic, so check it.)")

    def _reason_answer(self, text: str) -> str | None:
        """Work through a question step by step, or None to fall through to the
        other handlers. This is the main loop 'activating' the reasoning
        specialist."""
        return self._run_reason(text) if self._reason_request(text) else None

    def _code_answer(self, text: str) -> str | None:
        """A deterministic answer for plain-text questions about the last seen
        source file ("what language was that?"), or None to fall through."""
        import re

        low = text.lower()
        if not re.search(r"\b(language|lang|code|file|snippet|written in|in)\b", low):
            return None
        if "?" not in text and not re.search(
                r"\b(what|which|tell|guess|know|recognize|see|saw|seen|think)\b", low):
            return None
        if self.seen_code is None:
            return ("I haven't read any code yet — drop a source file into the "
                    "chat (or /code <file>) and I'll name its language.")
        seen = self.seen_code
        answer = f"{self._code_sentence(seen['label'], seen['conf'])} ({seen['name']})"
        self.history.extend([text, answer])
        return answer

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

    def _cmd_see(self, arg: str) -> str:
        """Recognize what an image shows — a handwritten digit or an everyday
        photo subject (dog, cat, airplane, ...) — via the expert model's vision
        specialist (see vision.py). Works in any mode. Dropping an image into
        the chat (no command) does the same thing."""
        raw = arg.strip()
        if not raw:
            return ("Usage: /see <image-file> — I'll say what's in it "
                    "(a digit, or a photo subject like a dog or a cat). "
                    "You can also just drop an image into the chat.")
        # Dragging a file into the terminal pastes its path wrapped in quotes
        # (e.g. '/Users/you/Desktop/Screenshot ....png') — strip a matching
        # pair before treating it as a path, or the quotes become part of it.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1]
        path = Path(raw).expanduser()
        if not path.exists():
            return f"No such file: {path}"
        return self._look_at(path)

    def _cmd_code(self, arg: str) -> str:
        """Name a source file's programming language (and, with `complete`,
        continue it) via the expert model's code specialist (see code.py). Works
        in any mode. Dropping a source file into the chat does the same thing."""
        raw = arg.strip()
        if not raw:
            return ("Usage: /code <source-file> [complete] — I'll name its "
                    "language (python, java, javascript, php, ruby, go). Add "
                    "`complete` and I'll continue it too. You can also just "
                    "drop a source file into the chat.")
        # `complete` is an optional trailing flag, not part of the path.
        want_complete = False
        parts = raw.split()
        if parts and parts[-1].lower() in ("complete", "finish", "continue"):
            want_complete = True
            raw = " ".join(parts[:-1])
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1]
        path = Path(raw).expanduser()
        if not path.exists():
            return f"No such file: {path}"
        flag = " complete" if want_complete else ""
        return self._read_code(path, flag)

    def _cmd_gen(self, arg: str) -> str:
        """Generate code from a description via the code-generation specialist:
        `/gen a python function that reverses a string`. In plain chat the same
        request ('write a function ...') is routed here automatically."""
        if not arg.strip():
            return ("Usage: /gen <description> — e.g. `/gen a python function "
                    "that reverses a string`. Or just ask in chat: "
                    "'write a js function to debounce a callback'.")
        lang, seed = self._codegen_seed(arg, force=True)
        return self._run_codegen(arg, lang, seed)

    def _cmd_think(self, arg: str) -> str:
        """Work a question out step by step via the reasoning specialist (see
        reason.py): `/think if I have 12 apples and eat 3, how many are left?`.
        In plain chat, questions that clearly want working-out are routed here
        automatically."""
        if not arg.strip():
            return ("Usage: /think <question> — e.g. `/think a shirt costs $15 "
                    "and jeans cost twice as much, how much for both?`. I'll "
                    "show my steps, then answer.")
        return self._run_reason(arg)

    def _cmd_duel(self, arg: str) -> str:
        # The live duel takes over the terminal, so the terminal loop runs it
        # (like /watch). Reached here only from non-terminal frontends.
        return ("Multiplayer snake is live and keyboard-driven — run /duel at the "
                "terminal prompt (python -m sodachat.agent), or play it standalone "
                "with python -m sodachat.games.versus.")

    # ------------------------------------------------------------- stats

    def _count_tokens(self, text: str) -> int:
        if self._counter is None:
            import torch

            from .model import DEFAULT_MODEL_PATH, tokenizer_from_payload

            ck = torch.load(DEFAULT_MODEL_PATH, map_location="cpu", weights_only=True)
            self._counter = tokenizer_from_payload(ck["tokenizer"])
        return max(1, len(self._counter.encode(text)))

    def record(self, reply: str, seconds: float) -> None:
        """Log the timing of one model reply (called by the terminal loop)."""
        n = self._count_tokens(reply)
        self.stats["replies"] += 1
        self.stats["tokens"] += n
        self.stats["seconds"] += seconds
        self.last = {"tokens": n, "ms": seconds * 1000,
                     "tok_s": n / max(seconds, 1e-9),
                     "hz": 1.0 / max(seconds, 1e-9)}

    def _cmd_stats(self, arg: str) -> str:
        s = self.stats
        if not s["replies"]:
            return "No model replies yet."
        return (f"session: {s['replies']} replies, {s['tokens']} tokens in "
                f"{s['seconds']:.1f}s\n"
                f"  avg {s['tokens'] / max(s['seconds'], 1e-9):.1f} tok/s | "
                f"{s['seconds'] / s['replies'] * 1000:.0f} ms/reply | "
                f"{s['replies'] / max(s['seconds'], 1e-9):.2f} replies/s\n"
                f"  last: {self.last['tokens']} tok, {self.last['ms']:.0f} ms, "
                f"{self.last['tok_s']:.1f} tok/s, {self.last['hz']:.1f} Hz")

    def _cmd_model(self, arg: str) -> str:
        a = arg.strip().lower()
        if a in ("expert", "routed", "moe"):
            return self._set_mode("expert")
        if a in ("unified", "one", "big"):
            return self._set_mode("unified")
        if a in ("instruct", "instruction", "vla"):
            return self._set_mode("instruct")
        if a in ("specialist", "specialists", "separate", "default"):
            return self._set_mode("specialist")
        if a:
            return (f"Unknown model '{arg}'. Use /model expert | specialist | "
                    f"unified | instruct, or /model with no argument for info.")
        return self._model_info()

    def _set_mode(self, mode: str) -> str:
        path = self._mode_path(mode)
        if path is not None and not path.exists():
            return f"No {mode} model trained yet ({path.name} not found)."
        self.mode = mode
        note = " (a running game keeps its current model until /stop + /play)" \
            if self.continuous is not None or self.game is not None else ""
        if mode == "expert":
            steer = (f"currently steering with goal {self.goal!r}" if self.goal_set
                     else "each game uses its natural goal; /goal to steer")
            return ("Now using the expert model — one model whose game and chat "
                    "weights are separated (task-routed experts). Chat/read use the "
                    f"text expert; moves use the game expert + action head ({steer})."
                    + note)
        if mode == "unified":
            return ("Now using the unified model — one 30M model handles chat, "
                    "reading, and moves." + note)
        if mode == "instruct":
            return (f"Now using the instruction model — game moves follow a goal "
                    f"(currently {self.goal!r}; change with /goal). Note: it reads "
                    f"and plays worse than the others." + note)
        return "Now using the specialist models — a separate model per task." + note

    def _cmd_goal(self, arg: str) -> str:
        if arg.strip():
            self.goal = arg.strip()
            self.goal_set = True
            return (f"Goal set to {self.goal!r} — steers game moves in the "
                    f"expert and instruct models (/model expert | instruct).")
        state = "steering moves" if self.goal_set else "not set (games use their natural goal)"
        return f"Current goal: {self.goal!r} ({state}). Set with /goal <instruction>."

    def _model_info(self) -> str:
        from .expert import DEFAULT_PATH as EXPERT_PATH
        from .instruct import OUT_PATH as INSTRUCT_PATH
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

        row("expert", EXPERT_PATH, EXPERT_PATH in self._lms)
        # Pluggable specialists the expert grafts on at load (expert.py) — e.g.
        # vision (image -> digit). Trained separately, auto-loaded with the expert.
        for spec in sorted(_MODELS.glob("specialist-*.pt")):
            name = spec.stem.removeprefix("specialist-")
            loaded = (EXPERT_PATH in self._lms
                      and name in getattr(self._lms[EXPERT_PATH], "specialists", {}))
            row(f"└ {name} specialist", spec, loaded)
        row("chat", DEFAULT_MODEL_PATH, self._chat is not None)
        row("reader", READER_PATH, self._reader is not None)
        row("unified", UNIFIED_PATH, UNIFIED_PATH in self._lms)
        row("instruct", INSTRUCT_PATH, INSTRUCT_PATH in self._lms)
        trained = [n for n in GAMES if game_model_path(n).exists()]
        lines.append(f"  games trained: {', '.join(trained) or 'none'}")
        active = (self.continuous.game_cls.NAME if self.continuous is not None
                  else self.game.NAME if self.game is not None else None)
        if active:
            lines.append(f"  active game: {active}")
        if self.mode in ("instruct", "expert"):
            steer = "steering moves" if self.goal_set else "off; games use natural goal"
            lines.append(f"  goal: {self.goal!r} ({steer}; /goal to change)")
        lines.append("switch with: /model expert | specialist | unified | instruct")
        return "\n".join(lines)

    def _cmd_help(self, arg: str) -> str:
        return "commands:\n" + "\n".join(f"  {c:16} {d}" for c, d in _HELP)


# name -> handler. The dispatch is a table, not a branch chain.
_COMMANDS = {
    "play": SodaAgent._cmd_play,
    "duel": SodaAgent._cmd_duel,
    "versus": SodaAgent._cmd_duel,
    "vs": SodaAgent._cmd_duel,
    "stop": SodaAgent._cmd_stop,
    "score": SodaAgent._cmd_score,
    "state": SodaAgent._cmd_state,
    "status": SodaAgent._cmd_state,
    "board": SodaAgent._cmd_board,
    "games": SodaAgent._cmd_games,
    "see": SodaAgent._cmd_see,
    "recognize": SodaAgent._cmd_see,
    "code": SodaAgent._cmd_code,
    "lang": SodaAgent._cmd_code,
    "gen": SodaAgent._cmd_gen,
    "generate": SodaAgent._cmd_gen,
    "think": SodaAgent._cmd_think,
    "reason": SodaAgent._cmd_think,
    "solve": SodaAgent._cmd_think,
    "model": SodaAgent._cmd_model,
    "info": SodaAgent._cmd_model,
    "goal": SodaAgent._cmd_goal,
    "stats": SodaAgent._cmd_stats,
    "help": SodaAgent._cmd_help,
}

# For /help. /watch and /exit are handled by the terminal loop, not the agent.
_HELP = [
    ("/play <game>", "start a game (snake, pong, dodge, tictactoe)"),
    ("/duel", "play snake against me — you steer, live (WASD/arrows)"),
    ("/stop", "stop the current game"),
    ("/score", "current score"),
    ("/state", "one-line game state (food, length, whose turn, ...)"),
    ("/board", "print the board now"),
    ("/watch", "stream the live board for a few seconds"),
    ("/games", "list games"),
    ("/see <image>", "say what an image shows (or just drop an image into the chat)"),
    ("/code <file>", "name a source file's language (or drop a .py/.js/... into the chat);"
     " add `complete` to continue it"),
    ("/gen <description>", "generate code from a description (or just ask in chat:"
     " 'write a python function that ...')"),
    ("/think <question>", "work a question out step by step, then answer (or just"
     " ask in chat: 'how many are left if ...')"),
    ("/model", "show models, or switch: expert | specialist | unified | instruct"),
    ("/goal", "set the instruction that steers moves (expert/instruct mode)"),
    ("/stats", "generation speed: tok/s, ms/reply, frequency"),
    ("/help", "this list"),
    ("/exit", "leave"),
]


# ------------------------------------------------------------------- terminal


def _watch(agent: SodaAgent, console) -> None:
    """Stream the live board for a few seconds. Input is paused here, so the
    animation never fights with the prompt. Ctrl-C returns early."""
    from rich.align import Align
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    from .games.core import framed_board

    console.print("[dim](watching — Ctrl-C to stop)[/]")
    try:
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            start = time.perf_counter()
            while agent.continuous is not None and time.perf_counter() - start < 8:
                # Half-block board: two grid rows per line, so even the tallest
                # board fits the terminal as it animates in place. Framed so the
                # play-field walls are visible around the empty cells.
                board = agent.continuous.board_text()
                # size the panel to the board so bigger resolutions aren't clipped
                cols = max((len(l) for l in board.plain.splitlines()), default=20)
                hud = Text(agent.continuous.hud(), style="dim")
                width = max(cols + 4, 34)
                live.update(Panel(Group(Align.center(framed_board(board, cols)),
                                        Text(""), hud),
                                  title="playing live", border_style="cyan",
                                  width=width))
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
        if low in {"/duel", "/versus", "/vs"}:  # live, keyboard-driven; takes the screen
            from .games.versus import play_duel

            play_duel(agent=agent, device=agent.device, console=console)
            continue
        if low == "/watch":
            _watch(agent, console) if agent.continuous else console.print(
                "[bold magenta]bot ›[/] No game running.")
            continue
        if not text.strip():
            continue
        is_command = text.strip().startswith("/")
        t0 = time.perf_counter()
        out = agent.handle(text)
        dt = time.perf_counter() - t0
        console.print(f"[bold magenta]bot ›[/] {escape(out)}")
        cg = agent.continuous
        if cg is not None and agent.continuous is cg:  # one-line pulse of the game
            console.print(f"[dim]  {cg.game_cls.NAME}: {cg.hud()}[/]")
        if not is_command:  # model reply — show speed
            agent.record(out, dt)
            console.print(f"[dim]  {agent.last['tokens']} tok · "
                          f"{agent.last['ms']:.0f} ms · {agent.last['tok_s']:.0f} tok/s "
                          f"· {agent.last['hz']:.1f} Hz[/]")

    if agent.continuous is not None:
        agent.continuous.stop()


if __name__ == "__main__":
    main()
