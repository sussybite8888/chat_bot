"""Shared plumbing for the chat-room frontends (Discord, Google Chat).

A room bot is the terminal agent ([agent.py](agent.py)) with four differences,
and this module is those four differences so the two frontends don't each grow
their own version:

  * **Many conversations at once.** Each room gets its own `SodaAgent` — its own
    history, its own running game, its own memory of the last image seen — while
    the loaded checkpoints are shared between them all (`SodaAgent(shared=...)`).
    One room per agent, one copy of the models per process.
  * **Files arrive as attachments, not paths.** The agent recognizes an image or
    a source file by finding an existing path in the message text, which is what
    a dropped file looks like in a terminal. `stage_attachments` downloads what a
    room sent to a scratch directory and rewrites the message to name it, so
    `/see`-style handling works with no change to the agent.
  * **The transport renders Markdown and caps message length.** A game board or
    a `/help` table is column-aligned text that a proportional font mangles, so
    `format_reply` fences the blocks (and only the blocks) and splits the result
    into transport-sized pieces without leaving a fence unclosed.
  * **The event loop must not block.** Generation takes seconds; `Rooms.reply`
    runs it in a worker thread, one at a time, so heartbeats and typing
    indicators keep flowing while a reply is being written.

Set `SODACHAT_AGENT=0` to get the plain chat engine instead — the frontends'
behaviour before the specialists existed.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

from .agent import SodaAgent

# Transport message-size caps.
DISCORD_LIMIT = 2000
GOOGLE_CHAT_LIMIT = 4096

# Attachments are downloaded to a scratch directory; anything bigger than this
# is skipped rather than pulled into the process.
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024


def agent_mode_enabled() -> bool:
    """Whether frontends should run the full agent (default) or the plain chat
    engine. `SODACHAT_AGENT=0` restores the pre-specialist behaviour."""
    return os.environ.get("SODACHAT_AGENT", "1").lower() not in {"0", "false", "no"}


def filtered_enabled() -> bool:
    return os.environ.get("SODACHAT_UNFILTERED", "").lower() not in {"1", "true", "yes"}


# ------------------------------------------------------------------ formatting


# Text that needs a monospace font to make sense: the block glyphs the grid
# games draw with, indented code, and column-aligned output like /help and
# /model. Prose almost never has two consecutive spaces mid-line, so that last
# one is a good enough tell.
_MONOSPACE = re.compile(
    r"[█▲●·■▓▒░│─┌┐└┘├┤┬┴┼]"       # snake, pong, dodge, sandbox
    r"|^[ \t]{2,}\S"                # indented code
    r"|\S {2,}\S",                  # aligned columns
    re.MULTILINE)


def _looks_like_a_grid(block: str) -> bool:
    """Two or more equal-width lines drawn from a small alphabet: a rendered
    board or image, whatever glyphs it happens to use. Needed because not every
    board has a telltale glyph — tic-tac-toe renders as "0 1 2 / 3 4 5 / 6 7 8",
    which has no block characters and no indentation and is still nonsense in a
    proportional font. Prose doesn't come in equal-width lines, and doesn't
    reuse a dozen characters for a whole paragraph."""
    lines = [line for line in block.split("\n") if line.strip()]
    if len(lines) < 2 or len({len(line) for line in lines}) != 1:
        return False
    return 3 <= len(lines[0]) <= 80 and len(set(block) - {"\n"}) <= 20


def _needs_monospace(block: str) -> bool:
    return "\n" in block.strip() and (bool(_MONOSPACE.search(block))
                                      or _looks_like_a_grid(block))


def fence_blocks(text: str) -> str:
    """Wrap the parts of a reply that need a monospace font in code fences,
    paragraph by paragraph. Only the blocks: a move reply is "I'll take my
    turn." + a board + "Your move:", and fencing the whole thing would put the
    sentences in a code block too."""
    parts = re.split(r"\n[ \t]*\n", text)
    return "\n\n".join(
        f"```\n{p.strip(chr(10))}\n```" if _needs_monospace(p) else p for p in parts)


def _hard_wrap(line: str, limit: int) -> list[str]:
    return [line[i:i + limit] for i in range(0, len(line), limit)] or [""]


def split_message(text: str, limit: int) -> list[str]:
    """Split a formatted reply into transport-sized pieces at line boundaries,
    keeping code fences balanced: a piece that ends inside a fence closes it, and
    the next one opens a new one."""
    budget = max(limit - 8, 16)  # room for the ``` a split fence costs
    pieces: list[str] = []
    current: list[str] = []
    open_fence = False

    def flush() -> None:
        if current:
            body = "\n".join(current)
            pieces.append(f"{body}\n```" if open_fence else body)
            current.clear()

    for raw in text.split("\n"):
        for line in _hard_wrap(raw, budget):
            length = sum(len(x) + 1 for x in current) + len(line)
            if current and length > budget:
                reopen = open_fence
                flush()
                if reopen:
                    current.append("```")
            current.append(line)
            if line.startswith("```"):
                open_fence = not open_fence
    flush()
    return [p for p in pieces if p.strip()] or [""]


def format_reply(text: str, limit: int) -> list[str]:
    """An agent reply as messages a chat room can render."""
    return split_message(fence_blocks(text.strip()), limit)


# ----------------------------------------------------------------- attachments


class Scratch:
    """A temporary directory for downloaded attachments, cleaned up on close.

    Files live only for the duration of one message: the agent reads an image or
    a source file immediately and keeps just the verdict (`seen_image` is a
    label, not a path), so nothing needs them afterwards."""

    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="sodachat-room-"))

    def path_for(self, filename: str) -> Path:
        # Trust nothing about a remote filename: keep the suffix, which is what
        # routes the file to a specialist, and drop the rest of the path.
        name = Path(filename or "file").name.replace(" ", "_")
        return self.dir / (re.sub(r"[^A-Za-z0-9._-]", "", name) or "file")

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def handled_suffixes() -> tuple[str, ...]:
    """The file types a specialist can do something with, straight from the
    agent so this list can't drift from the one it matches against."""
    return tuple(SodaAgent._IMAGE_EXTS) + tuple(SodaAgent._CODE_EXTS)


def with_paths(text: str, paths: list[Path]) -> str:
    """Rewrite a message so the agent finds the files that came with it. Paths
    are quoted because that is how a terminal pastes a dropped file, and
    `SodaAgent._find_file_path` already reads that form."""
    if not paths:
        return text
    named = " ".join(f"'{p}'" for p in paths)
    return f"{text.strip()} {named}".strip()


# ----------------------------------------------------------------------- rooms


class Rooms:
    """One agent per room, one set of models for the process."""

    def __init__(self, device: str | None = None, filtered: bool = True) -> None:
        self.device = device or os.environ.get("SODACHAT_DEVICE", "cpu")
        self.filtered = filtered
        self._shared: dict = {}
        self._agents: dict[str, SodaAgent] = {}
        # Generation is single-threaded per process: the models are shared, and
        # two rooms decoding through the same weights at once buys nothing.
        self._lock = asyncio.Lock()

    def agent(self, room: str) -> SodaAgent:
        if room not in self._agents:
            self._agents[room] = SodaAgent(device=self.device, filtered=self.filtered,
                                           shared=self._shared)
        return self._agents[room]

    def warm_up(self) -> None:
        """Load the models before the first message, so nobody waits ~10s for a
        reply that also had to read 190MB off disk."""
        self.agent("__warmup__").handle("hi")
        self._agents.pop("__warmup__", None)

    async def reply(self, room: str, text: str) -> str:
        """One reply, generated off the event loop and one at a time."""
        async with self._lock:
            return await asyncio.to_thread(self.agent(room).handle, text)

    def stop(self) -> None:
        """Stop anything still running in the background (a room that left a
        game going keeps a thread stepping it)."""
        for agent in self._agents.values():
            if agent.continuous is not None:
                agent.continuous.stop()
        self._agents.clear()
