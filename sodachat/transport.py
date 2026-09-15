"""Chat-transport plumbing, with nothing from the model side in it.

This is the half of the old rooms.py a *client* needs: message formatting for
a chat transport, the attachment staging a server does, and the environment
switches both ends read. It imports nothing heavier than the standard library
on purpose — a bot that talks to the master API server
([api.py](api.py)) should not pull 2GB of PyTorch into its process just to
split a message at 2000 characters.

The model-side half stayed in [rooms.py](rooms.py) (one agent per room over one
copy of the checkpoints), which re-exports everything here so the older imports
keep working.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Transport message-size caps.
DISCORD_LIMIT = 2000
GOOGLE_CHAT_LIMIT = 4096

# Attachments are staged into the room's own directory; anything bigger than
# this is skipped rather than pulled into the process. The client checks it
# before downloading, the server checks it again before writing.
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024


def room_id(frontend: str, room: str) -> str:
    """Namespace a room by the frontend that owns it ("discord:123",
    "googlechat:spaces/AAA"). One server now holds the rooms of every bot
    pointed at it, and two transports' ids are not from the same space."""
    return f"{frontend}:{room}"


def agent_mode_enabled() -> bool:
    """Whether the model server runs the full agent (default) or the plain chat
    engine. `SODACHAT_AGENT=0` restores the pre-specialist behaviour."""
    return os.environ.get("SODACHAT_AGENT", "1").lower() not in {"0", "false", "no"}


def filtered_enabled() -> bool:
    return os.environ.get("SODACHAT_UNFILTERED", "").lower() not in {"1", "true", "yes"}


def api_url() -> str | None:
    """The master API server a frontend should use, or None to load the models
    into the frontend's own process (the single-bot arrangement)."""
    url = os.environ.get("SODACHAT_API_URL", "").strip()
    return url.rstrip("/") or None


def api_key() -> str | None:
    key = os.environ.get("SODACHAT_API_KEY", "").strip()
    return key or None


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


@dataclass
class Attachment:
    """A file a room sent, in memory: what crosses the wire to the server.

    A frontend downloads the bytes from its transport, the server writes them
    into the room's own directory ([files.py](files.py)) and names the paths in
    the message, and the agent then sees exactly what a dropped file looks like
    in a terminal."""

    filename: str
    data: bytes = field(repr=False)

    @property
    def size(self) -> int:
        return len(self.data)


# Staging a room's files, and the sandbox they land in, live in files.py: they
# are about what the agent is allowed to open, not about the transport.
