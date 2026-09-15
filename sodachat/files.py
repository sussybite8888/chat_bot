"""Which files the agent may open, and how a room's files get to it.

The agent reads files: `/see photo.png` classifies an image, `/code app.py`
names a language, and a path sitting in a message is picked up and handled by
what it *is*. In a terminal that is exactly right — the person typing owns the
machine, and `/see ~/Desktop/cat.png` is no more access than `cat` would be.

Through a bot it is not. Whoever can message the bot is not whoever runs the
server, and the models now run in a **shared** server ([api.py](api.py)) that
several bots talk to, so "read any file" would mean any stranger in any channel
could walk the host filesystem — `/see /home/you/.ssh/id_rsa.png`, `/code
/etc/passwd`... — and the reply would hand back what it found.

So file access is a policy the frontend chooses, not something the agent decides:

    FileAccess.nothing()      the default — no file is readable
    FileAccess.rooted(dir)    only what's under `dir`, symlinks resolved
    FileAccess.anywhere()     no restriction; for a terminal the user owns

`Rooms` gives each room a `FileStore` directory of its own and roots that room's
agent there, and the only thing ever written into it is the attachments of the
message being answered, deleted again when the turn ends. So a room can read the
files it just sent and nothing else: not another room's files, not the server's,
not yours.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from .transport import Attachment

# What a refusal says. Deliberately the same whether the path is outside the
# sandbox, missing, or not a regular file: a bot that answers "no such file" for
# one path and "not allowed" for another is an oracle for what exists on the
# host, which is most of what a prober wants.
REFUSED = ("I can only read files sent to me in this conversation — post it "
           "here and I'll take a look.")


def _safe_name(name: str) -> str:
    """A single path component that can only ever name a child of a directory:
    no separators, no traversal, nothing a remote filename can smuggle in."""
    base = Path(name or "file").name.replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", base)
    return "file" if cleaned in ("", ".", "..") else cleaned


class FileAccess:
    """The agent's filesystem policy: which paths it is allowed to open."""

    def __init__(self, root: Path | None = None, unrestricted: bool = False) -> None:
        # Resolved once, because containment is only meaningful between two
        # fully-resolved paths — /tmp is a symlink to /private/tmp on macOS, and
        # comparing one resolved path against an unresolved root fails for
        # perfectly legitimate files.
        self.root = Path(root).resolve() if root is not None else None
        self.unrestricted = unrestricted

    @classmethod
    def nothing(cls) -> "FileAccess":
        """Read nothing. The default, so a new frontend that forgets to think
        about this is closed rather than open."""
        return cls()

    @classmethod
    def rooted(cls, root: Path) -> "FileAccess":
        """Read only what is under `root`."""
        return cls(root=root)

    @classmethod
    def anywhere(cls) -> "FileAccess":
        """Read any file the process can. Only for a frontend whose input comes
        from the person running it — the terminal agent, and nothing else."""
        return cls(unrestricted=True)

    @property
    def restricted(self) -> bool:
        return not self.unrestricted

    def allows(self, path: Path) -> bool:
        """Whether `path` may be opened. Symlinks are resolved first: a link
        inside the sandbox pointing at /etc/passwd is not inside the sandbox."""
        if self.unrestricted:
            return True
        if self.root is None:
            return False
        try:
            resolved = path.resolve()
        except OSError:  # a broken link, a loop, a path we can't walk
            return False
        return resolved == self.root or self.root in resolved.parents

    def resolve(self, raw: str) -> Path | None:
        """A path as written in a message, as a path this policy allows — or
        None. A bare name under a sandbox is taken as relative to it, so a file
        just posted to the room can be named without its staged directory.

        Existence is checked here too, so every caller gets one answer to "can I
        open this?" rather than each asking half the question."""
        if self.unrestricted:
            path = Path(raw).expanduser()
        elif self.root is None:
            return None
        else:
            # No expanduser(): "~" means the *server's* home, which is never
            # what a room is asking for and never somewhere it may read.
            path = Path(raw)
            if not path.is_absolute():
                path = self.root / path
        if not self.allows(path):
            return None
        try:
            # A regular file, not a directory, a device or a fifo: `/see` on
            # /dev/zero should not be a way to hang the server.
            return path if path.is_file() else None
        except OSError:
            return None

    def describe(self) -> str:
        if self.unrestricted:
            return "any file on this machine"
        if self.root is None:
            return "no files"
        return f"files under {self.root}"


class FileStore:
    """The directory tree a set of rooms is allowed to read: one subdirectory
    per room, private to it and holding only the message being answered.

    The base is a temporary directory this process creates and owns, so there is
    nothing to misconfigure into pointing at someone's home directory, and it
    goes away with `close()`."""

    def __init__(self, base: Path | None = None) -> None:
        self._owned = base is None
        self.base = Path(base or tempfile.mkdtemp(prefix="sodachat-files-")).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def room_dir(self, room: str) -> Path:
        """The directory that is `room`'s whole filesystem. Named after the room
        but not *by* it — a room id is remote input, and the hash keeps two ids
        that sanitize to the same name apart."""
        digest = hashlib.sha256(room.encode()).hexdigest()[:8]
        path = self.base / f"{_safe_name(room)}-{digest}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @contextmanager
    def staged(self, room: str, attachments: Sequence[Attachment]) -> Iterator[list[Path]]:
        """Write this turn's attachments inside the room's directory, and delete
        them when the turn ends — the agent keeps the verdict it read off a
        file, never the file, so nothing needs them afterwards."""
        if not attachments:
            yield []
            return
        turn = Path(tempfile.mkdtemp(dir=self.room_dir(room)))
        try:
            paths = []
            for attachment in attachments:
                path = turn / _safe_name(attachment.filename)
                path.write_bytes(attachment.data)
                paths.append(path)
            yield paths
        finally:
            shutil.rmtree(turn, ignore_errors=True)

    def forget(self, room: str) -> None:
        """Drop everything a room left behind."""
        shutil.rmtree(self.room_dir(room), ignore_errors=True)

    def close(self) -> None:
        if self._owned:
            shutil.rmtree(self.base, ignore_errors=True)


def with_paths(text: str, paths: Sequence[Path]) -> str:
    """Rewrite a message so the agent finds the files that came with it. Paths
    are quoted because that is how a terminal pastes a dropped file, and
    `SodaAgent._find_file_path` already reads that form."""
    if not paths:
        return text
    named = " ".join(f"'{p}'" for p in paths)
    return f"{text.strip()} {named}".strip()
