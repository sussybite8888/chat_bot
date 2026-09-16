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
    FileAccess.anywhere()     read anything; for a terminal the user owns

**Reading and writing are two policies, not one.** The agent can now keep files
as well as read them (`/write`, `/append`, `/rm`, `/files`), and the model can
reach for those itself ([actions.py](actions.py)) — so "may write" had to be
answerable separately from "may read", and the answer is always a single
directory: the `workspace`.

    read                                   write
    rooted(dir)   -> under dir             -> under dir (the same place)
    anywhere()    -> any file at all       -> only the workspace it was given
    nothing()     -> nothing               -> nothing

That asymmetry is the point of this file. A terminal's user owns the machine and
`/code ~/notes.md` is no more access than `cat` — but the *model* can now start
an operation on its own, and a sampled `[[rm ~/.ssh/id_rsa]]` is not something
anybody asked for. So reads stay as wide as the frontend wants and writes are
confined to one directory, always, for every frontend.

`Rooms` gives each room a `FileStore` directory of its own and roots that room's
agent there, read and write. A room can therefore read what it sent and
whatever it has written down since — and nothing else: not another room's
files, not the server's, not yours. The attachments of the message being
answered are still staged and deleted with the turn; what the agent writes
itself persists until the room is reset.

The workspace is quota'd (`MAX_FILE_BYTES`, `MAX_WORKSPACE_BYTES`,
`MAX_WORKSPACE_FILES`). A model that gets stuck in a loop writing files is a
thing that happens, and a full disk on the box holding the models takes every
room down with it.
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

# The terminal agent's workspace: next to models/ and data/, so what the bot
# wrote down is somewhere a person can go and look at it. Created on the first
# write, not before — a repo that never uses this grows no empty directory.
DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"

# What a workspace may hold. Small on purpose: this is a notepad, not a disk.
MAX_FILE_BYTES = 256 * 1024
MAX_WORKSPACE_BYTES = 4 * 1024 * 1024
MAX_WORKSPACE_FILES = 64
MAX_DEPTH = 3          # how deep a path inside the workspace may nest
READ_CHARS = 4000      # how much of a file one `/read` hands back


class FileRefused(Exception):
    """Why a file operation didn't happen, in words the room can be told. The
    message is the whole point: unlike a read, a refused *write* is about the
    agent's own directory, so there is nothing to give away by explaining."""


def _safe_name(name: str) -> str:
    """A single path component that can only ever name a child of a directory:
    no separators, no traversal, nothing a remote filename can smuggle in."""
    base = Path(name or "file").name.replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", base)
    return "file" if cleaned in ("", ".", "..") else cleaned


class FileAccess:
    """The agent's filesystem policy: which paths it is allowed to open."""

    def __init__(self, root: Path | None = None, unrestricted: bool = False,
                 workspace: Path | None = None) -> None:
        # Resolved once, because containment is only meaningful between two
        # fully-resolved paths — /tmp is a symlink to /private/tmp on macOS, and
        # comparing one resolved path against an unresolved root fails for
        # perfectly legitimate files.
        self.root = Path(root).resolve() if root is not None else None
        self.unrestricted = unrestricted
        # The one directory this agent may write in — never "wherever it can
        # read". It need not exist yet: the first write creates it.
        self.workspace = Path(workspace).resolve() if workspace is not None else None

    @classmethod
    def nothing(cls) -> "FileAccess":
        """Read nothing, write nothing. The default, so a new frontend that
        forgets to think about this is closed rather than open."""
        return cls()

    @classmethod
    def rooted(cls, root: Path) -> "FileAccess":
        """Read and write only what is under `root` — a room's own directory,
        which is both its whole filesystem and its notepad."""
        return cls(root=root, workspace=root)

    @classmethod
    def anywhere(cls, workspace: Path | None = None) -> "FileAccess":
        """Read any file the process can. Only for a frontend whose input comes
        from the person running it — the terminal agent, and nothing else.

        Writing is *not* opened up to match: it goes to `workspace` (by default
        none at all), because a read is something the user asked for and a
        write can now be something the model thought of."""
        return cls(unrestricted=True, workspace=workspace)

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

    # ------------------------------------------------------------- writing

    @property
    def writes(self) -> bool:
        return self.workspace is not None

    def for_write(self, raw: str) -> Path:
        """Where `raw` means, as somewhere this agent may write.

        Refuses anything that leaves the workspace by any spelling — `..`, an
        absolute path, a symlink planted inside it pointing out — and anything
        that isn't a plain file, so `/write /dev/sda ...` is a sentence rather
        than an incident. The parent is *not* created here; `write_text` does
        that once it knows the write is going ahead.
        """
        if self.workspace is None:
            raise FileRefused("I have nowhere to write — no workspace here.")
        # No expanduser(): "~" is the host's home, which is never what a room
        # means and never somewhere it may write.
        raw = raw.strip()
        path = Path(raw)
        if not raw or path.name in {"", ".", ".."}:
            raise FileRefused("that isn't a filename.")
        if raw.startswith("~"):
            # Contained either way — "~/x" is relative, so it would land in a
            # folder literally called "~" inside the workspace — but a path
            # that *looks* like a home directory and isn't one is a trap for
            # whoever reads the reply. Say no instead of quietly reinterpreting.
            raise FileRefused("I can't write to a home directory.")
        candidate = path if path.is_absolute() else self.workspace / path
        try:
            resolved = candidate.resolve()
        except OSError:
            raise FileRefused("I can't make sense of that path.") from None
        if not (self.workspace in resolved.parents):
            raise FileRefused("I can only write inside my own folder.")
        if len(resolved.relative_to(self.workspace).parts) > MAX_DEPTH:
            raise FileRefused(f"that's deeper than {MAX_DEPTH} folders.")
        if resolved.exists() and not resolved.is_file():
            raise FileRefused(f"{resolved.name} isn't a file I can write.")
        return resolved

    def listing(self) -> list[tuple[str, int]]:
        """Everything in the workspace, as (path relative to it, bytes), sorted.
        Staged attachments live in a subdirectory of a room's workspace and are
        listed too — for the turn they exist, they are as real as anything."""
        if self.workspace is None or not self.workspace.is_dir():
            return []
        out = []
        for path in sorted(self.workspace.rglob("*")):
            if path.is_file() and not path.is_symlink():
                out.append((str(path.relative_to(self.workspace)),
                            path.stat().st_size))
        return out

    def usage(self) -> tuple[int, int]:
        """(files, bytes) in the workspace right now."""
        sizes = [size for _, size in self.listing()]
        return len(sizes), sum(sizes)

    def write_text(self, raw: str, text: str, append: bool = False) -> Path:
        """Write (or append) `text`, inside the quota. Returns the path."""
        path = self.for_write(raw)
        existing = path.stat().st_size if path.is_file() else 0
        if append and existing and not self._ends_in_newline(path):
            # Appending to a notepad means adding a *line*. Without this,
            # `/append notes.txt and pong` glues itself onto the last word.
            text = "\n" + text
        data = text.encode("utf-8")
        size = existing + len(data) if append else len(data)
        if size > MAX_FILE_BYTES:
            raise FileRefused(f"that would make {path.name} bigger than "
                              f"{MAX_FILE_BYTES // 1024} KB.")
        files, total = self.usage()
        if not path.is_file() and files >= MAX_WORKSPACE_FILES:
            raise FileRefused(f"I'm already holding {MAX_WORKSPACE_FILES} files "
                              f"— /rm one first.")
        if total - existing + size > MAX_WORKSPACE_BYTES:
            raise FileRefused(f"my folder is full ({MAX_WORKSPACE_BYTES // 1024 // 1024} "
                              f"MB) — /rm something first.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a" if append else "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as e:
            raise FileRefused(f"I couldn't write that ({e.strerror}).") from None
        return path

    @staticmethod
    def _ends_in_newline(path: Path) -> bool:
        try:
            with path.open("rb") as fh:
                fh.seek(-1, 2)
                return fh.read(1) == b"\n"
        except OSError:
            return True  # unreadable: don't add a newline on top of a problem

    def remove(self, raw: str) -> Path:
        """Delete one file from the workspace. Only ever a file: a stray `/rm`
        should not be able to take a directory tree with it."""
        path = self.for_write(raw)
        if not path.is_file():
            raise FileRefused(f"there's no {raw.strip()} in my folder.")
        try:
            path.unlink()
        except OSError as e:
            raise FileRefused(f"I couldn't delete that ({e.strerror}).") from None
        return path

    def read_text(self, raw: str, limit: int = READ_CHARS) -> tuple[Path, str, bool]:
        """A text file as text: (path, contents, whether it was cut short).

        Goes through the *read* policy, not the workspace, so in a terminal this
        reads what `/code` reads and in a room it reads that room's folder.
        Undecodable bytes come back replaced rather than raising — the answer to
        "what's in this file" should be "that's not text", not a traceback.
        """
        path = self.resolve(raw)
        if path is None:
            raise FileRefused(REFUSED if self.restricted and self.root is None
                              else f"I can't find {raw.strip()} — /files lists "
                                   f"what I have.")
        try:
            data = path.read_bytes()
        except OSError as e:
            raise FileRefused(f"I couldn't read that ({e.strerror}).") from None
        if b"\x00" in data[:4096]:
            raise FileRefused(f"{path.name} isn't text.")
        text = data.decode("utf-8", errors="replace")
        return path, text[:limit], len(text) > limit

    def describe(self) -> str:
        if self.unrestricted:
            reads = "any file on this machine"
        elif self.root is None:
            reads = "no files"
        else:
            reads = f"files under {self.root}"
        if self.workspace is None:
            return reads
        if self.root == self.workspace:
            return f"{reads}, and writes there too"
        return f"{reads}; writes only in {self.workspace}"


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
