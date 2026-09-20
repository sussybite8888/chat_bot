"""The model side of the chat-room frontends: one agent per room, one copy of
the checkpoints for the whole process.

This is what the master API server ([api.py](api.py)) runs. A room bot is the
terminal agent ([agent.py](agent.py)) with six differences, and this module is
those six differences so no frontend grows its own version:

  * **Many conversations at once.** Each room gets its own `SodaAgent` — its own
    history, its own running game, its own memory of the last image seen — while
    the loaded checkpoints are shared between them all (`SodaAgent(shared=...)`).
    One room per agent, one copy of the models per process — and, with the API
    server in front, one copy for *every* bot rather than per bot.
  * **Files arrive as bytes, not paths.** The agent recognizes an image or a
    source file by finding an existing path in the message text, which is what a
    dropped file looks like in a terminal. `Rooms.reply` writes the attachments a
    room sent into a scratch directory and rewrites the message to name them, so
    `/see`-style handling works with no change to the agent — and so a bot on
    another machine can hand over a file it downloaded. That directory is also
    the room's *notepad*: what the agent writes there itself (`/write`) outlives
    the turn and is thrown away with the room.
  * **The transport renders Markdown and caps message length.** Handled by the
    frontend now, with the formatting in [transport.py](transport.py), because
    the cap belongs to the transport and not to the model.
  * **A turn can ask for more than words.** The agent names the acts it wants
    taken in the room it is speaking in — a reaction, a pin ([actions.py](
    actions.py)) — and `reply` returns them alongside the text. Nothing here
    performs one: this process has no gateway connection, and the frontend that
    does is the only thing that can.
  * **It can watch as well as answer.** `watch` is the other half of a chat
    room: what to do about a message nobody addressed to the bot. It generates
    nothing and keeps nothing, so a frontend can run a whole channel through
    it.
  * **The event loop must not block.** Generation takes seconds; `Rooms.reply`
    runs it in a worker thread, one at a time, so an HTTP server (or a gateway
    heartbeat, when the models are in-process) keeps flowing while a reply is
    being written.

Set `SODACHAT_AGENT=0` to serve the plain chat engine instead — the frontends'
behaviour before the specialists existed. That switch lives here rather than in
each frontend, so a bot is the same code either way.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from typing import Sequence

from .actions import Action, Reply, pick_reaction
from .agent import SodaAgent
from .files import MAX_WORKSPACE_BYTES, FileAccess, FileStore, with_paths
from .transport import (
    DISCORD_LIMIT,
    GOOGLE_CHAT_LIMIT,
    MAX_ATTACHMENT_BYTES,
    Attachment,
    agent_mode_enabled,
    fence_blocks,
    filtered_enabled,
    format_reply,
    room_id,
    split_message,
)

__all__ = [
    "DISCORD_LIMIT",
    "GOOGLE_CHAT_LIMIT",
    "MAX_ATTACHMENT_BYTES",
    "Action",
    "Attachment",
    "FileAccess",
    "FileStore",
    "Reply",
    "Rooms",
    "agent_mode_enabled",
    "fence_blocks",
    "filtered_enabled",
    "format_reply",
    "handled_suffixes",
    "room_id",
    "split_message",
    "with_paths",
]

_PLAIN_HISTORY = 8  # lines kept per room when serving plain chat


def handled_suffixes() -> tuple[str, ...]:
    """The file types a specialist can do something with, straight from the
    agent so this list can't drift from the one it matches against."""
    return tuple(SodaAgent._IMAGE_EXTS) + tuple(SodaAgent._CODE_EXTS)


class Rooms:
    """One agent per room, one set of models for the process."""

    def __init__(self, device: str | None = None, filtered: bool = True,
                 agent_mode: bool | None = None) -> None:
        self.device = device or os.environ.get("SODACHAT_DEVICE", "cpu")
        self.filtered = filtered
        self.agent_mode = agent_mode_enabled() if agent_mode is None else agent_mode
        self._shared: dict = {}
        self._agents: dict[str, SodaAgent] = {}
        # One directory per room, and that directory is the whole filesystem as
        # far as that room's agent is concerned (files.py). Rooms are strangers:
        # to each other, and to whoever owns the machine the models run on.
        self._files = FileStore()
        # Plain-chat mode (SODACHAT_AGENT=0): one engine for the process and a
        # few recent lines per room, which is all the state that mode has.
        self._engine = None
        self._histories: dict[str, deque[str]] = {}
        # Generation is single-threaded per process: the models are shared, and
        # two rooms decoding through the same weights at once buys nothing.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ per-room

    def agent(self, room: str) -> SodaAgent:
        if room not in self._agents:
            self._agents[room] = SodaAgent(
                device=self.device, filtered=self.filtered, shared=self._shared,
                files=FileAccess.rooted(self._files.room_dir(room)))
        return self._agents[room]

    def engine(self):
        """The plain chat engine, built once for the process."""
        if self._engine is None:
            from .engine import ChatEngine

            self._engine = ChatEngine(filtered=self.filtered)
        return self._engine

    def rooms(self) -> list[str]:
        """The rooms this server is holding a conversation with."""
        return sorted(self._agents if self.agent_mode else self._histories)

    def seed(self, room: str, history: Sequence[str]) -> None:
        """Give a room a past it didn't live through.

        The chat-room frontends never need this: a room accumulates its history
        one answered turn at a time. The OpenAI- and Anthropic-shaped endpoints
        ([compat.py](compat.py)) do, because those APIs are stateless — the
        client resends the whole conversation every request — so a history this
        server has no room for has to be *replayed* rather than re-generated.
        Generating the assistant's old turns again would cost one decode per
        turn and produce different words than the client already has.

        `history` is flat and alternating — user, assistant, user, ... — which
        is how both the agent and the plain-chat engine already store it, so
        seeding is an assignment rather than a translation.
        """
        if not self.agent_mode:
            self._histories[room] = deque(history, maxlen=_PLAIN_HISTORY)
            return
        self.agent(room).history = list(history)

    def reset(self, room: str) -> bool:
        """Forget a room: its history, its persona, and any game it left
        running. True if there was anything to forget."""
        if not self.agent_mode:
            return self._histories.pop(room, None) is not None
        agent = self._agents.pop(room, None)
        self._files.forget(room)
        if agent is None:
            return False
        if agent.continuous is not None:
            agent.continuous.stop()
        return True

    # --------------------------------------------------------------- info

    def info(self) -> dict:
        """What a client needs to know about this server: how it will behave,
        and what it will accept."""
        return {
            "agent_mode": self.agent_mode,
            "device": self.device,
            "filtered": self.filtered,
            "rooms": len(self.rooms()),
            "accepted_suffixes": list(handled_suffixes()) if self.agent_mode else [],
            "max_attachment_bytes": MAX_ATTACHMENT_BYTES,
            # Not the path — that is the server's business, and a client has no
            # use for it. Only that rooms are confined, which is a promise the
            # other end can check, and that a room can keep files there
            # (files.py), which is a thing a frontend may want to mention.
            "file_access": "per-room sandbox (read and write)",
            "workspace_bytes": MAX_WORKSPACE_BYTES,
        }

    # -------------------------------------------------------------- replies

    def warm_up(self) -> None:
        """Load the models before the first message, so nobody waits ~10s for a
        reply that also had to read 190MB off disk."""
        if not self.agent_mode:
            self.engine().reply("hi")
            return
        self.agent("__warmup__").handle("hi")
        self.reset("__warmup__")  # including the directory it was given

    async def reply(self, room: str, text: str,
                    attachments: Sequence[Attachment] = ()) -> Reply:
        """One reply — what to say, and what to do — generated off the event
        loop and one at a time.

        `attachments` are staged inside the room's own directory and named in
        the message before it reaches the agent, then deleted — the agent keeps
        the verdict it read off them, never the file, and that directory is the
        only place it can read from at all."""
        with self._files.staged(room, attachments) as paths:
            async with self._lock:
                return await asyncio.to_thread(self._reply, room,
                                               with_paths(text, paths))

    def _reply(self, room: str, text: str) -> Reply:
        """The blocking half of `reply`, run in a worker thread."""
        if self.agent_mode:
            agent = self.agent(room)
            # Drained in the same breath as the text is taken: the acts belong
            # to this turn, and a frontend that fails to post the reply should
            # not find them waiting on the next one.
            return Reply(agent.handle(text), agent.take_actions())
        history = self._histories.setdefault(room, deque(maxlen=_PLAIN_HISTORY))
        reply = self.engine().reply(text, history=history)
        history.extend([text, reply.text])
        # Plain chat is the engine and nothing else — no agent, so no tools.
        return Reply(reply.text)

    def watch(self, room: str, text: str) -> tuple[Action, ...]:
        """What to do about a message that wasn't addressed to the bot.

        The cheap path on purpose: no generation, no history, no lock, no
        thread — a frontend can run every message in a channel through here
        and the models never know. An existing room answers with its own
        persona and its own `/tools off`; a room nobody has talked in yet is
        answered from the process default rather than conjured into existence,
        because an agent means a history and a directory on disk and watching
        a channel should cost neither.
        """
        if not self.agent_mode:
            return ()
        if (agent := self._agents.get(room)) is not None:
            return agent.watch(text)
        from .persona import resolve_persona

        emoji = pick_reaction(text, None, resolve_persona().name)
        return (Action("react", emoji),) if emoji is not None else ()

    def stop(self) -> None:
        """Stop anything still running in the background (a room that left a
        game going keeps a thread stepping it)."""
        for agent in self._agents.values():
            if agent.continuous is not None:
                agent.continuous.stop()
        self._agents.clear()
        self._histories.clear()
        self._files.close()
