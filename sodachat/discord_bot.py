"""Discord frontend.

Replies to direct messages and to messages that @mention the bot. Set
DISCORD_RESPOND_ALL=1 to reply to every message the bot can read.

Runs the full agent by default ([agent.py](agent.py)), so a channel gets
everything the terminal does: the routing specialist picks which capability
answers each message, `/commands` work (`/play snake`, `/see`, `/gen`, `/think`,
`/route`, `/help`), and **images or source files posted to the channel are
handed to the vision and code specialists** — the room equivalent of dropping a
file into the terminal. Each channel keeps its own agent (its own history, its
own game); the models are loaded once and shared. `SODACHAT_AGENT=0` falls back
to the plain chat engine.

Requires the *Message Content Intent* to be enabled for the bot in the
Discord developer portal (Bot -> Privileged Gateway Intents).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import discord
from dotenv import load_dotenv

from .engine import ChatEngine
from .rooms import (
    DISCORD_LIMIT,
    MAX_ATTACHMENT_BYTES,
    Rooms,
    Scratch,
    agent_mode_enabled,
    filtered_enabled,
    format_reply,
    handled_suffixes,
    with_paths,
)

log = logging.getLogger("sodachat.discord")


def _strip_mentions(content: str, bot_user: discord.ClientUser) -> str:
    return re.sub(rf"<@!?{bot_user.id}>", "", content).strip()


async def _stage_attachments(message: discord.Message,
                             scratch: Scratch) -> tuple[list[Path], list[str]]:
    """Download the attachments a specialist can use (images for vision, source
    files for code) into the scratch directory.

    Returns the staged paths and a note for anything skipped, so posting a file
    at the bot gets an answer rather than silence."""
    suffixes = handled_suffixes()
    saved: list[Path] = []
    skipped: list[str] = []
    for attachment in message.attachments:
        name = attachment.filename
        if not name.lower().endswith(suffixes):
            skipped.append(f"i can't read {name} — images (png/jpg/...) so I can "
                           f"look at them, or source files (.py/.js/...) so I can "
                           f"name the language")
            continue
        if attachment.size > MAX_ATTACHMENT_BYTES:
            skipped.append(f"{name} is {attachment.size / 1e6:.0f} MB, past the "
                           f"{MAX_ATTACHMENT_BYTES // 1024 // 1024} MB I'll fetch")
            continue
        path = scratch.path_for(name)
        try:
            await attachment.save(path)
        except Exception as e:  # a failed download is not worth dropping the turn
            log.warning("could not download %s: %s", name, e)
            skipped.append(f"couldn't download {name}")
            continue
        saved.append(path)
    return saved, skipped


async def reply_to(message: discord.Message, text: str, rooms: Rooms) -> None:
    """Agent-mode handling of one message: stage whatever came attached, let the
    agent route it, and post the reply as however many renderable pieces it
    takes. Lives outside `main` so it can be exercised without a connection."""
    room = str(message.channel.id)
    scratch = Scratch()
    try:
        async with message.channel.typing():
            paths, skipped = await _stage_attachments(message, scratch)
            if skipped and not paths and not text:
                # Someone posted a file at the bot and nothing else: say why it
                # can't be read instead of going quiet. With text alongside it,
                # answer the text and let the file go.
                await message.reply(skipped[0], mention_author=False)
                return
            answer = await rooms.reply(room, with_paths(text, paths))
        for i, part in enumerate(format_reply(answer, DISCORD_LIMIT)):
            if i == 0:
                await message.reply(part, mention_author=False)
            else:
                await message.channel.send(part)
    except Exception:
        log.exception("failed to handle a message in channel %s", room)
        await message.reply("something went wrong on my end, sorry.",
                            mention_author=False)
    finally:
        scratch.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    load_dotenv()

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        sys.exit(
            "DISCORD_BOT_TOKEN is not set. Create a bot at "
            "https://discord.com/developers/applications, enable the Message "
            "Content Intent, and put the token in .env (see .env.example)."
        )

    respond_all = os.environ.get("DISCORD_RESPOND_ALL", "").lower() in {"1", "true", "yes"}
    filtered = filtered_enabled()
    agent_mode = agent_mode_enabled()

    rooms: Rooms | None = None
    engine: ChatEngine | None = None
    if agent_mode:
        log.info("loading the agent (chat, games, and the specialists)...")
        rooms = Rooms(filtered=filtered)
        rooms.warm_up()
        log.info("agent ready on %s — /help in a channel lists the commands",
                 rooms.device)
    else:
        log.info("loading the chat model (SODACHAT_AGENT=0: plain chat)...")
        engine = ChatEngine(filtered=filtered)

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    # Plain-chat mode only: recent lines per channel. In agent mode the per-room
    # agent keeps its own history (along with its game and what it last saw).
    histories: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=8))

    @client.event
    async def on_ready() -> None:
        log.info("logged in as %s (id %s)", client.user, client.user.id)

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
            return
        is_dm = message.guild is None
        mentioned = client.user in message.mentions
        if not (is_dm or mentioned or respond_all):
            return

        text = _strip_mentions(message.content, client.user)
        log.info("recieved: ", text)
        if not agent_mode:
            history = histories[message.channel.id]
            reply = engine.reply(text, history=history)
            history.extend([text, reply.text])
            async with message.channel.typing():
                await message.reply(reply.text, mention_author=False)
            return
        await reply_to(message, text, rooms)

    try:
        client.run(token, log_handler=None)
    finally:
        if rooms is not None:
            rooms.stop()


if __name__ == "__main__":
    main()
