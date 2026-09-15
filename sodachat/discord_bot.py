"""Discord frontend.

Replies to direct messages and to messages that @mention the bot. Set
DISCORD_RESPOND_ALL=1 to reply to every message the bot can read.

**The models live somewhere else.** Set `SODACHAT_API_URL` and this process is a
thin client of the master API server ([api.py](api.py)) — no PyTorch, no
checkpoints, no warm-up, and the same copy of the models as every other bot
pointed at it. Without that variable the models load here instead, which is what
this did before and is still the right answer for a single bot.

Either way a channel gets everything the terminal does: the routing specialist
picks which capability answers each message, `/commands` work (`/play snake`,
`/see`, `/gen`, `/think`, `/route`, `/persona`, `/help`), and **images or source
files posted to the channel are handed to the vision and code specialists** — the
attachment is downloaded here and its bytes go to the server, which stages it
where the agent can read it. Each channel keeps its own conversation (its own
history, its own game, its own persona), keyed as `discord:<channel id>` so two
frontends on one server never collide.

Requires the *Message Content Intent* to be enabled for the bot in the
Discord developer portal (Bot -> Privileged Gateway Intents).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import discord
from dotenv import load_dotenv

from .client import Backend, BackendError, open_backend
from .discord_text import clean_incoming
from .transport import DISCORD_LIMIT, Attachment, format_reply, room_id

log = logging.getLogger("sodachat.discord")

_OOPS = "something went wrong on my end, sorry."


def _skip_reason(backend: Backend, attachment: discord.Attachment) -> str | None:
    """Why an attachment won't be fetched, or None to fetch it. Checked before
    downloading: no point spending the bandwidth on a file the models can't use,
    and posting one should get a reason rather than silence."""
    name = attachment.filename
    if not backend.accepted_suffixes:  # plain-chat mode: no specialists loaded
        return f"i can't read {name} — i'm running in plain chat mode right now"
    if not backend.accepts(name):
        return (f"i can't read {name} — images (png/jpg/...) so I can look at "
                f"them, or source files (.py/.js/...) so I can name the language")
    if attachment.size > backend.max_attachment_bytes:
        return (f"{name} is {attachment.size / 1e6:.0f} MB, past the "
                f"{backend.max_attachment_bytes // 1024 // 1024} MB I'll fetch")
    return None


async def _collect_attachments(message: discord.Message,
                               backend: Backend) -> tuple[list[Attachment], list[str]]:
    """Download the attachments a specialist can use (images for vision, source
    files for code) into memory, to be sent along with the message.

    Returns them and a note for anything skipped, so posting a file at the bot
    gets an answer rather than silence."""
    taken: list[Attachment] = []
    skipped: list[str] = []
    for attachment in message.attachments:
        if (reason := _skip_reason(backend, attachment)) is not None:
            skipped.append(reason)
            continue
        try:
            data = await attachment.read()
        except Exception as e:  # a failed download is not worth dropping the turn
            log.warning("could not download %s: %s", attachment.filename, e)
            skipped.append(f"couldn't download {attachment.filename}")
            continue
        taken.append(Attachment(attachment.filename, data))
    return taken, skipped


async def reply_to(message: discord.Message, text: str, backend: Backend) -> None:
    """Handling of one message: pick up whatever came attached, let the backend
    answer, and post the reply as however many renderable pieces it takes. Lives
    outside `main` so it can be exercised without a connection."""
    room = room_id("discord", str(message.channel.id))
    try:
        async with message.channel.typing():
            attachments, skipped = await _collect_attachments(message, backend)
            if skipped and not attachments and not text:
                # Someone posted a file at the bot and nothing else: say why it
                # can't be read instead of going quiet. With text alongside it,
                # answer the text and let the file go.
                await message.reply(skipped[0], mention_author=False)
                return
            answer = await backend.reply(room, text, attachments)
    except BackendError as e:
        log.error("backend failed on channel %s: %s", room, e)
        await message.reply(_OOPS, mention_author=False)
        return
    except Exception:
        log.exception("failed to handle a message in channel %s", room)
        await message.reply(_OOPS, mention_author=False)
        return
    try:
        for i, part in enumerate(format_reply(answer, DISCORD_LIMIT)):
            if i == 0:
                await message.reply(part, mention_author=False)
            else:
                await message.channel.send(part)
    except discord.DiscordException:
        log.exception("could not post the reply in channel %s", room)


def build_client(backend: Backend, respond_all: bool = False) -> discord.Client:
    """The gateway client, wired to `backend`. Split out of `main` so a test (or
    another entry point) can build one without connecting."""
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

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

        # Not `message.content`: Discord's raw markup carries a snowflake id
        # for every emoji, mention and channel in the message, and the model
        # was trained on the cleaned form (see discord_text).
        text = clean_incoming(message.content, client.user.id)
        log.info("received: %s", text)
        await reply_to(message, text, backend)

    return client


async def _run(token: str, respond_all: bool) -> None:
    backend = await open_backend()
    log.info("using %s — /help in a channel lists the commands", backend.describe())
    client = build_client(backend, respond_all)
    try:
        async with client:
            await client.start(token)
    finally:
        await backend.close()


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
    try:
        asyncio.run(_run(token, respond_all))
    except BackendError as e:
        sys.exit(str(e))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
