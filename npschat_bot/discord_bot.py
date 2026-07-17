"""Discord frontend.

Replies to direct messages and to messages that @mention the bot. Set
DISCORD_RESPOND_ALL=1 to reply to every message the bot can read.

Requires the *Message Content Intent* to be enabled for the bot in the
Discord developer portal (Bot -> Privileged Gateway Intents).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import defaultdict, deque

import discord
from dotenv import load_dotenv

from .engine import ChatEngine

log = logging.getLogger("npschat.discord")


def _strip_mentions(content: str, bot_user: discord.ClientUser) -> str:
    return re.sub(rf"<@!?{bot_user.id}>", "", content).strip()


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
    filtered = os.environ.get("NPSCHAT_UNFILTERED", "").lower() not in {"1", "true", "yes"}

    log.info("warming up on the NPS Chat corpus...")
    engine = ChatEngine(filtered=filtered)

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    # Recent conversation lines per channel, used to condition the model.
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
        history = histories[message.channel.id]
        reply = engine.reply(text, history=history)
        history.extend([text, reply.text])
        async with message.channel.typing():
            await message.reply(reply.text, mention_author=False)

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
