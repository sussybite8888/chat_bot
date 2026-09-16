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

**A reply can also ask for something to be done**, not just said — react to that
message, pin it, rename whoever sent it, set the channel topic
([actions.py](actions.py)). The models name the act and this process performs
it, because this is the end holding the gateway connection. `perform` is that
half: one handler per act, every failure logged and none of them able to take
the turn down, and nothing attempted that `DISCORD_ALLOWED_ACTIONS` didn't
allow. The reversible, unprivileged acts are on by default; the ones wanting a
moderator-shaped permission are not (see `.env.example`). There is no kick, no
ban, no timeout and no deleting anyone else's message, at any setting.

**And it reacts to messages it was never sent.** Everything the bot can read
goes through `watch_message`, which is the cheap path in every direction: the
models generate nothing (a trigger table picks the emoji), the channel's
conversation is neither read nor added to, only a reaction can come of it
whatever the models ask for (`WATCH_ACTS`), and a channel that just got one is
left alone for `DISCORD_REACT_COOLDOWN` seconds — checked here, before the
request is sent. `DISCORD_WATCH=0` turns it off.

Requires the *Message Content Intent* to be enabled for the bot in the Discord
developer portal (Bot -> Privileged Gateway Intents), and — for reactions — the
*Add Reactions* permission on the channel.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time

import discord
from dotenv import load_dotenv

from .actions import (TOOLS, WATCH_ACTS, Action, allowed_actions,
                      watch_cooldown, watch_enabled)
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


# ------------------------------------------------------------------- acting


class _CantAct(Exception):
    """This act doesn't apply here — a thread in a DM, an emoji this server
    doesn't have. Not an error anyone needs a traceback for; the turn's words
    were posted and this is a footnote."""


def _emoji(message: discord.Message, arg: str):
    """The reaction to add, as discord.py wants it: a unicode emoji goes
    through as a string, while `:kekw:` has to be resolved to *this* server's
    emoji — the same name is a different image in every guild, and `:kekw:` is
    precisely the form the model reads and writes (discord_text.py)."""
    if not (arg.startswith(":") and arg.endswith(":")):
        return arg
    name = arg.strip(":")
    # This server's emoji only. The bot can technically reach for one from any
    # guild it is in, but a name that means one thing here and another thing
    # three servers over is a surprise nobody asked for.
    found = discord.utils.get(message.guild.emojis if message.guild else (), name=name)
    if found is None:
        raise _CantAct(f"no emoji called :{name}: in this server")
    return found


async def _act_react(message: discord.Message, arg: str, client: discord.Client) -> None:
    await message.add_reaction(_emoji(message, arg))


async def _act_unreact(message: discord.Message, arg: str,
                       client: discord.Client) -> None:
    await message.remove_reaction(_emoji(message, arg), client.user)


async def _act_say(message: discord.Message, arg: str, client: discord.Client) -> None:
    _remember(await message.channel.send(arg))


async def _act_delete(message: discord.Message, arg: str,
                      client: discord.Client) -> None:
    """Take back the last thing the *bot* said here. Deliberately not "delete
    that message": removing other people's words is moderation, and the only
    message this bot may unsay is its own."""
    mine = _LAST_SENT.pop(message.channel.id, None)
    if mine is None:
        raise _CantAct("I haven't said anything here yet")
    await mine.delete()


async def _act_dm(message: discord.Message, arg: str, client: discord.Client) -> None:
    try:
        await message.author.send(arg)
    except discord.Forbidden:  # their DMs are closed — not a permission of ours
        raise _CantAct("their DMs are closed") from None


async def _act_pin(message: discord.Message, arg: str, client: discord.Client) -> None:
    await message.pin(reason="asked for by the bot")


async def _act_unpin(message: discord.Message, arg: str,
                     client: discord.Client) -> None:
    await message.unpin(reason="asked for by the bot")


async def _act_thread(message: discord.Message, arg: str,
                      client: discord.Client) -> None:
    if message.guild is None:
        raise _CantAct("there are no threads in a DM")
    await message.create_thread(name=arg)


async def _act_nick(message: discord.Message, arg: str,
                    client: discord.Client) -> None:
    if message.guild is None:
        raise _CantAct("a nickname is a per-server thing; this is a DM")
    await message.guild.me.edit(nick=arg)


def _target(message: discord.Message, client: discord.Client):
    """Who an act about a *person* is about: whoever the message @mentioned, or
    else whoever sent it.

    The model can only ever mean the second one — mentions reach it as the word
    "@someone" with the id stripped out (discord_text.py), which is the whole
    point of that scrubbing. The first is for you: `/rename @bob stinky` typed
    into the channel arrives here with `bob` in `message.mentions`.
    """
    if message.guild is None:
        raise _CantAct("that's a per-server thing; this is a DM")
    # Off the message itself, never `guild.get_member` — that reads the member
    # cache, which is empty without the privileged Server Members intent this
    # bot deliberately doesn't ask for. A guild message already carries its
    # author and its mentions as members.
    for user in message.mentions:
        if user != client.user:
            if not isinstance(user, discord.Member):
                raise _CantAct(f"{user} isn't in this server")
            return user
    if not isinstance(message.author, discord.Member):
        raise _CantAct("I can't see who that is in this server")
    return message.author


def _nickname(arg: str) -> str:
    """The name out of a `/rename @bob stinky` argument. The mention is already
    "@someone" by the time the models see it, and it is addressing rather than
    a name, so it comes off."""
    name = re.sub(r"<@[!&]?\d+>|@someone", " ", arg).strip()
    if not name:
        raise _CantAct("that left me no name to set")
    return name[:32]


async def _act_rename(message: discord.Message, arg: str,
                      client: discord.Client) -> None:
    await _target(message, client).edit(nick=_nickname(arg),
                                        reason="asked for by the bot")


def _role(message: discord.Message, arg: str):
    if (role := discord.utils.get(message.guild.roles, name=arg.strip())) is None:
        raise _CantAct(f"no role called {arg.strip()!r} in this server")
    return role


async def _act_role(message: discord.Message, arg: str,
                    client: discord.Client) -> None:
    member = _target(message, client)
    await member.add_roles(_role(message, arg), reason="asked for by the bot")


async def _act_unrole(message: discord.Message, arg: str,
                      client: discord.Client) -> None:
    member = _target(message, client)
    await member.remove_roles(_role(message, arg), reason="asked for by the bot")


async def _act_topic(message: discord.Message, arg: str,
                     client: discord.Client) -> None:
    if not isinstance(message.channel, discord.TextChannel):
        raise _CantAct("this channel has no topic to set")
    await message.channel.edit(topic=arg, reason="asked for by the bot")


async def _act_slowmode(message: discord.Message, arg: str,
                        client: discord.Client) -> None:
    if not isinstance(message.channel, discord.TextChannel):
        raise _CantAct("this channel has no slowmode")
    await message.channel.edit(slowmode_delay=int(arg),
                               reason="asked for by the bot")


async def _act_status(message: discord.Message, arg: str,
                      client: discord.Client) -> None:
    """What the bot is shown as playing. The one act here that isn't about this
    channel — presence is per *bot*, so it changes in every server at once,
    which is a good reason for it to ship off."""
    await client.change_presence(activity=discord.Game(name=arg))


# The last thing the bot said in a channel, so it can take it back (`delete`).
# Bounded, because a busy bot sees a lot of channels and this is a convenience
# rather than a record.
_LAST_SENT: "dict[int, discord.Message]" = {}
_REMEMBERED = 256


def _remember(sent: discord.Message | None) -> None:
    if sent is None:
        return
    if len(_LAST_SENT) >= _REMEMBERED:
        _LAST_SENT.pop(next(iter(_LAST_SENT)), None)  # oldest channel seen
    _LAST_SENT[sent.channel.id] = sent


# name -> how to do it. Every key is a tool in actions.py; a name that isn't
# one never reaches here, because the agent drops it at the parse.
_ACTS = {
    "react": _act_react,
    "unreact": _act_unreact,
    "say": _act_say,
    "delete": _act_delete,
    "dm": _act_dm,
    "pin": _act_pin,
    "unpin": _act_unpin,
    "thread": _act_thread,
    "nick": _act_nick,
    "rename": _act_rename,
    "role": _act_role,
    "unrole": _act_unrole,
    "topic": _act_topic,
    "slowmode": _act_slowmode,
    "status": _act_status,
}
assert set(_ACTS) == set(TOOLS), "a tool exists that this frontend can't perform"


async def perform(message: discord.Message, actions: "tuple[Action, ...]",
                  client: discord.Client,
                  allowed: "frozenset[str] | None" = None) -> None:
    """Do what the turn asked for, in the channel the message came from.

    Runs after the words are posted, and is written so that it cannot cost the
    turn anything: an act that isn't allowed here is skipped, and one that
    fails — no permission, an emoji Discord won't take, a thread that already
    exists — is logged and the next one still runs. A reaction that never shows
    up is otherwise a silent nothing, and the log line is the only place the
    reason can be said out loud.
    """
    allowed = allowed_actions() if allowed is None else allowed
    for action in actions:
        if action.name not in allowed:
            log.info("not %sing: DISCORD_ALLOWED_ACTIONS doesn't include it",
                     action.name)
            continue
        try:
            await _ACTS[action.name](message, action.arg, client)
            log.info("acted: %s", action.render())
        except _CantAct as e:
            log.info("skipped %s: %s", action.render(), e)
        except discord.Forbidden:
            log.warning("can't %s here — the bot's role is missing %s",
                        action.name, TOOLS[action.name].needs or "the permission")
        except discord.HTTPException as e:
            log.warning("discord refused %s: %s", action.render(), e)
        except Exception:  # a broken act must never lose the conversation
            log.exception("failed to %s", action.name)


class _Cooldown:
    """One unprompted reaction per channel per `seconds`.

    Kept in the bot rather than in the models because it is about a *channel*,
    and because the cheapest request is the one never sent: a channel still
    cooling down is dropped here, before the message crosses the wire at all.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._last: dict[int, float] = {}

    def ready(self, channel: int) -> bool:
        return time.monotonic() - self._last.get(channel, -1e9) >= self.seconds

    def mark(self, channel: int) -> None:
        self._last[channel] = time.monotonic()
        if len(self._last) > 4096:  # a bot in a lot of channels; keep it bounded
            self._last.pop(next(iter(self._last)), None)


async def watch_message(message: discord.Message, text: str, backend: Backend,
                        client: discord.Client, allowed: frozenset[str],
                        cooldown: _Cooldown) -> None:
    """A message nobody sent to the bot: react, or (usually) do nothing.

    This runs over every message the bot can read, so it is built to be
    boring. The channel cooldown is checked before the request, the models are
    never asked to *generate* anything (`Rooms.watch` is a trigger table), the
    conversation in this channel is neither read nor added to, and only the
    acts in `WATCH_ACTS` can come back out. A failure anywhere costs a
    reaction, which is nothing.
    """
    if not text.strip() or not cooldown.ready(message.channel.id):
        return
    room = room_id("discord", str(message.channel.id))
    try:
        actions = await backend.watch(room, text)
    except BackendError as e:  # unreachable models: it can wait for the next one
        log.debug("could not ask about %s: %s", room, e)
        return
    if not actions:
        return
    # Marked whether or not the act lands: a channel the bot can't react in
    # should be tried once a minute, not once a message.
    cooldown.mark(message.channel.id)
    await perform(message, actions, client, allowed & WATCH_ACTS)


# ------------------------------------------------------------------ replying


async def reply_to(message: discord.Message, text: str, backend: Backend,
                   client: discord.Client | None = None,
                   allowed: "frozenset[str] | None" = None) -> None:
    """Handling of one message: pick up whatever came attached, let the backend
    answer, post the reply as however many renderable pieces it takes, and then
    do whatever the turn asked to have done. Lives outside `main` so it can be
    exercised without a connection."""
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
        for i, part in enumerate(format_reply(answer.text, DISCORD_LIMIT)):
            if i == 0:
                _remember(await message.reply(part, mention_author=False))
            else:
                _remember(await message.channel.send(part))
    except discord.DiscordException:
        log.exception("could not post the reply in channel %s", room)
    # After the words, not before: the reply is the part that matters, and an
    # act is a flourish on top of it. `client` is optional only so an older
    # caller still works — without it there is no bot user to react *as*.
    if answer.actions and client is not None:
        await perform(message, answer.actions, client, allowed)


def build_client(backend: Backend, respond_all: bool = False,
                 allowed: "frozenset[str] | None" = None,
                 watch: bool | None = None) -> discord.Client:
    """The gateway client, wired to `backend`. Split out of `main` so a test (or
    another entry point) can build one without connecting."""
    # Read once at startup rather than per message: what this bot may do in a
    # server is a property of how it was launched, and a log line at boot is
    # worth more than re-reading the environment every turn.
    allowed = allowed_actions() if allowed is None else allowed
    watch = watch_enabled() if watch is None else watch
    cooldown = _Cooldown(watch_cooldown())
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

        # Not `message.content`: Discord's raw markup carries a snowflake id
        # for every emoji, mention and channel in the message, and the model
        # was trained on the cleaned form (see discord_text).
        text = clean_incoming(message.content, client.user.id)
        if not (is_dm or mentioned or respond_all):
            # Not addressed to the bot. It still gets to react to it — a
            # channel is mostly people talking to each other, and reacting is
            # how you take part in that without interrupting.
            if watch and allowed & WATCH_ACTS:
                await watch_message(message, text, backend, client, allowed,
                                    cooldown)
            return
        log.info("received: %s", text)
        await reply_to(message, text, backend, client, allowed)

    return client


async def _run(token: str, respond_all: bool) -> None:
    backend = await open_backend()
    log.info("using %s — /help in a channel lists the commands", backend.describe())
    allowed = allowed_actions()
    log.info("acts allowed here: %s (DISCORD_ALLOWED_ACTIONS)",
             ", ".join(sorted(allowed)) or "none")
    watch = watch_enabled()
    log.info("messages it wasn't sent: %s", (
        f"reacting to them, at most one per channel per {watch_cooldown():.0f}s"
        if watch and allowed & WATCH_ACTS else "leaving them alone"))
    client = build_client(backend, respond_all, allowed, watch)
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
