"""What the bot can *do* in a room, besides talk — its tools.

Everything the agent could do until now came out as text: a reply, a board, a
verdict on a file. A chat room offers more than that. It can be reacted to,
pinned, split into a thread — small, visible acts that are part of how people
talk in a server, and that a bot which only ever posts paragraphs can't join
in on. This module is the vocabulary of those acts, and the plumbing that
carries them from the model to whichever frontend can perform them.

Three rules shape it.

**The model side names an act; it never performs one.** The models live in the
API server ([api.py](api.py)), which holds no gateway connection and knows
nothing about Discord. So an agent turn produces a `Reply` — text, plus the
`Action`s it wants taken — and the frontend that owns the transport is the only
thing that touches the transport. A frontend with no reactions (Google Chat,
the terminal) drops what it can't do, and nothing upstream has to care.

**A tool is data, like a persona.** `TOOLS` below is the whole list, each entry
carrying what it takes, what it needs permission-wise, and whether it runs by
default. A frontend performs the ones in `allowed()` and no others, so what the
bot may do in a server is a line in `.env` rather than a code change.

**The safe ones are on; the rest are opt-in.** Reacting is reversible, needs no
elevated permission, and is the act being asked for here. Pinning, threading
and renaming need *Manage Messages* / *Manage Threads* / *Change Nickname*, are
visible to the whole channel, and a small model choosing them unprompted is a
worse failure than a missed reaction — so they ship off, and
`DISCORD_ALLOWED_ACTIONS` turns them on per server.

How an action gets chosen
-------------------------
Two paths, and they meet here.

  * **The model asks.** Anything the model generates is scanned for a call in
    double brackets — `[[react :kekw:]]` — which `parse` lifts out of the reply
    and turns into an `Action`, leaving the prose behind. The syntax is on the
    generated side of the wire on purpose: a model fine-tuned on transcripts
    that contain these calls learns to emit them, and this parser is then the
    only part that has to already exist. Today the from-scratch model does not
    emit them; the instruct/gpt2 backends occasionally do, and either way the
    path is real rather than hypothetical.
  * **Nothing asked, so `pick_reaction` does.** A trigger table over the
    incoming message, seeded by the routing specialist's label when it has one.
    This is the same honest arrangement as routing before route.py was trained
    (see [route.py](route.py)): hand-written rules standing in for a specialist
    nobody has trained yet, which is *also* why they are a table of
    (pattern -> slot) pairs rather than a branch chain — a reaction specialist
    would slot in by replacing `pick_reaction` and nothing else.

Deliberately not here, at any setting: kicking, banning, timing anyone out, and
deleting other people's messages. Those are punishments and demolition rather
than conversation, and there is no environment variable for them because the
refusal is the point. `delete` takes back only what the bot itself said. The
frontend refuses a name it doesn't find in `TOOLS`, so adding one would be a
decision somebody has to make on purpose.

One more narrowing, in `WATCH_ACTS`: a message nobody addressed to the bot can
earn a *reaction* and nothing else, however the rest of this table is
configured.

Nothing in this module imports anything heavier than the standard library: the
thin client ([client.py](client.py)) and the model server both read it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Collection

# A turn's worth of acts. A model that has learned to emit calls and then gets
# stuck in a loop should not be able to paint a message with forty reactions,
# and Discord rejects the twentieth anyway.
MAX_PER_TURN = 4


@dataclass(frozen=True)
class Action:
    """One act, named and argued — `Action("react", "🔥")`.

    Transport-neutral by design: "react with this emoji" means something in
    Discord, something slightly different in Google Chat, and nothing at all in
    a terminal. What it costs a frontend to perform is the frontend's business.
    """

    name: str
    arg: str = ""

    def render(self) -> str:
        """The call as the model would write it — for `/tools`, for the
        terminal's echo of what just happened, and for building training data
        that teaches the syntax back to the model."""
        return f"[[{self.name} {self.arg}]]" if self.arg else f"[[{self.name}]]"


@dataclass
class Reply:
    """What one turn produces: what to say, and what to do.

    Frontends that can only say things use `.text` and ignore the rest, which
    is why this is a dataclass with a boring name rather than a tuple — adding
    a third thing later shouldn't touch every caller.
    """

    text: str
    actions: tuple[Action, ...] = ()

    def __str__(self) -> str:  # so a caller that only wants the words can
        return self.text       # still print one of these


@dataclass(frozen=True)
class Tool:
    """One entry in the vocabulary. `needs` is the Discord permission the act
    requires, quoted back in the log line when the bot is missing it — a
    reaction that silently never appears is the kind of thing you lose an
    evening to. `cap` is the length Discord will accept for the argument, and
    is applied here rather than at the call site so that one table decides it.
    """

    name: str
    takes: str        # a one-word description of the argument, "" if none
    doc: str
    default_on: bool  # performed unless DISCORD_ALLOWED_ACTIONS says otherwise
    needs: str = ""   # permission required, for the "I couldn't" log line
    cap: int = 0      # Discord's length limit on the argument
    aliases: tuple[str, ...] = ()


TOOLS: dict[str, Tool] = {
    t.name: t for t in (
        # --- talking, and the reversible marks on a message ---------------
        Tool("react", "emoji", "add a reaction to the message being answered",
             default_on=True, needs="Add Reactions", cap=40,
             aliases=("reaction",)),
        Tool("unreact", "emoji", "take one of its own reactions back off",
             default_on=True, needs="Add Reactions", cap=40),
        Tool("say", "text", "post a message of its own in the channel",
             default_on=True, needs="Send Messages", cap=1900),
        Tool("delete", "", "delete the last thing it said itself",
             default_on=True),
        Tool("dm", "text", "send that to the person privately instead",
             default_on=False, cap=1900, aliases=("whisper",)),
        # --- the message it is answering ----------------------------------
        Tool("pin", "", "pin the message being answered",
             default_on=False, needs="Manage Messages"),
        Tool("unpin", "", "unpin it again",
             default_on=False, needs="Manage Messages"),
        Tool("thread", "name", "start a thread on the message",
             default_on=False, needs="Create Public Threads", cap=100),
        # --- people -------------------------------------------------------
        Tool("nick", "name", "change its own nickname in this server",
             default_on=False, needs="Change Nickname", cap=32),
        Tool("rename", "name", "rename whoever it is answering",
             default_on=False, needs="Manage Nicknames", cap=32),
        Tool("role", "name", "give that person a role, by name",
             default_on=False, needs="Manage Roles", cap=100),
        Tool("unrole", "name", "take that role back off them",
             default_on=False, needs="Manage Roles", cap=100),
        # --- the room, and itself -----------------------------------------
        Tool("topic", "text", "set the channel topic",
             default_on=False, needs="Manage Channels", cap=1024),
        Tool("slowmode", "seconds", "set the channel's slowmode (0 turns it off)",
             default_on=False, needs="Manage Channels", cap=5),
        Tool("status", "text", "set what it is shown as playing",
             default_on=False, cap=128, aliases=("playing", "presence")),
    )
}

# Not here, on purpose: kicking, banning, timing anyone out, and deleting other
# people's messages. Those are punishments and demolition rather than
# conversation, and a 14M-parameter model that picks its words by sampling has
# no business holding that end of the stick — there is no environment variable
# for them because the refusal is the point. `role` is the one tool here that
# can hand out power, which is why it needs Manage Roles, ships off, and only
# ever reaches as high as the bot's own role (Discord enforces that part).

# Alias -> canonical name, so the model writing [[reaction 🔥]] still lands.
_ALIASES = {alias: t.name for t in TOOLS.values() for alias in t.aliases}


def resolve(name: str, extra: "Collection[str]" = ()) -> str | None:
    """The tool a name refers to, or None if it isn't one. Unknown names are
    dropped rather than guessed at: `[[delete]]` should do nothing at all.

    `extra` widens the vocabulary for one caller without widening it for
    everyone: the agent passes its `/command` names, so a model that writes
    `[[play snake]]` starts a game, while the Discord frontend — which performs
    acts and does not run commands — never resolves that name at all.
    """
    key = name.strip().lower().lstrip("/")
    key = _ALIASES.get(key, key)
    return key if key in TOOLS or key in extra else None


def allowed_actions(env: "dict[str, str] | None" = None) -> frozenset[str]:
    """Which acts a frontend will actually perform.

    `DISCORD_ALLOWED_ACTIONS` is a comma-separated list, and setting it replaces
    the defaults rather than adding to them — "what may this bot do" wants to be
    readable off one line, not assembled from a line plus a table. `none`
    disables tool use entirely; `all` enables everything in `TOOLS`.
    """
    raw = (env or os.environ).get("DISCORD_ALLOWED_ACTIONS", "").strip().lower()
    if not raw:
        return frozenset(name for name, t in TOOLS.items() if t.default_on)
    if raw in {"none", "off", "0"}:
        return frozenset()
    if raw in {"all", "*"}:
        return frozenset(TOOLS)
    names = (resolve(part) for part in raw.split(","))
    return frozenset(name for name in names if name)


# What may happen *unprompted* — to a message nobody addressed to the bot.
# Reactions and nothing else, and the narrowness is the feature: a reaction is
# silent, reversible and part of how a channel already talks, while a bot that
# starts posting in conversations it was not in is a different and much worse
# thing to be. `watch` on the model side only ever asks for a reaction anyway;
# this is the guard that keeps that true no matter what it asks for.
WATCH_ACTS = frozenset({"react", "unreact"})


def watch_enabled(env: "dict[str, str] | None" = None) -> bool:
    """Whether the bot reacts to messages that weren't addressed to it.
    `DISCORD_WATCH=0` turns it off and the bot only ever answers."""
    return (env or os.environ).get("DISCORD_WATCH", "1").strip().lower() \
        not in {"0", "false", "no", "off"}


def watch_cooldown(env: "dict[str, str] | None" = None) -> float:
    """Seconds a channel is left alone after an unprompted reaction.

    A busy channel is the case that matters: reacting to every message that
    trips a trigger would be a wall of emoji, so a channel gets one and then
    goes quiet for a while. 0 means react to everything that trips a trigger,
    which is only sensible in a quiet server.
    """
    raw = (env or os.environ).get("DISCORD_REACT_COOLDOWN", "").strip()
    try:
        return max(float(raw), 0.0) if raw else 60.0
    except ValueError:
        return 60.0


def describe(enabled: "frozenset[str] | None" = None) -> str:
    """The tool list as `/tools` prints it: what exists, what is on here."""
    enabled = allowed_actions() if enabled is None else enabled
    lines = ["tools (what I can do besides talk):"]
    for name, tool in TOOLS.items():
        call = f"{name} <{tool.takes}>" if tool.takes else name
        mark = "on " if name in enabled else "off"
        needs = f" (needs {tool.needs})" if tool.needs else ""
        lines.append(f"  [{mark}] {call:16} {tool.doc}{needs}")
    # on/off is read from *this* process's environment, and the process that
    # performs an act is the one holding the chat connection — which, with a
    # model server in front, is a different one. Say so rather than implying a
    # promise this end can't make.
    lines.append("on/off is DISCORD_ALLOWED_ACTIONS, and the bot process that "
                 "holds the connection has the last word"
                 if enabled else
                 "all off here — DISCORD_ALLOWED_ACTIONS=react,pin,... turns "
                 "them on, in the bot process")
    return "\n".join(lines)


# ------------------------------------------------------------------ parsing

# A call the model wrote: [[react 🔥]], [[pin]], [[thread what fish eat]].
# Double brackets because single ones appear in ordinary prose and in code the
# codegen specialist writes, and because a half-finished call at the end of a
# truncated reply ("[[react") then leaves no closing bracket to mistake for one.
_CALL = re.compile(r"\[\[\s*(/?\w+)([^\][\n]*)\]\]")
# The same call plus the space either side of it, which is what actually
# comes out: "sure [[react 🔥]] thing" should close up to "sure thing"
# rather than leave the two spaces behind.
_SPACED_CALL = re.compile(r"[ \t]*(?:" + _CALL.pattern + r")[ \t]*")
# A call that had a line to itself takes the line with it, newline included —
# otherwise lifting it leaves a blank line, and a blank line is a paragraph
# break to the code fencer (transport.py), which would cut a board in half.
_LINE_CALL = re.compile(r"^[ \t]*(?:" + _CALL.pattern + r")[ \t]*\n?", re.MULTILINE)


def parse(text: str, limit: int = MAX_PER_TURN,
          extra: "Collection[str]" = ()) -> tuple[str, list[Action]]:
    """Split generated text into the words to post and the calls to run.

    Unknown names are dropped, repeats collapse (two `[[react 🔥]]` in one reply
    is one reaction either way, and asking Discord twice earns a 400), and the
    markup always comes out of the text even when the call behind it is refused
    — a reader should never see the machinery.
    """
    actions: list[Action] = []
    seen: set[tuple[str, str]] = set()
    for raw_name, raw_arg in _CALL.findall(text):
        name = resolve(raw_name, extra)
        if name is None:
            continue
        arg = (clean_arg(TOOLS[name], raw_arg) if name in TOOLS
               else raw_arg.strip())
        if arg is None:
            continue  # nothing usable in it — a reaction to a non-emoji, say
        key = (name, arg)
        if key in seen or len(actions) >= limit:
            continue
        seen.add(key)
        actions.append(Action(name, arg))
    if not _CALL.search(text):
        # The overwhelmingly common case, and the one that must come back
        # *byte for byte*: a reply is often a rendered board or a column of
        # /help, where the runs of spaces are the content.
        return text, actions
    return tidy(_SPACED_CALL.sub(" ", _LINE_CALL.sub("", text))), actions


def tidy(text: str) -> str:
    """Close the hole a lifted call leaves behind, without reflowing the rest.

    Only the seam is touched — the space either side of the call went with it
    in `_SPACED_CALL`, and a whole line of call went with `_LINE_CALL`, so all
    that is left is a line end. Runs of spaces elsewhere are left exactly as they were: two
    spaces mid-line is how [transport.py](transport.py) recognizes text that
    needs a monospace font, so collapsing them would un-fence a board.
    """
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


# Emoji arrive two ways and only two are worth accepting. A unicode emoji is a
# short run of characters with no ASCII in it at all; a server's custom emoji
# is `:name:`, which is exactly the form discord_text.clean leaves in the text
# the model reads and therefore the form it would learn to write back.
_CUSTOM = re.compile(r"^:?([A-Za-z0-9_~]{2,32}):?$")
_ASCII = re.compile(r"[A-Za-z0-9 ]")


def clean_arg(tool: Tool, arg: str) -> str | None:
    """`arg` as the tool will take it, or None if there is nothing usable in it.

    Every caller goes through here — the parser, `/react` typed by hand, the
    frontend — so "what is a legal argument" is answered once. Discord's own
    limits are applied as a *trim* rather than a refusal (a nickname one
    character over the line should still land), except where a too-long value
    means the model lost the plot: an emoji is short or it isn't an emoji.
    """
    arg = arg.strip()
    if not tool.takes:
        return ""      # `[[pin]]` with something after it is still just a pin
    if not arg:
        return None    # and `[[react]]` with nothing after it is not an act
    if tool.takes == "emoji":
        return clean_emoji(arg)
    if tool.takes == "seconds":
        digits = re.sub(r"\D", "", arg)[:tool.cap]
        # Discord's ceiling is 6 hours, and 0 is the way slowmode is turned off.
        return str(min(int(digits), 21600)) if digits else None
    return arg[:tool.cap]


def clean_emoji(arg: str) -> str | None:
    """`arg` as an emoji to react with, or None if it isn't one.

    Custom emoji come back as `:name:` — the frontend resolves that against the
    server it is in, since the same name is a different image in every guild.
    """
    arg = arg.strip()
    if not arg or len(arg) > 40:
        return None
    if arg.startswith("<") and arg.endswith(">"):  # <:name:id>, already resolved
        return arg
    if not _ASCII.search(arg):
        return arg if len(arg) <= 16 else None  # unicode, incl. ZWJ sequences
    if (m := _CUSTOM.match(arg)) is not None:
        return f":{m.group(1)}:"
    return None


# ------------------------------------------------------------- picking one

# Reactions by *slot* rather than by emoji, so a persona can re-cast the whole
# set by overriding a few entries (persona.py does the same thing for how a
# reply sounds — data, not code).
SLOTS = ("laugh", "love", "wave", "party", "fire", "sad", "think", "eyes",
         "game", "code", "meh")

DEFAULT_EMOJI = {
    "laugh": "😂", "love": "❤️", "wave": "👋", "party": "🎉", "fire": "🔥",
    "sad": "😔", "think": "🤔", "eyes": "👀", "game": "🎮", "code": "💻",
    "meh": "😐",
}

# A persona that overrides nothing uses the table above. These are the ones
# where the default would be out of character — a deadpan bot does not post 🎉.
PERSONA_EMOJI: dict[str, dict[str, str]] = {
    "deadpan": {"laugh": "🙂", "love": "👍", "party": "👏", "fire": "👍",
                "sad": "😑", "meh": "😑"},
    "grumpy": {"laugh": "🙄", "love": "😒", "party": "🙄", "fire": "😤",
               "think": "😒", "meh": "🙄"},
    "lowkey": {"laugh": "😅", "love": "🫶", "party": "✨", "fire": "😎",
               "sad": "🥺", "meh": "😶"},
    "intimidating": {"laugh": "😏", "love": "🫡", "party": "🫡", "fire": "💀",
                     "sad": "🫥", "think": "🧐", "meh": "😑"},
    "ragebaiter": {"laugh": "💀", "love": "🤡", "party": "🤡", "fire": "🗿",
                   "sad": "🤣", "think": "🗿", "meh": "🥱"},
    "cheerful": {"think": "🤩", "meh": "😊", "sad": "🫂"},
}

# (pattern -> slot), tried in order, first match wins. Ordered by how sure the
# cue is rather than alphabetically: "thanks lol" is laughter with a thank-you
# in it, and "lol i'm stuck on this bug" is not a bug report to react sadly to.
_TRIGGERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(lol|lmao+|lmfao|rofl|kek+w?|ahaha|haha+|hehe|xd)\b|😂|🤣|💀"), "laugh"),
    (re.compile(r"\b(thanks|thank you|thx|ty|tysm|appreciate it|cheers)\b"), "love"),
    (re.compile(r"^\s*(hi|hey+|hello|yo+|sup|gm|good morning|good evening)\b"), "wave"),
    (re.compile(r"\b(congrats|congratulations|nailed it|shipped it|"
                r"it works|finally works|passed|we won|i won)\b"), "party"),
    (re.compile(r"\b(good bot|nice|based|goated|awesome|sick|let'?s go|lesgo|"
                r"banger|peak)\b"), "fire"),
    (re.compile(r"\b(rip|oof|sorry|that sucks|rough|i'?m tired|exhausted|"
                r"broke|broken|crashed|failed|it'?s down)\b|😔|😢|:\("), "sad"),
    (re.compile(r"\b(snake|pong|dodge|tic.?tac.?toe|high score|your turn)\b"), "game"),
    (re.compile(r"\b(bug|traceback|stack ?trace|compiles?|refactor|"
                r"pull request|merge conflict)\b"), "code"),
    (re.compile(r"\b(bad bot|shut up|you'?re wrong|stupid bot)\b"), "meh"),
    (re.compile(r"\?\s*$"), "think"),
)

# What the routing specialist's verdict implies, when the triggers found
# nothing. Reusing the trained classifier here is the point: it already read
# the message, and its label is a better signal than another regex would be.
# "chat" is absent on purpose — small talk is most of a channel, and a bot that
# reacts to every line of it is noise.
ROUTE_SLOT = {"reason": "think", "codegen": "code", "code": "code",
              "vision": "eyes", "game": "game"}


def emoji_for(slot: str, persona: str = "neutral") -> str:
    return PERSONA_EMOJI.get(persona, {}).get(slot) or DEFAULT_EMOJI[slot]


def pick_reaction(message: str, route: str | None = None,
                  persona: str = "neutral") -> str | None:
    """An emoji to react to `message` with, or None to stay quiet.

    None is the common answer and should stay that way: a reaction means "this
    one landed", and something posted under every message means nothing. So
    there is no fallback slot — a message that trips no trigger and routes to
    chat gets no reaction at all.
    """
    text = message.strip().lower()
    if not text or text.startswith("/"):  # a command is a control, not a remark
        return None
    for pattern, slot in _TRIGGERS:
        if pattern.search(text):
            return emoji_for(slot, persona)
    if (slot := ROUTE_SLOT.get(route or "")) is not None:
        return emoji_for(slot, persona)
    return None
