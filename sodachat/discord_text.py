"""Turning a raw Discord message into the text the model was trained on.

Discord does not hand over the message a human sees. Every entity in it arrives
as angle-bracket markup carrying a numeric snowflake id: a custom emoji is
`<:kekw:1234567890123456>`, a user mention is `<@1234567890123456>`, a channel
is `<#1234567890123456>`. What the sender saw was a picture and a name; what
the bot receives is eighteen digits.

Those digits are not words. Fed to the model they are a long run of numeric
tokens in the middle of a sentence — nothing it can learn from, and nothing it
should be echoing back into a channel.

**This module exists because two paths have to agree about that**, and once did
not. `tools/export_discord.py` cleaned the markup out of the exported training
corpus, so the model learned `:kekw:` and `@someone`. `discord_bot.py` passed
the raw message straight through, so at run time it was handed
`<:kekw:1234567890123456>` — markup the model had never seen in training, with
an id in the middle of it. Both now call `clean` here, so the text the model is
asked about is shaped like the text it was trained on, and neither side can
drift again without the other.

`clean` is deliberately transport-shaped rather than model-shaped: it knows
about Discord markup and nothing about tokenizers, so it is safe to run over a
corpus and over a live message alike.

`clean_outgoing` is the return leg, and exists because the scrubbing is not
symmetric: `@someone` carries no id, so nothing downstream can turn it back
into a mention. It comes out, and a real ping comes from the `ping` tool
instead.
"""

from __future__ import annotations

import re

# Custom emoji, static (`<:name:id>`) and animated (`<a:name:id>`). The name is
# the half a reader actually sees, so it is what survives.
_EMOJI = re.compile(r"<a?:(\w+):\d+>")
# Users (`<@id>`, `<@!id>` for the legacy nickname form) and roles (`<@&id>`).
# A mention often abuts the next word (`<@123>how many...`), so the replacement
# is padded and the whitespace collapse at the end puts the spacing right.
_MENTION = re.compile(r"<@[!&]?\d+>")
_CHANNEL = re.compile(r"<#\d+>")
# Slash-command mentions: `</play:123>`, or `</config set:123>` for a subcommand.
_COMMAND = re.compile(r"</([\w-]+(?: [\w-]+){0,2}):\d+>")
# A rendered timestamp (`<t:1700000000:R>` shows as "2 hours ago"). The number
# is a unix time, so the literal text is worth less than nothing.
_TIMESTAMP = re.compile(r"<t:\d+(?::[tTdDfFR])?>")
_SPOILER = re.compile(r"\|\|(.+?)\|\|", re.S)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*<?https?://[^)]*>?\s*\)")
_URL = re.compile(r"<?https?://\S+?>?(?=\s|$)")
_EMPTY_LINK = re.compile(r"\[([^\]]*)\]\(\s*\)")
_SPACES = re.compile(r"[ \t]+")
# The model's own output, on the way back out: the word `clean` leaves behind,
# plus the punctuation that was addressing someone with it.
_STRAY_MENTION = re.compile(r"@someone\b[,:;]?")
# Taking a word out of the middle of a line leaves the spacing around it: a gap
# before the punctuation that followed it ("hi !"), or a trailing space where
# it ended a line.
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.!?;:])")
_SPACE_EOL = re.compile(r"[ \t]+\n")


def strip_mention(text: str, user_id: int | str) -> str:
    """Remove one specific user's mention markup, left and right.

    The bot's own mention is addressing, not content: "@bot how are you" is the
    message "how are you". It has to come out *before* `clean`, which would
    otherwise turn it into "@someone" and leave the model reading a message
    addressed to a third party.
    """
    return re.sub(rf"<@!?{re.escape(str(user_id))}>", " ", text)


def clean(text: str) -> str:
    """Strip the parts of a Discord message that are markup rather than words.

    Entities keep the half a human reads and lose the id: an emoji becomes
    `:name:`, a mention becomes `@someone`, a channel becomes `#channel`. The
    generic replacements are deliberate — a model this small has no use for
    *which* user was mentioned, and putting a real display name in would teach
    it to address people who are not in the conversation.
    """
    text = _EMOJI.sub(r":\1:", text)
    text = _MENTION.sub(" @someone ", text)
    text = _CHANNEL.sub(" #channel ", text)
    text = _COMMAND.sub(r" /\1 ", text)
    text = _TIMESTAMP.sub(" ", text)
    text = _SPOILER.sub(r"\1", text)
    # A URL is a string of nothing this model can learn to produce. Markdown
    # links keep their label -- stripping the target out of `[Teto cinema](url)`
    # and leaving `[Teto cinema](` would teach broken syntax.
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _EMPTY_LINK.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()


def clean_outgoing(text: str) -> str:
    """A generated reply as a channel should see it: `@someone` taken back out.

    `clean` turns every inbound mention into `@someone`, so that is the shape a
    mention has in the training corpus and the model writes it — but the word
    addresses nobody. Discord renders it as grey text, and the id that would
    make it a real mention is exactly what was scrubbed on the way in, so there
    is nothing here to resolve it against.

    A mention that reaches a channel should therefore be one the model asked
    for on purpose, with `[[ping]]` (actions.py), which resolves to the person
    the turn is about. This takes out the rest, along with the comma or colon
    that was addressing them, so "@someone, how are you" reads as it should.
    """
    text = _STRAY_MENTION.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return _SPACE_EOL.sub("\n", text).strip()


def clean_incoming(content: str, bot_id: int | str | None = None) -> str:
    """A live message as the agent should see it: the bot's own @mention taken
    off the front, then the same cleaning the training corpus went through."""
    if bot_id is not None:
        content = strip_mention(content, bot_id)
    return clean(content)
