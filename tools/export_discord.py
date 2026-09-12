"""Export a Discord channel's history into `data/` as training data.

Reads a bot credential out of the repo's gitignored `bot_token` file (one
`name` line followed by its token, blank-line separated), talks to the Discord
REST API directly, and pages the whole channel backwards 100 messages at a
time.

    python tools/export_discord.py list
    python tools/export_discord.py export --guild flowrix.sussybite.dev \
                                          --channel oracle-training
    python tools/export_discord.py render models/oracle-training.jsonl

`export` writes JSONL — one message per line, full fidelity, kept out of
`data/` so a re-export never trains on itself. `render` turns that into the
prose documents `localdata.py` picks up: a transcript split into conversation
sessions on a gap in the timestamps, one `.txt` per session, because
`train.py` treats each file in `data/text/` as one document and masks attention
at its boundary. One 10,000-message wall of text would be a single document and
teach the model that conversations never end.

The bot needs *View Channel* and *Read Message History* on the channel, and the
*Message Content Intent* enabled on its application — without the intent the
API returns every message with an empty `content`, which `export` detects and
reports rather than writing a file of blanks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sodachat.discord_text import clean as _clean  # noqa: E402
from sodachat.model import SPEAKERS  # noqa: E402  (needs ROOT on the path)

TOKENS = ROOT / "bot_token"
API = "https://discord.com/api/v10"
UA = "DiscordBot (https://github.com/sussybite8888/chat_bot, 1.0)"

PAGE = 100              # the API's maximum per messages request
SESSION_GAP_MIN = 45    # a quiet gap this long ends a conversation
MIN_SESSION_MSGS = 4    # below this there is no conversation to learn from
MAX_TURN_CHARS = 320    # data._MAX_UTTERANCE_CHARS: the cap every loader applies

# Channel types worth reading text out of: text, announcement, and the two
# thread kinds. Voice/category/forum-container channels hold no transcript.
TEXT_CHANNELS = {0, 5, 10, 11, 12}


def shown(path: Path) -> str:
    """A path as it should be printed: relative to the repo when it is inside
    it, absolute when the caller sent output somewhere else."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def credentials(path: Path = TOKENS) -> dict[str, str]:
    """`{name: token}` from the `bot_token` file. Blank lines separate the
    entries; within one, the first line names it and the second is the token."""
    if not path.is_file():
        sys.exit(f"{path} not found — it holds the bot tokens (and is gitignored).")
    names: dict[str, str] = {}
    pending: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            pending = None
        elif pending is None:
            pending = line
        else:
            names[pending] = line
            pending = None
    return names


def token_for(name: str) -> str:
    creds = credentials()
    if name not in creds:
        sys.exit(f"no credential named {name!r} in {TOKENS.name} "
                 f"(have: {', '.join(sorted(creds)) or 'none'})")
    return creds[name]


def api(path: str, token: str, **params) -> object:
    """One GET against the Discord API, retrying through rate limits.

    Discord answers a 429 with the seconds to wait in the body, and warns of
    the next one in `X-RateLimit-Remaining`; honouring both keeps a long export
    from being throttled into failure halfway through."""
    url = f"{API}{path}" + (f"?{urlencode(params)}" if params else "")
    request = Request(url, headers={"Authorization": f"Bot {token}",
                                    "User-Agent": UA})
    for attempt in range(8):
        try:
            with urlopen(request, timeout=30) as response:
                body = json.loads(response.read() or b"null")
                if response.headers.get("X-RateLimit-Remaining") == "0":
                    time.sleep(float(response.headers.get(
                        "X-RateLimit-Reset-After", 1.0)) + 0.05)
                return body
        except HTTPError as e:
            if e.code == 429:
                payload = json.loads(e.read() or b"{}")
                time.sleep(float(payload.get("retry_after", 1.0)) + 0.1)
                continue
            if e.code == 401:
                sys.exit("Discord rejected the token (401). Check the "
                         f"credential in {TOKENS.name} is current.")
            if e.code == 403:
                sys.exit("the bot is not allowed to read that (403). It needs "
                         "View Channel + Read Message History there.")
            if e.code == 404:
                sys.exit(f"not found (404): {path}")
            raise
        except URLError as e:  # transient DNS/connection trouble
            if attempt == 7:
                raise
            time.sleep(1.5 * (attempt + 1))
    sys.exit("gave up after repeated rate limits from Discord.")


def find_guild(token: str, name: str) -> dict:
    guilds = api("/users/@me/guilds", token)
    exact = [g for g in guilds if g["name"].lower() == name.lower()]
    loose = [g for g in guilds if name.lower() in g["name"].lower()]
    hits = exact or loose
    if not hits:
        listing = "\n".join(f"  {g['name']}" for g in guilds) or "  (none)"
        sys.exit(f"the bot is not in a server matching {name!r}. It is in:\n{listing}")
    if len(hits) > 1:
        listing = "\n".join(f"  {g['name']}" for g in hits)
        sys.exit(f"{name!r} matches more than one server:\n{listing}")
    return hits[0]


def find_channel(token: str, guild_id: str, name: str) -> dict:
    channels = api(f"/guilds/{guild_id}/channels", token)
    wanted = name.lstrip("#").lower()
    hits = [c for c in channels
            if c.get("type") in TEXT_CHANNELS and c.get("name", "").lower() == wanted]
    if not hits:
        listing = "\n".join(f"  #{c['name']}" for c in channels
                            if c.get("type") in TEXT_CHANNELS) or "  (none visible)"
        sys.exit(f"no text channel named {name!r} the bot can see. It sees:\n{listing}")
    return hits[0]


def fetch_messages(token: str, channel_id: str, limit: int | None = None,
                   log=print) -> list[dict]:
    """Every message in the channel, oldest first.

    Pages backwards from the newest with `before`, which is the only ordering
    the API offers a complete walk in; the result is reversed at the end so the
    transcript reads forwards."""
    out: list[dict] = []
    before: str | None = None
    while True:
        params = {"limit": PAGE}
        if before:
            params["before"] = before
        batch = api(f"/channels/{channel_id}/messages", token, **params)
        if not batch:
            break
        out.extend(batch)
        before = batch[-1]["id"]
        log(f"  fetched {len(out):,} messages...")
        if len(batch) < PAGE or (limit and len(out) >= limit):
            break
    if limit:
        out = out[:limit]
    return list(reversed(out))


def slim(message: dict) -> dict:
    """The fields worth keeping: who said what, when, and what it replied to."""
    author = message.get("author") or {}
    return {
        "id": message["id"],
        "timestamp": message["timestamp"],
        "edited": message.get("edited_timestamp"),
        "type": message.get("type"),
        "author": {
            "id": author.get("id"),
            "name": author.get("global_name") or author.get("username"),
            "username": author.get("username"),
            "bot": bool(author.get("bot")),
        },
        "content": message.get("content") or "",
        "reply_to": (message.get("referenced_message") or {}).get("id"),
        "attachments": [a.get("filename") for a in message.get("attachments", [])],
    }


def cmd_list(args) -> None:
    token = token_for(args.bot)
    me = api("/users/@me", token)
    print(f"logged in as {me.get('username')} (id {me.get('id')})\n")
    for guild in api("/users/@me/guilds", token):
        print(f"{guild['name']}  (id {guild['id']})")
        for channel in api(f"/guilds/{guild['id']}/channels", token):
            if channel.get("type") in TEXT_CHANNELS:
                print(f"    #{channel['name']}  (id {channel['id']})")
        print()


def cmd_export(args) -> None:
    token = token_for(args.bot)
    guild = find_guild(token, args.guild)
    channel = find_channel(token, guild["id"], args.channel)
    print(f"{guild['name']} / #{channel['name']} (id {channel['id']})")

    messages = fetch_messages(token, channel["id"], args.limit)
    if not messages:
        sys.exit("the channel is empty (or the bot cannot read its history).")

    kept = [slim(m) for m in messages]
    if not any(m["content"] for m in kept):
        sys.exit(f"all {len(kept):,} messages came back with empty content — the "
                 f"application needs the Message Content Intent enabled "
                 f"(Developer Portal -> Bot -> Privileged Gateway Intents).")

    out = Path(args.out) if args.out else ROOT / "models" / f"{channel['name']}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for message in kept:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    humans = sum(1 for m in kept if not m["author"]["bot"])
    span = f"{kept[0]['timestamp'][:10]} .. {kept[-1]['timestamp'][:10]}"
    print(f"\nwrote {len(kept):,} messages ({humans:,} from humans) to "
          f"{shown(out)}\n  {span}")
    print(f"\nnext: python tools/export_discord.py render {shown(out)}")


def sessions(messages: list[dict], gap_minutes: int) -> list[list[dict]]:
    """Split a flat transcript into conversations on a gap in the timestamps."""
    out: list[list[dict]] = []
    current: list[dict] = []
    previous: datetime | None = None
    for message in messages:
        when = datetime.fromisoformat(message["timestamp"]).astimezone(timezone.utc)
        if previous and (when - previous).total_seconds() > gap_minutes * 60:
            out.append(current)
            current = []
        current.append(message)
        previous = when
    if current:
        out.append(current)
    return out


def turns(group: list[dict]) -> list[tuple[str, str]]:
    """Collapse a run of messages into conversational turns.

    Discord chat is bursty — one person sends four lines in a row where a
    dataset dialogue would have one utterance — so consecutive messages from
    the same author become a single turn. Without this, alternating A/B over
    raw messages invents turn-taking that never happened.
    """
    out: list[tuple[str, str]] = []
    for message in group:
        name = message["author"]["name"] or "someone"
        text = " ".join(message["content"].split())
        if out and out[-1][0] == name:
            out[-1] = (name, f"{out[-1][1]} {text}")
        else:
            out.append((name, text))
    return out


def render_lines(group: list[dict], speakers: str) -> tuple[list[str], int]:
    """One session's training text, plus how many turns had to be truncated.

    `ab` is the format the model actually speaks: `model.build_prompt` ends
    every inference prompt with "B:", so tagging turns A/B here makes a file
    from `data/` identical to what `data.format_dialogue` emits for SODA --
    `format_document` only strips and appends the separator, so the tags
    survive untouched. Real display names would instead teach a speaker
    convention that never appears at inference, and can be generated after
    "B:" as if they were words.
    """
    if speakers == "names":
        lines = []
        for message in group:
            name = message["author"]["name"] or "someone"
            for i, part in enumerate(message["content"].split("\n")):
                part = part.strip()
                if part:
                    lines.append(f"{name}: {part}" if i == 0 else f"  {part}")
        return lines, 0

    lines, clipped = [], 0
    for i, (_, text) in enumerate(turns(group)):
        if len(text) > MAX_TURN_CHARS:
            cut = text[:MAX_TURN_CHARS].rsplit(" ", 1)[0]
            text = (cut or text[:MAX_TURN_CHARS]).rstrip(" ,;:-") + "..."
            clipped += 1
        lines.append(text if speakers == "none" else f"{SPEAKERS[i % 2]}: {text}")
    return lines, clipped


def cmd_render(args) -> None:
    source = Path(args.jsonl)
    messages = [json.loads(line) for line in
                source.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Resolved against every message, including the ones about to be dropped,
    # so a reply can be told from what it was replying to.
    by_id = {m["id"]: m for m in messages}

    usable = []
    dangling = 0
    for message in messages:
        if message["author"]["bot"] and not args.include_bots:
            continue
        if message.get("type") not in (0, 19):   # plain message, or a reply
            continue
        parent = by_id.get(message.get("reply_to") or "")
        if parent and parent["author"]["bot"] and not args.include_bots:
            # This line answers a bot message that is not in the transcript.
            dangling += 1
            if args.drop_bot_replies:
                continue
        text = _clean(message["content"])
        if text:
            usable.append({**message, "content": text})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or source.stem

    written = chars = truncated = 0
    for group in sessions(usable, args.gap):
        if len(group) < args.min_messages:
            continue
        lines, clipped = render_lines(group, args.speakers)
        truncated += clipped
        if len(lines) < 2:            # a turn needs an answer to be a dialogue
            continue
        body = "\n".join(lines) + "\n"
        day = group[0]["timestamp"][:10]
        path = out_dir / f"{stem}-{day}-{group[0]['id'][-6:]}.txt"
        path.write_text(body, encoding="utf-8")
        written += 1
        chars += len(body)

    skipped = len(messages) - len(usable)
    print(f"{len(messages):,} messages in, {skipped:,} skipped "
          f"(bots, joins, empty after cleaning)")
    if dangling:
        verb = "dropped" if args.drop_bot_replies else "kept"
        print(f"{dangling:,} of them reply to a bot message that is not in the "
              f"transcript ({verb}; --drop-bot-replies flips this)")
    if truncated:
        print(f"{truncated:,} turn(s) truncated to {MAX_TURN_CHARS} chars "
              f"(the cap data.py applies to every shipped dialogue)")
    print(f"wrote {written:,} conversation files ({chars / 1000:.0f}k chars, "
          f"~{chars / 3500:.0f}k tokens) to {shown(out_dir)}/")
    print("\nnext: python -m sodachat.localdata")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bot", default="exporter",
                   help=f"which credential in {TOKENS.name} (default: exporter)")
    sub = p.add_subparsers(dest="cmd", required=True)

    listing = sub.add_parser("list", help="servers and channels the bot can see")
    listing.set_defaults(func=cmd_list)

    export = sub.add_parser("export", help="fetch a channel's history to JSONL")
    export.add_argument("--guild", required=True)
    export.add_argument("--channel", required=True)
    export.add_argument("--out", default=None)
    export.add_argument("--limit", type=int, default=None,
                        help="stop after this many (newest) messages")
    export.set_defaults(func=cmd_export)

    render = sub.add_parser("render", help="turn a JSONL export into data/text/")
    render.add_argument("jsonl")
    render.add_argument("--out-dir", default=str(ROOT / "data" / "text"))
    render.add_argument("--name", default=None, help="filename stem")
    render.add_argument("--gap", type=int, default=SESSION_GAP_MIN,
                        help=f"minutes of quiet that end a conversation "
                             f"(default: {SESSION_GAP_MIN})")
    render.add_argument("--min-messages", type=int, default=MIN_SESSION_MSGS)
    render.add_argument("--speakers", choices=("ab", "names", "none"),
                        default="ab",
                        help="how to tag turns: ab (default) matches the "
                             "A:/B: protocol model.build_prompt uses at "
                             "inference; names keeps Discord display names; "
                             "none writes bare lines")
    render.add_argument("--include-bots", action="store_true")
    render.add_argument("--drop-bot-replies", action="store_true",
                        help="also drop human lines that reply to a dropped "
                             "bot message, leaving only human-to-human turns")
    render.set_defaults(func=cmd_render)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
