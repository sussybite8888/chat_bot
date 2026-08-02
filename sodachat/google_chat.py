"""Google Chat frontend: an HTTP endpoint implementing the Chat app events API.

Google Chat delivers MESSAGE / ADDED_TO_SPACE events as JSON POSTs and renders
whatever ``{"text": ...}`` we return. Point a Chat app (Google Cloud console ->
Google Chat API -> Configuration) at this server's public HTTPS URL.

Runs the full agent by default ([agent.py](agent.py)), so a space gets what the
terminal does: the routing specialist picks which capability answers each
message, and `/commands` work (`/play snake`, `/gen`, `/think`, `/route`,
`/help`). Each space keeps its own agent — its own history, its own running
game — while the models are loaded once and shared. `SODACHAT_AGENT=0` falls
back to the plain chat engine.

**Attachments are not read here.** Google Chat sends a reference, not the file,
and fetching one needs the Chat API with service-account credentials — unlike
Discord, where the attachment comes with a URL the bot can already use. So an
uploaded image reaches this app as an empty message with metadata; the app says
so rather than ignoring it. `/see <path>` still works for files on the machine
running the server.

Optionally set GOOGLE_CHAT_AUDIENCE to your Cloud project number to verify the
bearer token Google attaches to each request (requires ``pip install
google-auth``); without it, requests are accepted unverified — fine for local
testing, not for production.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from .engine import ChatEngine
from .rooms import (
    GOOGLE_CHAT_LIMIT,
    Rooms,
    agent_mode_enabled,
    filtered_enabled,
    format_reply,
)

log = logging.getLogger("sodachat.googlechat")

_CHAT_ISSUER = "chat@system.gserviceaccount.com"

engine: ChatEngine | None = None
rooms: Rooms | None = None

# Plain-chat mode only: recent lines per Chat space. In agent mode each space's
# agent keeps its own history.
_histories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=8))

_GREETING = (
    "hey! i'm a small chatbot trained from scratch — chat, games, and a handful "
    "of specialists (write code, work a question out, recognize an image). "
    "say hi, or try /help."
)
_NO_ATTACHMENTS = (
    "i can see you attached something, but Google Chat only sends me a "
    "reference to it — i'd need the Chat API and credentials to fetch it. "
    "Send me a path with /see or /code if the file is on my machine."
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global engine, rooms
    load_dotenv()
    filtered = filtered_enabled()
    if agent_mode_enabled():
        log.info("loading the agent (chat, games, and the specialists)...")
        rooms = Rooms(filtered=filtered)
        rooms.warm_up()
        log.info("agent ready on %s", rooms.device)
    else:
        log.info("loading the chat model (SODACHAT_AGENT=0: plain chat)...")
        engine = ChatEngine(filtered=filtered)
        log.info("engine ready")
    try:
        yield
    finally:
        if rooms is not None:
            rooms.stop()


app = FastAPI(title="sodachat Google Chat app", lifespan=_lifespan)


def _verify_request(request: Request) -> None:
    audience = os.environ.get("GOOGLE_CHAT_AUDIENCE")
    if not audience:
        return
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
    except ImportError:
        raise HTTPException(
            500, "GOOGLE_CHAT_AUDIENCE is set but google-auth is not installed"
        )
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        claims = id_token.verify_oauth2_token(
            auth.removeprefix("Bearer "), google_requests.Request(), audience=audience
        )
    except ValueError:
        raise HTTPException(401, "invalid bearer token")
    if claims.get("iss") not in (_CHAT_ISSUER, f"https://{_CHAT_ISSUER}"):
        raise HTTPException(401, "unexpected token issuer")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "engine_ready": (rooms or engine) is not None,
            "agent_mode": rooms is not None}


@app.post("/")
async def on_event(request: Request) -> dict:
    _verify_request(request)
    event = await request.json()
    event_type = event.get("type")

    if event_type == "ADDED_TO_SPACE":
        return {"text": _GREETING}
    if event_type != "MESSAGE":
        return {}

    message = event.get("message", {})
    # argumentText excludes the leading @mention of the app, when present
    text = message.get("argumentText") or message.get("text") or ""
    space = event.get("space", {}).get("name", "dm")

    if rooms is None:  # plain chat
        history = _histories[space]
        reply = engine.reply(text, history=history)
        history.extend([text, reply.text])
        return {"text": reply.text}

    if not text.strip():
        attached = message.get("attachment") or message.get("attachments")
        return {"text": _NO_ATTACHMENTS} if attached else {}
    try:
        reply = await rooms.reply(space, text)
    except Exception:
        log.exception("failed to handle a message in space %s", space)
        return {"text": "something went wrong on my end, sorry."}
    # An event gets exactly one reply, so unlike Discord there is nowhere to put
    # the overflow — say it was cut rather than quietly dropping it.
    parts = format_reply(reply, GOOGLE_CHAT_LIMIT)
    return {"text": parts[0] if len(parts) == 1 else f"{parts[0]}\n_(cut short)_"}


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    load_dotenv()
    port = int(os.environ.get("GOOGLE_CHAT_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
