"""Master API server: **one** copy of the models, for as many bots as you like.

    python -m sodachat.api            # http://127.0.0.1:8765

Every frontend used to load its own checkpoints, so running the Discord bot and
the Google Chat app meant two copies of a ~190MB model set (and two warm-ups,
and two sets of CUDA context) for conversations that were never going to run at
the same instant anyway. This server holds the models — a `Rooms` from
[rooms.py](rooms.py), one agent per room — and the bots become thin HTTP
clients ([client.py](client.py)) that need neither PyTorch nor a checkpoint.

    discord_bot ─┐
    google_chat ─┼─ HTTP ─> api.py ─> Rooms ─> SodaAgent per room ─> one model
    your own    ─┘

A reply can also carry **actions** — "react with 🔥", "pin that" ([actions.py](
actions.py)). This server never performs one: it holds no gateway connection,
so it names them and the bot that owns the transport does the act. A frontend
that has no such thing ignores the field.

Rooms are namespaced by the frontend that owns them ("discord:123",
"googlechat:spaces/AAA"), so two bots on one server never land in each other's
conversation. Generation stays serialized inside `Rooms`: requests queue on its
lock, which is what you want when the models are one shared copy.

A room can read **only the files it sent** — `/see` and `/code` take a path from
whoever is talking, and here that is a stranger, so each room's agent is confined
to a directory of its own ([files.py](files.py)) holding just the attachments of
the message being answered. Not another room's files, not the host's.

Endpoints
    GET  /healthz                whether the models are loaded (unauthenticated)
    GET  /v1/info                device, mode, and what attachments are accepted
    POST /v1/reply               {room, text, attachments?, reset?}
                                     -> {text, actions}
    POST /v1/watch               {room, text} -> {actions} for a message that
                                     wasn't addressed to the bot (no generation)
    GET  /v1/rooms               the live conversations
    POST /v1/rooms/reset         {room} -> forget one room (or all of them)

Set `SODACHAT_API_KEY` and every route but /healthz requires it, as
`X-API-Key: <key>` or `Authorization: Bearer <key>`. Bind to 127.0.0.1 (the
default) unless the key is set: this endpoint runs a model for whoever asks.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .transport import MAX_ATTACHMENT_BYTES, Attachment, api_key, filtered_enabled

log = logging.getLogger("sodachat.api")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# One message can carry a handful of files; past that it is not a chat message.
MAX_ATTACHMENTS = 8

rooms = None  # the loaded models, set by the lifespan below
_started = 0.0


def api_host() -> str:
    return os.environ.get("SODACHAT_API_HOST", DEFAULT_HOST)


def api_port() -> int:
    return int(os.environ.get("SODACHAT_API_PORT", str(DEFAULT_PORT)))


# --------------------------------------------------------------- the app


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global rooms, _started
    load_dotenv()
    from .rooms import Rooms  # deferred: importing it loads torch

    rooms = Rooms(filtered=filtered_enabled())
    mode = "the agent (chat, games, and the specialists)" if rooms.agent_mode \
        else "the plain chat engine (SODACHAT_AGENT=0)"
    log.info("loading %s...", mode)
    rooms.warm_up()
    _started = time.time()
    log.info("ready on %s — point a bot at this server with SODACHAT_API_URL",
             rooms.device)
    if not api_key() and api_host() not in {"127.0.0.1", "localhost", "::1"}:
        log.warning("listening on %s with no SODACHAT_API_KEY set — anyone who "
                    "can reach this port can run the model", api_host())
    try:
        yield
    finally:
        rooms.stop()


app = FastAPI(title="sodachat model server", lifespan=_lifespan)


def _authorize(request: Request) -> None:
    """Shared-secret auth, when `SODACHAT_API_KEY` is set. Compared in constant
    time; the value never appears in a URL, so it stays out of access logs."""
    expected = api_key()
    if not expected:
        return
    header = request.headers.get("X-API-Key") or ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        header = header or auth.removeprefix("Bearer ").strip()
    if not header or not secrets.compare_digest(header, expected):
        raise HTTPException(401, "missing or invalid API key")


def _ready():
    if rooms is None:
        raise HTTPException(503, "models are still loading")
    return rooms


# ------------------------------------------------------------------ schema


class AttachmentIn(BaseModel):
    """A file from a room, base64 in JSON — attachments are a few MB at most and
    this keeps the client to httpx and the standard library."""

    filename: str = Field(default="file", max_length=255)
    content_b64: str


class ReplyRequest(BaseModel):
    room: str = Field(min_length=1, max_length=200)
    text: str = ""
    attachments: list[AttachmentIn] = []
    reset: bool = False  # start the room over before answering


class ActionOut(BaseModel):
    """One act the reply asks the *frontend* to perform. Kept as a plain
    name/arg pair rather than a variant per tool: the vocabulary lives in
    actions.py, and a new tool should not need a schema change here."""

    name: str
    arg: str = ""


class ReplyResponse(BaseModel):
    room: str
    text: str
    seconds: float
    actions: list[ActionOut] = []


class WatchRequest(BaseModel):
    room: str = Field(min_length=1, max_length=200)
    text: str = ""


class ResetRequest(BaseModel):
    room: str | None = None  # None: every room on the server


def _decode(items: list[AttachmentIn]) -> list[Attachment]:
    if len(items) > MAX_ATTACHMENTS:
        raise HTTPException(413, f"at most {MAX_ATTACHMENTS} attachments per message")
    out: list[Attachment] = []
    for item in items:
        try:
            data = base64.b64decode(item.content_b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, f"{item.filename}: content_b64 is not base64")
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                413, f"{item.filename} is {len(data) / 1e6:.0f} MB, past the "
                     f"{MAX_ATTACHMENT_BYTES // 1024 // 1024} MB limit")
        out.append(Attachment(item.filename, data))
    return out


# --------------------------------------------------------------- endpoints


@app.get("/healthz")
async def healthz() -> dict:
    """Unauthenticated on purpose: a load balancer or `docker healthcheck` asks
    this, and it says nothing a probe shouldn't see."""
    return {"ok": True, "ready": rooms is not None}


@app.get("/v1/info", dependencies=[Depends(_authorize)])
async def info() -> dict:
    return {**_ready().info(), "uptime_seconds": round(time.time() - _started, 1)}


@app.post("/v1/reply", dependencies=[Depends(_authorize)],
          response_model=ReplyResponse)
async def reply(request: ReplyRequest) -> ReplyResponse:
    live = _ready()
    if request.reset:
        live.reset(request.room)
    # An empty message is not an error: the agent answers it with a nudge
    # ("you there? say something"), which is what a bare @mention should get.
    attachments = _decode(request.attachments)
    t0 = time.perf_counter()
    try:
        answer = await live.reply(request.room, request.text, attachments)
    except Exception:
        # The room keeps its state; only this turn is lost. Log the traceback
        # here, where it is useful, and tell the bot something it can post.
        log.exception("failed to answer in room %s", request.room)
        raise HTTPException(500, "the model failed to answer that one") from None
    return ReplyResponse(
        room=request.room, text=answer.text,
        seconds=round(time.perf_counter() - t0, 3),
        actions=[ActionOut(name=a.name, arg=a.arg) for a in answer.actions])


@app.post("/v1/watch", dependencies=[Depends(_authorize)])
async def watch(request: WatchRequest) -> dict:
    """What to do about a message nobody sent to the bot — a reaction, usually
    nothing at all. No model runs here and no room is created, so a channel
    bot can ask this about every message it can see without the cost landing
    on the models or on this server's generation lock."""
    actions = _ready().watch(request.room, request.text)
    return {"actions": [{"name": a.name, "arg": a.arg} for a in actions]}


@app.get("/v1/rooms", dependencies=[Depends(_authorize)])
async def list_rooms() -> dict:
    live = _ready()
    return {"rooms": live.rooms()}


@app.post("/v1/rooms/reset", dependencies=[Depends(_authorize)])
async def reset(request: ResetRequest) -> dict:
    live = _ready()
    if request.room is None:
        count = len(live.rooms())
        live.stop()
        return {"reset": count}
    return {"reset": int(live.reset(request.room))}


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    load_dotenv()
    p = argparse.ArgumentParser(
        prog="sodachat.api",
        description="Serve one copy of the models to every bot (see client.py).")
    p.add_argument("--host", default=api_host(),
                   help=f"default: {DEFAULT_HOST}, or SODACHAT_API_HOST")
    p.add_argument("--port", type=int, default=api_port(),
                   help=f"default: {DEFAULT_PORT}, or SODACHAT_API_PORT")
    a = p.parse_args(argv)
    # Models load in the lifespan, so one worker: a second would be a second
    # copy of the weights, which is the whole thing this server exists to avoid.
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
