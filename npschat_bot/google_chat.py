"""Google Chat frontend: an HTTP endpoint implementing the Chat app events API.

Google Chat delivers MESSAGE / ADDED_TO_SPACE events as JSON POSTs and renders
whatever ``{"text": ...}`` we return. Point a Chat app (Google Cloud console ->
Google Chat API -> Configuration) at this server's public HTTPS URL.

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

log = logging.getLogger("npschat.googlechat")

_CHAT_ISSUER = "chat@system.gserviceaccount.com"

engine: ChatEngine | None = None

# Recent conversation lines per Chat space, used to condition the model.
_histories: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=8))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global engine
    load_dotenv()
    filtered = os.environ.get("NPSCHAT_UNFILTERED", "").lower() not in {"1", "true", "yes"}
    log.info("warming up on the NPS Chat corpus...")
    engine = ChatEngine(filtered=filtered)
    log.info("engine ready")
    yield


app = FastAPI(title="npschat Google Chat app", lifespan=_lifespan)


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
    return {"ok": True, "engine_ready": engine is not None}


@app.post("/")
async def on_event(request: Request) -> dict:
    _verify_request(request)
    event = await request.json()
    event_type = event.get("type")

    if event_type == "ADDED_TO_SPACE":
        return {"text": "hey! i learned everything i know from 2006 chat rooms. say hi!"}
    if event_type == "MESSAGE":
        message = event.get("message", {})
        # argumentText excludes the leading @mention of the app, when present
        text = message.get("argumentText") or message.get("text") or ""
        history = _histories[event.get("space", {}).get("name", "dm")]
        reply = engine.reply(text, history=history)
        history.extend([text, reply.text])
        return {"text": reply.text}
    return {}


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    load_dotenv()
    port = int(os.environ.get("GOOGLE_CHAT_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
