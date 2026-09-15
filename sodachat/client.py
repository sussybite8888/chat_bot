"""Talking to the models, from a bot that doesn't hold them.

A frontend wants one thing — "here's a message and maybe some files, give me a
reply" — and shouldn't care whether the model is in this process or behind an
HTTP call. That's `Backend`, with two implementations:

* `RemoteBackend` — the master API server ([api.py](api.py)). One copy of the
  models for every bot pointed at it, and this process never imports torch.
* `LocalBackend` — the models in the frontend's own process, which is what the
  bots did before this module existed. Still the right answer for one bot on one
  machine.

`open_backend()` picks between them from the environment: `SODACHAT_API_URL`
set means remote, unset means local. A frontend calls it once at startup and
then only ever calls `reply`, so both arrangements are the same code path.

Smoke-test a running server without a bot:

    python -m sodachat.client "hello there"
    python -m sodachat.client --room dev --file cat.png "what is this?"
"""

from __future__ import annotations

import base64
import logging
from typing import Protocol, Sequence

import httpx

from .transport import (
    MAX_ATTACHMENT_BYTES,
    Attachment,
    api_key,
    api_url,
    filtered_enabled,
)

log = logging.getLogger("sodachat.client")

# Generation is seconds, and requests queue behind each other on the server's
# single generation lock, so the read timeout is generous. Connecting is not.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=300.0)


class BackendError(RuntimeError):
    """The backend could not answer — unreachable, refused, or it failed."""


class Backend(Protocol):
    """What a frontend is allowed to assume about its models."""

    info: dict
    accepted_suffixes: tuple[str, ...]
    max_attachment_bytes: int

    async def start(self) -> None: ...
    async def reply(self, room: str, text: str,
                    attachments: Sequence[Attachment] = ()) -> str: ...
    async def reset(self, room: str) -> None: ...
    async def close(self) -> None: ...
    def accepts(self, filename: str) -> bool: ...
    def describe(self) -> str: ...


class _Common:
    """What both backends answer about themselves, from the same `info` dict, so
    a frontend's "can I send you this file?" check works either way."""

    info: dict

    @property
    def accepted_suffixes(self) -> tuple[str, ...]:
        return tuple(self.info.get("accepted_suffixes") or ())

    @property
    def max_attachment_bytes(self) -> int:
        return int(self.info.get("max_attachment_bytes") or MAX_ATTACHMENT_BYTES)

    def accepts(self, filename: str) -> bool:
        """Whether a specialist can do anything with this file — asked before a
        frontend spends bandwidth downloading it."""
        suffixes = self.accepted_suffixes
        return bool(suffixes) and filename.lower().endswith(suffixes)


class RemoteBackend(_Common):
    """The master API server, over HTTP."""

    def __init__(self, url: str, key: str | None = None,
                 timeout: httpx.Timeout | float = DEFAULT_TIMEOUT) -> None:
        self.url = url.rstrip("/")
        headers = {"X-API-Key": key} if key else {}
        self._http = httpx.AsyncClient(base_url=self.url, headers=headers,
                                       timeout=timeout)
        self.info = {}

    async def start(self) -> None:
        """Fetch what the server is, which doubles as the connection check: a
        bot that can't reach its models should fail at startup and say so, not
        on the first message in a channel."""
        try:
            response = await self._http.get("/v1/info")
        except httpx.HTTPError as e:
            await self.close()
            raise BackendError(
                f"can't reach the sodachat API server at {self.url} ({e}). Start "
                f"it with `python -m sodachat.api`, or unset SODACHAT_API_URL to "
                f"load the models into this process."
            ) from None
        if response.status_code == 401:
            await self.close()
            raise BackendError(f"{self.url} rejected the API key "
                               f"(set SODACHAT_API_KEY to the server's)")
        if response.status_code != 200:
            await self.close()
            raise BackendError(f"{self.url} is not a sodachat API server: "
                               f"{_detail(response)}")
        try:
            self.info = response.json()
        except ValueError:
            await self.close()
            raise BackendError(f"{self.url} answered /v1/info with something "
                               f"that isn't JSON") from None

    async def reply(self, room: str, text: str,
                    attachments: Sequence[Attachment] = ()) -> str:
        payload = {
            "room": room,
            "text": text,
            "attachments": [
                {"filename": a.filename,
                 "content_b64": base64.b64encode(a.data).decode()}
                for a in attachments
            ],
        }
        try:
            response = await self._http.post("/v1/reply", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = _detail(e.response)
            raise BackendError(f"the model server said: {detail}") from None
        except httpx.HTTPError as e:
            raise BackendError(f"the model server is unreachable ({e})") from None
        try:
            return response.json()["text"]
        except (ValueError, KeyError):
            raise BackendError("the model server sent a reply I can't read") from None

    async def reset(self, room: str) -> None:
        try:
            await self._http.post("/v1/rooms/reset", json={"room": room})
        except httpx.HTTPError as e:
            raise BackendError(f"could not reset {room} ({e})") from None

    async def close(self) -> None:
        await self._http.aclose()

    def describe(self) -> str:
        device = self.info.get("device", "?")
        mode = "agent" if self.info.get("agent_mode") else "plain chat"
        return f"shared models at {self.url} ({mode} on {device})"


class LocalBackend(_Common):
    """The models in this process — one bot, one copy, no server."""

    def __init__(self, filtered: bool | None = None,
                 device: str | None = None) -> None:
        self._filtered = filtered_enabled() if filtered is None else filtered
        self._device = device
        self._rooms = None
        self.info = {}

    async def start(self) -> None:
        import asyncio

        from .rooms import Rooms  # deferred: importing it loads torch

        self._rooms = Rooms(device=self._device, filtered=self._filtered)
        # Loading ~190MB of checkpoints blocks for several seconds; keep it off
        # the event loop so a gateway connection opened alongside stays alive.
        await asyncio.to_thread(self._rooms.warm_up)
        self.info = self._rooms.info()

    async def reply(self, room: str, text: str,
                    attachments: Sequence[Attachment] = ()) -> str:
        try:
            return await self._rooms.reply(room, text, attachments)
        except Exception as e:
            log.exception("failed to answer in room %s", room)
            raise BackendError(str(e)) from None

    async def reset(self, room: str) -> None:
        self._rooms.reset(room)

    async def close(self) -> None:
        if self._rooms is not None:
            self._rooms.stop()

    def describe(self) -> str:
        mode = "agent" if self.info.get("agent_mode") else "plain chat"
        return f"models in this process ({mode} on {self.info.get('device', '?')})"


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


async def open_backend(filtered: bool | None = None,
                       device: str | None = None) -> Backend:
    """The backend this process should use, started and ready.

    `SODACHAT_API_URL` points at a master API server; without it the models are
    loaded here. Either way the caller gets something with `.reply()`."""
    url = api_url()
    backend: Backend = (RemoteBackend(url, api_key()) if url
                        else LocalBackend(filtered=filtered, device=device))
    await backend.start()
    return backend


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    """A one-message client, for checking a server without wiring up a bot."""
    import argparse
    import asyncio
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()
    p = argparse.ArgumentParser(
        prog="sodachat.client",
        description="Send one message to the sodachat API server and print the reply.")
    p.add_argument("text", nargs="*", help="the message")
    p.add_argument("--room", default="cli", help="conversation to speak in (default: cli)")
    p.add_argument("--file", type=Path, action="append", default=[],
                   metavar="PATH", help="attach a file (repeatable)")
    p.add_argument("--url", default=None, help="server URL (default: SODACHAT_API_URL)")
    p.add_argument("--reset", action="store_true", help="forget the room first")
    a = p.parse_args(argv)

    url = (a.url or api_url())
    if not url:
        p.error("no server: pass --url or set SODACHAT_API_URL "
                "(start one with `python -m sodachat.api`)")

    async def run() -> None:
        backend = RemoteBackend(url, api_key())
        try:
            await backend.start()
            if a.reset:
                await backend.reset(a.room)
            attachments = [Attachment(f.name, f.read_bytes()) for f in a.file]
            print(await backend.reply(a.room, " ".join(a.text), attachments))
        finally:
            await backend.close()

    try:
        asyncio.run(run())
    except BackendError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
