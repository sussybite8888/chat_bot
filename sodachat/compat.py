"""OpenAI- and Anthropic-shaped endpoints, over the same rooms.

    POST /v1/chat/completions     OpenAI   (`openai` SDK, and everything that
                                            has grown to speak its dialect)
    POST /v1/messages             Anthropic (`anthropic` SDK)
    POST /v1/messages/count_tokens
    GET  /v1/models               both — see "One path, two shapes" below

The native API ([api.py](api.py)) is shaped like what this server *is*: a room,
a message, a reply, and the acts that reply wants taken in that room. It is the
right shape for a bot, and the wrong shape for everything else. Every editor
plugin, terminal client, eval harness and library already knows two other
shapes, so speaking them costs one module and buys all of that — point a thing
at `http://127.0.0.1:8765/v1` with any api key and it works.

Nothing here is a second implementation of anything. A request is translated
into the one call the rest of the server already answers (`Rooms.reply`), and
the reply is dressed in whichever wire format asked for it.

The stateless/stateful problem
------------------------------
Both APIs are **stateless**: the client holds the conversation and resends all
of it every request. `Rooms` is **stateful** — a room is a `SodaAgent` with its
own history, its own running game, its own persona and its own file sandbox,
and that state is the product, not an implementation detail. Bridging the two
naively costs one decode per prior turn, every turn, and throws the room's
non-textual state away each time.

So instead the history is used as an *identity* rather than as input. After
answering, this module fingerprints the conversation **including the reply it
just gave** and files the room under it. The next request arrives holding that
exact conversation plus one new user turn, so its `messages[:-1]` hash to the
fingerprint we stored, and it lands back in the same room — which then only has
to answer the newest turn, with its game and its persona and its notepad intact.

    turn 1   [u1]                 miss  -> new room R, seeded with nothing
             answer a1                  -> file fingerprint([u1, a1]) = R
    turn 2   [u1, a1, u2]         hit on fingerprint([u1, a1])  -> R, feed u2
             answer a2                  -> file fingerprint([u1, a1, u2, a2]) = R

A miss is not an error, it is the other half of the design: a client that edited
an earlier turn, branched the conversation, or is simply new here gets a fresh
room **seeded** with the history it sent (`Rooms.seed` — replayed, not
re-generated). So continuity is an optimization that also preserves game and
persona state, and correctness never depends on it.

The cache is bounded and evicts least-recently-used, resetting the room it drops
so the sandbox directory goes with it. A conversation whose room has been
evicted is not lost — it just takes the seeded path again on its next turn.

One path, two shapes
--------------------
`GET /v1/models` exists in both dialects with different response bodies:
OpenAI wants `{"object": "list", "data": [{"id", "object", "created",
"owned_by"}]}`, Anthropic wants `{"data": [{"type", "id", "display_name",
"created_at"}], "has_more", "first_id", "last_id"}`. One server cannot serve two
bodies at one path, so it serves their **union** — every field both dialects
look for, on one object. Each SDK finds what it parses and ignores the rest,
which is the only arrangement that needs no flag from the client.

What these endpoints do not have
--------------------------------
Honest list, because a compatible shape that quietly means something else is
worse than a missing feature:

  * **Actions are dropped.** `[[react :kekw:]]` is an act for the frontend that
    owns the room (actions.py); an HTTP client has no room to react in. The
    parser already lifts them out of the text, so what comes back is clean prose
    and nothing is silently half-performed.
  * **Sampling parameters are accepted and ignored** — `temperature`, `top_p`,
    `max_tokens`, `n`. Reply length and sampling are the *persona's* (persona.py),
    per room, and a from-scratch model that fits in 190MB is not improved by a
    stranger's `top_p`. `stop` is honoured, because it is post-hoc truncation
    and costs nothing to get right.
  * **Usage counts are estimates.** Rounded from character length, not from the
    tokenizer that actually ran. They are the right order of magnitude and they
    are not a billing record.
  * **A system prompt becomes the conversation's opening turn.** The agent has
    no system-prompt channel — persona is `/persona`, length is `/length`. So
    rather than drop it, it is seeded as the first thing said. It is context,
    not an instruction with authority over the model.
  * **Streaming generates first, then streams.** `Rooms.reply` returns a whole
    reply; there is no token callback to tap. The stream opens immediately and
    holds the connection (so no client times out waiting for a first byte),
    then emits the finished reply in word-sized deltas. Every event a client
    expects arrives in the right order — the text just isn't live.

Only the one model is served, so any `model` string is accepted and echoed back
rather than 404'd: clients hardcode `gpt-4o-mini` and `claude-sonnet-5` in
places you cannot always reach.

Auth is the server's own (`SODACHAT_API_KEY`, checked in api.py): both the
OpenAI `Authorization: Bearer` header and the Anthropic `x-api-key` header are
already what that check reads, so neither SDK needs anything special. With no
key set every request is allowed, which is why the default bind is loopback.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Callable, Iterable, Sequence

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .transport import MAX_ATTACHMENT_BYTES, Attachment

log = logging.getLogger("sodachat.compat")

# The one model this server has. Advertised under a stable id; anything else a
# client asks for is echoed back rather than refused (see the module docstring).
MODEL_ID = "sodachat"

# How many conversations keep their room. Each live room is an agent — a
# history, maybe a running game, a directory on disk — so this is a memory
# bound, not a cache-hit tuning knob. Past it, the least recently used
# conversation is reset and its next turn takes the seeded path.
MAX_CONVERSATIONS = 256

# Same ceiling the native endpoint uses: past a handful of files it is not a
# chat message.
MAX_ATTACHMENTS = 8

# Roughly a token. Used only for the `usage` fields, which are estimates and say
# so — the tokenizer that actually ran is inside the engine, not out here.
_CHARS_PER_TOKEN = 4

_DATA_URL = re.compile(r"^data:([\w.+-]+/[\w.+-]+)?(;[\w-]+=[\w.+-]+)*(;base64)?,", re.I)

# Extension to give a decoded image, so the agent's `/see` path recognizes it by
# suffix the same way it recognizes a dropped file in a terminal.
_IMAGE_SUFFIX = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp",
}


# --------------------------------------------------------------------- errors


class CompatError(Exception):
    """An error that has to come back in the dialect that asked for it.

    The SDKs parse the error body — `openai` reads `error.message`, `anthropic`
    reads `error.type` off a top-level `{"type": "error"}` — and a FastAPI
    `HTTPException` renders as `{"detail": ...}`, which neither recognizes. So
    errors carry the family they belong to and are rendered by the handler
    below, rather than being raised as HTTP exceptions and reshaped after.
    """

    def __init__(self, family: str, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.family = family      # "openai" | "anthropic"
        self.status = status
        self.kind = kind          # the API's own error-type vocabulary
        self.message = message

    def body(self) -> dict:
        if self.family == "anthropic":
            return {"type": "error",
                    "error": {"type": self.kind, "message": self.message}}
        return {"error": {"message": self.message, "type": self.kind,
                          "param": None, "code": None}}

    def response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status)


def _bad(family: str, message: str) -> CompatError:
    kind = "invalid_request_error"
    return CompatError(family, 400, kind, message)


# ----------------------------------------------------------------- the schema
#
# Deliberately permissive. These models exist to *find* the conversation in a
# request, not to re-specify two APIs this server does not own: a field we don't
# act on shouldn't be a 422, because the client is usually an SDK sending its
# whole surface and a rejected unknown field is an outage for something we were
# always going to ignore.


class ChatMessage(BaseModel):
    role: str
    # str, or a list of content parts. Both dialects allow both forms.
    content: Any = ""
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage] = []
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    stop: Any = None
    # Accepted so an SDK's full request body validates; not acted on.
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    user: str | None = None

    model_config = {"extra": "allow"}


class MessagesRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage] = []
    system: Any = None          # str or a list of text blocks
    stream: bool = False
    stop_sequences: list[str] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    metadata: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


# ------------------------------------------------------------- normalization


def _estimate_tokens(text: str) -> int:
    """A token count that is honest about being an estimate. The real tokenizer
    is inside the engine and is not on this path; a client that needs exact
    accounting should not be reading it off a compatibility shim."""
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _decode_data_url(url: str, family: str) -> Attachment | None:
    """An inline image from either dialect's content parts.

    Only `data:` URLs. A remote URL would mean this server fetching whatever a
    stranger names — an SSRF against the host that holds the models, reachable
    by anyone who can reach this port — so an `http(s)` image is skipped rather
    than retrieved, and the turn goes on without it.
    """
    match = _DATA_URL.match(url)
    if match is None:
        return None
    header, _, payload = url.partition(",")
    if ";base64" not in header.lower():
        return None
    media = (match.group(1) or "application/octet-stream").lower()
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise _bad(family, "image data is not valid base64") from None
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise CompatError(
            family, 413, "request_too_large",
            f"an image is {len(data) / 1e6:.0f} MB, past the "
            f"{MAX_ATTACHMENT_BYTES // 1024 // 1024} MB limit")
    suffix = _IMAGE_SUFFIX.get(media, ".png")
    return Attachment(f"image-{uuid.uuid4().hex[:8]}{suffix}", data)


def _flatten(content: Any, family: str) -> tuple[str, list[Attachment]]:
    """One message's content, as text plus any files it carried.

    Handles both dialects' part vocabularies in one pass — they differ in
    spelling (`image_url` vs `image`, a data URL vs a base64 `source`) and not
    in meaning, and a message can only be from one of them anyway.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text: list[str] = []
    files: list[Attachment] = []
    for part in content:
        if isinstance(part, str):
            text.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            text.append(str(part.get("text", "")))
        elif kind == "image_url":                      # OpenAI
            url = (part.get("image_url") or {}).get("url", "")
            if (file := _decode_data_url(str(url), family)) is not None:
                files.append(file)
        elif kind == "image":                          # Anthropic
            source = part.get("source") or {}
            if source.get("type") == "base64":
                url = (f"data:{source.get('media_type', 'image/png')};base64,"
                       f"{source.get('data', '')}")
                if (file := _decode_data_url(url, family)) is not None:
                    files.append(file)
            elif source.get("type") == "url":
                pass  # not fetched, for the reason in _decode_data_url
        elif kind == "tool_result":
            # No tools cross this wire, but a client replaying its own history
            # may send one. Keep whatever text is in it rather than dropping the
            # turn, so the conversation still reads as a conversation.
            inner, _ = _flatten(part.get("content"), family)
            if inner:
                text.append(inner)
    return "\n".join(t for t in text if t), files


def _conversation(messages: Sequence[ChatMessage], system: Any,
                  family: str) -> tuple[list[tuple[str, str]], list[Attachment]]:
    """A request's messages as alternating turns, plus the files of the turn
    being answered.

    Three things happen here, all of them because the agent's history is a flat
    alternating list and a client's `messages` is not:

      * `system` (top-level, Anthropic) and `role: "system"` messages (OpenAI)
        become the opening user turn — see the module docstring for why that,
        rather than dropping them or inventing authority they won't have.
      * consecutive same-role messages are merged, which both APIs permit and
        an alternating history cannot represent.
      * only the **last** user turn's attachments are returned. Earlier images
        belong to turns already answered; restaging them would hand the agent a
        file it was never sent this turn.
    """
    turns: list[tuple[str, str]] = []
    preamble: list[str] = []
    if isinstance(system, str) and system.strip():
        preamble.append(system.strip())
    elif isinstance(system, list):
        text, _ = _flatten(system, family)
        if text.strip():
            preamble.append(text.strip())

    latest: list[Attachment] = []
    for message in messages:
        text, files = _flatten(message.content, family)
        role = message.role
        if role == "system" or role == "developer":
            if text.strip():
                preamble.append(text.strip())
            continue
        if role not in {"user", "assistant"}:
            continue
        if role == "user":
            latest = files          # last one wins; earlier turns are history
        text = text.strip()
        if turns and turns[-1][0] == role:
            turns[-1] = (role, f"{turns[-1][1]}\n\n{text}".strip())
        else:
            turns.append((role, text))

    if preamble:
        opening = "\n\n".join(preamble)
        if turns and turns[0][0] == "user":
            turns[0] = ("user", f"{opening}\n\n{turns[0][1]}".strip())
        else:
            turns.insert(0, ("user", opening))

    # An alternating history starts with a user turn; a leading assistant turn
    # is a client replaying a greeting it generated, and has no user turn to
    # pair with.
    while turns and turns[0][0] == "assistant":
        turns.pop(0)
    if not turns or turns[-1][0] != "user":
        raise _bad(family, "messages must end with a user message")
    if len(latest) > MAX_ATTACHMENTS:
        raise CompatError(family, 413, "request_too_large",
                          f"at most {MAX_ATTACHMENTS} images per message")
    return turns, latest


def _apply_stop(text: str, stop: Any) -> tuple[str, str | None]:
    """Truncate at the earliest stop sequence, and say which one hit.

    Honoured — unlike the sampling parameters — because it is a string
    operation on a finished reply, so "supported" here means the same thing it
    means anywhere else. Anthropic reports the sequence that matched, hence the
    return of the string rather than a flag.
    """
    if stop is None:
        return text, None
    sequences = [stop] if isinstance(stop, str) else list(stop)
    cut, hit = len(text), None
    for sequence in sequences:
        if isinstance(sequence, str) and sequence and (i := text.find(sequence)) >= 0:
            if i < cut:
                cut, hit = i, sequence
    return (text[:cut], hit) if hit is not None else (text, None)


# ------------------------------------------------------- conversation memory


class Conversations:
    """Which room a stateless conversation belongs to.

    Keyed by a fingerprint of the conversation *so far*, refiled after every
    answer so the key moves forward with the conversation. One key per room at a
    time — the old key is dropped when the new one is filed — which is what lets
    eviction reset a room without the risk of dropping one that some other
    fingerprint still points at.
    """

    def __init__(self, forget: Callable[[str], Any],
                 capacity: int = MAX_CONVERSATIONS) -> None:
        self._rooms: OrderedDict[str, str] = OrderedDict()   # fingerprint -> room
        self._keys: dict[str, str] = {}                      # room -> fingerprint
        self._forget = forget
        self._capacity = capacity

    @staticmethod
    def fingerprint(turns: Sequence[tuple[str, str]]) -> str:
        """A conversation's identity. Hashed rather than kept whole: these are
        strangers' messages and this table outlives the request."""
        digest = hashlib.sha256()
        for role, text in turns:
            digest.update(role.encode())
            digest.update(b"\x00")
            digest.update(text.encode())
            digest.update(b"\x1e")
        return digest.hexdigest()

    def resolve(self, prior: Sequence[tuple[str, str]]) -> tuple[str, bool]:
        """The room holding this conversation, and whether it is new.

        `prior` is everything before the turn being answered. A hit means the
        room already lived through all of it; a miss means the caller should
        seed a fresh room with it.
        """
        key = self.fingerprint(prior)
        if (room := self._rooms.get(key)) is not None:
            self._rooms.move_to_end(key)
            return room, False
        return f"compat:{uuid.uuid4().hex}", True

    def remember(self, turns: Sequence[tuple[str, str]], room: str) -> None:
        """File a room under the conversation that now includes its own reply —
        the state the client will send back on the next turn."""
        if (stale := self._keys.pop(room, None)) is not None:
            self._rooms.pop(stale, None)
        key = self.fingerprint(turns)
        self._rooms[key] = room
        self._keys[room] = key
        self._rooms.move_to_end(key)
        while len(self._rooms) > self._capacity:
            _, dropped = self._rooms.popitem(last=False)
            self._keys.pop(dropped, None)
            try:
                self._forget(dropped)
            except Exception:  # a room we are throwing away anyway
                log.exception("failed to reset evicted room %s", dropped)

    def forget(self, room: str) -> None:
        """Drop one room's place in the table, without resetting it — the
        caller is already doing that. Unknown rooms are not an error: the
        native reset endpoint passes every room it is given, and most of them
        belong to a bot rather than to a conversation tracked here."""
        if (key := self._keys.pop(room, None)) is not None:
            self._rooms.pop(key, None)

    def rooms(self) -> list[str]:
        return list(self._keys)

    def clear(self) -> None:
        """Forget every conversation. The rooms themselves are the caller's to
        reset; this is only the map from a client's history to one of them."""
        self._rooms.clear()
        self._keys.clear()


# --------------------------------------------------------------- the replies


def _chunks(text: str, size: int = 24) -> Iterable[str]:
    """A finished reply as stream-sized pieces, split at word boundaries so a
    client rendering deltas as they arrive never shows half a word."""
    parts = re.findall(r"\S+\s*", text)
    if not parts:
        yield text
        return
    buffer = ""
    for part in parts:
        buffer += part
        if len(buffer) >= size:
            yield buffer
            buffer = ""
    if buffer:
        yield buffer


def _sse(data: dict, event: str | None = None) -> str:
    """One server-sent event. Anthropic names its events and OpenAI does not;
    an unnamed event is `message`, which is what the OpenAI SDK reads."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'))}\n\n"


class Answerer:
    """The one path a compat request takes to a reply, and back.

    Both dialects meet here: find or seed the room, answer the newest turn,
    refile the conversation. Everything above this is parsing and everything
    below it is formatting.
    """

    def __init__(self, ready: Callable[[], Any]) -> None:
        self._ready = ready
        self.conversations = Conversations(forget=lambda room: self._ready().reset(room))

    async def answer(self, turns: Sequence[tuple[str, str]],
                     attachments: Sequence[Attachment], family: str) -> str:
        live = self._ready()
        prior, latest = list(turns[:-1]), turns[-1][1]
        room, is_new = self.conversations.resolve(prior)
        if is_new and prior:
            # Replayed, not re-generated: the client already has these words and
            # generating them again would cost a decode per turn and produce
            # different ones (rooms.py `seed`).
            live.seed(room, [text for _, text in prior])
        try:
            reply = await live.reply(room, latest, attachments)
        except Exception:
            log.exception("failed to answer a %s-shaped request in room %s",
                          family, room)
            raise CompatError(family, 500, "api_error",
                              "the model failed to answer that one") from None
        # Actions are dropped here on purpose — an HTTP client has no room to
        # react in. `reply.text` is already free of them (actions.py parses them
        # out of the generated text), so nothing leaks into the response.
        self.conversations.remember(
            [*turns, ("assistant", reply.text)], room)
        return reply.text


# ------------------------------------------------------------------- routers


def _openai_body(text: str, model: str, finish: str, prompt_tokens: int) -> dict:
    completion = _estimate_tokens(text)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text, "refusal": None},
            "logprobs": None,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
        },
    }


def _anthropic_body(text: str, model: str, stop_reason: str,
                    stop_sequence: str | None, input_tokens: int) -> dict:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens,
                  "output_tokens": _estimate_tokens(text)},
    }


def build_router(ready: Callable[[], Any],
                 authorize: Callable[..., Any]) -> tuple[APIRouter, Answerer]:
    """The compat routes, over the same `Rooms` and the same auth as the native
    API. Both are passed in rather than imported: api.py owns the lifespan that
    loads the models and owns the shared-secret check, and this module having
    its own copy of either is how the two drift apart."""

    router = APIRouter(dependencies=[Depends(authorize)])
    answerer = Answerer(ready)

    # ---------------------------------------------------------- OpenAI

    @router.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        turns, attachments = _conversation(request.messages, None, "openai")
        prompt_tokens = sum(_estimate_tokens(text) for _, text in turns)
        model = request.model or MODEL_ID

        if not request.stream:
            text = await answerer.answer(turns, attachments, "openai")
            text, _hit = _apply_stop(text, request.stop)
            # OpenAI reports "stop" for both a natural end and a stop sequence;
            # only a `max_tokens` cut-off (which this server never applies)
            # would report anything else.
            return JSONResponse(_openai_body(text, model, "stop", prompt_tokens))

        include_usage = bool((request.stream_options or {}).get("include_usage"))

        async def stream() -> AsyncIterator[str]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            def frame(delta: dict, finish: str | None = None) -> dict:
                return {"id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta,
                                     "logprobs": None, "finish_reason": finish}]}

            # The role chunk goes out before generation starts, so the client
            # has its headers and a first event while the model is still
            # working rather than waiting out the whole decode on a silent
            # socket (see "Streaming generates first" in the module docstring).
            yield _sse(frame({"role": "assistant", "content": ""}))
            try:
                text = await answerer.answer(turns, attachments, "openai")
            except CompatError as error:
                yield _sse(error.body())
                yield "data: [DONE]\n\n"
                return
            text, _hit = _apply_stop(text, request.stop)
            for piece in _chunks(text):
                yield _sse(frame({"content": piece}))
            yield _sse(frame({}, finish="stop"))
            if include_usage:
                completion = _estimate_tokens(text)
                yield _sse({"id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model, "choices": [],
                            "usage": {"prompt_tokens": prompt_tokens,
                                      "completion_tokens": completion,
                                      "total_tokens": prompt_tokens + completion}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ------------------------------------------------------- Anthropic

    @router.post("/v1/messages")
    async def messages(request: MessagesRequest):
        turns, attachments = _conversation(
            request.messages, request.system, "anthropic")
        input_tokens = sum(_estimate_tokens(text) for _, text in turns)
        model = request.model or MODEL_ID

        if not request.stream:
            text = await answerer.answer(turns, attachments, "anthropic")
            text, hit = _apply_stop(text, request.stop_sequences)
            return JSONResponse(_anthropic_body(
                text, model, "end_turn" if hit is None else "stop_sequence",
                hit, input_tokens))

        async def stream() -> AsyncIterator[str]:
            message_id = f"msg_{uuid.uuid4().hex}"
            # message_start carries the whole message shell with empty content;
            # the SDKs accumulate onto it, so its usage block has to be present
            # even though output_tokens is not known until the end.
            yield _sse({"type": "message_start", "message": {
                "id": message_id, "type": "message", "role": "assistant",
                "model": model, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }}, "message_start")
            yield _sse({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}},
                       "content_block_start")
            # Keeps the connection warm across the decode, which is exactly what
            # a ping is for.
            yield _sse({"type": "ping"}, "ping")
            try:
                text = await answerer.answer(turns, attachments, "anthropic")
            except CompatError as error:
                yield _sse(error.body(), "error")
                return
            text, hit = _apply_stop(text, request.stop_sequences)
            for piece in _chunks(text):
                yield _sse({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": piece}},
                           "content_block_delta")
            yield _sse({"type": "content_block_stop", "index": 0},
                       "content_block_stop")
            yield _sse({"type": "message_delta",
                        "delta": {"stop_reason":
                                  "end_turn" if hit is None else "stop_sequence",
                                  "stop_sequence": hit},
                        "usage": {"output_tokens": _estimate_tokens(text)}},
                       "message_delta")
            yield _sse({"type": "message_stop"}, "message_stop")

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: MessagesRequest):
        turns, _ = _conversation(request.messages, request.system, "anthropic")
        return {"input_tokens": sum(_estimate_tokens(t) for _, t in turns)}

    # ------------------------------------------------------------ both

    @router.get("/v1/models")
    async def models() -> dict:
        live = ready()
        mode = "agent" if live.agent_mode else "chat"
        created = int(time.time())
        # The union of both dialects' model objects — see "One path, two shapes".
        entry = {
            "id": MODEL_ID, "object": "model", "created": created,
            "owned_by": "sodachat",
            "type": "model", "display_name": f"sodachat ({mode}, {live.device})",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(created)),
        }
        return {"object": "list", "data": [entry], "has_more": False,
                "first_id": MODEL_ID, "last_id": MODEL_ID}

    return router, answerer


def install(app: FastAPI, ready: Callable[[], Any],
            authorize: Callable[..., Any]) -> Answerer:
    """Add the compat endpoints to the server, and teach it to render their
    errors in the dialect that asked."""

    @app.exception_handler(CompatError)
    async def _compat_error(request: Request, error: CompatError) -> JSONResponse:
        return error.response()

    router, answerer = build_router(ready, authorize)
    app.include_router(router)
    return answerer
