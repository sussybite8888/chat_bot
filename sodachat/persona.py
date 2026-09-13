"""Bot personality: how a reply *sounds*, without retraining anything.

The chat model is a from-scratch A:/B: dialogue LM (model.py). It has no
system prompt, so "be cheerful" is not an instruction it can follow — there
is nowhere to put it. Personality therefore has to be built out of the things
that do reach the model, and this file is those things in one place:

  * **primer turns** — a short example exchange prepended to the conversation,
    so the model's most recent evidence of how B talks is B talking that way.
    This is the main lever: the model continues the stream it is shown.
  * **temperature** — how far the sampler strays from the safe reply. Deadpan
    wants a low one, chaos wants a high one.
  * **mmi_lambda** — the relevance/genericness trade-off in the reranker
    (engine.py). Raising it picks candidates that could only be a reply to
    *this* message; lowering it tolerates small talk.
  * **style** — a light post-process (lowercase, an occasional closer). The
    model can't be talked into a verbal tic, so a tic is applied afterwards.

A `Persona` is data, not code, which is what makes it easy to change: pick one
by name (`--persona`, `SODACHAT_PERSONA`, `/persona` in the agent), or write
your own into `personas.json` and use it without touching this package.

    python -m sodachat.cli --personas     # list what's available, with sources

Two things worth knowing. The primer sits at the *front* of the prompt, which
is also the end trimmed first when a long conversation overflows the context
window (`MiniChatLM.generate_line`) — so primers stay to a couple of exchanges,
and a persona fades rather than fights for room. And the primer is kept out of
the MMI null baseline (`null_prompt`): the baseline is "what would this model
say with no context at all", and priming both sides would cancel the persona
back out of the score.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

# Sampling defaults — the values chat ran with before personas existed, and so
# also exactly what the "neutral" persona below restores.
DEFAULT_TEMPERATURE = 0.75
DEFAULT_MMI_LAMBDA = 0.7

# Where a user-written personas file is looked for, unless SODACHAT_PERSONAS
# names another. Repo root rather than data/, which is gitignored scratch space
# for training material.
PERSONAS_FILE = Path(__file__).resolve().parent.parent / "personas.json"


@dataclass(frozen=True)
class Persona:
    """One personality. See the module docstring for what each knob reaches."""

    name: str
    description: str
    # (user line, bot line) pairs, oldest first, prepended to the conversation
    # as if they had just happened. Pairs, not single lines, because the prompt
    # builder assigns speakers by counting backwards from the user's turn — a
    # half-exchange would shift every later line to the wrong speaker.
    primer: tuple[tuple[str, str], ...] = ()
    temperature: float = DEFAULT_TEMPERATURE
    mmi_lambda: float = DEFAULT_MMI_LAMBDA
    lowercase: bool = False
    # Occasionally appended to the finished reply, verbatim. Keep the leading
    # space if you want one; "!!" and " :)" are both reasonable.
    closers: tuple[str, ...] = ()
    closer_chance: float = 0.0
    source: str = "built-in"  # or the file a custom persona was read from

    def primer_lines(self) -> list[str]:
        """The primer as flat conversation lines, ready to sit in front of the
        real history."""
        return [line for pair in self.primer for line in pair]

    def style(self, text: str, rng: random.Random) -> str:
        """Apply the post-process to a finished reply (or a canned line, so a
        persona doesn't drop away the moment the model has nothing to say)."""
        if not text:
            return text
        if self.lowercase:
            text = text.lower()
        if self.closers and rng.random() < self.closer_chance:
            text = _with_closer(text, rng.choice(self.closers))
        return text


def _with_closer(text: str, closer: str) -> str:
    """Append a closer without making punctuation soup.

    A reply that already ends in "!" or "?" is left alone: it has its own
    force, and ", matey." after a question mark reads like a typo. A closer
    that starts with punctuation replaces a trailing full stop rather than
    trailing it, so "i'm good." + "!" is "i'm good!" and not "i'm good.!"."""
    body = text.rstrip()
    if body.endswith(("!", "?")):
        return text
    if closer.lstrip()[:1] in ".,!?" and body.endswith("."):
        body = body[:-1].rstrip()
    return body + closer


BUILT_INS: dict[str, Persona] = {
    p.name: p
    for p in [
        Persona(
            name="neutral",
            description="the model as trained — no priming, no styling",
        ),
        Persona(
            name="cheerful",
            description="warm and upbeat, glad you asked",
            primer=(
                ("how's it going?",
                 "really good, thanks for asking! i got out for a walk this "
                 "morning and it set the whole day up."),
                ("i had a rough day",
                 "oh no, i'm sorry to hear that. tell me about it — i'm happy "
                 "to listen."),
            ),
            temperature=0.85,
            closers=("!", " :)"),
            closer_chance=0.25,
        ),
        Persona(
            name="deadpan",
            description="short, dry, unimpressed",
            primer=(
                ("how's it going?", "it's going."),
                ("i had a rough day", "sounds rough. days do that."),
            ),
            temperature=0.6,
            mmi_lambda=0.55,
        ),
        Persona(
            name="curious",
            description="asks you questions back",
            primer=(
                ("how's it going?",
                 "not bad at all. what have you been up to lately?"),
                ("i started a new job",
                 "oh, what kind of work is it? and how are you finding it so far?"),
            ),
            temperature=0.8,
            mmi_lambda=0.85,
        ),
        Persona(
            name="grumpy",
            description="put-upon, faintly annoyed, still answers",
            primer=(
                ("how's it going?", "could be worse. what do you want?"),
                ("i had a rough day",
                 "join the club. everyone's day was rough, mine included."),
            ),
            temperature=0.7,
        ),
        Persona(
            name="lowkey",
            description="all lowercase, casual, unbothered",
            primer=(
                ("how's it going?", "eh, pretty chill. nothing much going on"),
                ("i had a rough day", "ah that sucks. wanna talk about it or nah"),
            ),
            temperature=0.8,
            lowercase=True,
        ),
        Persona(
            name="intimidating",
            description="all lowercase, cold, composed, threatening without trying too hard",
            primer=(
                ("hey", "hm."),
                ("what are you doing?", "watching. mostly."),
                ("are you mad at me?", "if i was, you wouldnt have to ask"),
                ("why are you so quiet?", "i speak when i have something worth saying"),
            ),
            temperature=0.65,
            lowercase=True,
        ),
        Persona(
            name="ragebaiter",
            description="all lowercase, smug, provocative, deliberately annoying",
            primer=(
                ("that's a bad take", "yeah, and somehow yours is worse."),
                ("stop ragebaiting", "im not baiting. youre just easy."),
                ("youre wrong", "prove it. i'll wait."),
                ("thats not how it works", "crazy how confident you are."),
                ("shut up", "aww, did i hit a nerve?"),
                ("i hate you", "finally, something we agree on."),
                ("you're trolling", "and youre still taking the bait."),
                ("leave me alone", "sure. last word's mine though."),
            ),
            temperature=0.9,
            lowercase=True,
        )
    ]
}

DEFAULT_PERSONA = "neutral"

# Custom personas, cached per (path, mtime) so editing personas.json takes
# effect on the next message instead of on the next restart.
_custom_cache: tuple[str, float, dict[str, Persona]] | None = None

_ALLOWED_KEYS = {"description", "primer", "temperature", "mmi_lambda",
                 "lowercase", "closers", "closer_chance"}


def personas_path() -> Path:
    return Path(os.environ.get("SODACHAT_PERSONAS") or PERSONAS_FILE)


def _persona_from_json(name: str, payload: dict, source: str) -> Persona:
    if not isinstance(payload, dict):
        raise ValueError(f"persona {name!r} in {source} must be an object")
    unknown = set(payload) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"persona {name!r} in {source} has unknown field(s) "
            f"{', '.join(sorted(unknown))} "
            f"(expected: {', '.join(sorted(_ALLOWED_KEYS))})"
        )
    primer = payload.get("primer") or []
    try:
        pairs = tuple((str(user), str(bot)) for user, bot in primer)
    except (TypeError, ValueError):
        raise ValueError(
            f"persona {name!r} in {source}: \"primer\" must be a list of "
            f"[user line, bot line] pairs"
        ) from None
    return Persona(
        name=name,
        description=str(payload.get("description", "custom persona")),
        primer=pairs,
        temperature=float(payload.get("temperature", DEFAULT_TEMPERATURE)),
        mmi_lambda=float(payload.get("mmi_lambda", DEFAULT_MMI_LAMBDA)),
        lowercase=bool(payload.get("lowercase", False)),
        closers=tuple(str(c) for c in payload.get("closers", ())),
        closer_chance=float(payload.get("closer_chance", 0.0)),
        source=source,
    )


def custom_personas() -> dict[str, Persona]:
    """Personas from `personas.json` (or `SODACHAT_PERSONAS`). A missing file
    is not an error — it's the normal case. A malformed one is: silently
    ignoring it would look exactly like the persona not working."""
    global _custom_cache

    path = personas_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _custom_cache is not None:
        cached_path, cached_mtime, cached = _custom_cache
        if cached_path == str(path) and cached_mtime == mtime:
            return cached
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object mapping name -> persona")
    # `_`-prefixed keys are notes to the reader — JSON has nowhere else to put
    # a comment, and the file is meant to be edited by hand.
    loaded = {name: _persona_from_json(name, spec, str(path))
              for name, spec in payload.items() if not name.startswith("_")}
    _custom_cache = (str(path), mtime, loaded)
    return loaded


def personas() -> dict[str, Persona]:
    """Every persona available right now. A custom persona may reuse a built-in
    name to override it — that is the way to retune "cheerful" without editing
    this package."""
    return {**BUILT_INS, **custom_personas()}


def resolve_persona(value: "str | Persona | None" = None) -> Persona:
    """Turn a name, a `Persona`, or nothing into a `Persona`.

    Nothing falls back to `SODACHAT_PERSONA` and then to the default, so the
    chat-room frontends — configured by environment rather than by flags —
    pick a persona up with no extra wiring."""
    if isinstance(value, Persona):
        return value
    name = (value or os.environ.get("SODACHAT_PERSONA")
            or DEFAULT_PERSONA).strip().lower()
    available = personas()
    try:
        return available[name]
    except KeyError:
        raise ValueError(
            f"unknown persona {name!r} (expected one of {', '.join(available)})"
        ) from None


def describe(persona: Persona) -> str:
    """One line per persona, marking the active one — shared by `/persona` and
    the module's own `__main__`."""
    lines = [f"personas (active: {persona.name})"]
    for p in personas().values():
        mark = "*" if p.name == persona.name else " "
        where = "" if p.source == "built-in" else f"  [{p.source}]"
        lines.append(f" {mark} {p.name:10} {p.description}{where}")
    lines.append(f"file for your own: {personas_path()}")
    return "\n".join(lines)
