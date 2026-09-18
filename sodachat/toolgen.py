"""The tool-call specialist — the model deciding to *do* something, not just say it.

    python -m sodachat.toolgen sample        # look at the training text first
    python -m sodachat.toolgen train
    python -m sodachat.toolgen try "ping me when the build finishes"

`actions.py` has always had two ways for an act to happen: the model writes a
call in double brackets (`[[react :fire:]]`) and `parse` lifts it out, or
`pick_reaction` — a table of regexes — picks a reaction because no model asked
for one. The second path is the only one that has ever fired. Nothing in this
project was trained on the call syntax: `Action.render()` exists to build such
training data and is called by three log lines. So `react` works, because a
regex table can do it, and every other tool is unreachable.

This is the missing half: a **generator specialist** (see
`expert.scaffold_specialist`) that reads a message and writes the calls it
warrants, if any. The shape is codegen's — one fresh FFN expert per block on a
frozen trunk, no head, trained on next-token loss through the shared LM head —
so the chat model, the reader, the games and the other specialists cannot be
disturbed by it: their weights never receive a gradient. It ships as
`models/specialist-tools.pt` and `ExpertLM` grafts it on at startup.

The training document is one message and its verdict:

    <|acts|> ping me when the render is done
    acts: [[ping]]

    <|acts|> what time is it there
    acts: [[none]]

`[[none]]` is explicit rather than an empty line because "do nothing" is the
answer most of the time and a model needs something concrete to emit for it.
It is not a tool, so `actions.resolve` drops it on the way back out and the
turn ends with no acts — the same result as silence, arrived at deliberately.

Where the labels come from
--------------------------
Two sources, and the split is the honest part of this module.

  * **Reactions are labelled by `pick_reaction`.** Running the existing trigger
    table over real messages — your `data/` exports and SODA turns — is free
    ground truth, and it is *already the behaviour in production*, so a
    specialist that learns it has lost nothing. What it buys is generalization:
    the table fires on "good bot" and misses "yo that's actually sick", and a
    model trained on the table's verdicts over tens of thousands of real
    messages can learn the shape rather than the literals.

  * **Everything else is synthetic, and that is a real weakness.** No corpus
    exists of people asking a bot to pin a message. The templates below are a
    few dozen skeletons crossed with slot fillers, in the same spirit as
    route.py's synthesized classes — which its docstring calls "the honest weak
    point". A model trained on templates learns the templates; it will handle
    "ping me when X" and be shakier on phrasings nobody thought to write down.

The third source matters as much as either: **hard negatives**. "stop pinging
me", "i already pinged him", "my ping is 300ms" are messages about pinging that
must not produce one. A notification cannot be unsent, so the cost of a false
positive here is not symmetric with a missed call, and the corpus is weighted
accordingly — mostly `[[none]]`, with the near-misses written in on purpose.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .actions import MAX_PER_TURN, TOOLS, Action, parse, pick_reaction
from .blocks import make_amp, pick_device
from .expert import (DEFAULT_PATH as EXPERT_PATH, ExpertLM, TEXT, save_specialist,
                     scaffold_specialist, specialist_param_groups)
from .localdata import DATA_DIR

_MODELS = Path(__file__).resolve().parent.parent / "models"

NAME = "tools"
DEFAULT_PATH = _MODELS / f"specialist-{NAME}.pt"

# The specialist's private marker. A shared marker would collide with another
# specialist's trained embedding row at attach time (see `attach_specialist`),
# and this one is a prompt boundary rather than a classifier position, so it
# wants to mean exactly one thing.
ACTS = "<|acts|>"
NONE = "[[none]]"
PROMPT = "acts:"
BLOCK = 256

# What the corpus is made of. Sampling to explicit shares rather than taking
# whatever the sources happen to yield, because what they happen to yield is
# badly skewed: run `pick_reaction` over SODA and 34% of turns come back with a
# reaction, almost all of them from the `\?$` trigger — SODA is mostly
# questions. Train on that and the bot reacts to a third of everything said to
# it, which is the opposite of what the table's own docstring asks for ("None
# is the common answer and should stay that way").
SHARES = {"none": 0.58, "react": 0.14, "ping": 0.10, "other": 0.18}

# Above this, a message is prose rather than a chat line, and the trigger table
# was not built for it: its patterns are unanchored, so `\bnice\b` fires on
# "have a nice day" and `\?$` on any paragraph that ends in a question. Long
# messages are labelled quiet regardless of what the table says — a deliberate
# divergence from the running behaviour, and the one place this corpus tries to
# be better than its teacher rather than to copy it.
REACT_MAX_CHARS = 140


# --------------------------------------------------------------- the corpus


def doc(message: str, actions: "tuple[Action, ...]") -> str:
    """One training document: a message, and the calls it earns."""
    calls = " ".join(a.render() for a in actions) if actions else NONE
    return f"{ACTS} {message.strip()}\n{PROMPT} {calls}\n"


# Slot fillers. Deliberately mundane: the point of a template corpus is the
# shape of the request, and florid examples teach a vocabulary nobody uses.
_WHEN = ("the build finishes", "it's done", "the run ends", "you're finished",
         "training completes", "the deploy lands", "that finishes", "it works",
         "you figure it out", "the tests pass", "it's ready", "you get there")
# Two kinds of "who", and the difference is the whole lesson for `[[ping]]`'s
# argument. A *name* can be looked up in the server, so it belongs in the call:
# "tag the mods" -> `[[ping the mods]]`. A *pronoun* refers to the conversation
# rather than the member list — there is nobody in any guild called "him" — so
# it belongs nowhere, and the call goes out bare for the frontend to resolve
# against whoever the turn is about. Teaching both under one filler would teach
# the model to put "him" in the argument and the ping would find nobody.
# The named pool is generated rather than listed, and it is deliberately huge.
# A first pass used fourteen names and the specialist learned *them*: asked to
# "shout at the moderators", it answered `[[ping the mod team]]` — a filler it
# had memorized, not the name in front of it. Putting a name in the argument is
# a copying task, and a model can only be forced to copy by making memorizing
# useless. Thousands of distinct names, most seen once, do that.
_FIRST = ("priya", "hendrik", "wei", "amara", "tomas", "lena", "kofi", "sana",
          "dmitri", "yuki", "rosa", "olu", "ingrid", "hassan", "mei", "ravi",
          "bob", "alice", "dave", "sam", "jamie", "chris", "riley", "noor",
          "ana", "luca", "freya", "mateo", "zane", "iris", "otto", "nadia",
          "pablo", "greta", "hugo", "leila", "marcus", "elena", "jonas", "tara",
          "quinn", "vik", "bea", "soren", "maya", "felix", "ines", "aki")
_HANDLE_SUFFIX = ("", "", "", "_", "42", "88", "2000", "_dev", "xx", "99",
                  "_irl", "1234", "_tv", "07")
_ROLE_QUALIFIER = ("", "", "the ", "the ", "our ", "design ", "backend ",
                   "on-call ", "night ", "senior ", "core ", "new ", "server ")
_ROLE_NOUN = ("mods", "admins", "devs", "reviewers", "organisers", "team",
              "crew", "mod team", "support", "testers", "regulars", "staff",
              "maintainers", "artists", "writers", "players", "helpers")


def _name_pool() -> tuple[str, ...]:
    """Every name the templates may ask for: people, handles and roles."""
    people = list(_FIRST)
    handles = [f"{n}{suffix}" for n in _FIRST for suffix in _HANDLE_SUFFIX if suffix]
    roles = [f"{q}{noun}" for q in _ROLE_QUALIFIER for noun in _ROLE_NOUN]
    return tuple(dict.fromkeys(people + handles + roles))


_WHO_NAMED = _name_pool()
_WHO_PRONOUN = ("him", "her", "them", "that guy", "the others",
                "whoever wrote this", "whoever said that")
_WHO = _WHO_NAMED + _WHO_PRONOUN
_THING = ("this", "that", "this message", "that one", "it")
_NAME = ("standup", "bug hunt", "the snake thread", "planning", "random",
         "deploy chat", "help", "oracle talk")

# --- requests that want a ping, with nobody to name ---------------------
# The speaker wants reaching, or the target is a pronoun the member list cannot
# answer. Either way the call goes out bare and the frontend resolves it.
_PING_SELF = (
    "ping me when {when}", "ping me once {when}", "@ me when {when}",
    "let me know when {when}", "tell me when {when}", "notify me when {when}",
    "give me a shout when {when}", "ping me if {when}", "hit me up when {when}",
    "ping me", "tag me when {when}", "@ me", "buzz me when {when}",
    "can you ping {pronoun}", "ping {pronoun}", "tag {pronoun}",
    "get {pronoun}'s attention", "summon {pronoun}", "ping {pronoun} for me",
)

# --- requests that call the whole room ----------------------------------
# Whether these are *allowed* is not the model's business: the frontend gives
# the bot the asker's own reach and no more (`discord_bot._ping_block`), so a
# request from someone who cannot call the room is refused there. The model's
# job is only to notice that calling the room is what was asked for — and if it
# could not ask, the permission the frontend so carefully checks would never
# come up.
_PING_ALL = (
    "ping everyone about {when}", "let everyone know {when}",
    "tell everyone {when}", "@ everyone about this", "ping everyone",
    "let everyone in the server know", "tell everyone in this channel",
    "@ here", "get everyone's attention", "call everyone in",
    "ping everyone in the room", "announce this to everyone",
    "@ everyone please", "everyone needs to see this", "@ here please",
)

# The same rule the other tools follow, for the same reason: the frontend
# checks that the argument was copied out of the message
# (`discord_bot._copied_from`), so a template whose call says "everyone" while
# its message says "the whole server" would be quietly demoted to a ping of
# whoever asked. Phrasings that mean everyone without saying it are therefore
# not here — not because they are unnatural, but because this path cannot carry
# them honestly.
for _t in _PING_ALL:
    assert "everyone" in _t or "here" in _t, (
        f"{_t!r}: the word the call copies must be in the message")

# --- requests that name who to reach ------------------------------------
# The name goes in the argument, verbatim as the message gave it: the frontend
# matches it against the guild's roles and then its members, so "the mods"
# wants to arrive as "the mods" rather than as anything cleverer.
_PING_NAMED = (
    "can you ping {named}", "ping {named}", "tag {named}", "can you tag {named}",
    "get {named}'s attention", "summon {named}", "call {named} in here",
    "ping {named} for me", "would you tag {named}", "mention {named}",
    "@ {named} please", "let {named} know", "tell {named} about this",
    "someone should ping {named}", "give {named} a shout",
)

# --- the near misses, which must stay quiet -----------------------------
# Every one of these contains the vocabulary of a ping and asks for nothing.
# Without them a model that has seen "ping" a thousand times in the positive
# class learns the word, not the request.
_NOT_PING = (
    "stop pinging me", "don't ping me", "quit tagging me", "no need to ping",
    "i already pinged {who}", "i pinged {who} earlier", "{who} pinged me",
    "my ping is 300ms", "the ping is awful today", "ping is really high rn",
    "why does everyone keep pinging me", "he pinged the whole server",
    "sorry for the ping", "sorry for the late ping", "that ping woke me up",
    "please never @ everyone again", "who pinged me", "that was a mass ping",
    "i hate being tagged", "you got tagged in that thread",
    "don't tag {who} for this", "stop @ing people", "no pings please",
    # The whole-room negations. These matter more than the rest of the list:
    # the permission check downstream stops someone who *cannot* call the room,
    # which is exactly the wrong half of the problem here — a moderator saying
    # "don't tell everyone about it" is someone who can, asking not to.
    "dont tell everyone about it", "don't tell everyone", "don't ping everyone",
    "no need to tell everyone", "please don't @ everyone", "everyone is asleep",
    "not everyone needs to see this", "don't announce this to everyone",
    "keep this from everyone", "stop pinging everyone", "everyone already knows",
    "i don't want to bother everyone", "everyone saw it already",
    "let's not @ here for this", "no @ here please", "don't wake everyone up",
    "everyone left already", "is everyone still here",
)

# --- the other tools, in the phrasing a request actually arrives in ------
_REQUESTS: dict[str, tuple[str, ...]] = {
    "pin": ("pin {thing}", "can you pin {thing}", "pin this please",
            "worth pinning", "pin that for later", "someone pin {thing}"),
    "unpin": ("unpin {thing}", "can you unpin {thing}", "take that pin off",
              "unpin that please"),
    "thread": ("make a thread for {name}", "start a thread called {name}",
               "thread this as {name}", "can we get a {name} thread",
               "open a thread named {name}"),
    "nick": ("change your name to {name}", "call yourself {name}",
             "your nickname should be {name}", "rename yourself {name}"),
    "rename": ("rename {who} to {name}", "call {who} {name}",
               "change {who}'s nickname to {name}"),
    "role": ("give {who} the {name} role", "add {name} to {who}",
             "can {who} get the {name} role"),
    "unrole": ("take the {name} role off {who}", "remove {name} from {who}"),
    "topic": ("set the topic to {name}", "change the channel topic to {name}",
              "make the topic {name}"),
    "slowmode": ("turn on slowmode", "set slowmode to 10", "slow this channel down",
                 "turn slowmode off"),
    "status": ("set your status to {name}", "say you're playing {name}",
               "change what you're playing to {name}"),
    "delete": ("delete that", "take that back", "unsay that", "remove your last message"),
}

# What each of those should produce. The argument is the part the frontend
# needs; `[[pin]]` takes none, `[[thread standup]]` takes a name.
_ARG_FROM = {"thread": "{name}", "nick": "{name}", "rename": "{name}",
             "role": "{name}", "unrole": "{name}", "topic": "{name}",
             "status": "{name}", "slowmode": "10"}

# An argument the message does not contain is an argument the model has to
# invent, and a corpus that rewards inventing one gets a specialist that names
# every thread after whatever it saw in training. Enforced rather than
# remembered: a template for an arg-taking tool must carry that tool's slot.
for _tool, _templates in _REQUESTS.items():
    _slot = _ARG_FROM.get(_tool, "")
    if _slot.startswith("{"):
        assert all(_slot in _t for _t in _templates), (
            f"{_tool}: every template must contain {_slot} — otherwise the "
            f"argument is invented")


def _fill(rng: np.random.Generator, template: str) -> tuple[str, dict]:
    """A template with its slots filled, plus what went into them."""
    slots = {"when": rng.choice(_WHEN), "who": rng.choice(_WHO),
             "named": rng.choice(_WHO_NAMED), "pronoun": rng.choice(_WHO_PRONOUN),
             "thing": rng.choice(_THING), "name": rng.choice(_NAME)}
    return template.format(**slots), slots


def _synthetic(rng: np.random.Generator,
               per_template: int = 200) -> tuple[list[str], list[str], list[str]]:
    """Template-built examples, as three pools: ping requests, other tool
    requests, and the ping-shaped messages that must stay quiet.

    Kept apart so `corpus` can sample each to its target share — the pools have
    very different template counts (22 ping phrasings against 46 across eleven
    other tools), and mixing them first would let that accident set the blend.
    """
    ping, other, negative = [], [], []
    for template in _PING_SELF:
        for _ in range(per_template):
            text, _ = _fill(rng, template)
            ping.append(doc(text, (Action("ping"),)))
    for template in _PING_ALL:
        for _ in range(per_template):
            text, _ = _fill(rng, template)
            # The argument is the word the message used, so `_copied_from`
            # can check it the same way it checks a name.
            everyone = "here" if "@ here" in text else "everyone"
            ping.append(doc(text, (Action("ping", everyone),)))
    for template in _PING_NAMED:
        for _ in range(per_template):
            text, slots = _fill(rng, template)
            ping.append(doc(text, (Action("ping", slots["named"]),)))
    for template in _NOT_PING:
        for _ in range(per_template):
            text, _ = _fill(rng, template)
            negative.append(doc(text, ()))
    for tool, templates in _REQUESTS.items():
        for template in templates:
            for _ in range(per_template):
                text, slots = _fill(rng, template)
                arg = _ARG_FROM.get(tool, "")
                if arg:
                    arg = arg.format(**slots)
                if tool == "slowmode":
                    # The one tool whose argument is a number rather than a
                    # phrase: read it out of the message when it is there, and
                    # otherwise use the default that "turn on slowmode" means.
                    found = re.search(r"\b(\d+)\b", text)
                    arg = "0" if "off" in text else (found.group(1) if found else "10")
                other.append(doc(text, (Action(tool, arg),)))
    return ping, other, negative


_URL_RE = re.compile(r"https?://\S+")


def _usable(message: str) -> bool:
    """Whether a real chat line is worth a training document. Very short lines
    carry no intent to read, and a line that is mostly a URL is a link drop."""
    text = message.strip()
    if not 3 <= len(text) <= 200:
        return False
    return not _URL_RE.match(text)


def _labelled(messages: "list[str]", persona: str = "neutral") -> list[str]:
    """Real messages, labelled with what the trigger table would already do.

    This is the supervised half, and it is free: `pick_reaction` is the running
    behaviour, so its verdicts are the labels the bot already acts on. Most
    come back None, which is what makes these the negative class.
    """
    docs = []
    for message in messages:
        emoji = (pick_reaction(message, None, persona)
                 if len(message) <= REACT_MAX_CHARS else None)
        acts = (Action("react", emoji),) if emoji else ()
        docs.append(doc(message, acts))
    return docs


def _chat_messages(rng: np.random.Generator, soda_limit: int = 50000,
                   log=print) -> list[str]:
    """Real chat lines: your `data/` exports first, then SODA turns for breadth."""
    messages: list[str] = []
    for path in sorted(Path(DATA_DIR / "text").glob("*.txt")):
        for line in path.read_text(errors="ignore").splitlines():
            # The exports are already in A:/B: form; the tag is transport, not
            # content, and the specialist reads one message at a time.
            line = re.sub(r"^[AB]:\s*", "", line.strip())
            if _usable(line):
                messages.append(line)
    log(f"  {len(messages):,} lines from {DATA_DIR.name}/text/")
    try:
        from .data import soda_dialogues

        n = 0
        for dialogue in soda_dialogues("train"):
            for utterance in dialogue:
                if _usable(utterance):
                    messages.append(utterance)
                    n += 1
            if n >= soda_limit:
                break
        log(f"  {n:,} turns from SODA")
    except Exception as e:  # offline, or the parquet is not cached
        log(f"  (SODA unavailable: {e}) — templates and {DATA_DIR.name}/ only")
    return messages


def _take(rng: np.random.Generator, pool: list[str], n: int) -> list[str]:
    """`n` documents from `pool`, sampling with replacement only if the pool is
    short of it — a template pool repeating is fine, real messages repeating is
    a quieter way of overfitting."""
    if not pool:
        return []
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    reps = -(-n // len(pool))
    return (pool * reps)[:n]


def corpus(rng: np.random.Generator, holdout: float = 0.06, size: int | None = None,
           log=print) -> tuple[list[str], list[str]]:
    """The whole training corpus, sampled to `SHARES`, shuffled and split.

    The mix carries the lesson twice over. The synthetic requests teach the
    syntax and the rare tools; the labelled chat teaches when to stay quiet;
    and `_NOT_PING` — messages *about* pinging that ask for none — is folded in
    with the quiet class, because a model that has seen the word "ping" a
    thousand times in the positive class otherwise learns the word rather than
    the request.

    `size` defaults to whatever the quiet supply can fill at its target share,
    since real messages are the one ingredient that cannot be manufactured.
    """
    log("building the corpus...")
    ping, other, hard = _synthetic(rng)
    labelled = _labelled(_chat_messages(rng, log=log))
    quiet = [d for d in labelled if d.rstrip().endswith(NONE)]
    react = [d for d in labelled if not d.rstrip().endswith(NONE)]

    # The quiet class is part real messages, part near-miss templates: both are
    # "do nothing", and the near misses are the ones that are hard to get right.
    if size is None:
        size = int(len(quiet) / max(SHARES["none"] * 0.8, 1e-6))
    n_none = int(size * SHARES["none"])
    docs = (_take(rng, quiet, int(n_none * 0.8))
            + _take(rng, hard, n_none - int(n_none * 0.8))
            + _take(rng, react, int(size * SHARES["react"]))
            + _take(rng, ping, int(size * SHARES["ping"]))
            + _take(rng, other, int(size * SHARES["other"])))
    rng.shuffle(docs)
    log(f"  {len(docs):,} documents: " + ", ".join(
        f"{k} {v:.0%}" for k, v in SHARES.items())
        + f" (react labels bounded to messages <= {REACT_MAX_CHARS} chars)")
    cut = max(1, int(len(docs) * holdout))
    return docs[cut:], docs[:cut]


# -------------------------------------------------------------- training


def _encode_stream(tok, docs: list[str]) -> np.ndarray:
    ids: list[int] = []
    for enc in tok.encode_batch(docs):
        ids.extend(enc)
    return np.asarray(ids, dtype=np.int32)


def _batch(stream: np.ndarray, block: int, bs: int, task: int, device):
    ix = np.random.randint(0, max(len(stream) - block - 1, 1), size=bs)
    window = np.stack([stream[i:i + block + 1] for i in ix]).astype(np.int64)
    w = torch.from_numpy(window).to(device)
    t = torch.full((bs,), task, dtype=torch.long, device=device)
    return w[:, :-1].contiguous(), w[:, 1:].contiguous(), t


@torch.no_grad()
def _validate(model, stream, block, task, device, iters=40, bs=16) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y, t = _batch(stream, block, bs, task, device)
        logits = model(x, t)[0]
        losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def train(base=EXPERT_PATH, out=DEFAULT_PATH, steps=600, batch_size=12,
          lr=3e-4, block_size=BLOCK, device=None, seed=0, eval_every=50,
          log=print) -> Path:
    """Train the tool-call generator on top of a *frozen* expert model.

    Gradients reach only the new per-block FFN expert and the `<|acts|>` row of
    the embedding. Chat, reading, play and every other specialist are untouched
    — which is the reason to build this as a specialist rather than fine-tuning
    the chat model: a tool corpus is small, synthetic and repetitive, and that
    is exactly the diet that would overwrite a 77M chat model's voice.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, tok, task = scaffold_specialist(base, name=NAME, special_tokens=[ACTS],
                                           n_labels=None, seed_from=TEXT,
                                           device=device)
    block = min(block_size, model.cfg.block_size)
    per_block = sum(p.numel() for p in model.blocks[0].ffn.experts[task].parameters())
    log(f"specialist '{NAME}' (generator) on {Path(base).name}: expert slot {task} | "
        f"{per_block * model.cfg.n_layer / 1e6:.1f}M trainable of "
        f"{model.num_params() / 1e6:.1f}M (shared trunk + LM head frozen) | "
        f"seeded from TEXT | device {device}")

    train_docs, val_docs = corpus(rng, log=log)
    train_stream = _encode_stream(tok, train_docs)
    val_stream = _encode_stream(tok, val_docs)
    seen = steps * batch_size * block
    log(f"stream: {len(train_stream) / 1e6:.2f}M train / {len(val_stream) / 1e3:.0f}k "
        f"val tokens (block {block}) | schedule: {steps:,} x {batch_size} x {block} "
        f"= {seen / 1e6:.0f}M tokens (~{seen / max(len(train_stream), 1):.1f} epochs)")

    opt = torch.optim.AdamW(specialist_param_groups(model), lr=lr, betas=(0.9, 0.95))
    warmup = min(100, max(steps // 10, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1),
                                                    eta_min=lr * 0.1)],
        milestones=[warmup])

    amp = make_amp(device)
    model.train()
    started, best = time.time(), float("inf")
    for step in range(1, steps + 1):
        x, y, t = _batch(train_stream, block, batch_size, task, device)
        with amp.autocast():
            logits = model(x, t)[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % eval_every == 0 or step == steps:
            vloss = _validate(model, val_stream, block, task, device)
            mark = ""
            if vloss < best:
                best, mark = vloss, " <- saved"
                save_specialist(out, model, tok, name=NAME, task=task, labels=[],
                                special_tokens=[ACTS], kind="generate", steps=step,
                                val_acc=vloss, meta={"block": block,
                                                     "tools": sorted(TOOLS),
                                                     "shares": SHARES,
                                                     "val_loss": vloss})
            el = time.time() - started
            log(f"step {step:>5}/{steps} | train loss {loss.item():.3f} | "
                f"val loss {vloss:.3f} (ppl {np.exp(vloss):.1f}) | {el / 60:.0f}m, "
                f"eta {el / step * (steps - step) / 60:.0f}m{mark}")

    log(f"done — best val loss {best:.3f}, saved to {out}")
    return out


# ------------------------------------------------------------- inference


def available(lm: ExpertLM) -> bool:
    """Whether this specialist is attached to `lm`."""
    return NAME in getattr(lm, "specialists", {})


def suggest(lm: ExpertLM, message: str, temperature: float = 0.05,
            max_new_tokens: int = 24) -> tuple[Action, ...]:
    """The acts a message warrants, as the specialist reads it.

    Sampled nearly greedily (temperature 0.05) on purpose. This is a decision,
    not prose: there is nothing to be gained from the model being surprising
    about whether to notify someone, and something to lose. Measured over ten
    samples each, 0.2 answered "tag the mods" with `[[ping the mods]]` five
    times and a bare `[[ping]]` the other five — the same message pinging a
    different person depending on the roll. At 0.05 and below it is 10/10, and
    the calls it is confident about (`[[ping priya]]`) never wavered at any
    setting, so the cold sampling costs nothing it was getting right.

    Unknown names — `[[none]]` above all — are dropped by `actions.parse`, so
    "nothing to do" needs no special case here.
    """
    if not available(lm):
        return ()
    task = lm.specialists[NAME]["task"]
    prompt = f"{ACTS} {message.strip()}\n{PROMPT}"
    device = lm.model.tok_emb.weight.device
    idx = torch.tensor([lm.tok.encode(prompt)], dtype=torch.long, device=device)
    # `stop_tokens` are ids, not strings: the line the calls sit on ends at the
    # newline, and everything after it would be the model carrying on inventing
    # the next message in the transcript.
    newline = lm.tok.encode("\n")
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature=temperature,
                                 top_k=20, stop_tokens=newline)
    ids = out[0].tolist() if hasattr(out, "shape") else out[0]
    text = lm.tok.decode(ids[idx.shape[1]:])  # just the continuation
    _, actions = parse(text.split("\n")[0])
    return actions[:MAX_PER_TURN]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="The tool-call specialist.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="print training documents and stop")
    s.add_argument("-n", type=int, default=25)
    s.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--batch-size", type=int, default=12)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--device", default=None)
    t.add_argument("--out", type=Path, default=DEFAULT_PATH)
    t.add_argument("--base", type=Path, default=EXPERT_PATH)
    y = sub.add_parser("try", help="what the trained specialist would do")
    y.add_argument("message", nargs="+")
    a = p.parse_args()

    if a.cmd == "sample":
        rng = np.random.default_rng(a.seed)
        train_docs, val_docs = corpus(rng)
        print(f"\n--- {a.n} of {len(train_docs):,} training documents ---\n")
        for d in train_docs[:a.n]:
            print(d.rstrip() + "\n")
    elif a.cmd == "train":
        train(base=a.base, out=a.out, steps=a.steps, batch_size=a.batch_size,
              lr=a.lr, device=a.device, eval_every=a.eval_every)
    else:
        lm = ExpertLM(EXPERT_PATH)
        message = " ".join(a.message)
        acts = suggest(lm, message)
        print(f"{message!r} -> " + (" ".join(x.render() for x in acts) or "(nothing)"))


if __name__ == "__main__":
    main()
