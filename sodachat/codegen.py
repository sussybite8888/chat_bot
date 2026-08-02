"""Code *generation*, added to the expert model as a second code specialist.

Where `code.py` is a **classifier** — it names a snippet's language off a class
head and only *borrows* the frozen LM head for a weak, untrained completion —
this module is a **generator**. It trains a fresh per-block FFN expert on a
next-token language-modelling objective over real code, so the expert learns to
shape the frozen trunk's features toward code continuations. The shared trunk
and the tied LM head stay frozen (chat, reading, play, and the code classifier
are untouched by construction); only the new expert's ~17M weights move.

    <|code classifier|>  code.py       -> language label (reliable)
    <|code generator|>   codegen.py    -> next-token completion (this file)

Design choices that make generation work as well as a ~14M frozen trunk allows:

  * The expert is seeded from the **TEXT** expert, not GAME — TEXT already
    decodes fluent next-token text through the LM head, so it is the right warm
    start for generating code (vs. code.py's classifier, which seeds from GAME
    because dense glyph grids resemble source the *most* for a feature head).
  * **No new tokens and no head.** Routing is by task id, not by a tag token, so
    the generator needs no vocabulary of its own — it reads and writes ordinary
    text tokens and stops at the existing `<|end|>`. That also sidesteps the
    special-token id collisions two specialists trained from one base can hit.
  * Training packs cleaned function bodies into one stream, snippets separated by
    `<|end|>`, and samples fixed-width windows — the standard LM diet, but every
    token routed to the code-generator expert.
  * **Each snippet carries a language header** (`// javascript`, `# python`) —
    an ordinary comment line, so still no new tokens. Six languages in one
    undifferentiated stream left `function foo(` as likely to continue in PHP or
    Java as in JavaScript; the header makes the language something generation can
    ask for (`generate(..., lang="javascript")`) instead of guess.
  * **Machine-generated source is dropped** from both corpora — minified,
    bundled, transpiled and obfuscated code is valid and licence-clean, so
    nothing else rejects it, but it is precisely what teaches a model to write
    mangled names and helper-call soup (see `_machine_generated`).

Quality is bounded by the frozen trunk (a ~14M model on a chat/game diet) and a
BPE vocabulary built for dialogue, so treat this as a small, code-flavoured
autocomplete — markedly better than code.py's untrained-expert completion, not a
real code model. Saved as `specialist-codegen.pt`; `ExpertLM` grafts it on at
startup alongside the classifier and the vision specialist.

Corpus: [CodeSearchNet](https://huggingface.co/datasets/code_search_net), the
same six languages code.py classifies (python, java, javascript, php, ruby, go).

    python -m sodachat.codegen train     # needs models/expert.pt (train it first)
    python -m sodachat.codegen eval       # held-out perplexity
    python -m sodachat.codegen complete --file snippet.py
    python -m sodachat.codegen demo       # a few prompts + continuations
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import make_amp, pick_device
from .code import LANGS, _hf_snippets
from .expert import (
    _MODELS,
    DEFAULT_PATH as EXPERT_PATH,
    END,
    ExpertLM,
    TEXT,
    save_specialist,
    scaffold_specialist,
    specialist_param_groups,
)

NAME = "codegen"
DEFAULT_PATH = _MODELS / "specialist-codegen.pt"
# A permissively-licensed local code corpus (built off-box from MIT/BSD/Apache
# projects on disk by tools/build_codegen_corpus.py — one file per NUL-separated
# chunk, each chunk `<path>\n<source>`), mixed in with CodeSearchNet when
# present. Absent -> CodeSearchNet only.
LOCAL_PATH = _MODELS / "codegen-local.txt"
BLOCK = 512   # code snippets are short; a 512-token window spans a few functions

# Every snippet enters the stream under a one-line header written in its own
# language's comment syntax:
#
#     // javascript
#     function debounce(fn, ms) { ... }
#     <|end|>
#
# Without it the six languages are one undifferentiated stream, and a `function
# foo(` prompt is as likely to be continued as PHP or Java as JavaScript. The
# header costs one line, needs no new tokens (it is ordinary source text, which
# is the whole point — see the module docstring), and gives generation a knob:
# `generate(..., lang="javascript")` prepends the same line it was trained on.
COMMENT = {"python": "#", "ruby": "#", "javascript": "//", "typescript": "//",
           "java": "//", "php": "//", "go": "//", "c": "//", "cpp": "//"}
# Local-corpus extensions, mapped onto the languages this specialist writes.
# Anything not listed here (.c/.h/.cpp/.rs/.lua ...) is dropped: C was 68% of
# the local blob by bytes, and untagged C in a six-language stream is where JS
# completions picked up `#endif` and template syntax.
LOCAL_EXTS = {".py": "python", ".js": "javascript", ".mjs": "javascript",
              ".jsx": "javascript", ".ts": "typescript", ".go": "go",
              ".rb": "ruby", ".php": "php", ".java": "java"}

# Machine-generated JavaScript — minifier, bundler, transpiler and obfuscator
# output. It is syntactically valid and licence-clean, so nothing upstream
# rejects it, but training on it teaches the model to *write* mangled names and
# helper-call soup, which is what obfuscated output looks like.
_MACHINE = [
    re.compile(r"_0x[0-9a-f]{3,}"),                       # obfuscator.io names
    re.compile(r"(\\x[0-9a-fA-F]{2}){3,}"),               # hex-escaped blobs
    re.compile(r"(\\u[0-9a-fA-F]{4}){3,}"),               # unicode-escaped blobs
    re.compile(r"\+!\+\[\]|\[\]\[\(!\[\]"),               # jsfuck
    re.compile(r"\beval\s*\(\s*(function|atob|unescape|String\.fromCharCode)"),
    re.compile(r"String\.fromCharCode\(\s*(?:0x[0-9a-f]+|\d+)\s*"
               r"(?:,\s*(?:0x[0-9a-f]+|\d+)\s*){4,}\)"),
    re.compile(r"_WEBPACK_IMPORTED_MODULE|__webpack_"),   # bundler output
    re.compile(r"_interopRequireDefault|\b_\w+2\.default\b"),   # babel interop
    re.compile(r"\b_(classCallCheck|createClass|possibleConstructorReturn|"
               r"slicedToArray|toConsumableArray|objectSpread|inherits)\b"),
]
_IDENT = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")
_KEYWORDS = frozenset(
    "var let const function return if else for while new this typeof in of class "
    "extends super null true false undefined try catch throw switch case break "
    "continue do delete void instanceof yield await async static get set import "
    "export from default require module exports def self end elsif nil puts func "
    "package type struct interface map range go defer chan public private static "
    "int void string bool echo".split())


# Shape heuristics (dense lines, mangled names) are for the languages that are
# actually shipped minified. Go names its receivers `p` and its buffers `b`, and
# Java is generic-heavy, so "mostly short identifiers" is idiom there, not
# mangling — applying it to them threw away 41% of the Go corpus.
_MINIFIED_LANGS = frozenset({"javascript", "typescript"})


def _machine_generated(code: str, lang: str | None = None) -> bool:
    """True for minified/bundled/transpiled/obfuscated source. Fingerprints and
    absurd line lengths count against any language; the shape heuristics (dense
    throughout, mostly 1-2 character identifiers) only against the ones that get
    minified in practice."""
    if any(rx.search(code) for rx in _MACHINE):
        return True
    lines = [ln for ln in code.split("\n") if ln.strip()]
    if not lines:
        return True
    if max(len(ln) for ln in lines) > 400:          # a bundled one-liner
        return True
    if lang is not None and lang not in _MINIFIED_LANGS:
        return False
    if sum(len(ln) for ln in lines) / len(lines) > 120:   # dense throughout
        return True
    names = [i for i in _IDENT.findall(code) if i not in _KEYWORDS]
    return bool(names) and sum(1 for i in names if len(i) <= 2) / len(names) > 0.5


def _tag(lang: str, code: str) -> str:
    """One snippet under its language header, ready for the stream."""
    return f"{COMMENT.get(lang, '//')} {lang}\n{code.strip()}"


def _csn_texts(split: str, per_lang: int, rng: np.random.Generator,
               log=None) -> list[str]:
    """CodeSearchNet function bodies across all six languages, each under its
    language header and with machine-generated source dropped. `split` is
    "train" or "test" (held-out)."""
    texts: list[str] = []
    for lang in LANGS:
        raw = _hf_snippets(lang, split, per_lang, rng)
        kept = [_tag(lang, s) for s in raw if not _machine_generated(s, lang)]
        if log and len(kept) < len(raw):
            log(f"  {lang}: dropped {len(raw) - len(kept)} of {len(raw)} "
                f"machine-generated snippets")
        texts += kept
    return texts


def _local_chunks(path, rng: np.random.Generator, holdout: float = 0.08, log=None):
    """The local permissive corpus split into (train, val) file lists, each file
    tagged with the language its extension names and machine-generated source
    dropped. Empty lists if the blob isn't present, so training falls back to
    CodeSearchNet.

    The builder (tools/build_codegen_corpus.py) writes `<path>\\n<source>` per
    chunk so the language is recoverable here; blobs from the older, pathless
    builder are skipped rather than fed in untagged."""
    if not Path(path).exists():
        return [], []
    blob = Path(path).read_text(encoding="utf-8", errors="ignore")
    tagged, untagged = [], 0
    for chunk in blob.split("\0"):
        if not chunk.strip():
            continue
        head, _, body = chunk.partition("\n")
        lang = LOCAL_EXTS.get(Path(head.strip()).suffix.lower()) if head else None
        if lang is None or not body.strip():
            untagged += 1
            continue
        if _machine_generated(body, lang):
            continue
        tagged.append(_tag(lang, body))
    if log and untagged:
        log(f"  skipped {untagged} local chunks with no usable language header "
            f"(rebuild with tools/build_codegen_corpus.py)")
    rng.shuffle(tagged)
    n_val = max(1, int(len(tagged) * holdout)) if tagged else 0
    return tagged[n_val:], tagged[:n_val]


def _encode_stream(tok, texts: list[str], rng: np.random.Generator) -> np.ndarray:
    """Pack a list of code chunks into one token stream, each terminated with
    `<|end|>` so the model learns where a unit of code stops."""
    end = tok.token_id(END)
    order = list(texts)
    rng.shuffle(order)
    ids: list[int] = []
    for enc in tok.encode_batch(order):
        ids.extend(enc)
        ids.append(end)
    return np.asarray(ids, dtype=np.int32)


def _batch(stream: np.ndarray, block: int, bs: int, task: int, device):
    """Sample `bs` random `block`-wide windows and their next-token targets, all
    routed to the code-generator expert."""
    ix = np.random.randint(0, len(stream) - block - 1, size=bs)
    x = np.stack([stream[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([stream[i + 1:i + block + 1] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    t = torch.full_like(x, task)
    if device == "cuda":
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True), t.to(device))
    return x.to(device), y.to(device), t.to(device)


@torch.no_grad()
def _validate(model, stream, block, task, device, iters=40, bs=16) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(iters):
        x, y, _ = _batch(stream, block, bs, task, device)
        logits = model(x, y.new_full(x.shape, task))[0]
        losses.append(F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    if was_training:
        model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


# ------------------------------------------------------------------- training


def train(base=EXPERT_PATH, out=DEFAULT_PATH, steps=3000, batch_size=24, lr=3e-4,
          block_size=BLOCK, per_lang=4000, device=None, seed=0, eval_every=250,
          log=print) -> Path:
    """Train the code generator on top of a *frozen* expert model: gradients
    reach only the new per-block FFN expert (seeded from TEXT). Chat, reading,
    play, and the code classifier are untouched — their weights never receive a
    gradient. The generator has no head and no tokens of its own; it is trained
    on next-token loss through the shared, frozen (tied) LM head, and generates
    with `generate_text` routed to its expert."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, tok, task = scaffold_specialist(base, name=NAME, special_tokens=[],
                                           n_labels=None, seed_from=TEXT, device=device)
    block = min(block_size, model.cfg.block_size)
    per_block = sum(p.numel() for p in model.blocks[0].ffn.experts[task].parameters())
    trainable = per_block * model.cfg.n_layer
    log(f"specialist '{NAME}' (generator) on {Path(base).name}: expert slot {task} | "
        f"{trainable / 1e6:.1f}M trainable of {model.num_params() / 1e6:.1f}M "
        f"(shared trunk + LM head frozen) | seeded from TEXT | device {device}")

    log(f"loading CodeSearchNet ({', '.join(LANGS)})...")
    csn_train = _csn_texts("train", per_lang, rng, log=log)
    csn_val = _csn_texts("test", per_lang // 4 + 50, rng)
    local_train, local_val = _local_chunks(LOCAL_PATH, rng, log=log)
    if local_train:
        log(f"+ local permissive corpus: {len(local_train):,} train / "
            f"{len(local_val):,} val files from {LOCAL_PATH.name}")
    else:
        log(f"no local corpus at {LOCAL_PATH} — CodeSearchNet only")
    train_stream = _encode_stream(tok, csn_train + local_train, rng)
    val_stream = _encode_stream(tok, csn_val + local_val, rng)
    seen = steps * batch_size * block
    log(f"stream: {len(train_stream) / 1e6:.2f}M train / {len(val_stream) / 1e3:.0f}k "
        f"val tokens (block {block}) | schedule: {steps:,} steps x {batch_size} x "
        f"{block} = {seen / 1e6:.0f}M tokens (~{seen / max(len(train_stream), 1):.1f} epochs)")

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
                                special_tokens=[], kind="generate", steps=step,
                                val_acc=vloss, meta={"langs": LANGS, "block": block,
                                                     "source": "code_search_net",
                                                     "lang_headers": True,
                                                     "val_loss": vloss})
            el = time.time() - started
            log(f"step {step:>5}/{steps} | train loss {loss.item():.3f} | val loss "
                f"{vloss:.3f} (ppl {np.exp(vloss):.1f}) | {el / 60:.0f}m, "
                f"eta {el / step * (steps - step) / 60:.0f}m{mark}")

    log(f"done — best val loss {best:.3f} (ppl {np.exp(best):.1f}), saved to {out}")
    return out


# ------------------------------------------------------------------ inference


# Code repeats itself by nature — the loop counter and the accumulator recur
# every other line — so the intuition is that a repetition penalty should hurt
# here. Measured over 56 samples x 7 prompts on this checkpoint, it does not:
# the share of 1-2 character identifiers is flat (0.176 / 0.174 / 0.176) across
# penalties 1.0 / 1.1 / 1.15, while degenerate looping falls off steeply
# (repeated 4-grams 0.024 -> 0.023 -> 0.009, duplicate lines 0.065 -> 0.034).
# At 1.2 looping is lowest but short names start climbing (0.195). Mangled
# output is a *training-data* problem, not a sampling one; 1.15 is where a
# 17M expert stops looping without paying for it.
REPETITION_PENALTY = 1.15


def _header(lang: str | None) -> str:
    """The language header a snippet was trained under, or "" for unconditional
    generation (a checkpoint trained before headers existed)."""
    return f"{COMMENT.get(lang, '//')} {lang}\n" if lang else ""


def _tagged(lm: ExpertLM) -> bool:
    """Whether this checkpoint was trained with language headers."""
    return bool(lm.specialists[NAME].get("lang_headers"))


@torch.no_grad()
def complete(lm: ExpertLM, prefix: str, max_new_tokens: int = 128,
             temperature: float = 0.6, top_p: float = 0.95,
             lang: str | None = None) -> str:
    """Continue a code snippet. Every token — the prefix and each sampled token —
    is routed through the code-generator expert and read off the shared LM head,
    stopping at `<|end|>`. Pass `lang` to condition on a language header (see
    `_tag`); it is ignored by checkpoints trained without one. Quality is bounded
    by the frozen trunk; treat it as code-flavoured autocomplete."""
    task = lm.specialists[NAME]["task"]
    end = lm.tok.token_id(END)
    head = _header(lang) if _tagged(lm) else ""
    idx = lm._ids(head + prefix.rstrip("\n") + "\n")
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature,
                                 top_k=40, top_p=top_p,
                                 repetition_penalty=REPETITION_PENALTY,
                                 stop_tokens=[end])
    text = lm.tok.decode(out[0][idx.shape[1]:].tolist())
    return text.split("<|end|>")[0].rstrip()


@torch.no_grad()
def generate(lm: ExpertLM, seed: str, max_new_tokens: int = 160,
             temperature: float = 0.6, top_p: float = 0.95,
             lang: str | None = None) -> str:
    """Continue `seed` **verbatim** (no reformatting or appended newline, unlike
    `complete`), routed through the codegen expert — used to turn a seed (a
    comment describing the task plus a function header) into code. `lang`
    prepends the training-time language header so the continuation commits to
    one language instead of drifting across the six in the stream. Returns just
    the generated continuation, stopping at `<|end|>`."""
    task = lm.specialists[NAME]["task"]
    end = lm.tok.token_id(END)
    head = _header(lang) if _tagged(lm) else ""
    idx = lm._ids(head + seed)
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature,
                                 top_k=40, top_p=top_p,
                                 repetition_penalty=REPETITION_PENALTY,
                                 stop_tokens=[end])
    return lm.tok.decode(out[0][idx.shape[1]:].tolist()).split("<|end|>")[0].rstrip()


def evaluate(path=DEFAULT_PATH, base=EXPERT_PATH, per_lang=800, device=None,
             log=print) -> float:
    """Held-out next-token perplexity over the six-language test stream."""
    device = device or pick_device()
    lm = ExpertLM(base, device, specialists=[path])
    task = lm.specialists[NAME]["task"]
    # CodeSearchNet-only test stream, so perplexity stays comparable across runs.
    stream = _encode_stream(lm.tok, _csn_texts("test", per_lang, np.random.default_rng(0)),
                            np.random.default_rng(0))
    block = lm.specialists[NAME].get("block", BLOCK)
    vloss = _validate(lm.model, stream, block, task, device, iters=100)
    log(f"{NAME}: val loss {vloss:.3f} | perplexity {np.exp(vloss):.1f} "
        f"over {len(stream) / 1e3:.0f}k test tokens")
    return vloss


_PROMPTS = [
    ("python", "def quicksort(arr):"),
    ("python", "def is_prime(n):"),
    ("javascript", "function debounce(fn, delay) {"),
    ("javascript", "function shuffle(arr) {"),
    ("java", "public static int gcd(int a, int b) {"),
]


def demo(path=DEFAULT_PATH, base=EXPERT_PATH, device="cpu",
         max_new_tokens=80) -> None:
    lm = ExpertLM(base, device, specialists=[path])
    for lang, prompt in _PROMPTS:
        cont = complete(lm, prompt, max_new_tokens=max_new_tokens, lang=lang)
        print(f"{_header(lang) if _tagged(lm) else ''}{prompt}\n{cont}\n{'-' * 60}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Code-generation specialist for the expert model: continue a "
                    "snippet (CodeSearchNet, 6 languages, next-token LM).")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=3000)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--block-size", type=int, default=BLOCK)
    tr.add_argument("--per-lang", type=int, default=4000,
                    help="snippets per language to pack into the train stream")
    tr.add_argument("--device", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--per-lang", type=int, default=800)
    ev.add_argument("--device", default=None)
    de = sub.add_parser("demo")
    de.add_argument("--max-new-tokens", type=int, default=80)
    de.add_argument("--device", default="cpu")
    cp = sub.add_parser("complete", help="continue a snippet from a file or stdin")
    cp.add_argument("--file", default=None, help="file to read the prefix from "
                    "(default: read stdin)")
    cp.add_argument("--max-new-tokens", type=int, default=128)
    cp.add_argument("--device", default="cpu")
    cp.add_argument("--lang", default=None, choices=sorted(COMMENT),
                    help="language to condition on (default: unconditional)")
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, lr=a.lr,
              block_size=a.block_size, per_lang=a.per_lang, device=a.device)
    elif a.cmd == "eval":
        evaluate(per_lang=a.per_lang, device=a.device)
    elif a.cmd == "demo":
        demo(max_new_tokens=a.max_new_tokens, device=a.device)
    elif a.cmd == "complete":
        import sys

        text = Path(a.file).read_text() if a.file else sys.stdin.read()
        lm = ExpertLM(EXPERT_PATH, a.device, specialists=[DEFAULT_PATH])
        print(complete(lm, text, max_new_tokens=a.max_new_tokens, lang=a.lang))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
