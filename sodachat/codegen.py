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
# projects on disk, one file per NUL-separated chunk), mixed in with
# CodeSearchNet when present. Absent -> CodeSearchNet only.
LOCAL_PATH = _MODELS / "codegen-local.txt"
BLOCK = 512   # code snippets are short; a 512-token window spans a few functions


# --------------------------------------------------------------- data → stream


def _csn_texts(split: str, per_lang: int, rng: np.random.Generator) -> list[str]:
    """CodeSearchNet function bodies across all six languages. `split` is
    "train" or "test" (held-out)."""
    texts: list[str] = []
    for lang in LANGS:
        texts += _hf_snippets(lang, split, per_lang, rng)
    return texts


def _local_chunks(path, rng: np.random.Generator, holdout: float = 0.08):
    """The local permissive corpus split into (train, val) file lists. Empty
    lists if the blob isn't present, so training falls back to CodeSearchNet."""
    if not Path(path).exists():
        return [], []
    blob = Path(path).read_text(encoding="utf-8", errors="ignore")
    chunks = [c for c in blob.split("\0") if c.strip()]
    rng.shuffle(chunks)
    n_val = max(1, int(len(chunks) * holdout))
    return chunks[n_val:], chunks[:n_val]


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
    csn_train = _csn_texts("train", per_lang, rng)
    csn_val = _csn_texts("test", per_lang // 4 + 50, rng)
    local_train, local_val = _local_chunks(LOCAL_PATH, rng)
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
                                                     "val_loss": vloss})
            el = time.time() - started
            log(f"step {step:>5}/{steps} | train loss {loss.item():.3f} | val loss "
                f"{vloss:.3f} (ppl {np.exp(vloss):.1f}) | {el / 60:.0f}m, "
                f"eta {el / step * (steps - step) / 60:.0f}m{mark}")

    log(f"done — best val loss {best:.3f} (ppl {np.exp(best):.1f}), saved to {out}")
    return out


# ------------------------------------------------------------------ inference


@torch.no_grad()
def complete(lm: ExpertLM, prefix: str, max_new_tokens: int = 128,
             temperature: float = 0.6, top_p: float = 0.95) -> str:
    """Continue a code snippet. Every token — the prefix and each sampled token —
    is routed through the code-generator expert and read off the shared LM head,
    stopping at `<|end|>`. Quality is bounded by the frozen trunk; treat it as
    code-flavoured autocomplete."""
    task = lm.specialists[NAME]["task"]
    end = lm.tok.token_id(END)
    idx = lm._ids(prefix.rstrip("\n") + "\n")
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature,
                                 top_k=40, top_p=top_p, repetition_penalty=1.15,
                                 stop_tokens=[end])
    text = lm.tok.decode(out[0][idx.shape[1]:].tolist())
    return text.split("<|end|>")[0].rstrip()


@torch.no_grad()
def generate(lm: ExpertLM, seed: str, max_new_tokens: int = 160,
             temperature: float = 0.6, top_p: float = 0.95) -> str:
    """Continue `seed` **verbatim** (no reformatting or appended newline, unlike
    `complete`), routed through the codegen expert — used to turn a seed (a
    comment describing the task plus a function header) into code. Returns just
    the generated continuation, stopping at `<|end|>`."""
    task = lm.specialists[NAME]["task"]
    end = lm.tok.token_id(END)
    idx = lm._ids(seed)
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature,
                                 top_k=40, top_p=top_p, repetition_penalty=1.15,
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
    "def quicksort(arr):",
    "def is_prime(n):",
    "function debounce(fn, delay) {",
    "public static int gcd(int a, int b) {",
]


def demo(path=DEFAULT_PATH, base=EXPERT_PATH, device="cpu",
         max_new_tokens=80) -> None:
    lm = ExpertLM(base, device, specialists=[path])
    for prompt in _PROMPTS:
        cont = complete(lm, prompt, max_new_tokens=max_new_tokens)
        print(f"{prompt}\n{cont}\n{'-' * 60}")


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
        print(complete(lm, text, max_new_tokens=a.max_new_tokens))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
