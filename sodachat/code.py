"""Code recognition + completion, added to the expert model as a pluggable
specialist.

The expert model routes every token through a per-task FFN expert (expert.py);
a **specialist** is a new expert trained *after the fact* with the whole shared
trunk frozen — so the model gains a capability without any risk to the ones it
already has. This module is the second specialist (after vision.py): it reads a
code snippet and (1) names its programming language and (2) continues it. It
trains in minutes, saved as a standalone checkpoint (`specialist-code.pt`) that
`ExpertLM` grafts on at startup.

Code becomes tokens the same way everything else does — as text. A snippet is
wrapped in tag tokens:

    <|code|>
    def hello(name):
        return f"hi {name}"
    <|cls|>

The language label reads off a 6-way class head at the trailing `<|cls|>` token
in a single forward pass — no sampling — exactly how game moves read off the
action head at `<|act|>` and how vision reads its label at `<|cls|>`. The whole
document is routed to the code expert.

**Completion** is delegated to the dedicated code-*generation* specialist
(`sodachat.codegen`, `specialist-codegen.pt`) when it is attached — its expert is
LM-trained on code for exactly this. If codegen isn't trained, `complete()` falls
back to routing generation through *this* classifier expert, which learned only
to read a label off `<|cls|>` and so emits gibberish — a stub, not a real
completion. Classification is this module's reliable win; generation lives in
`codegen`.

The corpus is [CodeSearchNet](https://huggingface.co/datasets/code_search_net)
(function bodies across six languages: python, java, javascript, php, ruby,
go). Each language config is loaded separately and subsampled to a few thousand
snippets, cleaned and length-capped so a whole doc fits inside the block.

    python -m sodachat.code train      # needs models/expert.pt (train it first)
    python -m sodachat.code eval       # held-out accuracy, per language
    python -m sodachat.code demo       # print a few snippets + predictions
    python -m sodachat.code complete --file snippet.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import make_amp, pick_device
from .expert import (
    _MODELS,
    DEFAULT_PATH as EXPERT_PATH,
    END,
    ExpertLM,
    save_specialist,
    scaffold_specialist,
    specialist_param_groups,
)

NAME = "code"
CODE, CLS = "<|code|>", "<|cls|>"
# CodeSearchNet's six configs map 1:1 to these labels (config name == language).
LANGS = ["python", "java", "javascript", "php", "ruby", "go"]
LABELS = list(LANGS)
CSN_REPO = "code-search-net/code_search_net"
FIELD = "whole_func_string"   # the full function (docstring + signature + body)
MAX_CHARS = 800               # cap a snippet so the doc fits the block
DEFAULT_PATH = _MODELS / "specialist-code.pt"


# ----------------------------------------------------------------- data loading


def _clean(text: str) -> str:
    """Strip a snippet and normalize trailing whitespace per line. Returns ""
    for snippets too short to identify or too long to keep the doc in-block."""
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()
    if len(s) < 16 or len(s) > MAX_CHARS:   # degenerate or oversized
        return ""
    return s


def _hf_snippets(lang: str, split: str, cap: int, rng: np.random.Generator) -> list[str]:
    """One CodeSearchNet language config -> a cleaned, capped list of snippets.
    `split` is "train" or "test" (CodeSearchNet's validation split is named
    "validation"; we follow the convention the rest of the package uses and
    treat "test" as the held-out probe, falling back to "validation")."""
    from datasets import load_dataset

    use = "test" if split == "test" else "validation" if split == "val" else split
    ds = load_dataset(CSN_REPO, lang, split=use)
    key = FIELD if FIELD in ds.column_names else ds.column_names[0]
    out: list[str] = []
    for row in ds:
        s = _clean(row.get(key) or "")
        if s:
            out.append(s)
        if len(out) >= cap:
            break
    if len(out) > cap:  # deterministic subsample so train/val don't overlap-ish
        idx = rng.choice(len(out), size=cap, replace=False)
        out = [out[i] for i in sorted(idx.tolist())]
    return out


# --------------------------------------------------------------- doc rendering


def code_doc(snippet: str) -> str:
    # No <|end|>: docs are never packed into a stream (each is its own
    # sequence), and the class head fires at the final <|cls|> token.
    return f"{CODE}\n{snippet.strip()}\n{CLS}"


def _build(split: str, per_lang: int, rng: np.random.Generator):
    """Combined six-language corpus -> (docs, labels over LABELS)."""
    docs, labels = [], []
    for i, lang in enumerate(LANGS):
        for s in _hf_snippets(lang, split, per_lang, rng):
            docs.append(code_doc(s))
            labels.append(i)
    return docs, np.asarray(labels, dtype=np.int64)


def _tokenize(tok, docs) -> list[np.ndarray]:
    return [np.asarray(ids, dtype=np.int32) for ids in tok.encode_batch(docs)]


def _batch(ids_list, labels, pick, device):
    """Right-pad a batch of docs; the class head reads each doc at its own
    final (<|cls|>) position, so padding after it is causally invisible."""
    seqs = [ids_list[i] for i in pick]
    width = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), width), dtype=np.int64)
    last = np.empty(len(seqs), dtype=np.int64)
    for j, s in enumerate(seqs):
        x[j, : len(s)] = s
        last[j] = len(s) - 1
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(last).to(device),
            torch.from_numpy(labels[np.asarray(pick)]).to(device))


def _logits(model, task: int, x, last):
    """Class-head logits at each doc's <|cls|> position, the whole batch routed
    to the code expert."""
    feats = model._features(x, torch.full_like(x, task))
    return model.heads[NAME](feats[torch.arange(x.shape[0], device=x.device), last])


@torch.no_grad()
def _accuracy(model, task: int, ids_list, labels, device, bs=64, pick=None) -> float:
    was_training = model.training
    model.eval()
    idxs = np.arange(len(ids_list)) if pick is None else np.asarray(pick)
    hit = 0
    for s in range(0, len(idxs), bs):
        x, last, y = _batch(ids_list, labels, idxs[s:s + bs], device)
        hit += int((_logits(model, task, x, last).argmax(-1) == y).sum())
    if was_training:
        model.train()
    return hit / max(len(idxs), 1)


# ------------------------------------------------------------------- training


def train(base=EXPERT_PATH, out=DEFAULT_PATH, steps=4000, batch_size=32, lr=3e-4,
          per_lang=3000, device=None, seed=0, eval_every=500, eval_n=300,
          log=print) -> Path:
    """Train the code specialist on top of a *frozen* expert model: gradients
    reach only the new per-block FFN expert, the 6-way class head, and the two
    new tokens' embedding rows (see `scaffold_specialist`). Chat, reading, and
    play are untouched by construction — their weights never receive a gradient.

    The new expert is seeded from the game expert (its training diet, dense
    token grids, is the closest thing to source code), then specializes. The LM
    head is shared and frozen, so completion quality is bounded by what the
    pretrained trunk already knows — classify is what this specialist actually
    learns."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, tok, task = scaffold_specialist(base, name=NAME, special_tokens=[CODE, CLS],
                                           n_labels=len(LABELS), device=device)
    per_block = sum(p.numel() for p in model.blocks[0].ffn.experts[task].parameters())
    trainable = (per_block * model.cfg.n_layer
                 + sum(p.numel() for p in model.heads[NAME].parameters())
                 + 2 * model.cfg.n_embd)  # the <|code|>/<|cls|> embedding rows
    log(f"specialist '{NAME}' on {Path(base).name}: expert slot {task} | "
        f"{trainable / 1e6:.1f}M trainable of {model.num_params() / 1e6:.1f}M "
        f"(shared trunk frozen) | device {device}")

    log(f"loading CodeSearchNet ({', '.join(LANGS)})...")
    train_docs, train_y = _build("train", per_lang, rng)
    test_docs, test_y = _build("test", per_lang // 3 + 50, rng)
    log(f"tokenizing {len(train_docs):,} train + {len(test_docs):,} test snippets...")
    train_ids = _tokenize(tok, train_docs)
    test_ids = _tokenize(tok, test_docs)
    lens = [len(s) for s in train_ids]
    assert max(lens) <= model.cfg.block_size, "code docs overflow the block"
    assert train_ids[0][-1] == tok.token_id(CLS)
    probe = np.random.default_rng(seed + 1).permutation(len(test_ids))[:eval_n]
    seen = steps * batch_size
    log(f"docs: {min(lens)}-{max(lens)} tokens (block {model.cfg.block_size}) | "
        f"labels: {len(LABELS)} | schedule: {steps:,} steps x {batch_size} = "
        f"{seen / 1e3:.0f}k snippets (~{seen / max(len(train_ids), 1):.1f} epochs)")

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
    started, best = time.time(), 0.0
    for step in range(1, steps + 1):
        pick = np.random.randint(0, len(train_ids), size=batch_size)
        x, last, y = _batch(train_ids, train_y, pick, device)
        with amp.autocast():
            loss = F.cross_entropy(_logits(model, task, x, last), y)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % eval_every == 0 or step == steps:
            acc = _accuracy(model, task, test_ids, test_y, device, pick=probe)
            mark = ""
            if acc > best:
                best, mark = acc, " <- saved"
                save_specialist(out, model, tok, name=NAME, task=task, labels=LABELS,
                                special_tokens=[CODE, CLS], steps=step, val_acc=acc,
                                meta={"langs": LANGS, "field": FIELD,
                                      "max_chars": MAX_CHARS,
                                      "source": "code_search_net"})
            el = time.time() - started
            log(f"step {step:>5}/{steps} | loss {loss.item():.3f} | test acc {acc:.1%} "
                f"| {el / 60:.0f}m, eta {el / step * (steps - step) / 60:.0f}m{mark}")

    lm = ExpertLM(base, device, specialists=[out])
    task = lm.specialists[NAME]["task"]
    for lang, pick in ((l, np.flatnonzero(test_y == i))
                       for i, l in enumerate(LABELS)):
        acc = _accuracy(lm.model, task, test_ids, test_y, device, pick=pick)
        log(f"final {lang} test accuracy: {acc:.2%} over {len(pick):,} snippets")
    log(f"done — best mixed test accuracy {best:.1%}, saved to {out}")
    return out


# ------------------------------------------------------------------ inference


def classify(lm: ExpertLM, snippet: str) -> tuple[str, float]:
    """Name the language of one code snippet, using the code specialist loaded
    in an ExpertLM. Returns (label, confidence)."""
    return lm.classify(NAME, code_doc(snippet))


@torch.no_grad()
def complete(lm: ExpertLM, prefix: str, max_new_tokens: int = 96,
             temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Continue a code snippet. Prefers the dedicated code-*generation*
    specialist (`codegen`) when it is attached — its expert is LM-trained for
    exactly this. Without it, this falls back to routing generation through
    *this* classifier expert, which was only trained to read a language label
    off <|cls|>, so its output is gibberish — a stub, not a real completion."""
    if "codegen" in lm.specialists:
        from . import codegen
        return codegen.complete(lm, prefix, max_new_tokens=max_new_tokens)
    task = lm.specialists[NAME]["task"]
    end = lm.tok.token_id(END)
    idx = lm._ids(code_doc(prefix).rsplit(CLS, 1)[0].rstrip())  # drop the <|cls|>
    out = lm.model.generate_text(idx, task, max_new_tokens, temperature,
                                 top_k=40, top_p=top_p, repetition_penalty=1.1,
                                 stop_tokens=[end])
    text = lm.tok.decode(out[0][idx.shape[1]:].tolist())
    # Cut at the first blank line: sampled tails wander, and a function usually
    # ends where the next blank line would be.
    cut = text.find("\n\n")
    return (text[:cut] if cut != -1 else text).rstrip()


def evaluate(path=DEFAULT_PATH, base=EXPERT_PATH, n=300, device=None, log=print) -> float:
    """Held-out accuracy, reported per language (`n` snippets from each)."""
    device = device or pick_device()
    lm = ExpertLM(base, device, specialists=[path])
    task = lm.specialists[NAME]["task"]
    docs, labels = _build("test", n, np.random.default_rng(0))
    ids = _tokenize(lm.tok, docs)
    accs = []
    for lang, pick in ((l, np.flatnonzero(labels == i)) for i, l in enumerate(LANGS)):
        acc = _accuracy(lm.model, task, ids, labels, device, pick=pick[:n])
        accs.append(acc)
        log(f"{lang}: {acc:.2%} over {len(pick[:n]):,} test snippets")
    return sum(accs) / len(accs)


def demo(path=DEFAULT_PATH, base=EXPERT_PATH, n=6, device="cpu", seed=0) -> None:
    lm = ExpertLM(base, device, specialists=[path])
    rng = np.random.default_rng(seed)
    shown: set[int] = set()
    for i in rng.integers(0, sum(1 for _ in LANGS), size=n):
        i = int(i) % len(LANGS)
        if i in shown:  # try to show a mix of languages
            continue
        shown.add(i)
        lang = LANGS[i]
        snippets = _hf_snippets(lang, "test", 5, rng)
        if not snippets:
            continue
        snippet = snippets[0]
        # Show a prefix and predict — the realistic case (you've typed some code).
        lines = snippet.split("\n")
        prefix = "\n".join(lines[: max(2, len(lines) // 3)])
        label, conf = classify(lm, prefix)
        verdict = "correct" if label == lang else f"wrong (it's {lang})"
        print(f"{prefix}\n-> {label}  ({conf:.0%}, {verdict})\n")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Code specialist for the expert model: name a snippet's "
                    "language and continue it (CodeSearchNet, 6 languages).")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=4000)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--per-lang", type=int, default=3000,
                    help="snippets per language to train on")
    tr.add_argument("--device", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--n", type=int, default=300, help="test snippets per language")
    ev.add_argument("--device", default=None)
    de = sub.add_parser("demo")
    de.add_argument("--n", type=int, default=6)
    de.add_argument("--seed", type=int, default=0)
    cp = sub.add_parser("complete", help="continue a snippet from a file or stdin")
    cp.add_argument("--file", default=None, help="file to read the prefix from "
                    "(default: read stdin)")
    cp.add_argument("--max-new-tokens", type=int, default=96)
    cp.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, lr=a.lr,
              per_lang=a.per_lang, device=a.device)
    elif a.cmd == "eval":
        evaluate(n=a.n, device=a.device)
    elif a.cmd == "demo":
        demo(n=a.n, seed=a.seed)
    elif a.cmd == "complete":
        import sys

        text = Path(a.file).read_text() if a.file else sys.stdin.read()
        # auto-load all specialists so completion delegates to codegen if trained
        lm = ExpertLM(EXPERT_PATH, a.device)
        print(complete(lm, text, max_new_tokens=a.max_new_tokens))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
