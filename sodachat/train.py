"""Train the from-scratch GPT on a dialogue dataset.

    python -m sodachat.train [--dataset soda|dailydialog|nps] [--steps N]

Dialogues are rendered as tagged turns (see data.py), tokenized once into a
flat uint16 file under models/, and trained on as random crops of that token
stream. Keeping tokens on disk rather than in RAM is what makes the ~200M-token
SODA corpus trainable on modest hardware. A crop almost always straddles a
dialogue boundary, so each token is tagged with the dialogue it came from and
attention is masked to it (see `_batch`).

Any plaintext files you drop in `data/` (see localdata.py) are mixed into the
same stream as untagged documents — that is the way to train this model on your
own writing without adding a dataset loader. `--no-local-data` turns it off.

The default (soda) schedule is ~4 epochs of the corpus — about 820M tokens.
Chinchilla answers "best loss per unit of *training* compute", which is the
wrong question for a model that gets trained once and then run forever, so its
~20 tokens/param is a reference point here and not a target. Repeating a corpus
up to ~4 times is worth nearly as much as fresh data, and stays positive out to
~16 (Muennighoff et al. 2023), so extra epochs cost wall-clock and little else.

What the ratio still tells you is whether the corpus can support the model. At
13.7M params those 820M tokens were ~60 tokens/param — comfortably fed. At the
current 76.9M they are ~11, which is under Chinchilla rather than past it: the
same corpus now has to fill 5.6x the parameters. SODA is ~200M tokens, so
reaching even 20 tokens/param means ~1.5B tokens (~8 epochs), and matching the
old 60 would take ~23x the corpus. Either raise --steps, add corpora, or expect
memorization. DailyDialog alone is ~1.5M tokens, which leaves a model this size
hopelessly under-fed: fluent, but with nothing to say.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator

# Cap the MPS allocator (read when it first initializes) so a memory-hungry
# run raises an OOM error instead of swap-freezing the whole machine. The low
# watermark must stay <= the high one (its default is 1.4). Harmless on CUDA.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import torch

from .data import (
    DIALOG_SEP,
    dailydialog_dialogues,
    format_dialogue,
    format_document,
    nps_dialogues,
    soda_dialogues,
)
from .blocks import BPETokenizer, CharTokenizer, GPTConfig, make_amp, pick_device
from . import localdata
from .model import MiniGPT, default_model_path, save_checkpoint
from .optim import MUON_LR, build_optimizers, lr_multiplier, set_lr

# Model size and schedule per dataset. Dropout earns its keep once the model
# sees the data more than once, which the ~4-epoch soda schedule now does.
_DATASET_PRESETS: dict[str, dict] = {
    # 76.9M params (10/12/768, GPT-2-small width at 10 layers). At the default
    # 100k steps this sees the same ~820M tokens as the old 13.7M config did,
    # which is ~11 tokens/param — BELOW Chinchilla's ~20, not past it. The
    # module docstring explains why that is the wrong number to optimize, but
    # the direction still matters: 5.6x the parameters against an unchanged
    # corpus is a model with more room to memorize, which is why dropout goes
    # up. Feed it more data (--steps for more epochs, or a bigger corpus) or
    # it will overfit SODA rather than generalize.
    #
    # bs=40 / 160k steps sees 1.64B tokens = 21 tokens/param over ~8.6 epochs
    # of SODA's 190M, rather than the 10.6 that 100k x 32 would give. Batch is
    # what buys those tokens cheaply (tokens = steps x batch x block_size) but
    # it is also what fills VRAM: activations dominate here, because relu2's
    # MLP is 4*n_embd = 3072 wide. Measured peak on a 12GB RTX 4000 Ada
    # (11.6GB usable) under bf16 AMP: bs=32 7.7GB (66%), bs=40 9.1GB (78%),
    # bs=48 10.9GB (94%), bs=56 OOM. 40 is the largest with enough headroom
    # that a 20h run will not die of fragmentation at hour 15.
    # The AdamW lr is sqrt(40/32) above the bs=32 figure; MUON_LR is swept
    # separately at this width (see optim.py).
    # A WSD run stopped early never reaches its decay phase, so prefer
    # lowering --steps over killing the run.
    "soda": {
        "tokenizer": "bpe", "vocab_size": 8000,
        "n_layer": 10, "n_head": 12, "n_embd": 768, "dropout": 0.15,
        "steps": 160000, "batch_size": 40, "bpe_sample": 60000,
    },
    "dailydialog": {
        "tokenizer": "bpe", "vocab_size": 8000,
        "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.2,
        "steps": 4000, "batch_size": 16, "bpe_sample": None,
    },
    "nps": {
        "tokenizer": "char",
        "n_layer": 4, "n_head": 4, "n_embd": 192, "dropout": 0.1,
        "steps": 2500, "batch_size": 64, "bpe_sample": None,
    },
}

_ENCODE_CHUNK = 2000


def _dialogues(dataset: str, split: str) -> Iterator[list[str]]:
    """Yield dialogues (lists of alternating utterances) for a split."""
    if dataset == "soda":
        return soda_dialogues("train" if split == "train" else "validation")
    if dataset == "dailydialog":
        return iter(dailydialog_dialogues("train" if split == "train" else "validation"))
    if dataset == "nps":
        all_d = nps_dialogues()
        cut = max(1, int(0.9 * len(all_d)))
        return iter(all_d[:cut] if split == "train" else all_d[cut:])
    raise ValueError(f"unknown dataset {dataset!r} (expected soda, dailydialog or nps)")


def _chunks(it: Iterable, size: int) -> Iterator[list]:
    it = iter(it)
    while chunk := list(islice(it, size)):
        yield chunk


def _local_texts(
    data_dir: Path | None, log: Callable[[str], None]
) -> tuple[list[str], list[str], str]:
    """Your own prose from `data/`, formatted and split into (train, val),
    plus a fingerprint of the corpus.

    Text files are documents, not conversations, so they are rendered without
    speaker tags (`data.format_document`) — but still terminated by the
    dialogue separator, so document-boundary masking treats each file as one
    unit. Returns empty lists when there is nothing there, which is the common
    case: `data/` is opt-in.
    """
    docs = localdata.text_docs(data_dir) if data_dir else []
    if not docs:
        return [], [], ""
    log(f"local text data ({Path(data_dir).name}/): {localdata.summarize(docs)}")
    train, val = localdata.split(docs)
    return (
        [format_document(d.text) for d in train],
        [format_document(d.text) for d in val],
        localdata.fingerprint(docs),
    )


def _texts(dataset: str, split: str, local: dict[str, list[str]]) -> Iterator[str]:
    """The training text for a split: local documents first, then the dataset's
    dialogues. Local goes first so that the BPE vocabulary — trained on the
    first `bpe_sample` documents of the stream — actually sees your files; a few
    hundred of them behind 60,000 SODA dialogues would never be reached."""
    yield from local.get(split, ())
    for dialogue in _dialogues(dataset, split):
        yield format_dialogue(dialogue)


def _write_tokens(tokenizer, texts: Iterator[str], path: Path, log) -> int:
    """Tokenize formatted documents into a flat uint16 file. Returns the token
    count."""
    total, doc_count, started = 0, 0, time.time()
    with open(path, "wb") as f:
        for batch in _chunks(texts, _ENCODE_CHUNK):
            if hasattr(tokenizer, "encode_batch"):
                encoded = tokenizer.encode_batch(batch)
            else:
                encoded = [tokenizer.encode(t) for t in batch]
            flat = [i for ids in encoded for i in ids]
            np.asarray(flat, dtype=np.uint16).tofile(f)
            total += len(flat)
            doc_count += len(batch)
            if doc_count % 100_000 < _ENCODE_CHUNK:
                log(
                    f"  tokenized {doc_count:,} documents "
                    f"({total:,} tokens, {time.time() - started:.0f}s)"
                )
    return total


def prepare_data(
    dataset: str,
    preset: dict,
    cache_dir: Path,
    log: Callable[[str], None] = print,
    data_dir: Path | None = localdata.DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, CharTokenizer | BPETokenizer]:
    """Tokenize the dataset — plus any plaintext documents in `data_dir` — to
    disk once, then memory-map it. Pass `data_dir=None` to train on the dataset
    alone."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_train, local_val, local_id = _local_texts(data_dir, log)
    local = {"train": local_train, "val": local_val}
    # A mixed corpus gets its own cache files, so switching local data off does
    # not force a re-tokenization of the plain dataset (and vice versa).
    name = f"{dataset}+local" if local_id else dataset
    tok_path = cache_dir / f"{name}-tokenizer.json"
    meta_path = cache_dir / f"{name}-meta.json"
    bins = {s: cache_dir / f"{name}-{s}.bin" for s in ("train", "val")}

    cached = (
        tok_path.exists()
        and meta_path.exists()
        and all(p.exists() for p in bins.values())
    )
    if cached:
        meta = json.loads(meta_path.read_text())
        # The cache is keyed by dataset name, which cannot notice that a file
        # in data/ was edited, added or removed — the fingerprint can.
        if meta.get("local", "") != local_id:
            log("local data changed since the cache was built — re-tokenizing")
            cached = False
    if cached:
        tokenizer = _load_tokenizer(tok_path, meta)
        log(f"reusing tokenized cache ({meta['train_tokens']:,} train tokens)")
    else:
        if preset["tokenizer"] == "bpe":
            sample_n = preset.get("bpe_sample")
            log(
                "training BPE vocabulary"
                + (f" on {sample_n:,} sampled documents..." if sample_n else "...")
            )
            sample = islice(_texts(dataset, "train", local), sample_n)
            tokenizer = BPETokenizer.train(
                sample, preset["vocab_size"], special_tokens=[DIALOG_SEP]
            )
        else:
            text = "".join(_texts(dataset, "train", local))
            tokenizer = CharTokenizer(sorted(set(text)))

        meta = {"tokenizer": preset["tokenizer"], "local": local_id}
        for split, path in bins.items():
            log(f"tokenizing {split} split -> {path.name}")
            meta[f"{split}_tokens"] = _write_tokens(
                tokenizer, _texts(dataset, split, local), path, log
            )
        _save_tokenizer(tok_path, tokenizer, meta)
        meta_path.write_text(json.dumps(meta))

    train = np.memmap(bins["train"], dtype=np.uint16, mode="r")
    val = np.memmap(bins["val"], dtype=np.uint16, mode="r")
    return train, val, tokenizer


def _save_tokenizer(path: Path, tokenizer, meta: dict) -> None:
    path.write_text(json.dumps(tokenizer.to_payload()))


def _load_tokenizer(path: Path, meta: dict):
    from .blocks import tokenizer_from_payload

    return tokenizer_from_payload(json.loads(path.read_text()))


def _batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str,
    sep_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One batch of (inputs, targets, document ids).

    Windows are random crops of a stream of packed dialogues, so a window
    nearly always contains the tail of one conversation and the head of the
    next. Given `sep_id` (the id of DIALOG_SEP) each token is numbered with the
    dialogue it belongs to, which lets the model mask attention to a single
    conversation instead of learning to ignore the previous one. Without it
    the returned document ids are None and attention stays plainly causal.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    # Read block_size + 1 tokens so inputs and targets are two views of one
    # window — and so a target's document id is known even at the last position.
    window = np.stack([data[i : i + block_size + 1] for i in ix]).astype(np.int64)
    w = torch.from_numpy(window)
    xt, yt = w[:, :-1].contiguous(), w[:, 1:].contiguous()

    doc = None
    if sep_id is not None:
        sep = (w == sep_id).long()
        # Inclusive cumsum numbers each dialogue; subtracting `sep` puts the
        # separator itself in the dialogue it closes, so the model still learns
        # to *emit* it at the end of a conversation.
        ids = sep.cumsum(1) - sep
        doc = ids[:, :-1].contiguous()
        # Drop the loss wherever the target belongs to the next dialogue. That
        # is exactly one position per boundary — the separator, whose "next
        # token" is the opening of an unrelated conversation and unpredictable
        # by construction. It is also the only target that would cross the
        # boundary the attention mask just closed.
        yt = yt.masked_fill(ids[:, 1:] != doc, -100)

    def to_device(t: torch.Tensor) -> torch.Tensor:
        if device == "cuda":  # overlap the host->device copy with compute
            return t.pin_memory().to(device, non_blocking=True)
        return t.to(device)

    return to_device(xt), to_device(yt), None if doc is None else to_device(doc)


@torch.no_grad()
def _eval_loss(
    model: MiniGPT,
    data: np.ndarray,
    block_size: int,
    device: str,
    sep_id: int | None = None,
    iters: int = 40,
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y, doc = _batch(data, block_size, 16, device, sep_id)
        _, loss = model(x, y, doc_ids=doc)
        losses.append(loss.item())
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


def train_model(
    dataset: str = "soda",
    out_path: Path | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    lr: float = 3.4e-4,
    muon_lr: float = MUON_LR,
    optimizer: str = "muon",
    schedule: str = "wsd",
    device: str | None = None,
    seed: int = 1337,
    data_dir: Path | None = localdata.DATA_DIR,
    log: Callable[[str], None] = print,
) -> Path:
    preset = _DATASET_PRESETS.get(dataset)
    if preset is None:
        raise ValueError(f"unknown dataset {dataset!r}")
    steps = steps or preset["steps"]
    batch_size = batch_size or preset["batch_size"]
    out_path = Path(out_path) if out_path else default_model_path(dataset)

    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, val_data, tokenizer = prepare_data(
        dataset, preset, out_path.parent, log, data_dir=data_dir
    )

    cfg = GPTConfig(
        vocab_size=len(tokenizer),
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        n_embd=preset["n_embd"],
        dropout=preset["dropout"],
    )
    model = MiniGPT(cfg).to(device)
    seen = steps * batch_size * cfg.block_size
    log(
        f"data: {len(train_data):,} train / {len(val_data):,} val tokens "
        f"(vocab {len(tokenizer)}) | model: {model.num_params():,} params | "
        f"device: {device}"
    )
    log(
        f"schedule: {steps:,} steps x {batch_size} x {cfg.block_size} = "
        f"{seen/1e6:.0f}M tokens seen (~{seen/max(len(train_data),1):.1f} epochs) | "
        f"{seen/model.num_params():.1f} tokens/param | "
        f"arch: {cfg.mlp}, qk_norm={cfg.qk_norm}, softcap={cfg.logit_softcap:g}"
    )

    # Muon for the hidden matrices, AdamW for embeddings and norms — see
    # optim.py for why the split is required rather than merely tidy. The
    # cosine schedule and the single-AdamW path are kept behind flags so a
    # change can be bisected against the old recipe.
    optimizers = build_optimizers(
        model, lr=lr, muon_lr=muon_lr, weight_decay=0.01, use_muon=optimizer == "muon"
    )
    warmup = min(1000, max(steps // 20, 1))
    min_frac = 0.0 if schedule == "wsd" else 0.1
    amp = make_amp(device)
    log(
        f"optimizer: {optimizer} "
        + (f"(muon lr {muon_lr:g} / adamw lr {lr:g})" if optimizer == "muon"
           else f"(lr {lr:g})")
        + f" | schedule: {schedule}, {warmup} warmup steps | {amp.tag}"
    )

    # The separator's token id turns on intra-dialogue attention masking; char
    # and legacy tokenizers have no such token, so they keep plain causal
    # attention.
    try:
        sep_id = tokenizer.token_id(DIALOG_SEP)
    except (AttributeError, KeyError):
        sep_id = None
        log("note: tokenizer has no separator token — training without "
            "dialogue-boundary masking")

    eval_every = max(250, steps // 40)
    model.train()
    started = time.time()
    best_val = float("inf")
    for step in range(1, steps + 1):
        set_lr(
            optimizers,
            lr_multiplier(
                step - 1, steps, warmup=warmup, schedule=schedule, min_frac=min_frac
            ),
        )
        x, y, doc = _batch(train_data, cfg.block_size, batch_size, device, sep_id)
        with amp.autocast():
            _, loss = model(x, y, doc_ids=doc)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(optimizers, model)

        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

        if step % eval_every == 0 or step == steps:
            val_loss = _eval_loss(model, val_data, cfg.block_size, device, sep_id)
            marker = ""
            if val_loss < best_val:  # keep only the best-generalizing weights
                best_val = val_loss
                save_checkpoint(out_path, model, tokenizer, step, val_loss)
                marker = " <- saved"
            elapsed = time.time() - started
            eta = elapsed / step * (steps - step)
            log(
                f"step {step:>6}/{steps} | train {loss.item():.3f} | "
                f"val {val_loss:.3f} | {elapsed/60:.0f}m elapsed, {eta/60:.0f}m left"
                f"{marker}"
            )

    log(f"done — best checkpoint at {out_path} (val loss {best_val:.3f})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mini chat GPT.")
    parser.add_argument(
        "--dataset", choices=list(_DATASET_PRESETS), default="soda"
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3.4e-4,
                        help="AdamW rate, for embeddings and norms")
    parser.add_argument("--muon-lr", type=float, default=MUON_LR,
                        help="Muon rate, for the hidden matrices")
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    parser.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=localdata.DATA_DIR,
        help="folder of your own plaintext documents to mix in "
        f"(default: {localdata.DATA_DIR.name}/, ignored when empty or absent)",
    )
    parser.add_argument(
        "--no-local-data",
        action="store_true",
        help=f"train on the dataset alone, ignoring {localdata.DATA_DIR.name}/",
    )
    args = parser.parse_args()
    train_model(
        dataset=args.dataset,
        out_path=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        muon_lr=args.muon_lr,
        optimizer=args.optimizer,
        schedule=args.schedule,
        device=args.device,
        seed=args.seed,
        data_dir=None if args.no_local_data else args.data_dir,
    )


if __name__ == "__main__":
    main()
