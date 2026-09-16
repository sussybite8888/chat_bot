"""Train the from-scratch GPT on a dialogue dataset.

    python -m sodachat.train [--dataset soda|dailydialog|nps] [--steps N]

Dialogues are rendered as tagged turns (see data.py), tokenized once into a
flat uint16 file under models/, and trained on as random crops of that token
stream. Keeping tokens on disk rather than in RAM is what makes the ~200M-token
SODA corpus trainable on modest hardware. A crop almost always straddles a
dialogue boundary, so each token is tagged with the dialogue it came from and
attention is masked to it (see `_batch`).

The stream is not one corpus but three, each tokenized to its own file and
sampled from per batch (see `_batch` and `mix_weights`):

    dialog   the dataset above — how a turn works
    books    public-domain prose from Pre-1929 Books — how English works
    local    any plaintext files you drop in `data/` (see localdata.py) — how
             *you* write. `--no-local-data` turns it off.

`data/` is the corpus the run is ultimately for, and it is also by far the
smallest, so its share of each batch is scheduled rather than fixed: it stays
at its natural token share for the bulk of training and ramps up over the final
`--local-ramp-frac` of the run, the same stretch the WSD schedule spends
decaying the learning rate. Data seen while the rate decays is what the final
weights are shaped by, so this is where a small corpus buys the most voice for
the least memorization; `--local-end-weight` sets how far it ramps, and the
startup log prints how many epochs of `data/` that works out to.

Validation stays at the natural blend all run so that "best val loss" compares
like with like — but for the same reason, checkpoint selection stops following
it once the ramp begins: a run that is deliberately trading natural-blend loss
for your voice would otherwise save a mid-run checkpoint and discard the ending.

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
hopelessly under-fed: fluent, but with nothing to say. The book corpus is the
answer to that: `--books-tokens` is the one dial that adds fresh tokens rather
than re-reading the ones already there.
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
    BOOKS_REPO,
    CHARS_PER_TOKEN,
    DIALOG_SEP,
    books_passages,
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
    #
    # books_tokens is the fresh-token dial (data.books_passages). 150M nearly
    # doubles the 190M-token SODA corpus, taking the default schedule from ~8.6
    # epochs to ~4.7 and giving the model long-form English to learn syntax
    # from, which dialogue corpora barely contain. It costs ~525MB of download
    # and ~300MB of cached uint16 tokens, once.
    "soda": {
        "tokenizer": "bpe", "vocab_size": 8000,
        "n_layer": 10, "n_head": 12, "n_embd": 768, "dropout": 0.15,
        "steps": 160000, "batch_size": 40, "bpe_sample": 60000,
        "books_tokens": 150_000_000,
    },
    # The two small presets are smoke tests and comparison runs, not models
    # anyone talks to: books are off by default there because streaming a book
    # corpus for a 4000-step run costs more than the run does. --books-tokens
    # turns them on anyway.
    "dailydialog": {
        "tokenizer": "bpe", "vocab_size": 8000,
        "n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.2,
        "steps": 4000, "batch_size": 16, "bpe_sample": None,
        "books_tokens": 0,
    },
    "nps": {
        "tokenizer": "char",
        "n_layer": 4, "n_head": 4, "n_embd": 192, "dropout": 0.1,
        "steps": 2500, "batch_size": 64, "bpe_sample": None,
        "books_tokens": 0,
    },
}

_ENCODE_CHUNK = 2000

# The streams a run mixes, in the order they are logged and sampled.
_SOURCES = ("dialog", "books", "local")
# Validation books, as a fraction of the training budget. The held-out shard is
# ~2.5GB of text; a val split only has to be big enough to measure.
_BOOKS_VAL_FRAC = 0.01
# Where the ramp ends: `data/`'s share of the final batch. A small corpus at a
# large share is a lot of repetition — the startup log turns this into a number
# of passes over data/ so the trade is visible before the run starts.
DEFAULT_LOCAL_END_WEIGHT = 0.05
DEFAULT_LOCAL_RAMP_FRAC = 0.2


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


def _source_texts(
    source: str,
    dataset: str,
    split: str,
    local: dict[str, list[str]],
    books_chars: int,
    log: Callable[[str], None],
) -> Iterator[str]:
    """Formatted training documents for one stream of one split."""
    if source == "local":
        yield from local.get(split, ())
    elif source == "books":
        budget = books_chars if split == "train" else max(
            int(books_chars * _BOOKS_VAL_FRAC), 1
        )
        for passage in books_passages(split, budget, log):
            yield format_document(passage)
    elif source == "dialog":
        for dialogue in _dialogues(dataset, split):
            yield format_dialogue(dialogue)
    else:
        raise ValueError(f"unknown source {source!r}")


def _bpe_sample(
    sources: list[str],
    dataset: str,
    local: dict[str, list[str]],
    books_chars: int,
    log: Callable[[str], None],
) -> Iterator[str]:
    """The document stream the BPE vocabulary is trained on.

    Local documents come first and the other streams are then interleaved,
    because the vocabulary only sees the first `bpe_sample` documents: a few
    hundred of your files queued behind 60,000 SODA dialogues would never be
    reached, and a books-then-dialogue ordering would spend the whole sample on
    1890s prose and tokenize modern chat badly.
    """
    yield from local.get("train", ())
    streams = [
        iter(_source_texts(s, dataset, "train", local, books_chars, log))
        for s in sources
        if s != "local"
    ]
    while streams:
        for stream in list(streams):
            try:
                yield next(stream)
            except StopIteration:
                streams.remove(stream)


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
    books_tokens: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], CharTokenizer | BPETokenizer]:
    """Tokenize every stream a run mixes — the dataset's dialogues, the book
    corpus, and any plaintext documents in `data_dir` — to disk once, then
    memory-map them.

    Returns `(train, val, tokenizer)` where train and val are dicts keyed by
    stream name. One file per stream rather than one concatenated stream is
    what lets the trainer change the blend as the run progresses; cropping a
    single packed file can only ever sample it in proportion to its length.

    Pass `data_dir=None` to train without your own documents, `books_tokens=0`
    without the books.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    if books_tokens is None:
        books_tokens = preset.get("books_tokens", 0)
    books_chars = int(books_tokens * CHARS_PER_TOKEN)
    local_train, local_val, local_id = _local_texts(data_dir, log)
    local = {"train": local_train, "val": local_val}
    books_id = f"{BOOKS_REPO}@{books_tokens}" if books_tokens else ""
    sources = [
        s
        for s in _SOURCES
        if s == "dialog" or (s == "books" and books_id) or (s == "local" and local_id)
    ]
    if books_id:
        log(
            f"book corpus ({BOOKS_REPO}): budget {books_tokens:,} tokens "
            f"(~{books_chars:,} chars, streamed — nothing is kept on disk but "
            "the tokens)"
        )
    # A mixed corpus gets its own cache files, so switching a stream off does
    # not force a re-tokenization of the others (and vice versa). The blend is
    # in the name because the BPE vocabulary is trained on the blend: the same
    # SODA tokens are not the same tokens once books are in the sample.
    name = dataset + ("+books" if books_id else "") + ("+local" if local_id else "")
    tok_path = cache_dir / f"{name}-tokenizer.json"
    meta_path = cache_dir / f"{name}-meta.json"
    bins = {
        (source, split): cache_dir / f"{name}-{source}-{split}.bin"
        for source in sources
        for split in ("train", "val")
    }

    cached = (
        tok_path.exists()
        and meta_path.exists()
        and all(p.exists() for p in bins.values())
    )
    if cached:
        meta = json.loads(meta_path.read_text())
        # The cache is keyed by dataset name, which cannot notice that a file
        # in data/ was edited, added or removed — the fingerprint can. The book
        # budget is in the same position: a bigger budget is a bigger corpus.
        if meta.get("local", "") != local_id:
            log("local data changed since the cache was built — re-tokenizing")
            cached = False
        elif meta.get("books", "") != books_id:
            log("book budget changed since the cache was built — re-tokenizing")
            cached = False
    if cached:
        tokenizer = _load_tokenizer(tok_path, meta)
        log(f"reusing tokenized cache ({_token_summary(meta, 'train')})")
    else:
        if preset["tokenizer"] == "bpe":
            sample_n = preset.get("bpe_sample")
            log(
                "training BPE vocabulary"
                + (f" on {sample_n:,} sampled documents..." if sample_n else "...")
            )
            sample = islice(
                _bpe_sample(sources, dataset, local, books_chars, log), sample_n
            )
            tokenizer = BPETokenizer.train(
                sample, preset["vocab_size"], special_tokens=[DIALOG_SEP]
            )
        else:
            text = "".join(
                t
                for source in sources
                for t in _source_texts(
                    source, dataset, "train", local, books_chars, log
                )
            )
            tokenizer = CharTokenizer(sorted(set(text)))

        meta = {
            "tokenizer": preset["tokenizer"],
            "local": local_id,
            "books": books_id,
            "tokens": {},
        }
        for (source, split), path in bins.items():
            log(f"tokenizing {source}/{split} -> {path.name}")
            meta["tokens"][f"{source}/{split}"] = _write_tokens(
                tokenizer,
                _source_texts(source, dataset, split, local, books_chars, log),
                path,
                log,
            )
        _save_tokenizer(tok_path, tokenizer, meta)
        meta_path.write_text(json.dumps(meta))

    train = {source: _open_bin(bins[(source, "train")]) for source in sources}
    val = {source: _open_bin(bins[(source, "val")]) for source in sources}
    return train, val, tokenizer


def _open_bin(path: Path) -> np.ndarray:
    """Memory-map a token file. An empty one maps to an empty array rather than
    raising: a stream can legitimately come out empty (a `data/` holding a
    single file has nothing left to hold out for validation), and the trainer
    drops empty streams from the mixture with a note."""
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=np.uint16)
    return np.memmap(path, dtype=np.uint16, mode="r")


def _token_summary(meta: dict, split: str) -> str:
    counts = {
        key.split("/")[0]: n
        for key, n in meta.get("tokens", {}).items()
        if key.endswith(f"/{split}")
    }
    total = sum(counts.values())
    by = ", ".join(f"{k} {v:,}" for k, v in counts.items())
    return f"{total:,} {split} tokens" + (f" [{by}]" if len(counts) > 1 else "")


def _save_tokenizer(path: Path, tokenizer, meta: dict) -> None:
    path.write_text(json.dumps(tokenizer.to_payload()))


def _load_tokenizer(path: Path, meta: dict):
    from .blocks import tokenizer_from_payload

    return tokenizer_from_payload(json.loads(path.read_text()))


def mix_weights(
    sizes: dict[str, int], local_share: float | None = None
) -> dict[str, float]:
    """How much of each batch is drawn from each stream.

    With no `local_share` every stream is sampled in proportion to its token
    count — exactly what cropping one concatenated stream used to do, and the
    right yardstick for validation. Given one, `data/` takes that share of the
    batch and the remaining streams divide the rest between them by token count
    as before, so raising your own data's weight costs the dialogue and book
    streams proportionally rather than starving one of them.
    """
    total = sum(sizes.values())
    if not sizes:
        return {}
    if not total:
        return {k: 1.0 / len(sizes) for k in sizes}
    weights = {k: v / total for k, v in sizes.items()}
    if local_share is None or "local" not in weights:
        return weights
    rest = {k: v for k, v in weights.items() if k != "local"}
    rest_total = sum(rest.values())
    if not rest_total:
        return {"local": 1.0}
    scale = (1.0 - local_share) / rest_total
    mixed = {k: v * scale for k, v in rest.items()}
    mixed["local"] = local_share
    return mixed


def local_share(
    step: int,
    steps: int,
    *,
    start: float,
    end: float,
    ramp_frac: float = DEFAULT_LOCAL_RAMP_FRAC,
) -> float:
    """`data/`'s share of the batch at `step` (1-based) of a `steps`-step run.

    Flat at `start` for the bulk of the run, then linear to `end` over the
    final `ramp_frac` — the same stretch WSD spends decaying the learning rate,
    which is the point. Early training is where the model learns what English
    and a conversation are, and it wants the big corpora for that; the decay
    phase is where the weights stop moving far and the data still arriving is
    what the final model sounds like. Ramping rather than switching keeps the
    other streams present throughout, so the ending is a shift in emphasis and
    not a fine-tune that forgets.
    """
    ramp_steps = max(int(steps * ramp_frac), 1)
    stable_until = steps - ramp_steps
    if step <= stable_until:
        return start
    progress = min((step - stable_until) / ramp_steps, 1.0)
    return start + (end - start) * progress


def _windows(data: np.ndarray, block_size: int, n: int) -> np.ndarray:
    """`n` random crops of `block_size + 1` tokens.

    One token more than a window so inputs and targets are two views of it —
    and so a target's document id is known even at the last position.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=n)
    return np.stack([data[i : i + block_size + 1] for i in ix]).astype(np.int64)


def _batch(
    parts: list[np.ndarray],
    weights: list[float],
    block_size: int,
    batch_size: int,
    device: str,
    sep_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One batch of (inputs, targets, document ids), drawn from the streams in
    `parts` according to `weights`.

    How many rows come from which stream is a multinomial draw per batch, so
    the blend is right in expectation without any stream having to divide the
    batch size evenly — which matters at the start of the ramp, where `data/`
    is owed a fraction of one row.

    Windows are random crops of a stream of packed dialogues, so a window
    nearly always contains the tail of one conversation and the head of the
    next. Given `sep_id` (the id of DIALOG_SEP) each token is numbered with the
    dialogue it belongs to, which lets the model mask attention to a single
    conversation instead of learning to ignore the previous one. Without it
    the returned document ids are None and attention stays plainly causal.
    Document ids are numbered within a row, so rows from different streams mix
    in one batch without their ids colliding.
    """
    if len(parts) == 1:
        counts = [batch_size]
    else:
        total = sum(weights)
        counts = np.random.multinomial(batch_size, [w / total for w in weights])
    window = np.concatenate(
        [_windows(part, block_size, n) for part, n in zip(parts, counts) if n]
    )
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


def _usable(
    parts: dict[str, np.ndarray],
    block_size: int,
    split: str,
    log: Callable[[str], None],
) -> dict[str, np.ndarray]:
    """Drop streams too short to crop a single window from, with a note.

    A `data/` of two short files has a validation split of one file, which can
    easily be shorter than `block_size`; that is a stream to leave out of the
    mixture, not a crash in `np.random.randint`.
    """
    keep = {}
    for name, data in parts.items():
        if len(data) < block_size + 2:
            log(
                f"note: {split} stream {name!r} holds {len(data):,} tokens — too "
                f"short for a {block_size}-token window, leaving it out"
            )
        else:
            keep[name] = data
    return keep


@torch.no_grad()
def _eval_loss(
    model: MiniGPT,
    parts: dict[str, np.ndarray],
    weights: dict[str, float],
    block_size: int,
    device: str,
    sep_id: int | None = None,
    iters: int = 40,
) -> tuple[float, dict[str, float]]:
    """Validation loss per stream, and their blend under fixed `weights`.

    The blend is deliberately not the training mixture: that one changes as the
    run progresses, and a moving yardstick would make "best val loss so far"
    compare two different measurements — the saved checkpoint would end up
    being whichever step happened to be sampling the easiest corpus.
    """
    model.eval()
    each = max(10, iters // max(len(parts), 1))
    per_part = {}
    for name, data in parts.items():
        losses = []
        for _ in range(each):
            x, y, doc = _batch([data], [1.0], block_size, 16, device, sep_id)
            _, loss = model(x, y, doc_ids=doc)
            losses.append(loss.item())
        per_part[name] = sum(losses) / len(losses)
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    blended = sum(per_part[k] * weights.get(k, 0.0) for k in per_part)
    return blended, per_part


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
    books_tokens: int | None = None,
    local_start_weight: float | None = None,
    local_end_weight: float = DEFAULT_LOCAL_END_WEIGHT,
    local_ramp_frac: float = DEFAULT_LOCAL_RAMP_FRAC,
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

    train_parts, val_parts, tokenizer = prepare_data(
        dataset, preset, out_path.parent, log, data_dir=data_dir,
        books_tokens=books_tokens,
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

    train_parts = _usable(train_parts, cfg.block_size, "train", log)
    val_parts = _usable(val_parts, cfg.block_size, "val", log)
    if not train_parts or not val_parts:
        raise ValueError("no stream is long enough to train on")
    train_names = list(train_parts)
    train_sizes = {k: len(v) for k, v in train_parts.items()}
    train_total = sum(train_sizes.values())
    # Validation is measured at the natural blend and stays there all run; only
    # the training mixture moves (see `_eval_loss` and `local_share`).
    val_weights = mix_weights({k: len(v) for k, v in val_parts.items()})

    by_stream = ", ".join(f"{k} {v/1e6:.1f}M" for k, v in train_sizes.items())
    log(
        f"data: {train_total:,} train / {sum(len(v) for v in val_parts.values()):,} "
        f"val tokens [{by_stream}] (vocab {len(tokenizer)}) | model: "
        f"{model.num_params():,} params | device: {device}"
    )
    log(
        f"schedule: {steps:,} steps x {batch_size} x {cfg.block_size} = "
        f"{seen/1e6:.0f}M tokens seen (~{seen/max(train_total,1):.1f} epochs) | "
        f"{seen/model.num_params():.1f} tokens/param | "
        f"arch: {cfg.mlp}, qk_norm={cfg.qk_norm}, softcap={cfg.logit_softcap:g}"
    )

    # `data/`'s share of the batch: its natural token share until the ramp,
    # `local_end_weight` by the last step. A start weight below the natural one
    # would be strange (it would hide your files early to reveal them late), so
    # the default simply is the natural one: the run is unchanged up to the
    # ramp, and the ramp is the whole of the new behaviour.
    natural = mix_weights(train_sizes)
    share_start = (
        natural.get("local", 0.0) if local_start_weight is None else local_start_weight
    )
    ramping = "local" in train_parts
    ramp_steps = max(int(steps * local_ramp_frac), 1)
    if ramping:
        ramp_tokens = (
            ramp_steps * batch_size * cfg.block_size
            * (share_start + local_end_weight) / 2
        )
        passes = ramp_tokens / max(train_sizes["local"], 1)
        log(
            f"data/ curriculum: {share_start:.3%} of each batch through step "
            f"{steps - ramp_steps:,}, then ramping to {local_end_weight:.1%} by "
            f"step {steps:,} (~{passes:,.0f} passes over data/ during the ramp, "
            f"{train_sizes['local']:,} tokens). Checkpoints inside the ramp "
            "follow the run rather than the best validation loss."
        )
        if passes > 100:
            log(
                "  warning: that is a lot of repetition of a small corpus — it "
                "will be memorized rather than learned from. Lower "
                "--local-end-weight, or put more text in data/."
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
    best_val = saved_val = float("inf")
    for step in range(1, steps + 1):
        set_lr(
            optimizers,
            lr_multiplier(
                step - 1, steps, warmup=warmup, schedule=schedule, min_frac=min_frac
            ),
        )
        share = (
            local_share(
                step, steps, start=share_start, end=local_end_weight,
                ramp_frac=local_ramp_frac,
            )
            if ramping
            else None
        )
        weights = mix_weights(train_sizes, share)
        x, y, doc = _batch(
            [train_parts[n] for n in train_names],
            [weights[n] for n in train_names],
            cfg.block_size,
            batch_size,
            device,
            sep_id,
        )
        with amp.autocast():
            _, loss = model(x, y, doc_ids=doc)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(optimizers, model)

        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()

        if step % eval_every == 0 or step == steps:
            val_loss, per_part = _eval_loss(
                model, val_parts, val_weights, cfg.block_size, device, sep_id
            )
            # Best-val selection, until the ramp starts. After that the run
            # is deliberately trading natural-blend loss for your voice, so the
            # yardstick it would be selected on is no longer the thing being
            # optimized — keeping the "best" checkpoint there would reliably
            # throw away the end of the run, which is the whole point of the
            # ramp. Inside it the newest weights win.
            in_ramp = ramping and step > steps - ramp_steps
            marker = ""
            if val_loss < best_val:
                marker = " <- saved"
            elif in_ramp:
                marker = " <- saved (ramp)"
            if marker:
                save_checkpoint(out_path, model, tokenizer, step, val_loss)
                saved_val = val_loss
            best_val = min(best_val, val_loss)
            elapsed = time.time() - started
            eta = elapsed / step * (steps - step)
            detail = ""
            if len(per_part) > 1:
                detail = " (" + " ".join(
                    f"{k} {v:.3f}" for k, v in per_part.items()
                ) + ")"
            mix = f" | data/ {share:.1%}" if ramping and share > 0.001 else ""
            log(
                f"step {step:>6}/{steps} | train {loss.item():.3f} | "
                f"val {val_loss:.3f}{detail}{mix} | "
                f"{elapsed/60:.0f}m elapsed, {eta/60:.0f}m left{marker}"
            )

    log(
        f"done — checkpoint at {out_path} (val loss {saved_val:.3f}"
        + (f", best seen {best_val:.3f})" if saved_val > best_val else ")")
    )
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
    parser.add_argument(
        "--books-tokens",
        type=int,
        default=None,
        help="budget of public-domain book tokens to mix in from "
        f"{BOOKS_REPO} (default: the preset's; 0 disables). Long-form English "
        "prose, to teach the model the language the dialogue corpora assume.",
    )
    parser.add_argument(
        "--local-end-weight",
        type=float,
        default=DEFAULT_LOCAL_END_WEIGHT,
        help=f"share of the final batches drawn from {localdata.DATA_DIR.name}/ "
        f"(default: {DEFAULT_LOCAL_END_WEIGHT:g}). Your data is weighted up over "
        "the end of the run, where it shapes the finished model most.",
    )
    parser.add_argument(
        "--local-start-weight",
        type=float,
        default=None,
        help=f"share of the batch drawn from {localdata.DATA_DIR.name}/ before "
        "the ramp (default: its natural token share, i.e. no upweighting)",
    )
    parser.add_argument(
        "--local-ramp-frac",
        type=float,
        default=DEFAULT_LOCAL_RAMP_FRAC,
        help="fraction of the run spent ramping to --local-end-weight "
        f"(default: {DEFAULT_LOCAL_RAMP_FRAC:g}, matching the WSD decay phase)",
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
        books_tokens=args.books_tokens,
        local_start_weight=args.local_start_weight,
        local_end_weight=args.local_end_weight,
        local_ramp_frac=args.local_ramp_frac,
    )


if __name__ == "__main__":
    main()
