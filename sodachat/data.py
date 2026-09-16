"""Dialogue dataset loaders and the training text format.

Loaders return a list of dialogues, each a list of alternating utterances
(speaker A, speaker B, A, ...). `format_dialogue` renders one into the text
the model trains on:

    A: Say, Jim, how about going for a few beers after dinner?
    B: You know that is tempting but is really not good for our fitness.
    A: What do you mean? It will help us to relax.
    <|endofdialog|>

The speaker tags teach the model that turns alternate and which side it is
answering as; the separator marks where a conversation ends, so the model
does not learn that abruptly switching topic is a valid reply.

`format_document` is the same idea for plain prose — your own files from
`data/` (see localdata.py) and the public-domain books of
`books_passages`, which have no turns to tag but still end at the separator so
each one is a document of its own.
"""

from __future__ import annotations

import re
from typing import Iterator

from .model import DIALOG_SEP, SPEAKERS

_MAX_UTTERANCE_CHARS = 320
DAILYDIALOG_REPO = "li2017dailydialog/daily_dialog"
SODA_REPO = "allenai/soda"
BOOKS_REPO = "common-pile/pre_1929_books_filtered"

# ~3.5 chars per token is what this package's BPE vocabulary averages on
# English prose (the same figure localdata.summarize sizes runs with). Budgets
# for the book corpus are given in tokens and converted with it, because the
# only way to know the true count is to tokenize, and the point of the budget
# is to stop before doing that to 60GB of text.
CHARS_PER_TOKEN = 3.5


def clean_utterance(text: str) -> str:
    """Undo DailyDialog's tokenized spacing ("I ’ m fine , thanks !")."""
    text = re.sub(r"\s*’\s*", "'", text)
    text = text.replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # DailyDialog often omits the space after sentence-final punctuation.
    text = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+'\s*s\b", "'s", text)
    return re.sub(r"\s+", " ", text).strip()


def format_dialogue(utterances: list[str]) -> str:
    """Render one dialogue as tagged training text."""
    lines = [
        f"{SPEAKERS[i % 2]}: {utterance}" for i, utterance in enumerate(utterances)
    ]
    return "\n".join(lines) + f"\n{DIALOG_SEP}\n"


def format_document(text: str) -> str:
    """Render one plaintext document (a file from `data/`, see localdata.py) as
    training text.

    Prose has no speaker tags — it is not a dialogue, and labelling it `A:`
    would teach the model that monologue is a turn. It still ends with the
    dialogue separator, because that token is what `train._batch` counts to
    number documents: terminating a file with it keeps the attention mask from
    spilling one document into the next, exactly as it does for a conversation.
    """
    return text.strip() + f"\n{DIALOG_SEP}\n"


def dailydialog_dialogues(split: str) -> list[list[str]]:
    from datasets import load_dataset

    # The canonical repo still hosts a (now unsupported) loading script;
    # the auto-converted parquet branch is the supported way in.
    ds = load_dataset(DAILYDIALOG_REPO, revision="refs/convert/parquet", split=split)
    dialogues = []
    for row in ds:
        utterances = [clean_utterance(u) for u in row["dialog"]]
        utterances = [u for u in utterances if u and len(u) <= _MAX_UTTERANCE_CHARS]
        if len(utterances) >= 2:
            dialogues.append(utterances)
    return dialogues


_SODA_FILES = {"train": "train.parquet", "validation": "valid.parquet"}


def soda_dialogues(split: str, limit: int | None = None) -> Iterator[list[str]]:
    """SODA: ~1.2M narrative-grounded two-party dialogues (CC-BY-4.0).

    The parquet file is fetched once and read locally in batches — streaming
    row-by-row over the network is ~100x slower, and the split is far too
    large (~200M tokens) to materialize as Python objects.

    ~97% of dialogues are two speakers strictly alternating; the rest are
    dropped so that position alone determines the speaker (`format_dialogue`).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SODA_REPO, _SODA_FILES[split], repo_type="dataset")
    kept = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=2048, columns=["dialogue", "speakers"]
    ):
        for dialogue, speakers in zip(
            batch.column("dialogue").to_pylist(),
            batch.column("speakers").to_pylist(),
        ):
            if limit is not None and kept >= limit:
                return
            if len(dialogue) < 2 or len(set(speakers)) != 2:
                continue
            if any(speakers[i] == speakers[i + 1] for i in range(len(speakers) - 1)):
                continue
            utterances = [clean_utterance(u) for u in dialogue]
            if not all(utterances):
                continue
            if max(len(u) for u in utterances) > _MAX_UTTERANCE_CHARS:
                continue
            kept += 1
            yield utterances


# Pre-1929 Books (Common Pile v0.1): ~130k US books published before 1929 and
# in the public domain since 2024, OCR'd by the Internet Archive for HathiTrust.
# 26 dolma-format shards, ~19.5GB gzipped — far too much to download, let alone
# tokenize, so a run takes a token budget off the front of the stream and stops.
_BOOKS_SHARD = "public_library_1929_dolma-{:04d}.json.gz"
_BOOKS_SHARDS = 26
# The corpus ships one split; the last shard is held out so validation prose is
# books the run never trained on, not a sample of the ones it did.
_BOOKS_VAL_SHARDS = 1
# A passage is the training document, cut on paragraph boundaries. ~4000 chars
# is ~1100 tokens — several block_size windows, so most crops stay inside one
# passage, without holding a whole 600KB book in memory as one document.
_BOOKS_PASSAGE_CHARS = 4000
_BOOKS_MIN_PASSAGE_CHARS = 400
# Non-space characters that must be letters for a paragraph to be prose. Running
# English is ~93% letters outside whitespace; indexes, page-number runs, tables
# of contents and OCR noise fall far below this.
_BOOKS_MIN_ALPHA = 0.8
_BOOKS_MIN_WORDS = 3

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_LINE_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")


def _shard_files(split: str) -> list[str]:
    names = [_BOOKS_SHARD.format(i) for i in range(_BOOKS_SHARDS)]
    if split == "train":
        return names[:-_BOOKS_VAL_SHARDS]
    return names[-_BOOKS_VAL_SHARDS:]


def clean_book_text(text: str) -> str:
    """Undo OCR page layout: hard-wrapped lines become paragraphs again.

    The scans are wrapped at the width of the printed page, so a raw book is a
    column of ~70-character lines. Fed in as-is it teaches the model to break
    a line every seventy characters mid-sentence, which is a property of the
    1890s typesetter and not of English. Line breaks inside a paragraph are
    therefore joined (rejoining the word where the break fell on a hyphen) and
    only blank-line paragraph breaks survive.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    for para in _PARAGRAPH_RE.split(text):
        para = _LINE_HYPHEN_RE.sub(r"\1\2", para)   # "knowl-\nedge" -> "knowledge"
        para = re.sub(r"\s*\n\s*", " ", para)       # unwrap the rest of the column
        para = re.sub(r"\s+([,.;:!?])", r"\1", para)  # OCR's spaced-off " ;" and " ."
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


def _is_prose(paragraph: str) -> bool:
    """Whether a cleaned paragraph is running text rather than page furniture.

    Every book carries a title page, a table of contents, page headers, an
    index and a scattering of OCR wreckage. None of it is English worth
    learning, and an index in particular is thousands of near-identical lines
    — the exact shape of text a small model will happily memorize.
    """
    if len(paragraph.split()) < _BOOKS_MIN_WORDS:
        return False
    body = [c for c in paragraph if not c.isspace()]
    if sum(c.isalpha() for c in body) < _BOOKS_MIN_ALPHA * len(body):
        return False
    # Running heads and chapter titles ("THE ARABIAN NIGHTS. 147") are set in
    # capitals; a paragraph without a single lowercase letter is one of those.
    return any(c.islower() for c in paragraph)


def _passages(text: str) -> Iterator[str]:
    """Group a cleaned book's paragraphs into passage-sized documents."""
    buffer: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        if not _is_prose(para):
            continue
        buffer.append(para)
        size += len(para) + 2
        if size >= _BOOKS_PASSAGE_CHARS:
            yield "\n\n".join(buffer)
            buffer, size = [], 0
    if size >= _BOOKS_MIN_PASSAGE_CHARS:
        yield "\n\n".join(buffer)


def _stream_shard(name: str) -> Iterator[dict]:
    """JSONL rows of one gzipped dolma shard, read straight off the wire.

    Hand-rolled rather than `load_dataset(..., streaming=True)` for the same
    reason `soda_dialogues` reads parquet itself: the shape of the data. Each
    row carries a free-form `metadata` object whose keys differ from book to
    book, and Arrow infers a struct schema from the first block of a file and
    then fails to cast the rest — a crash hours into tokenizing. Only `text`
    is wanted here, so the gzip is decompressed line by line and nothing is
    written to disk (the shards are ~640MB each; see the note in
    data/README.md about not materializing this corpus).
    """
    import gzip
    import json
    import urllib.request

    from huggingface_hub import hf_hub_url

    url = hf_hub_url(BOOKS_REPO, name, repo_type="dataset")
    with urllib.request.urlopen(url, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as lines:
            for line in lines:
                try:
                    yield json.loads(line)
                except ValueError:  # a truncated final line, nothing more here
                    return


def books_passages(
    split: str, char_budget: int | None = None, log=None
) -> Iterator[str]:
    """Pre-1929 Books: public-domain English prose, in passage-sized documents.

    The chat corpora are dialogue — short turns, contemporary register, and in
    SODA's case machine-written. They teach the model to take a turn; they are
    thin on the long, syntactically varied sentences that teach it English.
    This is the counterweight: 130k books of edited prose, capped at
    `char_budget` characters (see CHARS_PER_TOKEN) because the full corpus is
    ~60GB and a run wants a slice of it, not all of it.

    The text is OCR, so it is unwrapped (`clean_book_text`) and filtered to
    paragraphs that look like prose (`_is_prose`) before being cut into
    passages. A shard that fails mid-stream is logged and skipped rather than
    killing a tokenization pass that has already run for an hour.
    """
    used = 0
    for name in _shard_files(split):
        try:
            for row in _stream_shard(name):
                for passage in _passages(clean_book_text(row.get("text") or "")):
                    yield passage
                    used += len(passage)
                    if char_budget is not None and used >= char_budget:
                        return
        except (OSError, EOFError) as exc:
            if log:
                log(f"  books: {name} failed ({exc}) — skipping to the next shard")


def nps_dialogues() -> list[list[str]]:
    # Imported lazily: only the NPS dataset needs nltk.
    from .corpus import load_sessions

    dialogues = []
    for session in load_sessions():
        utterances = [
            p.text
            for p in session
            if p.act != "System" and p.text and len(p.text) <= _MAX_UTTERANCE_CHARS
        ]
        if len(utterances) >= 2:
            dialogues.append(utterances)
    return dialogues


# Back-compat helpers returning already-formatted text.
def dailydialog_texts(split: str) -> list[str]:
    return [format_dialogue(d) for d in dailydialog_dialogues(split)]


def nps_texts() -> list[str]:
    return [format_dialogue(d) for d in nps_dialogues()]
