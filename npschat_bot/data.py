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
"""

from __future__ import annotations

import re
from typing import Iterator

from .model import DIALOG_SEP, SPEAKERS

_MAX_UTTERANCE_CHARS = 320
DAILYDIALOG_REPO = "li2017dailydialog/daily_dialog"
SODA_REPO = "allenai/soda"


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
