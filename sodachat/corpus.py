"""Loading and cleaning of the NLTK NPS Chat corpus.

The corpus is ~10.5k posts scraped from age-segregated chat rooms in 2006,
with usernames anonymized to tokens like ``10-19-20sUser7`` and every post
labeled with a dialogue act (Greet, Statement, whQuestion, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import nltk

# Anonymized handles embedded in post text, e.g. "10-19-20sUser7", "11-08-adultsUser12"
_USER_TOKEN_RE = re.compile(
    r"\d{1,2}-\d{1,2}-(?:teens|adults|\d{1,3}s(?:\d+)?)User\d+", re.IGNORECASE
)
_ACTION_RE = re.compile(r"^\s*\.ACTION\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Post:
    text: str
    act: str


def ensure_corpus() -> None:
    """Download the nps_chat corpus on first use."""
    try:
        nltk.data.find("corpora/nps_chat")
    except LookupError:
        nltk.download("nps_chat", quiet=True)


def clean_text(text: str) -> str:
    """Strip anonymization artifacts and normalize whitespace."""
    text = _ACTION_RE.sub("", text or "")
    text = _USER_TOKEN_RE.sub("friend", text)
    return _WS_RE.sub(" ", text).strip()


def load_sessions() -> list[list[Post]]:
    """Return the corpus as one ordered list of posts per chat-room session."""
    ensure_corpus()
    from nltk.corpus import nps_chat

    sessions: list[list[Post]] = []
    for fileid in nps_chat.fileids():
        posts = [
            Post(text=clean_text(p.text or ""), act=p.get("class", "Other"))
            for p in nps_chat.xml_posts(fileid)
        ]
        sessions.append(posts)
    return sessions
