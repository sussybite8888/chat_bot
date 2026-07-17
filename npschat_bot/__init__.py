"""A chatbot trained on the NPS Chat corpus (NLTK), with terminal, Discord,
and Google Chat frontends sharing one engine."""

from .engine import ChatEngine, Reply

__version__ = "0.1.0"
__all__ = ["ChatEngine", "Reply", "__version__"]
