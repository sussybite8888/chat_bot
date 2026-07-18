"""A small GPT trained from scratch: a chatbot (terminal, Discord, and Google
Chat frontends sharing one engine) and a real-time game controller."""

from .engine import ChatEngine, Reply

__version__ = "0.1.0"
__all__ = ["ChatEngine", "Reply", "__version__"]
