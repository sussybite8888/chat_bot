"""A small GPT trained from scratch: a chatbot (terminal, Discord, and Google
Chat frontends sharing one engine) and a real-time game controller.

MODEL MAP — every model, where its parts live, and its checkpoint
================================================================

Shared foundation (owned by no single model):
    blocks.py   building blocks (RMSNorm, RoPE, attention, SwiGLU, Block),
                GPTConfig, the tokenizers, pick_device, pad_load
    model.py    MiniGPT — the base decoder LM every text model reuses

Each model = an architecture + its data/training/inference, in one file:

    model         module        architecture    inference class   checkpoint
    ------------  ------------  --------------   ---------------   ----------------------
    chat          model.py      MiniGPT          MiniChatLM        minigpt-soda.pt
    reader        reader.py     MiniGPT          Reader            reader.pt
    game (each)   game_train.py MiniGPT          GamePlayer        <game>-gpt.pt
    narrator      narrate.py    MultiHeadGPT     NarratingPlayer   snake-narrator.pt
    unified       unified.py    MiniGPT          UnifiedLM         unified.pt
    instruct      instruct.py   MiniGPT          InstructPlayer    unified-instruct.pt
    expert        expert.py     ExpertGPT        ExpertLM          expert.pt
    vision        vision.py     (expert add-on)  via ExpertLM      specialist-vision.pt

Architectures:
    MiniGPT       (model.py)   base decoder LM — chat, reader, games, unified, instruct
    MultiHeadGPT  (narrate.py) MiniGPT + an action head (text + move, one pass)
    ExpertGPT     (expert.py)  task-routed FFN experts (text vs game) + action head;
                               loads *specialists* (frozen-trunk add-on experts with
                               their own heads, e.g. vision) from specialist-*.pt

The agent (agent.py) ties chat + games together and can run any of the single
models via /model. Frontends: cli.py (terminal), discord_bot.py, google_chat.py.
"""

from .engine import ChatEngine, Reply

__version__ = "0.1.0"
__all__ = ["ChatEngine", "Reply", "__version__"]
