"""A small model that *reads* a game's state and answers questions about it.

The chat model can't reliably report a score — it was never trained to. This
model is: given the game state written out as fields (``score 7 len 4 food up``)
plus a question, it generates the answer by reading the right field out of the
state. It genuinely reads rather than memorizes — trained with the fields in
random order and a wide range of values, and evaluated on scores held out of
training (see `train`).

    python -m sodachat.reader train      # ~a few minutes
    # then, in the agent, game questions are answered by this model

Training targets are templated (question -> answer that quotes a state field),
so the task is extractive: locate the field named in the question, copy its
value. A ~1M-param char model learns this well.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import GPTConfig, make_amp, pick_device
from .model import MiniGPT

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "reader.pt"
_MAX_LEN = 160
_HOLDOUT_SCORES = {17, 33, 54, 76, 91}  # never shown in training; used to test reading

_DIRECTIONS = ["up", "down", "left", "right", "up and left", "up and right",
               "down and left", "down and right"]

# Each field: how to render a random value, the questions that ask for it, and
# how to phrase the answer from the value.
_QUESTIONS = {
    "score": ["whats the score", "what's the score", "score", "how many points",
              "how am i doing", "am i winning", "what's my score", "how many points do i have"],
    "length": ["how long am i", "how long is the snake", "what's my length",
               "how big am i", "how long"],
    "food": ["where is the food", "where's the food", "where's the apple",
             "which way to the food", "where do i go"],
    "over": ["is it over", "is the game over", "are we done", "still going",
             "is it finished", "did it end"],
    "turn": ["whose turn is it", "whose turn", "is it my turn", "whose move"],
}
_ANSWERS = {
    "score": lambda v: random.choice([f"the score is {v}", f"you have {v} points",
                                       f"score is {v}", f"it's {v}"]),
    "length": lambda v: random.choice([f"you're {v} long", f"length is {v}",
                                       f"{v} segments"]),
    "food": lambda v: random.choice([f"the food is {v}", f"food is {v}", f"it's {v}"]),
    "over": lambda v: ("yes, it's over" if v == "yes" else "no, still going"),
    "turn": lambda v: f"it's {v}'s turn",
}


# Non-game messages. The reader learns to stay silent (empty answer) on these
# and on questions about fields the state doesn't have, so the agent can fall
# back to chat instead of forcing a game answer.
_CHITCHAT = [
    "hi", "hey", "hello", "how are you", "nice", "nice moves", "good job",
    "cool", "lol", "haha", "thanks", "thank you", "you're great", "wow",
    "what should i eat", "tell me a joke", "what's the weather", "i'm bored",
    "do you like music", "that's funny", "interesting", "ok", "sure", "yeah",
    "what's your name", "how old are you", "keep it up", "you can do it",
    "what's up", "good morning", "see you later", "amazing", "hmm", "really",
]


def _random_state(rng: random.Random, allow_holdout: bool) -> dict:
    while True:
        score = rng.randint(0, 99)
        if allow_holdout or score not in _HOLDOUT_SCORES:
            break
    # A random subset of fields is present (games differ), so the reader must
    # rely on what's actually there rather than assuming every field exists.
    fields = {"score": score, "over": rng.choice(["yes", "no"])}
    if rng.random() < 0.7:
        fields["length"] = rng.randint(1, 40)
    if rng.random() < 0.7:
        fields["food"] = rng.choice(_DIRECTIONS)
    if rng.random() < 0.4:
        fields["turn"] = rng.choice(["x", "o"])
    return fields


def state_block(fields: dict, rng: random.Random | None = None) -> str:
    """Render fields as a shuffled 's: k v k v ...' block, so the model must
    read by field name, not position."""
    items = list(fields.items())
    (rng or random).shuffle(items)
    return "s: " + " ".join(f"{k} {v}" for k, v in items)


def _example(rng: random.Random, allow_holdout: bool) -> tuple[str, int]:
    state = _random_state(rng, allow_holdout)
    present = [f for f in _QUESTIONS if f in state]
    roll = rng.random()
    if roll < 0.20:  # chit-chat → stay silent (empty answer → agent chats)
        q, a = rng.choice(_CHITCHAT), ""
    elif roll < 0.30 and len(present) < len(_QUESTIONS):  # asked about a missing field
        absent = [f for f in _QUESTIONS if f not in state]
        q, a = rng.choice(_QUESTIONS[rng.choice(absent)]), ""
    else:  # a question about a field that's present → read it
        field = rng.choice(present)
        q, a = rng.choice(_QUESTIONS[field]), _ANSWERS[field](state[field])
    text = f"{state_block(state, rng)} | q: {q} | a: {a}\n"
    answer_start = text.index("| a: ") + len("| a: ")
    return text, answer_start


class CharTok:
    _CHARS = "\n |:.,'0123456789abcdefghijklmnopqrstuvwxyz"

    def __init__(self, charset: str | None = None):
        self.charset = charset or self._CHARS
        self._stoi = {c: i for i, c in enumerate(self.charset)}
        self.PAD = len(self.charset)
        self.vocab_size = self.PAD + 1

    def encode(self, s: str) -> list[int]:
        return [self._stoi.get(c, self._stoi[" "]) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.charset[i] for i in ids if i < len(self.charset))


def _build(tok: CharTok, text: str, answer_start: int):
    ids = tok.encode(text)[:_MAX_LEN]
    x = np.full(_MAX_LEN, tok.PAD, dtype=np.int16)
    x[: len(ids)] = ids
    y = np.full(_MAX_LEN, -100, dtype=np.int64)
    for p in range(answer_start - 1, len(ids) - 1):  # predict each answer char
        y[p] = ids[p + 1]
    return x, y


def _dataset(n: int, seed: int, allow_holdout: bool, tok: CharTok):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(n):
        text, start = _example(rng, allow_holdout)
        x, y = _build(tok, text, start)
        X.append(x); Y.append(y)
    return np.stack(X), np.stack(Y)


def train(path: Path = DEFAULT_PATH, n=120000, steps=3000, batch_size=128,
          lr=3e-4, device=None, seed=0, log=print) -> Path:
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = CharTok()
    log("generating (state, question, answer) examples...")
    X, Y = _dataset(n, seed, allow_holdout=False, tok=tok)  # holdout scores excluded

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=_MAX_LEN,
                    n_layer=4, n_head=4, n_embd=128, dropout=0.0)
    model = MiniGPT(cfg).to(device)
    log(f"reader {model.num_params():,} params | {len(X):,} examples | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    amp = make_amp(device)
    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        idx = np.random.randint(0, len(X), size=batch_size)
        xb = torch.from_numpy(X[idx].astype(np.int64)).to(device)
        yb = torch.from_numpy(Y[idx]).to(device)
        with amp.autocast():
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1),
                                   ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % 500 == 0 or step == steps:
            log(f"step {step:>5}/{steps} | loss {loss.item():.3f} | "
                f"{time.time() - started:.0f}s")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": vars(cfg), "charset": tok.charset,
                "state_dict": model.state_dict()}, path)
    log(f"saved reader to {path}")
    return path


class Reader:
    def __init__(self, path: Path = DEFAULT_PATH, device=None):
        device = device or "cpu"
        if device == "cpu":
            torch.set_num_threads(1)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.tok = CharTok(ckpt["charset"])
        self.model = MiniGPT(GPTConfig(**ckpt["config"]))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self._nl = self.tok.encode("\n")[0]

    @torch.inference_mode()
    def answer(self, fields: dict, question: str, max_new=40) -> str:
        prompt = f"{state_block(fields)} | q: {question} | a: "
        ids = self.tok.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new, temperature=0.3,
                                  top_k=20, stop_tokens=[self._nl])
        return self.tok.decode(out[0][len(ids):].tolist()).strip()


def _facts_to_fields(facts: dict) -> dict:
    """Map a game's facts() dict to the reader's field names."""
    fields = {}
    if "score" in facts:
        fields["score"] = facts["score"]
    if "length" in facts:
        fields["length"] = facts["length"]
    if "food" in facts:
        fields["food"] = facts["food"]
    if "turn" in facts:
        fields["turn"] = facts["turn"].lower()
    fields["over"] = facts.get("over", "no")
    return fields


def main() -> None:
    p = argparse.ArgumentParser(description="Game-state reader model.")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=3000)
    tr.add_argument("--device", default=None)
    sub.add_parser("test")
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, device=a.device)
    else:  # quick read of held-out scores
        r = Reader()
        for s in sorted(_HOLDOUT_SCORES):
            fields = {"score": s, "length": 5, "food": "up", "over": "no"}
            print(f"score={s:2d} (held out)  ->  {r.answer(fields, 'whats the score')!r}")


if __name__ == "__main__":
    main()
