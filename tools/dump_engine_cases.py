"""Generate the fixtures `tools/check_web_engine.mjs` tests the browser against.

The ONNX graphs are checked against PyTorch by `sodachat.export_onnx --check`,
and the tokenizer by `tools/check_tokenizer.mjs`. This closes the last gap:
that the *JavaScript* — prompts, cache handling, log-probs, heads — reproduces
what the Python inference wrappers do, end to end.

Everything recorded here is deterministic. Sampled replies cannot be compared
across two implementations of a random number generator, so the reference is
greedy decoding (argmax, no warping) plus the values that feed the sampler:
MMI log-probs, classifier heads, and action-head moves.

    python tools/dump_engine_cases.py            # -> web/models/engine-cases.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sodachat.expert import ACT, CHAT, GAME_TAG, TEXT, ExpertLM  # noqa: E402
from sodachat.export_onnx import DEFAULT_OUT  # noqa: E402
from sodachat.model import DEFAULT_MODEL_PATH, MiniChatLM, build_prompt, null_prompt  # noqa: E402

CHATS = [
    {"history": [], "message": "hey there"},
    {"history": ["hi", "hello! how are you?"], "message": "i'm good, what about you"},
    {"history": [], "message": "what did you do today?"},
]
SCORES = [
    ("i'm doing well, thanks!", "that sounds nice."),
    ("what's your name?", "i don't know."),
]
GREEDY_TOKENS = 24


@torch.no_grad()
def _greedy(forward, tok, prompt: str, block: int, stop: set[int]) -> list[int]:
    """Argmax decoding against a full-context forward — the reference the JS
    greedy path (an incremental one, over a cache) has to reproduce."""
    ids = tok.encode(prompt)[-(block - GREEDY_TOKENS):]
    idx = torch.tensor([ids], dtype=torch.long)
    out: list[int] = []
    for _ in range(GREEDY_TOKENS):
        nxt = int(forward(idx[:, -block:])[0, -1].argmax())
        if nxt in stop:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return out


def _chat_cases(cases: dict) -> None:
    if not DEFAULT_MODEL_PATH.exists():
        print(f"skipping chat: no {DEFAULT_MODEL_PATH}")
        return
    lm = MiniChatLM(DEFAULT_MODEL_PATH, device="cpu")
    block = lm.model.cfg.block_size
    stop = set(lm._stop_ids)
    entry = {"greedy": [], "logprob": []}
    for case in CHATS:
        prompt = build_prompt(case["history"], case["message"])
        entry["greedy"].append(
            {
                "prompt": prompt,
                "ids": _greedy(lambda x: lm.model(x)[0], lm.tokenizer, prompt, block, stop),
            }
        )
    for context, continuation in SCORES:
        prompt = build_prompt([], context)
        entry["logprob"] += [
            {"context": prompt, "continuation": continuation,
             "value": lm.logprob(prompt, continuation)},
            {"context": null_prompt(), "continuation": continuation,
             "value": lm.logprob(null_prompt(), continuation)},
        ]
    cases["chat"] = entry
    print(f"chat: {len(entry['greedy'])} greedy, {len(entry['logprob'])} logprob")


def _expert_cases(cases: dict) -> None:
    from sodachat.expert import DEFAULT_PATH

    if not DEFAULT_PATH.exists():
        print(f"skipping expert: no {DEFAULT_PATH}")
        return
    lm = ExpertLM(DEFAULT_PATH, device="cpu")
    block = lm.block
    stop = {lm._nl, lm._end}

    def forward(task):
        return lambda x: lm.model(x, torch.full_like(x, task))[0]

    entry = {"greedy": [], "logprob": []}
    for case in CHATS:
        prompt = f"{CHAT}\n" + build_prompt(case["history"], case["message"])
        entry["greedy"].append(
            {"prompt": prompt, "ids": _greedy(forward(TEXT), lm.tok, prompt, block, stop)}
        )
    for context, continuation in SCORES:
        prompt = build_prompt([], context)
        # ExpertLM.logprob re-tags the frame itself, so the fixture records the
        # untagged prompt and the JS side adds the same prefix from the manifest.
        entry["logprob"] += [
            {"context": f"{CHAT}\n{prompt}", "continuation": continuation,
             "value": lm.logprob(prompt, continuation)},
        ]
    cases["expert-text"] = entry
    print(f"expert-text: {len(entry['greedy'])} greedy, {len(entry['logprob'])} logprob")

    # A move off the action head, on a board the game module actually produces.
    from sodachat.games.snake import SnakeGame

    game = SnakeGame(seed=7)
    board = game.model_board()
    legal = [a for a in game.ACTIONS if game.legal(a)]
    goal = "eat the food"
    cases["expert-game"] = {
        "move": [
            {
                "prompt": f"{GAME_TAG} snake | goal: {goal}\n{board}\n{ACT}",
                "legal": legal,
                "action": lm.move("snake", board, legal, goal=goal),
            }
        ]
    }
    print(f"expert-game: move -> {cases['expert-game']['move'][0]['action']}")

    # Classifier specialists: the label and confidence off each one's head.
    classifiers = {
        "route": ["what is 12 * 7?", "hey how are you", "write me a python function"],
        "code": ["def add(a, b):\n    return a + b\n", "function add(a, b) { return a + b; }"],
    }
    manifest = json.loads((DEFAULT_OUT / "manifest.json").read_text())["models"]
    for name, inputs in classifiers.items():
        if name not in lm.specialists or f"expert-{name}" not in manifest:
            continue
        frame = manifest[f"expert-{name}"]["frame"]
        rows = []
        for text in inputs:
            value = " ".join(text.split()) if frame.get("collapse_whitespace") else text
            value = value[: frame.get("max_chars", len(value))]
            prompt = frame["template"].replace("{input}", value)
            label, confidence = lm.classify(name, prompt)
            rows.append({"prompt": prompt, "label": label, "confidence": confidence})
        cases[f"expert-{name}"] = {"classify": rows}
        print(f"expert-{name}: {[r['label'] for r in rows]}")


def main() -> None:
    cases: dict = {}
    _chat_cases(cases)
    _expert_cases(cases)
    if not cases:
        raise SystemExit("no checkpoints found — nothing to compare against")
    path = DEFAULT_OUT / "engine-cases.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
