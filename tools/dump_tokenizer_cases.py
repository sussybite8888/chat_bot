"""Generate the token-id fixtures `tools/check_tokenizer.mjs` tests against.

The JS tokenizer in `web/js/tokenizer.js` has to agree with the Python one
exactly — a single id off and the browser is prompting the model with something
it never saw in training. This writes what Python produces for a set of strings
chosen to hit the parts that are easy to get wrong (special tokens, leading
spaces, unicode, the chat and specialist frames), and the JS side asserts it
reproduces them.

    python tools/dump_tokenizer_cases.py            # -> web/models/tokenizer-cases.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sodachat.blocks import tokenizer_from_payload  # noqa: E402
from sodachat.export_onnx import DEFAULT_OUT  # noqa: E402

CASES = [
    "",
    "hello",
    " hello",
    "hello there, how are you?",
    "A: hi\nB:",
    "A: what did you do today?\nB: i went to the park\nA: nice\nB:",
    " i went to the park.\n",
    "<|chat|>\nA: hey\nB:",
    "<|game|> snake | goal: eat the food\n....#\n<|act|>",
    "<|route|>\nwhat is 12 * 7?\n<|dest|>",
    "<|reason|> what is 12 * 7?\n<|think|>",
    "<|img|>\n..:+#\n<|cls|>",
    "<|code|>\ndef f(x):\n    return x + 1\n<|cls|>",
    "# python\ndef add(a, b):\n    return a + b\n<|end|>",
    "back-to-back specials: <|chat|><|end|><|chat|>",
    "numbers 0 1 42 1234567890",
    "emoji 🙂 and accents café naïve",
    "tabs\tand   runs   of  spaces",
    "trailing space ",
    "CAPS and MiXeD CaSe",
    "punctuation!!! ??? ... --- ***",
    "a" * 300,
]


def main() -> None:
    out = DEFAULT_OUT
    manifest = json.loads((out / "manifest.json").read_text())
    fixtures = {}
    for name in {m["tokenizer"] for m in manifest["models"].values()}:
        tok = tokenizer_from_payload({"type": "bpe", "json": (out / name).read_text()})
        fixtures[name] = [
            {"text": text, "ids": tok.encode(text), "decoded": tok.decode(tok.encode(text))}
            for text in CASES
        ]
        print(f"{name}: {len(fixtures[name])} cases, vocab {len(tok)}")
    path = out / "tokenizer-cases.json"
    path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
