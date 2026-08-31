"""Plaintext training data you drop in `data/`.

Everything else in this package trains on datasets fetched from Hugging Face
(SODA, DailyDialog, CodeSearchNet) or minted from the games. This module is the
way in for *your own* files: put plain text and source files under `data/` and
they are mixed into training, no dataset loader and no conversion step.

    data/
      text/            prose -> the chat model     (train.py)
        notes.md
      code/            source -> the code generator (codegen.py)
        utils.py

The two subfolders are a convention, not a requirement — the *extension* is what
decides which trainer a file feeds, so `data/anything/foo.py` is code and
`data/foo.txt` is text (see `classify`). The folders exist so a file whose
extension says nothing (`README`, `notes.log`) can still be claimed as text by
sitting under `data/text/`; an unrecognised extension anywhere else is skipped
and reported rather than fed in as untagged filler — the codegen specialist's
one hard-won lesson is that untagged source in a tagged stream is how JavaScript
completions learn to emit `#endif` (see codegen.py).

What this module does *not* do is quality filtering: it reads, normalizes
whitespace, drops binaries/empties/duplicates, and hands over documents. The
consumer applies its own diet — `codegen.py` runs the same machine-generated
source filter over these files as over everything else it ingests.

    python -m sodachat.localdata          # inventory: what would be picked up
    python -m sodachat.localdata --root some/other/dir
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Source extensions, mapped onto the languages the codegen specialist writes
# (the six CodeSearchNet languages plus TypeScript). Anything outside this map
# is not code *as far as training is concerned*: a .rs or .c file in `data/`
# is skipped rather than shuffled into a stream whose language headers could
# not name it. codegen.LOCAL_EXTS is this same table.
CODE_EXTS = {".py": "python", ".js": "javascript", ".mjs": "javascript",
             ".jsx": "javascript", ".ts": "typescript", ".go": "go",
             ".rb": "ruby", ".php": "php", ".java": "java"}
TEXT_EXTS = {".txt", ".text", ".md", ".markdown", ".rst"}

# Directory names that never hold hand-written training data.
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
             "build", "out", "coverage", ".cache", ".ipynb_checkpoints"}

# One source file past ~250KB is generated, vendored or a data table; prose
# runs much longer legitimately (a whole book is ~1MB), so it gets more room.
MAX_CODE_BYTES = 250_000
MAX_TEXT_BYTES = 4_000_000
MIN_CHARS = 40          # below this a file carries no signal worth a document


@dataclass(frozen=True)
class Doc:
    """One file from `data/`, normalized and ready to train on."""

    path: Path
    kind: str            # "code" or "text"
    lang: str | None     # the code language; None for text
    text: str

    @property
    def name(self) -> str:
        """The path as it should be shown — relative to `data/` when it is
        under it, so logs stay readable and don't leak a home directory."""
        try:
            return str(self.path.relative_to(DATA_DIR))
        except ValueError:
            return str(self.path)


def classify(path: Path, root: Path) -> tuple[str, str | None] | None:
    """(kind, lang) for a file, or None if it isn't training data.

    The extension decides first — a .py under `data/text/` is still code — so
    that the same file means the same thing wherever it is filed. Only when the
    extension says nothing does the folder get a say, and only `data/text/`
    gets one: an extensionless or oddly-suffixed file there (`README`,
    `chat.log`) is prose, while the same file elsewhere is skipped.
    """
    ext = path.suffix.lower()
    if ext in CODE_EXTS:
        return "code", CODE_EXTS[ext]
    if ext in TEXT_EXTS:
        return "text", None
    try:
        top = path.relative_to(root).parts[0]
    except ValueError:
        return None
    return ("text", None) if top == "text" else None


def _normalize(text: str) -> str:
    """Light, safe whitespace normalization, identical for code and prose:
    one newline convention, tabs expanded, no trailing whitespace, no runs of
    3+ blank lines, exactly one trailing newline. Nothing here changes the
    meaning of a line — the model should learn the author's style, not the
    editor's."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.expandtabs(4).rstrip() for line in lines]
    out: list[str] = []
    blanks = 0
    for line in lines:
        blanks = blanks + 1 if line == "" else 0
        if blanks > 2:
            continue
        out.append(line)
    return "\n".join(out).strip("\n") + "\n"


def _read(path: Path, kind: str) -> str | None:
    """The file's text, or None if it is oversized, binary, or not UTF-8.
    Decoding strictly is the binary guard: a stray byte that isn't valid UTF-8
    means this is not a plaintext file, and silently mangling it into the
    training stream is worse than skipping it."""
    cap = MAX_CODE_BYTES if kind == "code" else MAX_TEXT_BYTES
    try:
        if path.stat().st_size > cap:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def load(root: Path | str = DATA_DIR, kind: str | None = None,
         log=None) -> list[Doc]:
    """Every usable document under `root`, sorted by path so a run is
    reproducible. `kind` filters to "code" or "text". Duplicate contents are
    kept once — the same file copied into two folders is one document, not two
    epochs of one.

    A missing `root` is not an error: `data/` is optional, and every caller
    treats an empty list as "no local data, carry on"."""
    root = Path(root)
    if not root.is_dir():
        return []
    docs: list[Doc] = []
    seen: set[str] = set()
    skipped = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if any(part in SKIP_DIRS or part.startswith(".") for part in
               path.relative_to(root).parts[:-1]):
            continue
        if path.name.startswith("."):
            continue
        # The folder documents itself (data/README.md); that is instructions
        # for you, not prose for the model. A README one level down is yours.
        if path.parent == root and path.stem.upper() == "README":
            continue
        what = classify(path, root)
        if what is None:
            skipped += 1
            continue
        this_kind, lang = what
        if kind is not None and this_kind != kind:
            continue
        raw = _read(path, this_kind)
        if raw is None:
            skipped += 1
            continue
        text = _normalize(raw)
        if len(text) < MIN_CHARS or (this_kind == "code" and text.count("\n") < 3):
            skipped += 1
            continue
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        docs.append(Doc(path=path, kind=this_kind, lang=lang, text=text))
    if log and skipped:
        log(f"  skipped {skipped} file(s) in {root} (unsupported extension, "
            f"binary, empty, or oversized)")
    return docs


def code_docs(root: Path | str = DATA_DIR, log=None) -> list[Doc]:
    """Source files, each tagged with the language its extension names."""
    return load(root, kind="code", log=log)


def text_docs(root: Path | str = DATA_DIR, log=None) -> list[Doc]:
    """Prose files."""
    return load(root, kind="text", log=log)


def split(docs: list[Doc], holdout: float = 0.08,
          seed: int = 0) -> tuple[list[Doc], list[Doc]]:
    """Shuffle into (train, val). The shuffle is seeded, so the two halves are
    the same across the separate train- and val-stream builds a trainer runs.
    A single document goes entirely to train — a validation set of "everything
    you have" measures nothing."""
    order = list(docs)
    random.Random(seed).shuffle(order)
    if len(order) < 2:
        return order, []
    n_val = max(1, int(len(order) * holdout))
    return order[n_val:], order[:n_val]


def fingerprint(docs: list[Doc]) -> str:
    """A digest of the whole local corpus, for cache invalidation. Tokenized
    caches under models/ are keyed by dataset name alone, which cannot notice
    that a file in `data/` changed; storing this next to the cache can."""
    h = hashlib.sha1()
    for doc in docs:
        h.update(doc.name.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha1(doc.text.encode("utf-8")).digest())
    return h.hexdigest()


def summarize(docs: list[Doc]) -> str:
    """A one-line description of a document list, for training logs."""
    if not docs:
        return "no local documents"
    chars = sum(len(d.text) for d in docs)
    langs: dict[str, int] = {}
    for doc in docs:
        key = doc.lang or "text"
        langs[key] = langs.get(key, 0) + 1
    by = ", ".join(f"{k} {v}" for k, v in
                   sorted(langs.items(), key=lambda kv: -kv[1]))
    # ~3.5 chars/token is what this package's BPE vocabulary averages on code
    # and prose alike; close enough to size a run against.
    return (f"{len(docs):,} files, {chars / 1e6:.2f}M chars "
            f"(~{chars / 3500:.0f}k tokens) [{by}]")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Inventory the plaintext training data in data/ — what the "
                    "trainers would pick up, and what they would skip.")
    p.add_argument("--root", default=DATA_DIR, type=Path)
    p.add_argument("--kind", choices=("code", "text"), default=None)
    p.add_argument("--files", action="store_true", help="list every document")
    a = p.parse_args()

    if not Path(a.root).is_dir():
        print(f"{a.root} does not exist — no local training data.")
        return
    docs = load(a.root, kind=a.kind, log=print)
    code = [d for d in docs if d.kind == "code"]
    text = [d for d in docs if d.kind == "text"]
    print(f"{a.root}:")
    print(f"  code -> codegen.py : {summarize(code)}")
    print(f"  text -> train.py   : {summarize(text)}")
    if a.files:
        for doc in docs:
            print(f"    {doc.kind:4} {doc.lang or '-':10} {len(doc.text):>9,}c  "
                  f"{doc.name}")


if __name__ == "__main__":
    main()
