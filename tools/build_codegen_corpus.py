"""Gather a permissively-licensed local code corpus for the codegen specialist.
Only project roots whose OWN license is permissive (MIT/BSD/Apache/ISC); no VEX
code (per user); minified/bundled/generated files filtered out; light whitespace
normalization.

Output: one NUL-separated blob, each chunk `<repo-relative path>\\n<source>`.
The path line is what `codegen._local_chunks` reads the language off, so the
snippet can go into the training stream under a language header instead of
being shuffled in untagged.

Only extensions the codegen specialist actually writes are collected — the six
CodeSearchNet languages plus TypeScript. C/C++ used to be 68% of this blob
(micropython's core, quickjs, pico-sdk), and untagged C in a six-language
stream is how JavaScript completions ended up emitting `#endif`."""
import hashlib
import os
import re
import sys

HOME = os.path.expanduser("~")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "models", "codegen-local.txt")

# (root, allowed_subdirs or None for whole tree) — permissive roots only.
ROOTS = [
    ("micropython-1.24.1", ["py", "extmod", "shared", "drivers"]),  # MIT core only
    ("quickjs", None),                    # MIT
    ("Ogar", None),                       # Apache-2.0
    ("sb3topy", None),                    # BSD
    ("agar.io-clone", None),              # MIT
    ("raspberrypi-pico", None),           # MIT
    ("unfinished-florr-clone", None),     # MIT
    ("html-css-javascript-games", None),  # MIT
    ("flooooio", None),                   # MIT (dense client code; filters trim it)
    # restringer (MIT) is deliberately absent: it is a *deobfuscator*, so
    # tests/resources is a directory of obfuscated JavaScript — _0x names,
    # jsfuck, eval-packed payloads — and its own tests embed those samples as
    # string literals. Exactly what this corpus must not teach.
]

# The languages the codegen specialist writes (six CodeSearchNet + TypeScript).
# Anything else is left out rather than fed in as untagged filler.
EXTS = {".py", ".js", ".mjs", ".jsx", ".ts", ".java", ".go", ".rb", ".php"}
SKIP_DIRS = {"node_modules", ".git", "build", "dist", "out", "__pycache__",
             ".venv", "coverage", ".cache", "vendor", "min", "resources"}
MAX_BYTES = 250_000       # skip generated/huge single files
MAX_LINE = 400            # skip minified/bundled (one giant line)
MAX_AVG_LINE = 120        # skip dense/minified even without one huge line
MAX_SHORT_IDENT = 0.5     # skip mangled names (mostly 1-2 character identifiers)
# The last two are shape heuristics, and only JS/TS is shipped minified: Go
# names its receivers `p` and its buffers `b`, so "mostly short identifiers" is
# idiom there rather than mangling.
MINIFIED_EXTS = {".js", ".mjs", ".jsx", ".ts"}

# Minifier / bundler / transpiler / obfuscator fingerprints. Such code is valid
# and licence-clean, so nothing else here rejects it, but training on it teaches
# the model to write mangled names and helper-call soup.
MACHINE = [
    re.compile(r"_0x[0-9a-f]{3,}"),                       # obfuscator.io names
    re.compile(r"(\\x[0-9a-fA-F]{2}){3,}"),               # hex-escaped blobs
    re.compile(r"(\\u[0-9a-fA-F]{4}){3,}"),               # unicode-escaped blobs
    re.compile(r"\+!\+\[\]|\[\]\[\(!\[\]"),               # jsfuck
    re.compile(r"\beval\s*\(\s*(function|atob|unescape|String\.fromCharCode)"),
    re.compile(r"String\.fromCharCode\(\s*(?:0x[0-9a-f]+|\d+)\s*"
               r"(?:,\s*(?:0x[0-9a-f]+|\d+)\s*){4,}\)"),
    re.compile(r"_WEBPACK_IMPORTED_MODULE|__webpack_"),   # bundler output
    re.compile(r"_interopRequireDefault|\b_\w+2\.default\b"),   # babel interop
    re.compile(r"\b_(classCallCheck|createClass|possibleConstructorReturn|"
               r"slicedToArray|toConsumableArray|objectSpread|inherits)\b"),
]
IDENT = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b")
KEYWORDS = frozenset(
    "var let const function return if else for while new this typeof in of class "
    "extends super null true false undefined try catch throw switch case break "
    "continue do delete void instanceof yield await async static get set import "
    "export from default require module exports def self end elsif nil puts func "
    "package type struct interface map range go defer chan public private static "
    "int void string bool echo".split())


def _norm(text: str) -> str:
    """Light, safe style normalization: expand tabs, strip trailing whitespace,
    collapse 4+ blank-line runs to one, ensure a single trailing newline."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.expandtabs(4).rstrip() for ln in lines]
    out, blanks = [], 0
    for ln in lines:
        blanks = blanks + 1 if ln == "" else 0
        if blanks > 2:  # collapse 3+ consecutive blank lines
            continue
        out.append(ln)
    return "\n".join(out).strip("\n") + "\n"


def _keep(path: str, text: str) -> bool:
    if len(text) < 40 or text.count("\n") < 3:
        return False
    lines = text.split("\n")
    mx = max((len(x) for x in lines), default=0)
    if mx > MAX_LINE:
        return False
    # crude binary/text guard
    if "\x00" in text:
        return False
    if any(rx.search(text) for rx in MACHINE):
        return False
    if os.path.splitext(path)[1].lower() not in MINIFIED_EXTS:
        return True
    if len(text) / max(len(lines), 1) > MAX_AVG_LINE:
        return False
    names = [i for i in IDENT.findall(text) if i not in KEYWORDS]
    if names and sum(1 for i in names if len(i) <= 2) / len(names) > MAX_SHORT_IDENT:
        return False
    return True


def main() -> None:
    seen: set[str] = set()
    chunks: list[str] = []
    per_ext: dict[str, int] = {}
    kept = skipped = 0
    for root, subdirs in ROOTS:
        base = os.path.join(HOME, root)
        walk_roots = ([os.path.join(base, s) for s in subdirs] if subdirs
                      else [base])
        for wr in walk_roots:
            for dirpath, dirnames, filenames in os.walk(wr):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in EXTS:
                        continue
                    fp = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(fp) > MAX_BYTES:
                            skipped += 1
                            continue
                        with open(fp, encoding="utf-8", errors="ignore") as f:
                            raw = f.read()
                    except Exception:
                        skipped += 1
                        continue
                    if not _keep(fp, raw):
                        skipped += 1
                        continue
                    text = _norm(raw)
                    h = hashlib.md5(text.encode()).hexdigest()
                    if h in seen:
                        skipped += 1
                        continue
                    seen.add(h)
                    # First line is the path: codegen._local_chunks reads the
                    # language off its extension and strips it back off.
                    chunks.append(f"{os.path.relpath(fp, HOME)}\n{text}")
                    per_ext[ext] = per_ext.get(ext, 0) + 1
                    kept += 1
    blob = "\0".join(chunks)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(blob)
    nbytes = len(blob.encode())
    print(f"kept {kept} files, skipped {skipped}")
    print(f"bytes: {nbytes:,}  (~{nbytes // 3500}k tokens est.)")
    print("by ext:", dict(sorted(per_ext.items(), key=lambda kv: -kv[1])))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
