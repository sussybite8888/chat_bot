"""Gather a permissively-licensed local code corpus for the codegen specialist.
Only project roots whose OWN license is permissive (MIT/BSD/Apache/ISC); no VEX
code (per user); minified/bundled/generated files filtered out; light whitespace
normalization. Output: one NUL-separated blob of cleaned source files."""
import hashlib
import os
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
    ("restringer", None),                 # MIT
    ("agar.io-clone", None),              # MIT
    ("raspberrypi-pico", None),           # MIT
    ("unfinished-florr-clone", None),     # MIT
    ("html-css-javascript-games", None),  # MIT
    ("flooooio", None),                   # MIT (minified -> filtered to nothing)
]

EXTS = {".py", ".js", ".mjs", ".ts", ".jsx", ".c", ".cc", ".cpp", ".cxx",
        ".h", ".hpp", ".hh", ".java", ".go", ".rb", ".php", ".rs", ".lua"}
SKIP_DIRS = {"node_modules", ".git", "build", "dist", "out", "__pycache__",
             ".venv", "coverage", ".cache", "vendor", "min"}
MAX_BYTES = 250_000       # skip generated/huge single files
MAX_LINE = 2000           # skip minified/bundled (one giant line)
MAX_AVG_LINE = 250        # skip dense/minified even without one huge line


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
    if len(text) / max(len(lines), 1) > MAX_AVG_LINE:
        return False
    # crude binary/text guard
    if "\x00" in text:
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
                    chunks.append(text)
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
