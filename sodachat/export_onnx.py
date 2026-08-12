"""Export the trained models to ONNX, so they can run in a browser.

This is the build step behind the **web backend** (`web/`): onnxruntime-web
loads the graphs written here and runs generation entirely client-side, with no
Python and no server round-trip. Nothing at inference time is shared with the
PyTorch path, so everything the models need to be reproduced in JavaScript —
the tokenizer, the special-token ids, the label lists — is written out
alongside the graphs as `manifest.json` (see `web/js/` for the other half).

    python -m sodachat.export_onnx                 # everything, fp32
    python -m sodachat.export_onnx --tasks text    # just what the chat UI needs
    python -m sodachat.export_onnx --quantize int8 # ~4x smaller, lossy

Three decisions shape what comes out, all of them forced by the browser:

**KV cache.** The PyTorch models re-run the whole context for every token
(`MiniGPT.generate`), which is fine on a GPU and hopeless in WASM: a reply is
12 candidates x 48 tokens, and at ~3 GFLOP per uncached forward that is minutes
of compute. The exported graphs therefore take the attention cache as an input
and return it grown by one step, turning each token into ~30 MFLOP. The JS
engine prefills the prompt once and reuses that cache across all 12 candidates.

**Explicit attention.** `CausalSelfAttention` calls `F.scaled_dot_product_attention`
with `is_causal=True`; the wrappers below spell the same computation out as
matmul/softmax with an additive mask, because the fused op has no counterpart
in the ORT web build. `--check` proves the two agree numerically.

**One graph per task, not one routed graph.** `RoutedFFN` picks a per-token
expert, but at inference every token in a sequence carries the *same* task
(chat routes to TEXT, a move to GAME, a specialist to its own slot), so each
graph is exported with one expert's weights baked in. The alternative — a
`task` input gathering over stacked experts — would put all seven experts'
weights (117M params) in every download when the chat UI only ever uses one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from .blocks import _apply_rope, _rope_cache, rms_normalize
from .expert import (
    ACT,
    ACTION_VOCAB,
    CHAT,
    END,
    GAME,
    GAME_TAG,
    READ,
    TEXT,
)
from .expert import DEFAULT_PATH as EXPERT_PATH
from .model import BOT, DIALOG_SEP, USER

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "models"

# Batch is fixed at 1: the JS engine runs one stream at a time (candidates are
# generated in sequence off a shared prefill, and MMI scores them one by one).
# Only the sequence length and the cache length are dynamic.
_BATCH = 1


# ------------------------------------------------------------ export wrappers
#
# These re-express a trained model's forward pass in the subset of torch that
# exports cleanly, reusing the *same parameter tensors* as the original modules
# (they are assigned as submodules, not copied), so there is no second source of
# truth for the weights — only for the control flow around them.


class _ExportBlock(nn.Module):
    """One transformer block, reading and writing the attention cache.

    `mlp` is whichever feed-forward this graph is being specialized to: a
    `MiniGPT` block's dense `mlp`, or the one expert of an `ExpertBlock.ffn`
    that this task routes to.
    """

    def __init__(self, ln1, attn, ln2, mlp, qk_norm: bool, n_head: int, head_dim: int):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.mlp = ln1, attn, ln2, mlp
        self.qk_norm = qk_norm
        self.n_head = n_head
        self.head_dim = head_dim

    def forward(self, x, cos, sin, bias, past_k, past_v):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.attn.qkv(h).split(C, dim=2)
        q, k, v = (
            t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        if self.qk_norm:
            q, k = rms_normalize(q), rms_normalize(k)
        # The cached keys were rotated at their own absolute positions when they
        # were written, so only the new ones are rotated here.
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5) + bias
        y = att.softmax(dim=-1) @ v
        x = x + self.attn.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x)), k, v


class _ExportGPT(nn.Module):
    """A decoder stack with a cache, plus whichever heads this graph exposes.

    Inputs  idx (1, T) int64, pos (T,) int64 — the absolute position of each
            token, past_k/past_v (L, 1, n_head, P, head_dim).
    Outputs logits (1, T, V), the extra head logits this graph carries, and
            present_k/present_v (L, 1, n_head, P + T, head_dim).

    `pos` is passed in rather than derived from the cache length because the
    caller already knows it, and a graph that never has to reason about "how
    long is the cache" exports without any shape arithmetic in it.
    """

    def __init__(
        self,
        cfg,
        tok_emb: nn.Embedding,
        blocks: list[_ExportBlock],
        ln_f,
        lm_head,
        *,
        logit_softcap: float = 0.0,
        action_head=None,
        specialist_head=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = tok_emb
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = ln_f
        self.lm_head = lm_head
        self.action_head = action_head
        self.specialist_head = specialist_head
        self.logit_softcap = logit_softcap
        head_dim = cfg.n_embd // cfg.n_head
        # RoPE angles are fixed, so the whole table ships as a constant and each
        # step gathers the rows it needs. Buffers, not parameters: excluded from
        # the tied-weight dedup and from quantization.
        cos, sin = _rope_cache(
            cfg.block_size, head_dim, cfg.rope_theta, torch.device("cpu"), torch.float32
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def output_names(self) -> list[str]:
        names = ["logits"]
        if self.action_head is not None:
            names.append("action_logits")
        if self.specialist_head is not None:
            names.append("head_logits")
        return [*names, "present_k", "present_v"]

    def forward(self, idx, pos, past_k, past_v):
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[pos], self.rope_sin[pos]
        # Additive causal mask over the cache *and* this chunk. Built once from
        # absolute positions, which sidesteps needing the cache length as a
        # number: a query at position p may look at every key up to p.
        k_pos = torch.arange(past_k.shape[3] + idx.shape[1], device=idx.device)
        bias = torch.where(
            pos.unsqueeze(1) >= k_pos.unsqueeze(0),
            0.0,
            torch.finfo(torch.float32).min,
        )
        present_k, present_v = [], []
        for i, block in enumerate(self.blocks):
            x, k, v = block(x, cos, sin, bias, past_k[i], past_v[i])
            present_k.append(k)
            present_v.append(v)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.logit_softcap:
            cap = self.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        out = [logits]
        if self.action_head is not None:
            out.append(self.action_head(x))
        if self.specialist_head is not None:
            out.append(self.specialist_head(x))
        return (*out, torch.stack(present_k), torch.stack(present_v))

    def empty_cache(self, past: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (
            self.cfg.n_layer,
            _BATCH,
            self.cfg.n_head,
            past,
            self.cfg.n_embd // self.cfg.n_head,
        )
        return torch.zeros(shape), torch.zeros(shape)


def _wrap_minigpt(model, cfg) -> _ExportGPT:
    head_dim = cfg.n_embd // cfg.n_head
    blocks = [
        _ExportBlock(b.ln1, b.attn, b.ln2, b.mlp, cfg.qk_norm, cfg.n_head, head_dim)
        for b in model.blocks
    ]
    return _ExportGPT(
        cfg, model.tok_emb, blocks, model.ln_f, model.head,
        logit_softcap=cfg.logit_softcap,
    )


def _wrap_expert(model, cfg, task: int, specialist: str | None) -> _ExportGPT:
    head_dim = cfg.n_embd // cfg.n_head
    blocks = [
        _ExportBlock(
            b.ln1, b.attn, b.ln2, b.ffn.experts[task], cfg.qk_norm, cfg.n_head, head_dim
        )
        for b in model.blocks
    ]
    return _ExportGPT(
        cfg,
        model.tok_emb,
        blocks,
        model.ln_f,
        model.lm_head,
        # ExpertGPT.forward applies no softcap whatever the config says, so
        # neither does its graph.
        logit_softcap=0.0,
        action_head=model.action_head if task == GAME else None,
        specialist_head=(
            model.heads[specialist] if specialist and specialist in model.heads else None
        ),
    )


# -------------------------------------------------------------------- exporting


def _export_one(wrapper: _ExportGPT, path: Path, log) -> None:
    """Trace `wrapper` to ONNX with sequence length and cache length dynamic."""
    wrapper.eval()
    # Example shapes are deliberately >1 in every dynamic axis: torch.export
    # specializes on sizes of 0 and 1, which would bake them into the graph.
    idx = torch.randint(0, wrapper.cfg.vocab_size, (_BATCH, 4), dtype=torch.long)
    pos = torch.arange(3, 7, dtype=torch.long)
    past_k, past_v = wrapper.empty_cache(3)
    seq = torch.export.Dim("seq", min=1, max=wrapper.cfg.block_size)
    cache = torch.export.Dim("cache", min=0, max=wrapper.cfg.block_size)
    dynamic_shapes = ({1: seq}, {0: seq}, {3: cache}, {3: cache})

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (idx, pos, past_k, past_v),
            str(path),
            input_names=["idx", "pos", "past_k", "past_v"],
            output_names=wrapper.output_names,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
            optimize=True,
            # One self-contained file per graph. The exporter defaults to
            # spilling weights into a sidecar `.onnx.data`, which onnxruntime-web
            # can only load if the page hands it the extra file by name — a
            # second fetch to keep in sync for no benefit at these sizes (the
            # largest graph here is ~120MB, far under the 2GB protobuf limit).
            external_data=False,
        )
    log(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


def _quantize(path: Path, log) -> None:
    """Dynamic int8 weights — roughly a quarter of the download.

    Lossy, and not covered by `--check`: the point of the flag is that a 123 MB
    fp32 graph is not something a browser should be asked to fetch.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = path.with_suffix(".fp32.onnx")
    path.rename(tmp)
    try:
        quantize_dynamic(tmp, path, weight_type=QuantType.QInt8)
        log(f"  quantized {path.name} -> {path.stat().st_size / 1e6:.1f} MB")
    finally:
        tmp.unlink(missing_ok=True)


# fp32 logits reproduce to ~3e-5 across every graph here; anything approaching
# this is a bug in the export, not accumulated rounding.
_PARITY_TOLERANCE = 1e-3


@torch.no_grad()
def _check(wrapper: _ExportGPT, path: Path, log) -> float:
    """Compare the graph against the wrapper it came from, prefill *and* decode.

    Run in two shapes that matter and are easy to get wrong: a prefill with an
    empty cache (a dynamic axis of length 0), and a single-token step reading a
    cache — the two calls the JS engine actually makes. Raises if they disagree:
    a graph that is close but not equal is a browser quietly running a different
    model from the terminal.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def run(idx, pos, past_k, past_v):
        outputs = session.run(
            None,
            {
                "idx": idx.numpy(),
                "pos": pos.numpy(),
                "past_k": past_k.numpy(),
                "past_v": past_v.numpy(),
            },
        )
        return dict(zip(wrapper.output_names, outputs))

    worst = 0.0
    prompt = torch.randint(0, wrapper.cfg.vocab_size, (_BATCH, 17), dtype=torch.long)
    pos = torch.arange(17, dtype=torch.long)
    empty_k, empty_v = wrapper.empty_cache(0)

    ref = wrapper(prompt, pos, empty_k, empty_v)
    got = run(prompt, pos, empty_k, empty_v)
    worst = max(worst, float((ref[0] - torch.from_numpy(got["logits"])).abs().max()))

    # ...then one decode step off the cache the prefill just produced.
    step = torch.randint(0, wrapper.cfg.vocab_size, (_BATCH, 1), dtype=torch.long)
    step_pos = torch.tensor([17], dtype=torch.long)
    ref_step = wrapper(step, step_pos, ref[-2], ref[-1])
    got_step = run(
        step,
        step_pos,
        torch.from_numpy(got["present_k"]),
        torch.from_numpy(got["present_v"]),
    )
    worst = max(
        worst, float((ref_step[0] - torch.from_numpy(got_step["logits"])).abs().max())
    )

    # A cached step must also agree with the uncached forward it replaces —
    # this is what catches a RoPE offset or mask that is subtly wrong.
    full = torch.cat([prompt, step], dim=1)
    ref_full = wrapper(full, torch.arange(18, dtype=torch.long), empty_k, empty_v)
    drift = float((ref_full[0][:, -1] - ref_step[0][:, -1]).abs().max())
    worst = max(worst, drift)
    log(f"  parity: max |Δlogit| {worst:.2e} (cache vs full-context {drift:.2e})")
    # Written as `not (worst < t)` so a NaN fails rather than slipping through.
    if not (worst < _PARITY_TOLERANCE):
        raise SystemExit(
            f"{path.name} does not reproduce PyTorch: max |Δlogit| {worst:.3e} "
            f"exceeds {_PARITY_TOLERANCE:.0e}"
        )
    return worst


# --------------------------------------------------------------------- manifest


def _tokenizer_specials(tok, names: dict[str, str]) -> dict[str, int]:
    """Resolve the special tokens the JS side needs by name, skipping any a
    given checkpoint's vocabulary does not have."""
    out = {}
    for key, token in names.items():
        try:
            out[key] = tok.token_id(token)
        except (AttributeError, KeyError):
            # Not a *added* token. A plain character still has an id — "\n" is
            # stored byte-level as "Ċ" and so misses a name lookup — so fall
            # back to encoding it, the way the inference wrappers resolve it.
            ids = tok.encode(token)
            if len(ids) == 1:
                out[key] = ids[0]
    return out


def export_chat(
    out: Path, model_path: Path | None, log
) -> tuple[dict, _ExportGPT] | None:
    """The `mini` backend: a dense MiniGPT with a single LM head."""
    from .model import DEFAULT_MODEL_PATH, load_checkpoint

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        log(f"skipping chat: no checkpoint at {path}")
        return None
    log(f"chat model {path.name}")
    model, tok = load_checkpoint(path, device="cpu")
    model.eval()
    if tok.kind != "bpe":
        raise SystemExit(
            f"{path.name} has a {tok.kind} tokenizer; the web backend ships the "
            "byte-level BPE vocabulary only (see web/js/tokenizer.js)"
        )
    (out / "tokenizer-chat.json").write_text(tok.to_payload()["json"])
    wrapper = _wrap_minigpt(model, model.cfg)
    _export_one(wrapper, out / "chat.onnx", log)
    return {
        "chat": {
            "file": "chat.onnx",
            "tokenizer": "tokenizer-chat.json",
            "kind": "chat",
            "label": "mini — chat GPT trained from scratch on SODA",
            "params": model.num_params(),
            "config": _config_summary(model.cfg),
            "specials": _tokenizer_specials(
                tok, {"newline": "\n", "dialog_sep": DIALOG_SEP}
            ),
            "prompt": {"user": USER, "bot": BOT, "prefix": ""},
            "max_new_tokens": 48,
        }
    }, wrapper


def export_expert(
    out: Path, model_path: Path | None, tasks: set[str] | None, log
) -> tuple[dict, list]:
    """The expert model, one graph per routed task.

    Every graph comes from a *single* fully-attached `ExpertLM`, so they all
    share one vocabulary and one trunk — a specialist's token ids depend on
    what else is attached, and would not line up if each were exported alone.
    """
    from .expert import ExpertLM

    path = Path(model_path) if model_path else EXPERT_PATH
    if not path.exists():
        log(f"skipping expert: no checkpoint at {path}")
        return {}, []
    log(f"expert model {path.name}")
    lm = ExpertLM(path, device="cpu")
    lm.model.eval()
    (out / "tokenizer-expert.json").write_text(lm.tok.to_payload()["json"])

    specials = _tokenizer_specials(
        lm.tok,
        {
            "newline": "\n",
            "chat": CHAT,
            "read": READ,
            "game": GAME_TAG,
            "act": ACT,
            "end": END,
        },
    )

    # TEXT and GAME are the base model's two experts; everything else is a
    # specialist that `ExpertLM` grafted on, and knows its own slot.
    wanted: list[tuple[str, int, str | None]] = [("text", TEXT, None), ("game", GAME, None)]
    wanted += [(name, spec["task"], name) for name, spec in lm.specialists.items()]

    entries, wrappers = {}, []
    for name, task, specialist in wanted:
        if tasks is not None and name not in tasks:
            continue
        info = lm.specialists.get(specialist or "", {})
        wrapper = _wrap_expert(lm.model, lm.model.cfg, task, specialist)
        _export_one(wrapper, out / f"expert-{name}.onnx", log)
        wrappers.append((f"expert-{name}", wrapper))
        entries[f"expert-{name}"] = {
            "file": f"expert-{name}.onnx",
            "tokenizer": "tokenizer-expert.json",
            "kind": _task_kind(name, info),
            "label": _task_label(name, info),
            "task": task,
            "params": sum(p.numel() for p in wrapper.parameters()),
            "config": _config_summary(lm.model.cfg),
            "specials": specials,
            # The chat frame the expert was trained on is tagged; MiniGPT's is not.
            "prompt": {"user": USER, "bot": BOT, "prefix": f"{CHAT}\n"},
            "max_new_tokens": 48,
            "labels": info.get("labels", []),
            "actions": lm.actions if task == GAME else [],
            "frame": _specialist_entry(lm, specialist) if specialist else {},
            "meta": {k: v for k, v in info.items() if k not in ("task", "labels")},
        }
    return entries, wrappers


# How each specialist frames its prompt at inference, and how it samples —
# mirrored from the module that trained it (vision.render_image / code.doc /
# route.doc / reason.think / codegen.generate) so the JS side has one place to
# read the format from instead of five hardcoded copies. `{input}` is where the
# caller's text goes; a classifier reads its head at the final token, so its
# frame must end on the marker the head was trained at.
_SPECIALIST_FRAMES: dict[str, dict] = {
    "vision": {"template": "<|img|>\n{input}\n<|cls|>"},
    "code": {"template": "<|code|>\n{input}\n<|cls|>", "max_chars": 800},
    "route": {
        "template": "<|route|>\n{input}\n<|dest|>",
        "max_chars": 400,
        "collapse_whitespace": True,
    },
    "reason": {
        "template": "<|reason|> {input}\n<|think|>",
        "collapse_whitespace": True,
        "split_on": "<|answer|>",
        "sampling": {
            "max_new_tokens": 220,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
        },
    },
    "codegen": {
        # The language header the snippets were trained under; without one the
        # continuation drifts between the six languages in the stream.
        "template": "{header}{input}",
        "headers": {
            "python": "# python\n", "ruby": "# ruby\n", "javascript": "// javascript\n",
            "java": "// java\n", "php": "// php\n", "go": "// go\n",
        },
        "sampling": {
            "max_new_tokens": 160,
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.95,
            "repetition_penalty": 1.15,
        },
    },
}


def _specialist_tokens(name: str) -> list[str]:
    """The special tokens a specialist introduced, recovered from its file (the
    attached model keeps labels and task id, but not the token strings)."""
    from .expert import _MODELS

    path = _MODELS / f"specialist-{name}.pt"
    if not path.exists():
        return []
    return list(torch.load(path, map_location="cpu", weights_only=True)["token_ids"])


def _specialist_entry(lm, name: str) -> dict:
    """The prompt frame and token ids a specialist needs on the JS side."""
    frame = dict(_SPECIALIST_FRAMES.get(name, {"template": "{input}"}))
    frame["tokens"] = _tokenizer_specials(
        lm.tok, {t: t for t in _specialist_tokens(name)}
    )
    return frame


def _task_kind(name: str, info: dict) -> str:
    if name == "text":
        return "chat"
    if name == "game":
        return "move"
    return "classify" if info.get("kind", "classify") == "classify" else "generate"


def _task_label(name: str, info: dict) -> str:
    return {
        "text": "expert — chat/read, routed to the text expert",
        "game": "expert — board to move, read off the action head",
    }.get(name, f"specialist — {name}")


def _config_summary(cfg) -> dict:
    return {
        "vocab_size": cfg.vocab_size,
        "block_size": cfg.block_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "head_dim": cfg.n_embd // cfg.n_head,
    }


def export(
    out: Path = DEFAULT_OUT,
    *,
    models: set[str] = frozenset({"chat", "expert"}),
    tasks: set[str] | None = None,
    quantize: str = "none",
    check: bool = True,
    chat_path: Path | None = None,
    expert_path: Path | None = None,
    log=print,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    wrappers: list[tuple[str, _ExportGPT]] = []

    if "chat" in models:
        result = export_chat(out, chat_path, log)
        if result:
            entry, wrapper = result
            entries.update(entry)
            wrappers.append(("chat", wrapper))
    if "expert" in models:
        expert_entries, expert_wrappers = export_expert(out, expert_path, tasks, log)
        entries.update(expert_entries)
        wrappers.extend(expert_wrappers)

    if not entries:
        raise SystemExit("nothing exported — no checkpoints found (see README)")

    if check:
        log("checking graphs against PyTorch...")
        for name, wrapper in wrappers:
            log(f"  {name}")
            _check(wrapper, out / entries[name]["file"], log)

    if quantize == "int8":
        log("quantizing...")
        for name in entries:
            _quantize(out / entries[name]["file"], log)

    for name in entries:
        entries[name]["bytes"] = (out / entries[name]["file"]).stat().st_size
        entries[name]["quantization"] = quantize

    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "generator": "sodachat.export_onnx",
                "batch": _BATCH,
                "models": entries,
            },
            indent=2,
        )
        + "\n"
    )
    total = sum(e["bytes"] for e in entries.values()) / 1e6
    log(f"wrote {manifest} — {len(entries)} model(s), {total:.0f} MB total")
    if quantize == "none" and total > 100:
        log("  (--quantize int8 cuts that to roughly a quarter, at some quality cost)")
    return manifest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="sodachat.export_onnx",
        description="Export the trained models to ONNX for the browser (web/).",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument(
        "--models",
        default="chat,expert",
        help="which checkpoints to export: chat, expert, or both (default: both)",
    )
    p.add_argument(
        "--tasks",
        default="all",
        help="expert graphs to write: all, or a comma list of text,game,"
        "<specialist> — one graph is ~120MB fp32, and the chat UI needs only 'text'",
    )
    p.add_argument("--quantize", choices=("none", "int8"), default="none")
    p.add_argument(
        "--no-check",
        action="store_true",
        help="skip the PyTorch/onnxruntime parity check",
    )
    p.add_argument("--chat-path", type=Path, default=None)
    p.add_argument("--expert-path", type=Path, default=None)
    a = p.parse_args(argv)
    export(
        a.out,
        models=set(a.models.split(",")),
        tasks=None if a.tasks == "all" else set(a.tasks.split(",")),
        quantize=a.quantize,
        check=not a.no_check,
        chat_path=a.chat_path,
        expert_path=a.expert_path,
    )


if __name__ == "__main__":
    main()
