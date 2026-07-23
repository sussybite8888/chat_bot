"""Image recognition, added to the expert model as a pluggable specialist.

The expert model routes every token through a per-task FFN expert (expert.py);
a **specialist** is a new expert trained *after the fact* with the whole shared
trunk frozen — so the model gains a capability without any risk to the ones it
already has. This module is the first specialist: it recognizes handwritten
digits (MNIST) and everyday photo subjects (CIFAR-10: airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, truck) — 20 labels total — trained in
under an hour, saved as a standalone checkpoint (`models/specialist-vision.pt`)
that `ExpertLM` grafts on at startup.

Images become tokens the same way game boards do — rendered as a glyph grid.
An image is pooled to at most 16x16 and each cell becomes one character: a
gray ramp `.:+#` for achromatic cells, or a hue letter `r y g c b m`
(uppercase = bright) where the cell has real color — so a tiny model gets the
color signal that separates sky from fur. Dense rows keep a whole image at
~75-280 BPE tokens, well inside the block:

    <|img|>
    BBBBBBBBBBBBBBBB
    BBBBBBBBBBBBBBBB
    ::+rrr+::+##+:::
    :+rrrrr+:+##+:::
    gggggggggggggggg
    <|cls|>

The label reads off a 20-way class head at the trailing `<|cls|>` token in a
single forward pass — no sampling — exactly how game moves read off the action
head at `<|act|>`. The whole document is routed to the vision expert.

Digits train with random polarity (light-on-dark and dark-on-light), so a
photographed pen-and-paper digit works without preprocessing; CIFAR trains
with mirror augmentation. Expectations: MNIST is ~97%; CIFAR-10 through a
16x16 glyph grid and a frozen text trunk is genuinely hard — treat object
labels as a good guess, not an oracle.

    python -m sodachat.vision train      # needs models/expert.pt (train it first)
    python -m sodachat.vision eval       # held-out accuracy, per dataset
    python -m sodachat.vision demo       # render a few test images + predictions
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .blocks import make_amp, pick_device
from .expert import (
    _MODELS,
    DEFAULT_PATH as EXPERT_PATH,
    ExpertLM,
    save_specialist,
    scaffold_specialist,
    specialist_param_groups,
)

NAME = "vision"
IMG, CLS = "<|img|>", "<|cls|>"
CIFAR_LABELS = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]
LABELS = [str(d) for d in range(10)] + CIFAR_LABELS
GLYPHS = ".:+#"    # achromatic ramp, dark -> bright
HUES = "rygcbm"    # chromatic letters by hue; uppercase = bright
CHROMA_MIN = 48.0  # channel spread below this reads as gray
MAX_SIDE = 16      # grids are pooled/resized to at most this side
DEFAULT_PATH = _MODELS / "specialist-vision.pt"
MNIST_REPO = "ylecun/mnist"
CIFAR_REPO = "uoft-cs/cifar10"


# ------------------------------------------------------------------ rendering


def _pool(px: np.ndarray) -> np.ndarray:
    """(N,H,W,3) float -> (N,S,S,3) with S <= MAX_SIDE. Square images whose
    side divides cleanly are mean-pooled (28->14, 32->16, 640->16); anything
    else goes through a PIL resize."""
    n, h, w = px.shape[:3]
    f = -(-h // MAX_SIDE)  # ceil: the smallest factor that fits
    if h == w and h % f == 0:
        return px.reshape(n, h // f, f, w // f, f, 3).mean(axis=(2, 4))
    from PIL import Image

    return np.stack([
        np.asarray(Image.fromarray(im.astype(np.uint8))
                   .resize((MAX_SIDE, MAX_SIDE), Image.BILINEAR), dtype=np.float32)
        for im in px])


def _glyph_grids(px: np.ndarray) -> list[str]:
    """(N,S,S,3) float in 0..255 -> one glyph grid per image. Achromatic cells
    use the gray ramp; colorful cells use a hue letter, uppercased when bright.
    Dense rows, deliberately: space-separated cells (the game-board style)
    tokenize to ~3x more tokens and would overflow the expert's block."""
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx, mn = px.max(axis=-1), px.min(axis=-1)
    spread = mx - mn
    d = np.where(spread == 0, 1.0, spread)
    hue = np.select([mx == r, mx == g],
                    [((g - b) / d) % 6, (b - r) / d + 2], (r - g) / d + 4)
    letters = np.array(list(HUES))[np.round(hue).astype(int) % 6]
    letters = np.where(lum >= 128, np.char.upper(letters), letters)
    gray = np.array(list(GLYPHS))[
        np.clip((lum * len(GLYPHS) / 256).astype(int), 0, len(GLYPHS) - 1)]
    grid = np.where(spread >= CHROMA_MIN, letters, gray)
    return ["\n".join("".join(row) for row in im) for im in grid]


def _render_docs(images) -> list[str]:
    """Render a stack of images (N,H,W) grayscale or (N,H,W,3) color into
    tagged docs, in chunks so a full dataset never materializes as float32."""
    docs = []
    for s in range(0, len(images), 4096):
        px = np.asarray(images[s:s + 4096], dtype=np.float32)
        if px.ndim == 3:
            px = np.repeat(px[..., None], 3, axis=-1)
        docs += [f"{IMG}\n{grid}\n{CLS}" for grid in _glyph_grids(_pool(px))]
    return docs


def render_image(pixels) -> str:
    """One image (grayscale or RGB, any size) -> its glyph grid."""
    px = np.asarray(pixels, dtype=np.float32)
    if px.ndim == 2:
        px = np.repeat(px[..., None], 3, axis=-1)
    return _glyph_grids(_pool(px[None]))[0]


def image_doc(pixels) -> str:
    # No <|end|>: docs are never packed into a stream (each is its own
    # sequence), and the class head fires at the final <|cls|> token.
    return f"{IMG}\n{render_image(pixels)}\n{CLS}"


# ----------------------------------------------------------------------- data


def _hf_images(repo: str, split: str):
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    key = "img" if "img" in ds.column_names else "image"
    images = np.stack([np.asarray(im, dtype=np.uint8) for im in ds[key]])
    return images, np.asarray(ds["label"], dtype=np.int64)


def mnist(split: str):
    """MNIST -> (images uint8 [N,28,28], labels 0-9)."""
    return _hf_images(MNIST_REPO, split)


def cifar(split: str):
    """CIFAR-10 -> (images uint8 [N,32,32,3], labels 0-9 in CIFAR_LABELS order)."""
    return _hf_images(CIFAR_REPO, split)


def _build(split: str, rng: np.random.Generator | None = None):
    """The combined corpus -> (docs, labels over LABELS). With `rng` (train),
    each digit gets a random polarity — so dark-on-light photos of pen-and-
    paper digits are in-distribution — and CIFAR is doubled with mirrors."""
    m_img, m_y = mnist(split)
    c_img, c_y = cifar(split)
    if rng is not None:
        flip = rng.random(len(m_img)) < 0.5
        m_img = np.where(flip[:, None, None], 255 - m_img, m_img)
        c_img = np.concatenate([c_img, c_img[:, :, ::-1]])
        c_y = np.concatenate([c_y, c_y])
    docs = _render_docs(m_img) + _render_docs(c_img)
    labels = np.concatenate([m_y, c_y + 10])
    return docs, labels


def _tokenize(tok, docs) -> list[np.ndarray]:
    return [np.asarray(ids, dtype=np.int32) for ids in tok.encode_batch(docs)]


def _batch(ids_list, labels, pick, device):
    """Right-pad a batch of docs; the class head reads each doc at its own
    final (<|cls|>) position, so padding after it is causally invisible."""
    seqs = [ids_list[i] for i in pick]
    width = max(len(s) for s in seqs)
    x = np.zeros((len(seqs), width), dtype=np.int64)
    last = np.empty(len(seqs), dtype=np.int64)
    for j, s in enumerate(seqs):
        x[j, : len(s)] = s
        last[j] = len(s) - 1
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(last).to(device),
            torch.from_numpy(labels[np.asarray(pick)]).to(device))


def _logits(model, task: int, x, last):
    """Class-head logits at each doc's <|cls|> position, the whole batch routed
    to the vision expert."""
    feats = model._features(x, torch.full_like(x, task))
    return model.heads[NAME](feats[torch.arange(x.shape[0], device=x.device), last])


@torch.no_grad()
def _accuracy(model, task: int, ids_list, labels, device, bs=64, pick=None) -> float:
    was_training = model.training
    model.eval()
    idxs = np.arange(len(ids_list)) if pick is None else np.asarray(pick)
    hit = 0
    for s in range(0, len(idxs), bs):
        x, last, y = _batch(ids_list, labels, idxs[s:s + bs], device)
        hit += int((_logits(model, task, x, last).argmax(-1) == y).sum())
    if was_training:
        model.train()
    return hit / max(len(idxs), 1)


# ------------------------------------------------------------------- training


def train(base=EXPERT_PATH, out=DEFAULT_PATH, steps=6000, batch_size=32, lr=3e-4,
          device=None, seed=0, eval_every=500, eval_n=1000, log=print) -> Path:
    """Train the vision specialist on top of a *frozen* expert model: gradients
    reach only the new per-block FFN expert, the 20-way class head, and the two
    new tokens' embedding rows (see `scaffold_specialist`). Chat, reading, and
    play are untouched by construction — their weights never receive a gradient.

    The new expert is seeded from the game expert (its training diet, glyph
    boards, is the closest thing to a rendered image), then specializes."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model, tok, task = scaffold_specialist(base, name=NAME, special_tokens=[IMG, CLS],
                                           n_labels=len(LABELS), device=device)
    per_block = sum(p.numel() for p in model.blocks[0].ffn.experts[task].parameters())
    trainable = (per_block * model.cfg.n_layer
                 + sum(p.numel() for p in model.heads[NAME].parameters())
                 + 2 * model.cfg.n_embd)  # the <|img|>/<|cls|> embedding rows
    log(f"specialist '{NAME}' on {Path(base).name}: expert slot {task} | "
        f"{trainable / 1e6:.1f}M trainable of {model.num_params() / 1e6:.1f}M "
        f"(shared trunk frozen) | device {device}")

    log("loading + rendering MNIST and CIFAR-10...")
    train_docs, train_y = _build("train", rng)
    test_docs, test_y = _build("test")
    log(f"tokenizing {len(train_docs):,} train + {len(test_docs):,} test images...")
    train_ids = _tokenize(tok, train_docs)
    test_ids = _tokenize(tok, test_docs)
    lens = [len(s) for s in train_ids]
    assert max(lens) <= model.cfg.block_size, "image docs overflow the block"
    assert train_ids[0][-1] == tok.token_id(CLS)
    probe = np.random.default_rng(seed + 1).permutation(len(test_ids))[:eval_n]
    seen = steps * batch_size
    log(f"docs: {min(lens)}-{max(lens)} tokens (block {model.cfg.block_size}) | "
        f"labels: {len(LABELS)} | schedule: {steps:,} steps x {batch_size} = "
        f"{seen / 1e3:.0f}k images (~{seen / len(train_ids):.1f} epochs)")

    opt = torch.optim.AdamW(specialist_param_groups(model), lr=lr, betas=(0.9, 0.95))
    warmup = min(100, max(steps // 10, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warmup),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps - warmup, 1),
                                                    eta_min=lr * 0.1)],
        milestones=[warmup])

    amp = make_amp(device)
    model.train()
    started, best = time.time(), 0.0
    for step in range(1, steps + 1):
        pick = np.random.randint(0, len(train_ids), size=batch_size)
        x, last, y = _batch(train_ids, train_y, pick, device)
        with amp.autocast():
            loss = F.cross_entropy(_logits(model, task, x, last), y)
        opt.zero_grad(set_to_none=True)
        amp.backward(loss)
        amp.step(opt, model)
        sched.step()
        if device == "mps" and step % 100 == 0:
            torch.mps.empty_cache()
        if step % eval_every == 0 or step == steps:
            acc = _accuracy(model, task, test_ids, test_y, device, pick=probe)
            mark = ""
            if acc > best:
                best, mark = acc, " <- saved"
                save_specialist(out, model, tok, name=NAME, task=task, labels=LABELS,
                                special_tokens=[IMG, CLS], steps=step, val_acc=acc,
                                meta={"glyphs": GLYPHS, "hues": HUES,
                                      "max_side": MAX_SIDE, "chroma_min": CHROMA_MIN,
                                      "source": "mnist+cifar10"})
            el = time.time() - started
            log(f"step {step:>5}/{steps} | loss {loss.item():.3f} | test acc {acc:.1%} "
                f"| {el / 60:.0f}m, eta {el / step * (steps - step) / 60:.0f}m{mark}")

    lm = ExpertLM(base, device, specialists=[out])
    task = lm.specialists[NAME]["task"]
    for name, pick in (("mnist", np.flatnonzero(test_y < 10)),
                       ("cifar10", np.flatnonzero(test_y >= 10))):
        acc = _accuracy(lm.model, task, test_ids, test_y, device, pick=pick)
        log(f"final {name} test accuracy: {acc:.2%} over {len(pick):,} images")
    log(f"done — best mixed test accuracy {best:.1%}, saved to {out}")
    return out


# ------------------------------------------------------------------ inference


def classify(lm: ExpertLM, pixels) -> tuple[str, float]:
    """Recognize one image (grayscale or RGB array, any size) with the vision
    specialist loaded in an ExpertLM. Returns (label, confidence)."""
    return lm.classify(NAME, image_doc(pixels))


def evaluate(path=DEFAULT_PATH, base=EXPERT_PATH, n=2000, device=None, log=print) -> float:
    """Held-out accuracy, reported per dataset (`n` images from each)."""
    device = device or pick_device()
    lm = ExpertLM(base, device, specialists=[path])
    task = lm.specialists[NAME]["task"]
    docs, labels = _build("test")
    ids = _tokenize(lm.tok, docs)
    accs = []
    for name, pick in (("mnist", np.flatnonzero(labels < 10)),
                       ("cifar10", np.flatnonzero(labels >= 10))):
        acc = _accuracy(lm.model, task, ids, labels, device, pick=pick[:n])
        accs.append(acc)
        log(f"{name}: {acc:.2%} over {len(pick[:n]):,} test images")
    return sum(accs) / len(accs)


def demo(path=DEFAULT_PATH, base=EXPERT_PATH, n=6, device="cpu", seed=0) -> None:
    lm = ExpertLM(base, device, specialists=[path])
    m_img, m_y = mnist("test")
    c_img, c_y = cifar("test")
    rng = np.random.default_rng(seed)
    shows = ([(m_img[i], str(m_y[i])) for i in rng.choice(len(m_img), n // 2, replace=False)]
             + [(c_img[i], CIFAR_LABELS[c_y[i]]) for i in rng.choice(len(c_img), n - n // 2, replace=False)])
    for px, want in shows:
        label, conf = classify(lm, px)
        verdict = "correct" if label == want else f"wrong (it's {want})"
        print(f"{render_image(px)}\n-> {label}  ({conf:.0%}, {verdict})\n")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Image-recognition specialist for the expert model "
                    "(MNIST digits + CIFAR-10 objects).")
    sub = p.add_subparsers(dest="cmd")
    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=6000)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--device", default=None)
    ev = sub.add_parser("eval")
    ev.add_argument("--n", type=int, default=2000, help="test images per dataset")
    ev.add_argument("--device", default=None)
    de = sub.add_parser("demo")
    de.add_argument("--n", type=int, default=6)
    de.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch_size=a.batch_size, lr=a.lr, device=a.device)
    elif a.cmd == "eval":
        evaluate(n=a.n, device=a.device)
    elif a.cmd == "demo":
        demo(n=a.n, seed=a.seed)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
