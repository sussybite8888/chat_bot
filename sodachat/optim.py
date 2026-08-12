"""Optimizers and learning-rate schedules — the training-loop toolkit.

Model-agnostic in the same way `blocks.py` is: nothing here knows what is
being trained, only how to update it.

    Muon                the orthogonalized-momentum optimizer for hidden layers
    build_optimizers    the Muon/AdamW parameter split every trainer wants
    lr_multiplier       warmup-stable-decay (and cosine) schedules
    set_lr              drive several optimizers off one schedule

Why two optimizers rather than one: Muon is only valid for the 2D weights of
hidden layers. Embeddings, the LM head, and every scalar/vector parameter stay
on AdamW — that split is an empirical requirement reported by Muon's author,
not a convenience, and skipping it costs quality.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Muon's defaults. The learning rate is an order of magnitude above AdamW's
# because an orthogonalized update has unit-ish singular values regardless of
# the gradient's magnitude — it is a direction, not a step size, so the step
# size is entirely `lr`.
#
# 0.0025, not the 0.02 the nanoGPT speedrun uses: that figure is for a 768-dim
# model, and this one is 384-dim. Swept at n_embd=384 over 600 steps of
# DailyDialog, val loss was 3.369 / 3.353 / 3.370 / 3.384 / 3.401 / 3.427 at
# lr 0.0015 / 0.0025 / 0.005 / 0.0075 / 0.02 / 0.04 — a flat optimum around
# 0.0025 and clearly worse by 0.02. The optimum is only weakly constrained by
# a run that short, so re-sweep before committing to a long one.
MUON_LR = 0.0025
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5


def _newton_schulz(G: torch.Tensor, steps: int = MUON_NS_STEPS, eps: float = 1e-7):
    """Approximately orthogonalize `G` — replace it with the `U @ V.T` of its
    own SVD, without ever computing an SVD.

    A fixed quintic polynomial is applied to the singular values `steps` times.
    The coefficients (Jordan, 2024) are deliberately *not* the ones that
    converge fastest to the true answer: they are tuned to pull small singular
    values up hard, at the price of never settling exactly on 1. Orthogonalized
    momentum is what the optimizer wants and slop in the third decimal is not,
    which is why five bfloat16 iterations are enough to be worth their cost.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)  # the iteration only converges for ||X|| <= 1
    # The iteration works on the smaller Gram matrix; transpose so it is the
    # short side that gets squared.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if transposed else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz (Jordan et al., 2024).

    Plain SGD-momentum, except that each update matrix is replaced by its
    nearest semi-orthogonal matrix before being applied. Momentum built from
    many similar gradients is dominated by a few directions; orthogonalizing
    spreads the step evenly over all of them, so the rarely-updated directions
    of a weight matrix stop being starved. In the nanoGPT speedrun this was
    worth ~1.35x fewer tokens to the same loss than AdamW, and it is now the
    optimizer behind Kimi K2 and GLM-4.5.

    Pass **only 2D hidden-layer weights** (see `build_optimizers`). The Newton-
    Schulz iteration costs five extra matmuls per matrix per step, which is
    paid back many times over at any reasonable batch size but is not free —
    it scales as model_dim / batch_tokens, so small batches pay the most.
    """

    def __init__(
        self,
        params,
        lr: float = MUON_LR,
        momentum: float = MUON_MOMENTUM,
        nesterov: bool = True,
        ns_steps: int = MUON_NS_STEPS,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon takes 2D parameters only, got {tuple(p.shape)} — "
                        "route embeddings, heads and norms to AdamW"
                    )
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(p)
                buf.lerp_(p.grad, 1.0 - group["momentum"])
                # Nesterov: step from where the momentum is already heading.
                update = (
                    p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                )
                update = _newton_schulz(update, group["ns_steps"])
                # After orthogonalization the update's size depends on the
                # matrix shape, not the gradient, so rescale to keep tall and
                # wide matrices moving by comparable amounts.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)
        return loss


def build_optimizers(
    model: nn.Module,
    *,
    lr: float = 3e-4,
    muon_lr: float = MUON_LR,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    use_muon: bool = True,
) -> list[torch.optim.Optimizer]:
    """Split `model`'s parameters by what each one needs, and return the
    optimizers to step together.

    Three kinds of parameter, three treatments:

    * **hidden 2D weights** (attention and FFN matrices) — Muon, decayed.
    * **embeddings and the LM head** — AdamW, decayed. Muon is not valid here:
      a row of an embedding table is a token, not a direction in a hidden
      space, and orthogonalizing across rows mixes unrelated tokens.
    * **everything 1D** (norm gains, biases) — AdamW, *never* decayed. Decay on
      an RMSNorm gain shrinks a parameter whose whole job is to set a scale.

    Weight tying is handled: `named_parameters` yields a shared weight once, so
    a tied head follows its embedding into the AdamW group.
    """
    embedding_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for p in module.parameters(recurse=False)
    }

    hidden, decayed, undecayed = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            undecayed.append(p)
        elif p.ndim > 2 or id(p) in embedding_ids or "head" in name:
            decayed.append(p)
        else:
            hidden.append(p)

    optimizers: list[torch.optim.Optimizer] = []
    if use_muon and hidden:
        optimizers.append(Muon(hidden, lr=muon_lr, weight_decay=weight_decay))
    elif hidden:
        decayed = decayed + hidden

    groups = [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": undecayed, "weight_decay": 0.0},
    ]
    optimizers.append(
        torch.optim.AdamW(
            [g for g in groups if g["params"]], lr=lr, betas=betas
        )
    )
    return optimizers


def lr_multiplier(
    step: int,
    total: int,
    *,
    warmup: int,
    schedule: str = "wsd",
    decay_frac: float = 0.2,
    min_frac: float = 0.0,
) -> float:
    """The learning-rate factor at `step` (0-based) of a `total`-step run.

    `wsd` (warmup-stable-decay) holds the peak rate for the bulk of training
    and only decays over the final `decay_frac`. It matches or beats cosine,
    and unlike cosine it does not bake the total step count into every earlier
    step — so a run can be *extended* without having spent its schedule, and
    one stable checkpoint can be branched and decayed separately per task.
    The cost is that a WSD run stopped early never decayed, and an undecayed
    checkpoint is worse than a decayed one; stop at `total`, not before.

    `cosine` is the previous behaviour, kept for comparison runs.
    """
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if schedule == "cosine":
        progress = min((step - warmup) / max(total - warmup, 1), 1.0)
        return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    if schedule != "wsd":
        raise ValueError(f"unknown schedule {schedule!r} (expected wsd or cosine)")
    decay_steps = max(int(total * decay_frac), 1)
    stable_until = total - decay_steps
    if step < stable_until:
        return 1.0
    progress = min((step - stable_until) / decay_steps, 1.0)
    return 1.0 - (1.0 - min_frac) * progress


def set_lr(optimizers, multiplier: float) -> None:
    """Scale every param group's learning rate by `multiplier`, relative to the
    rate it was built with (remembered on first call)."""
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            base = group.setdefault("initial_lr", group["lr"])
            group["lr"] = base * multiplier
