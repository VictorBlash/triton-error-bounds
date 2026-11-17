#!/usr/bin/env python3
"""Training-loop check: are the Triton LayerNorm and softmax drop-in inside a real model?

The kernel checks test each kernel alone against fp64. This trains a small transformer
with the autograd wrappers in the loop and compares it with the same model built from
PyTorch's own operations. Same initial weights, same batches, mixed precision with a
gradient scaler, as a real fp16 training run would be.

Three variants of the model differ only in LayerNorm and the attention softmax:
  torch    F.layer_norm and torch.softmax as PyTorch runs them under autocast.
  emulate  PyTorch operations under the kernels' contract: fp16 in, fp32 arithmetic, fp16
           out, fp16 weight and bias. What the kernels should compute, without Triton.
  triton   layer_norm_autograd and softmax_autograd.

Checks:
  1. Gradients of every parameter on a held-out batch, at two states of the model: at
     initialization, where attention is close to uniform, and partly trained (the first
     step at which the torch variant's loss is below half its starting value), where
     attention is doing the work. Triton must agree with emulate (they differ only by
     fp32 rounding) and with torch (they also differ by the fp16 contract).
  2. Mutation tests: a softmax whose backward drops the sum(dy * y) term, and a LayerNorm
     whose backward drops the two mean terms, must each fail check 1. If one passed,
     check 1 could not see a wrong gradient from that kernel.
  3. The first steps of training: triton and emulate losses stay together. Later the
     runs drift apart, as any two runs that differ in rounding do, so the whole curves
     are printed and not held to a limit.
  4. Every variant learns the task: the loss falls and held-out accuracy is high.

The task is to output the next token of the input sequence, which needs attention.

--emulate runs emulate against torch on the CPU, with the mutants built on emulate. It
tests this script and needs neither a GPU nor Triton.

Usage: python train_check.py [--emulate] [--steps 300]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-5
VOCAB, D_MODEL, HEADS, LAYERS, SEQ_LEN, BATCH = 64, 256, 4, 2, 64, 64
LEARNING_RATE = 1e-3
POSITION_STD = 1.0  # as nn.Embedding initializes the token embeddings
LOSS_SCALE = 65536.0  # GradScaler's starting scale
SEED = 0

# Limit on the largest relative gradient gap over the parameters. It sits between two
# things measured with --emulate, which needs no kernel: the fp16 contract alone moves
# gradients by under 1e-3, and the two wrong backwards move them by 0.2 to 4.5.
GRAD_RTOL = 1e-2
EARLY_STEPS = 10
EARLY_LOSS_ATOL = 1e-3  # --emulate measures 2e-5 between emulate and torch
CURVE_STEPS = (1, 10, 20, 30, 40, 50, 100, 200, 300)
MIN_ACCURACY = 0.99

Norm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
Softmax = Callable[[torch.Tensor], torch.Tensor]
Ops = tuple[Norm, Softmax]
State = dict[str, torch.Tensor]
Batch = tuple[torch.Tensor, torch.Tensor]


def _torch_ops() -> Ops:
    return (lambda x, w, b: F.layer_norm(x, x.shape[-1:], w, b, EPS)), (lambda s: torch.softmax(s, dim=-1))


def _emulate_ops() -> Ops:
    half = torch.float16

    def norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.to(half).float(), x.shape[-1:], w.to(half).float(), b.to(half).float(), EPS).to(half)

    return norm, (lambda s: torch.softmax(s.to(half).float(), dim=-1).to(half))


def _triton_ops() -> Ops:
    from layernorm_triton import EPS as KERNEL_EPS
    from layernorm_triton_bwd import layer_norm_autograd
    from softmax_triton import softmax_autograd

    assert KERNEL_EPS == EPS, f"the kernel's eps is {KERNEL_EPS}, this script assumes {EPS}"
    half = torch.float16
    return (
        lambda x, w, b: layer_norm_autograd(x.to(half), w.to(half), b.to(half)),
        lambda s: softmax_autograd(s.to(half)),
    )


class _SoftmaxNoSum(torch.autograd.Function):
    """Wrong on purpose: the given softmax forward, with the backward y * dy."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, softmax: Softmax) -> torch.Tensor:
        y = softmax(scores)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (y,) = ctx.saved_tensors
        return (y.float() * dy.float()).to(dy.dtype), None  # bug: no - y * sum(dy * y)


class _LayerNormNoMeans(torch.autograd.Function):
    """Wrong on purpose: the given LayerNorm forward, with the backward for x w * rstd * dy."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, norm: Norm) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        return norm(x, w, b)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w = ctx.saved_tensors
        x32, dy32 = x.float(), dy.float()
        centered = x32 - x32.mean(-1, keepdim=True)
        rstd = (centered.square().mean(-1, keepdim=True) + EPS).rsqrt()
        rows = tuple(range(dy.dim() - 1))
        dx = dy32 * w.float() * rstd  # bug: no - mean(g) - xhat * mean(g * xhat), with g = dy * w
        return dx.to(x.dtype), (dy32 * centered * rstd).sum(rows).to(w.dtype), dy32.sum(rows).to(w.dtype), None


def _mutants(ops: Ops) -> dict[str, Ops]:
    norm, softmax = ops
    half = torch.float16
    return {
        "softmax backward without the sum term": (norm, lambda s: _SoftmaxNoSum.apply(s.to(half), softmax)),
        "LayerNorm backward without the mean terms": (
            lambda x, w, b: _LayerNormNoMeans.apply(x.to(half), w.to(half), b.to(half), norm),
            softmax,
        ),
    }


class Block(nn.Module):
    """Pre-norm transformer block whose LayerNorm and softmax come from `ops`."""

    def __init__(self, ops: Ops) -> None:
        super().__init__()
        self.norm, self.softmax = ops
        self.w1, self.b1 = nn.Parameter(torch.ones(D_MODEL)), nn.Parameter(torch.zeros(D_MODEL))
        self.w2, self.b2 = nn.Parameter(torch.ones(D_MODEL)), nn.Parameter(torch.zeros(D_MODEL))
        # Separate, so that the q and k gradients, which exist only through the softmax
        # backward, are compared on their own. No bias: the k bias has a zero gradient.
        self.q, self.k, self.v = (nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(3))
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, length, HEADS, D_MODEL // HEADS).transpose(1, 2)

        # .to(x.dtype) is exact. It lets the fp32 layers of a run without autocast take the fp16 result.
        h = self.norm(x, self.w1, self.b1).to(x.dtype)
        q, k, v = heads(self.q(h)), heads(self.k(h)), heads(self.v(h))
        scores = q @ k.transpose(-2, -1) * (D_MODEL // HEADS) ** -0.5
        mixed = (self.softmax(scores).to(v.dtype) @ v).transpose(1, 2).reshape(b, length, D_MODEL)
        x = x + self.proj(mixed)
        return x + self.down(F.gelu(self.up(self.norm(x, self.w2, self.b2).to(x.dtype))))


class Model(nn.Module):
    """Token and position embeddings, LAYERS blocks, a final LayerNorm and a linear head."""

    def __init__(self, ops: Ops) -> None:
        super().__init__()
        self.norm = ops[0]
        self.tokens = nn.Embedding(VOCAB, D_MODEL)
        self.positions = nn.Parameter(torch.randn(SEQ_LEN, D_MODEL) * POSITION_STD)
        self.blocks = nn.ModuleList(Block(ops) for _ in range(LAYERS))
        self.wf, self.bf = nn.Parameter(torch.ones(D_MODEL)), nn.Parameter(torch.zeros(D_MODEL))
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tokens(tokens) + self.positions
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x, self.wf, self.bf).to(x.dtype))


def make_batches(count: int, device: str) -> list[Batch]:
    """`count` batches of (tokens, targets): the target at each position is the next token, wrapping round."""
    g = torch.Generator().manual_seed(SEED + 1)
    tokens = torch.randint(0, VOCAB, (count, BATCH, SEQ_LEN), generator=g)
    return [(t.to(device), t.roll(-1, dims=-1).to(device)) for t in tokens]


class _StaticScaler:
    """A fixed loss scale behind GradScaler's interface, for the CPU self-test.

    fp16 gradients need a scaled loss: unscaled, most of them are subnormal in fp16 and lose
    their digits. GradScaler does this on CUDA and has no CPU form before torch 2.3.
    """

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * LOSS_SCALE

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.div_(LOSS_SCALE)
        optimizer.step()

    def update(self) -> None:
        pass

    def get_scale(self) -> float:
        return LOSS_SCALE


def _scaler(device: str):
    if device != "cuda":
        return _StaticScaler()
    make = getattr(torch.amp, "GradScaler", None)  # torch 2.3 or later
    return make("cuda", init_scale=LOSS_SCALE) if make else torch.cuda.amp.GradScaler(init_scale=LOSS_SCALE)


class Run:
    """One variant of the model with its optimizer and gradient scaler."""

    def __init__(self, ops: Ops, state: State, device: str) -> None:
        self.device = device
        self.model = Model(ops).to(device)
        self.model.load_state_dict(state)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.scaler = _scaler(device)

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            logits = self.model(tokens)
        return F.cross_entropy(logits.float().reshape(-1, VOCAB), targets.reshape(-1))

    def gradients(self, tokens: torch.Tensor, targets: torch.Tensor) -> State:
        """Gradient of every parameter on one batch, through the scaled loss, without taking a step."""
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(self.loss(tokens, targets)).backward()
        scale = self.scaler.get_scale()
        grads = {name: p.grad.detach().double() / scale for name, p in self.model.named_parameters()}
        self.optimizer.zero_grad(set_to_none=True)
        return grads

    def step(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        """One optimizer step. Returns the loss before the step."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(tokens, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def train(self, batches: list[Batch]) -> tuple[list[float], State | None]:
        """One step per batch.

        Returns:
            The loss before each step, and the weights after the first step whose loss was
            below half the starting loss (None if no step was).
        """
        losses: list[float] = []
        halfway = None
        for batch in batches:
            losses.append(self.step(*batch))
            if halfway is None and losses[-1] < 0.5 * losses[0]:
                halfway = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        return losses, halfway

    @torch.no_grad()
    def accuracy(self, tokens: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            predicted = self.model(tokens).argmax(-1)
        return float((predicted == targets).float().mean())


def grad_gap(a: State, b: State) -> tuple[float, str]:
    """Largest ||a - b|| / ||b|| over the parameters, and the parameter it occurs at.

    A non-finite gradient, or a gap that is not a number, counts as infinite.
    """
    worst, where = 0.0, ""
    for name, gb in b.items():
        gap = float((a[name] - gb).norm() / gb.norm())
        if not torch.isfinite(a[name]).all() or gap != gap:
            gap = float("inf")
        if gap >= worst:
            worst, where = gap, name
    return worst, where


class Checker:
    """Counts failures and holds what every gradient check needs."""

    def __init__(self, variants: dict[str, Ops], pairs: list[tuple[str, str, float]], device: str) -> None:
        self.variants, self.pairs, self.device = variants, pairs, device
        self.mutants = _mutants(variants[pairs[0][0]])
        self.failures = 0

    def report(self, ok: bool, text: str) -> None:
        self.failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {text}")

    def gradients(self, label: str, state: State, batch: Batch) -> None:
        """Checks 1 and 2 with the model's weights set to `state`."""
        grads = {name: Run(ops, state, self.device).gradients(*batch) for name, ops in self.variants.items()}
        for tested, reference, limit in self.pairs:
            gap, where = grad_gap(grads[tested], grads[reference])
            self.report(
                gap <= limit,
                f"gradients {label}, {tested} against {reference}: {gap:.2e} at {where} (limit {limit:.0e})",
            )
        _, reference, limit = self.pairs[0]
        for name, ops in self.mutants.items():
            gap, where = grad_gap(Run(ops, state, self.device).gradients(*batch), grads[reference])
            self.report(gap > limit, f"  mutant, {name}: {gap:.2e} at {where} (must exceed the limit)")


def print_curves(losses: dict[str, list[float]], tested: str, reference: str) -> None:
    """Print each variant's loss at a few steps and where the tested and reference curves differ most."""
    print("     step  " + "  ".join(f"{name:>9}" for name in losses))
    for step in CURVE_STEPS:
        if step <= len(losses[reference]):
            print(f"     {step:>4}  " + "  ".join(f"{curve[step - 1]:9.4f}" for curve in losses.values()))
    gaps = [abs(a - b) for a, b in zip(losses[tested], losses[reference], strict=True)]
    worst = max(range(len(gaps)), key=gaps.__getitem__)
    print(f"     largest difference over all steps, {tested} against {reference}: {gaps[worst]:.2e} (step {worst + 1})")


def main() -> int:
    """Run the four checks and return 0 only if all pass."""
    parser = argparse.ArgumentParser(description="Training-loop check for the Triton LayerNorm and softmax")
    parser.add_argument("--emulate", action="store_true", help="CPU only: emulate against torch, no Triton")
    parser.add_argument("--steps", type=int, default=300, help="training steps per variant")
    args = parser.parse_args()

    variants = {"torch": _torch_ops(), "emulate": _emulate_ops()}
    if args.emulate:
        device, pairs = "cpu", [("emulate", "torch", GRAD_RTOL)]
        print("EMULATED on the CPU: this tests the script, not the kernels.")
    else:
        if not torch.cuda.is_available():
            print("A CUDA GPU is required; use --emulate to test the script without one.", file=sys.stderr)
            return 1
        device = "cuda"
        variants["triton"] = _triton_ops()
        pairs = [("triton", "emulate", GRAD_RTOL), ("triton", "torch", GRAD_RTOL)]
        print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print(
        f"model: {LAYERS} blocks, d_model {D_MODEL}, {HEADS} heads, sequence {SEQ_LEN}, batch {BATCH}; "
        f"LayerNorm width {D_MODEL}, softmax width {SEQ_LEN}"
    )

    torch.manual_seed(SEED)
    start = Model(_torch_ops()).state_dict()
    batches = make_batches(args.steps + 1, device)
    held_out = batches.pop()
    check = Checker(variants, pairs, device)

    check.gradients("at initialization", start, held_out)

    losses, accuracy, halfway = {}, {}, None
    for name, ops in variants.items():
        run = Run(ops, start, device)
        losses[name], reached = run.train(batches)
        accuracy[name] = run.accuracy(*held_out)
        if name == "torch":
            halfway = reached
    tested, reference, _ = pairs[0]
    early = max(abs(a - b) for a, b in zip(losses[tested][:EARLY_STEPS], losses[reference][:EARLY_STEPS], strict=True))
    check.report(
        early <= EARLY_LOSS_ATOL,
        f"first {EARLY_STEPS} steps, {tested} against {reference}: "
        f"largest loss difference {early:.2e} (limit {EARLY_LOSS_ATOL:.0e})",
    )
    print_curves(losses, tested, reference)
    for name in variants:
        check.report(
            accuracy[name] >= MIN_ACCURACY and losses[name][-1] < losses[name][0],
            f"{name:<8} learns the task: loss {losses[name][0]:.3f} -> {losses[name][-1]:.4f} in {args.steps} steps, "
            f"held-out accuracy {accuracy[name]:.4f} (need {MIN_ACCURACY})",
        )

    if halfway is None:
        check.report(False, "the torch variant never reached half its starting loss: no partly trained state to check")
    else:
        step = next(i for i, loss in enumerate(losses["torch"]) if loss < 0.5 * losses["torch"][0])
        check.gradients(f"partly trained (torch variant after step {step + 1})", halfway, held_out)

    print(f"\n{'FAIL' if check.failures else 'OK'}: {check.failures} failures")
    return 1 if check.failures else 0


if __name__ == "__main__":
    sys.exit(main())
