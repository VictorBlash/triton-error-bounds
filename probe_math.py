#!/usr/bin/env python3
"""Measure the GPU's fp32 exp, division and 1 / sqrt against the budgets in common.py.

The per-element bounds charge approximate GPU math a budget: EXP_ULPS and EXP_ARG_ULPS
for tl.exp, DIV_ULPS for one division, APPROX_MATH_ULPS for 1 / sqrt. The kernels store
fp16 or bf16, and that rounding is hundreds of times larger than an fp32 error of a few
ulps. So the kernel checks can show a budget is large enough, but not how much of it is
used. This probe runs each operation alone in fp32, written the way the kernels write it,
and compares every element with fp64.

Numbers per row:
  - max ulps: the largest error in true fp32 ulps of the exact result, the unit NVIDIA's
    documentation uses.
  - used: the largest error / budget over the elements, with the budget exactly as the
    error model charges it. The model's unit is 2**-23 relative to the result, which is
    between 1 and 2 true ulps, so `used` can be well below max ulps / budget ulps.
  - exp only, vs doc: the largest error in true ulps / (2 + floor(1.173 |x|)), the bound
    NVIDIA documents for __expf. Above 1 means that bound does not describe this
    lowering. It is reported, not failed on: the model's budget is what the checks use.
  - exp only, worst x: the argument of the element that uses most of its budget.
A probe fails if any element is over its budget.

--emulate runs the same measurement on the CPU with torch's own fp32 math in place of the
kernels. It tests this script, not the GPU, and needs neither a GPU nor Triton.

Usage: python probe_math.py [--emulate]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import types
from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # --emulate on a machine without Triton: the kernels are never launched
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn  # type: ignore[attr-defined]
    tl = types.ModuleType("triton.language")

from common import (
    APPROX_MATH_ULPS,
    DIV_ULPS,
    EXP_ARG_ULPS,
    EXP_ULPS,
    FP32_MIN_SUBNORMAL,
    ULP_REL_FP32,
    exp_err,
    require_cuda,
)

BLOCK = 1024
SEED = 11

# Softmax only ever takes exp of x - max, which is at most 0. Below -80 the result nears
# the fp32 subnormal range, where a relative budget stops meaning anything.
EXP_MIN = -80.0
EXP_POINTS = 1 << 23
EXP_BIN_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 80.0)

# Operands of the division and the square root, spread evenly over 40 binades.
POW2_RANGE = 20.0
POW2_POINTS = 1 << 22

# A result this many model units off must be reported as over every budget here.
SELF_TEST_UNITS = 64


@triton.jit
def _exp_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    """y = exp(x) in fp32, the way the softmax kernels call it."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.exp(x), mask=mask)


@triton.jit
def _recip_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    """y = 1.0 / x in fp32, the softmax kernels' division by the denominator."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    tl.store(y_ptr + offs, 1.0 / x, mask=mask)


@triton.jit
def _rsqrt_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    """y = 1.0 / sqrt(x) in fp32, the LayerNorm kernel's rstd."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    tl.store(y_ptr + offs, 1.0 / tl.sqrt(x), mask=mask)


def _launch(kernel: Callable, x: torch.Tensor) -> torch.Tensor:
    """Run a one-input fp32 kernel over a 1-D tensor and return its output."""
    y = torch.empty_like(x)
    n = x.numel()
    kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK_SIZE=BLOCK)
    return y


# name: (kernel, the same operation in torch fp32 for --emulate)
OPERATIONS: dict[str, tuple[Callable, Callable[[torch.Tensor], torch.Tensor]]] = {
    "exp": (_exp_kernel, torch.exp),
    "recip": (_recip_kernel, lambda x: 1.0 / x),
    "rsqrt": (_rsqrt_kernel, lambda x: 1.0 / torch.sqrt(x)),
}


def fp32_ulp(ref: torch.Tensor) -> torch.Tensor:
    """Spacing of fp32 at |ref|, for exact fp64 values."""
    _, exponent = torch.frexp(ref.abs())
    return torch.ldexp(torch.ones_like(ref), exponent - 24).clamp(min=FP32_MIN_SUBNORMAL)


def measure(got: torch.Tensor, ref: torch.Tensor, budget_rel: torch.Tensor | float) -> tuple[float, float]:
    """Compare fp32 results with exact ones.

    Args:
        got: fp32 results.
        ref: exact results in fp64, same shape.
        budget_rel: the error model's relative error budget, a scalar or one per element.

    Returns:
        (largest error in true fp32 ulps, largest error / budget). A NaN result counts as
        infinite error.
    """
    err = abs_error(got, ref)
    return float((err / fp32_ulp(ref)).max()), float((err / (budget_rel * ref.abs())).max())


def abs_error(got: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """|got - ref| in fp64, with a NaN result counted as infinite error."""
    inf = float("inf")
    return torch.nan_to_num((got.double() - ref).abs(), nan=inf, posinf=inf)


def _inputs(points: int, low: float, high: float, device: str) -> torch.Tensor:
    """`points` fp32 values spread uniformly over [low, high], the same on every run."""
    g = torch.Generator(device=device).manual_seed(SEED)
    return low + (high - low) * torch.rand(points, generator=g, device=device, dtype=torch.float32)


def probe_exp(run: Callable[[torch.Tensor], torch.Tensor], device: str) -> float:
    """Print the exp table by |x| range and return the largest error / budget."""
    x = _inputs(EXP_POINTS, EXP_MIN, 0.0, device)
    got = run(x)
    x64 = x.double()
    ref = torch.exp(x64)
    err = abs_error(got, ref)
    ulps = err / fp32_ulp(ref)
    used = err / (exp_err(x64.abs(), 0.0) * ref)
    vs_doc = ulps / (EXP_ULPS + torch.floor(EXP_ARG_ULPS * x64.abs()))
    print(f"\nexp(x): {EXP_POINTS} points, x in [{EXP_MIN:g}, 0]. Budget {EXP_ULPS} + {EXP_ARG_ULPS} |x| model units.")
    print(f"{'|x| in':>12} {'max ulps':>9} {'vs doc':>7} {'used':>7} {'worst x':>9}")
    worst_used, worst_doc = 0.0, 0.0
    for low, high in itertools.pairwise(EXP_BIN_EDGES):
        sel = (x64.abs() >= low) & (x64.abs() < high)
        if not sel.any():
            continue
        at = int(used[sel].argmax())
        bin_used, bin_doc = float(used[sel][at]), float(vs_doc[sel].max())
        worst_used, worst_doc = max(worst_used, bin_used), max(worst_doc, bin_doc)
        print(
            f"{f'[{low:g}, {high:g})':>12} {float(ulps[sel].max()):9.2f} {bin_doc:7.3f} "
            f"{bin_used:7.3f} {float(x64[sel][at]):9.3f}"
        )
    if worst_doc > 1.0:
        print(
            f"note: the error reaches {worst_doc:.2f} times NVIDIA's documented __expf bound of "
            f"{EXP_ULPS} + floor({EXP_ARG_ULPS} |x|) true ulps. The budget above is what the checks rely on."
        )
    return worst_used


def probe_pow2(name: str, run: Callable[[torch.Tensor], torch.Tensor], device: str) -> tuple[float, float]:
    """Run recip or rsqrt over operands spread across 40 binades.

    Returns:
        (largest error in true ulps, largest error / budget).
    """
    x = torch.exp2(_inputs(POW2_POINTS, -POW2_RANGE, POW2_RANGE, device))
    got = run(x)
    x64 = x.double()
    if name == "recip":
        return measure(got, 1.0 / x64, DIV_ULPS * ULP_REL_FP32)
    return measure(got, 1.0 / torch.sqrt(x64), APPROX_MATH_ULPS * ULP_REL_FP32)


def self_test(device: str) -> bool:
    """Return True if a result SELF_TEST_UNITS model units off is reported as over budget.

    Without this, a mistake in `measure` that always reported a small number would look
    like a GPU that is very accurate.
    """
    x = _inputs(1 << 12, -1.0, 0.0, device)
    ref = torch.exp(x.double())
    bad = (ref * (1.0 + SELF_TEST_UNITS * ULP_REL_FP32)).float()
    _, used = measure(bad, ref, exp_err(x.double().abs(), 0.0))
    nan = bad.clone()
    nan[0] = float("nan")
    _, used_nan = measure(nan, ref, exp_err(x.double().abs(), 0.0))
    return used > 1.0 and used_nan == float("inf")


def main() -> int:
    """Run every probe and return 0 only if each stayed inside its budget."""
    parser = argparse.ArgumentParser(description="fp32 exp, division and 1/sqrt error against the model's budgets")
    parser.add_argument("--emulate", action="store_true", help="use torch's CPU fp32 math instead of the kernels")
    args = parser.parse_args()

    if args.emulate:
        device = "cpu"
        run = {name: torch_op for name, (_, torch_op) in OPERATIONS.items()}
        print("EMULATED on the CPU: this tests the probe, not the GPU.")
    else:
        cuda = require_cuda()
        if cuda is None:
            return 1
        device = cuda
        run = {name: (lambda x, k=kernel: _launch(k, x)) for name, (kernel, _) in OPERATIONS.items()}

    failed = []
    if not self_test(device):
        print(f"FAIL  self-test: a result {SELF_TEST_UNITS} model units off was not reported as over budget")
        failed.append("self-test")
    else:
        print(f"ok  self-test: a result {SELF_TEST_UNITS} model units off, and a NaN, are reported as over budget")

    used = probe_exp(run["exp"], device)
    print(f"{'ok ' if used <= 1.0 else 'FAIL'} exp: the worst element uses {used:.3f} of its budget")
    if used > 1.0:
        failed.append("exp")

    print(f"\n1 / x and 1 / sqrt(x): {POW2_POINTS} points each, x in [2^-{POW2_RANGE:g}, 2^{POW2_RANGE:g}].")
    for name, label, budget in (
        ("recip", "1 / x", DIV_ULPS),
        ("rsqrt", "1 / sqrt(x)", APPROX_MATH_ULPS),
    ):
        ulps, used = probe_pow2(name, run[name], device)
        print(
            f"{'ok ' if used <= 1.0 else 'FAIL'} {label}: max {ulps:.2f} ulps, "
            f"the worst element uses {used:.3f} of its budget of {budget} model units"
        )
        if used > 1.0:
            failed.append(name)

    if failed:
        print(f"\nOver budget: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
