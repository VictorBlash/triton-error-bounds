"""Shared pieces for the kernels in this repo.

1. The fp32 rounding-error model behind the per-element correctness bounds. Each helper
   takes exact magnitudes (fp64, non-negative) and bounds on how far the kernel's fp32
   values are from exact, and returns a bound for the next result, including that
   operation's own rounding. Sums assume no order, so one bound covers every launch
   configuration. A fused multiply-add rounds once where these count two roundings, so a
   compiler contraction stays inside the bound. "Beyond atol and rtol" (arXiv 2610.38983)
   is the reference for the approach. The constants below are the assumptions, stated
   once; the bound holds if no fp32 intermediate overflows or underflows and the
   approximate GPU math stays within its budget.
2. Fixed tolerances for the eager-mode parity checks.
3. Launcher and benchmark helpers shared by the kernels.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import torch

U_FP32 = 2.0**-24
# One ulp relative to the value: at most 2**-23 for fp32, twice the unit roundoff.
ULP_REL_FP32 = 2.0**-23
FP32_MIN_SUBNORMAL = 2.0**-149
UNIT_ROUNDOFF = {torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8}
HALF_MIN_SUBNORMAL = {torch.float16: 2.0**-25, torch.bfloat16: 2.0**-134}

# GPU math that can lower to approximate instructions. Each budget is in ulps of the
# result, beyond whatever error the operation inherits from its inputs. Raise one if a GPU
# or Triton version disagrees, and record why in RESULTS.md.
APPROX_MATH_ULPS = 4  # tl.sqrt followed by an fp32 division, combined (LayerNorm's rstd)
DIV_ULPS = 2  # one fp32 division: div.full.f32, documented at 2 ulps
# fp32 tl.exp lowers to ex2.approx(x * log2e) on NVIDIA (Triton ElementwiseOpToLLVM.cpp).
# CUDA documents that same sequence as __expf: max error 2 + floor(|1.173 x|) ulps (CUDA
# C++ Programming Guide, Mathematical Functions appendix, intrinsic table). The argument
# term exists because the scaled exponent x * log2e is rounded, so the error grows with |x|.
EXP_ULPS = 2  # fixed part per tl.exp call
EXP_ARG_ULPS = 1.173  # per unit of |argument|, per tl.exp call

# Fixed tolerances for the eager-mode parity check. They suit fp16, whose spacing near 1.0
# is about 1e-3. bf16 spacing near 1.0 is 8x larger, so the same tolerance would flag
# correct kernels that round one ulp differently from eager. bf16 tolerances are scaled by 8.
PARITY_ATOL = 1e-3
PARITY_RTOL = 1e-3


def parity_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return (atol, rtol) for the eager parity check, scaled to the dtype's spacing."""
    scale = UNIT_ROUNDOFF[dtype] / UNIT_ROUNDOFF[torch.float16]
    return PARITY_ATOL * scale, PARITY_RTOL * scale


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------


def gamma(k: int) -> float:
    """Higham's gamma_k for fp32: worst-case relative error of k chained roundings."""
    return k * U_FP32 / (1.0 - k * U_FP32)


def mul_err(
    a: torch.Tensor | float, ea: torch.Tensor | float, b: torch.Tensor | float, eb: torch.Tensor | float
) -> torch.Tensor:
    """Bound for fl(a' * b') where a = |exact a|, b = |exact b|, |a' - a| <= ea, |b' - b| <= eb.

    At least one argument must be a tensor; every caller passes one.
    """
    return a * eb + b * ea + ea * eb + U_FP32 * (a + ea) * (b + eb)


def add_err(s: torch.Tensor, ea: torch.Tensor | float, eb: torch.Tensor | float) -> torch.Tensor:
    """Bound for fl(a' + b') where s = |exact a + b| and ea, eb bound the input errors."""
    e = ea + eb
    return e + U_FP32 * (s + e)


def sum_err(t: torch.Tensor, et: torch.Tensor) -> torch.Tensor:
    """Bound for fl(sum(t')) over the last dim, for any summation order.

    Args:
        t: exact terms (signed), shape (..., N).
        et: bound on each computed term's error, same shape.

    Returns:
        Bound of shape (..., 1). Adding a masked zero lane is exact, so N can be the row
        width even when some lanes are masked.
    """
    n = t.shape[-1]
    return et.sum(-1, keepdim=True) + gamma(n - 1) * (t.abs() + et).sum(-1, keepdim=True)


def mean_err(t: torch.Tensor, et: torch.Tensor, n: int) -> torch.Tensor:
    """Bound for fl(sum(t') / n) over the last dim: sum_err plus one fp32 division."""
    e = sum_err(t, et) / n
    return e + DIV_ULPS * ULP_REL_FP32 * (t.sum(-1, keepdim=True).abs() / n + e)


def exp_err(d_abs: torch.Tensor, ed: torch.Tensor | float, calls: int = 1, muls: int = 0) -> torch.Tensor:
    """Relative error bound on a computed exp(d) product, given exact |d| and argument error.

    Covers a chain of `calls` exp evaluations multiplied together with `muls` fp32
    multiplies, as in online softmax where exp(x - m_t) is later rescaled by
    exp(m_t - m_{t+1}) and so on. Each call costs EXP_ULPS plus EXP_ARG_ULPS per unit of
    its own |argument|. The chained arguments telescope to d when the running maxes are
    monotone, so the argument terms sum to EXP_ARG_ULPS * |d| over the whole chain, and
    one absolute argument error `ed` covers every link.

    Args:
        d_abs: exact |d|, where the chain computes exp(d).
        ed: bound on the total absolute error in the arguments.
        calls: number of exp evaluations in the chain, at least 1.
        muls: number of fp32 multiplies joining them.

    Returns:
        Relative error bound, same shape as d_abs.
    """
    arg = torch.expm1(ed + EXP_ARG_ULPS * ULP_REL_FP32 * (d_abs + ed))
    return (1.0 + arg) * (1.0 + EXP_ULPS * ULP_REL_FP32) ** calls * (1.0 + U_FP32) ** muls - 1.0


def store_err(v: torch.Tensor, ev: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Bound after rounding an fp32 value to dtype: round to nearest plus the subnormal floor."""
    return ev + UNIT_ROUNDOFF[dtype] * (v.abs() + ev) + HALF_MIN_SUBNORMAL[dtype]


# ---------------------------------------------------------------------------
# Launcher and benchmark helpers
# ---------------------------------------------------------------------------


def num_warps_for(block_size: int) -> int:
    """About 256 elements per warp, between 1 and 8 warps.

    Depends only on the block size, so every launch with the same N runs the same
    reduction order. Batch invariance needs that. 8 is the cap because, on a T4, 8 warps
    beat 16 and 32 for a row kernel at N = 32768 and tied at 16384 (RESULTS.md).
    """
    return min(max(block_size // 256, 1), 8)


def compile_fresh(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """torch.compile of `fn`, specialized to the next shape it sees.

    Resets the compiler cache first so each benchmark width gets its own kernels instead
    of hitting the recompile limit. Default inductor mode. The first call compiles, which
    takes a few seconds; do_bench's warmup absorbs it.
    """
    torch.compiler.reset()
    return torch.compile(fn, dynamic=False)


def require_cuda() -> str | None:
    """Return "cuda" if a GPU is available, else print why not and return None."""
    if not torch.cuda.is_available():
        print("A CUDA or ROCm GPU is required: Triton compiles kernels for the GPU.", file=sys.stderr)
        return None
    return "cuda"


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def assert_close(name: str, got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float) -> None:
    """torch.allclose with a message that names the check and the worst difference."""
    diff = (got.double() - want.double()).abs()
    assert torch.allclose(got.double(), want.double(), atol=atol, rtol=rtol), (
        f"{name}: max |diff| {diff.max().item():.3e} (atol={atol}, rtol={rtol})"
    )


def bound_ratio(got: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor) -> float:
    """Largest |got - ref| / bound over all elements: how much of the bound the worst element uses.

    At most 1 is inside the bound. An exact element counts as 0 even where its bound is
    zero. A NaN, or any error over a zero bound, counts as infinity so it cannot read as a pass.
    """
    err = (got.double() - ref).abs()
    ratio = torch.where(err == 0, torch.zeros_like(err), err / bound)
    inf = float("inf")
    return float(torch.nan_to_num(ratio, nan=inf, posinf=inf).max())


def assert_within_bound(name: str, got: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor) -> float:
    """Assert |got - ref| <= bound elementwise; the message names the worst element.

    Returns:
        bound_ratio of the result, so a passing check can also report its slack.
    """
    excess = (got.double() - ref).abs() - bound
    worst = int(excess.argmax())
    assert (excess <= 0).all(), (
        f"{name}: per-element bound exceeded at flat index {worst}: "
        f"got={got.flatten()[worst].item():.6g}, ref={ref.flatten()[worst].item():.6g}, "
        f"bound={bound.flatten()[worst].item():.3e}"
    )
    return bound_ratio(got, ref, bound)
