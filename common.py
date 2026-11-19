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

import statistics
import sys
from collections.abc import Callable, Sequence
from typing import NamedTuple

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
# CUDA documents __expf, which has the same form, at 2 + floor(|1.173 x|) true ulps (CUDA
# C++ Programming Guide, Mathematical Functions appendix, intrinsic table). The argument
# term exists because the scaled exponent x * log2e is rounded, so the error grows with |x|.
# The budget applies those two numbers in this model's unit, 2**-23 relative to the result,
# which is 1 to 2 true ulps. That matters: probe_math.py measured Triton's exp on a T4 at up
# to 1.09 times the documented figure in true ulps, and at 0.50 of this budget (RESULTS.md).
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


class Timing(NamedTuple):
    """One benchmark cell: statistics over the timed runs of one do_bench call, in ms."""

    mean: float
    median: float
    runs: int


def time_fn(fn: Callable[[], object]) -> Timing:
    """Time `fn` with triton.testing.do_bench and summarize every timed run.

    do_bench returns the mean unless asked for another statistic. Asking for every run
    lets one measurement give the mean, which the main tables print, and the median, which
    shows whether a few slow runs pulled that mean up. do_bench sets the number of runs to
    100 ms over its estimate of one iteration, at least 1, and the estimate includes the
    cache flush it does before each run (about 1.2 ms on a T4). So a fast kernel gets about
    80 runs and a slow one few: 12 at 6.6 ms, 1 at 65 ms (RESULTS.md).

    Returns:
        Mean, median and number of timed runs.
    """
    import triton  # here, not at the top: this module also loads where Triton is not installed

    times = triton.testing.do_bench(fn, return_mode="all")
    return Timing(statistics.mean(times), statistics.median(times), len(times))


def mean_cell(timing: Timing | None, nbytes: int) -> str:
    """Main-table cell: mean ms and effective GB/s. None, a path that cannot run, prints dashes."""
    if timing is None:
        return f"{'-':>10} {'-':>6}"
    return f"{timing.mean:10.4f} {nbytes / (timing.mean * 1e-3) / 1e9:6.1f}"


def print_median_table(title: str, names: Sequence[str], rows: Sequence[tuple[int, Sequence[Timing | None]]]) -> None:
    """Print the companion table: median ms, median / mean and timed runs for every cell.

    The cells come from the same timed runs as the main table. A ratio well under 1 means
    a few slow runs raised the mean there.
    """
    print(f"\n{title}, median ms, median / mean and timed runs")
    print(f"{'N':>6}  " + "  ".join(f"{nm + ' ms':>10} {'ratio':>6} {'runs':>5}" for nm in names))
    for n, timings in rows:
        cells = [
            f"{'-':>10} {'-':>6} {'-':>5}" if t is None else f"{t.median:10.4f} {t.median / t.mean:6.3f} {t.runs:5d}"
            for t in timings
        ]
        print(f"{n:>6}  {'  '.join(cells)}")


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


def strided_views(base: torch.Tensor) -> dict[str, torch.Tensor]:
    """Two non-contiguous (M, N) views of a (2M, 2N) tensor, one for each way a launcher treats strides.

    "every other row" is unit-stride along each row, so a launcher copies nothing and
    passes the row stride to the kernel. "every other column" is not, so a launcher has
    to copy it before the kernel can index it. Either way the elements between the
    view's own are valid memory holding plausible numbers: reading them raises nothing.

    Raises:
        AssertionError: a view is contiguous or the wrong shape, so it would test nothing.
    """
    m, n = base.shape[0] // 2, base.shape[1] // 2
    views = {"every other row": base[::2, :n], "every other column": base[:m, ::2]}
    for name, view in views.items():
        assert view.shape == (m, n) and not view.is_contiguous(), f"{name}: not a strided (M, N) view"
    return views


class MutantCase(NamedTuple):
    """One planted fault, with everything needed to judge it.

    Each mutation test builds its cases once, so the test and probe_tolerances.py judge
    the same tensors.
    """

    label: str
    good: dict[str, torch.Tensor]  # the correct kernels' output on the same input, by path
    bad: torch.Tensor  # the faulty output
    eager: torch.Tensor  # PyTorch eager's result, in the kernel's dtype
    ref: torch.Tensor  # fp64 reference
    bound: torch.Tensor  # per-element bound on |correct output - ref|
    atol_scale: int  # the parity check multiplies atol by this: the row count for dw and db, else 1


def bound_rejects_mutant(case: MutantCase) -> bool:
    """Mutation test: the per-element bound must accept the correct outputs and reject the fault.

    The correct kernels are checked on the same input first, so a rejection means the
    bound saw the fault, not that it is too tight for this input.

    Returns:
        True if the fixed-tolerance parity check (torch.allclose against eager with
        parity_tolerances) also caught the fault, False if only the bound did.

    Raises:
        AssertionError: the bound rejected a correct output or accepted the faulty one.
    """
    for path, good in case.good.items():
        assert_within_bound(f"{case.label}: correct {path} output", good, case.ref, case.bound)
    assert ((case.bad.double() - case.ref).abs() > case.bound).any(), f"per-element bound accepted mutant: {case.label}"
    atol, rtol = parity_tolerances(case.bad.dtype)
    return not torch.allclose(case.bad.double(), case.eager.double(), atol=atol * case.atol_scale, rtol=rtol)
