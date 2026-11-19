#!/usr/bin/env python3
"""Correctness checks, mutation tests and benchmark for the Triton LayerNorm backward.

Checks per input, run on every path in layernorm_triton_bwd.PATHS:
  - torch.allclose against eager PyTorch autograd (fixed tolerance),
  - per-element error bound against an fp64 reference, for dx, dw and db,
  - bitwise determinism across repeated launches,
  - batch invariance of dx.
Plus: the autograd wrapper matches the default path, and mutation tests on real kernels
with gradients of size ~1e-3: a dx kernel that drops the xhat * mean(wdy * xhat) term, and
dw/db that skip the last row. The bound must reject both; the output reports whether
allclose also caught them.

The per-element bound uses the same error model as the forward file (mul_err, add_err,
mean_err, store_err, row_stats_with_error) and the same assumptions: no fp32 overflow or
underflow, approximate sqrt and division within APPROX_MATH_ULPS and DIV_ULPS. The sum
bounds assume no order, so one bound covers every path.

Benchmark: every path vs eager PyTorch vs torch.compile across widths.

Usage: python layernorm_bwd_checks.py [--skip-bench] [--no-compile]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

import torch
import triton
import triton.language as tl

from common import (
    U_FP32,
    MutantCase,
    add_err,
    assert_close,
    assert_within_bound,
    bound_rejects_mutant,
    compile_fresh,
    gamma,
    mean_cell,
    mean_err,
    mul_err,
    parity_tolerances,
    print_median_table,
    require_cuda,
    store_err,
    time_fn,
)
from layernorm_triton import (
    BENCH_WIDTHS,
    CASES,
    EPS,
    RowStats,
    layer_norm_eager,
    layer_norm_triton_fwd,
    make_inputs,
    row_stats_with_error,
)
from layernorm_triton_bwd import PATHS, default_path, layer_norm_autograd, layer_norm_triton_bwd

Grads = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# fp64 reference with per-element bound
# ---------------------------------------------------------------------------


def _dx_with_bound(dy64: torch.Tensor, w64: torch.Tensor, st: RowStats, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact dx and its fp32 error bound, before the final store. Follows _row_dx step by step."""
    rstd_err = st.rstd * st.rstd_rel
    xhat = st.xc * st.rstd
    xhat_err = mul_err(st.xc.abs(), st.xc_err, st.rstd, rstd_err)
    wdy = w64 * dy64
    wdy_err = U_FP32 * wdy.abs()

    # c1 = mean(xhat * wdy), c2 = mean(wdy).
    p = xhat * wdy
    c1 = p.mean(-1, keepdim=True)
    c1_err = mean_err(p, mul_err(xhat.abs(), xhat_err, wdy.abs(), wdy_err), n)
    c2 = wdy.mean(-1, keepdim=True)
    c2_err = mean_err(wdy, wdy_err, n)

    # dx = (wdy - (xhat * c1 + c2)) * rstd.
    r = xhat * c1 + c2
    r_err = add_err(r.abs(), mul_err(xhat.abs(), xhat_err, c1.abs(), c1_err), c2_err)
    s = wdy - r
    s_err = add_err(s.abs(), wdy_err, r_err)
    return s * st.rstd, mul_err(s.abs(), s_err, st.rstd, rstd_err)


def _dwdb_with_bound(dy64: torch.Tensor, st: RowStats) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact dw and db and their fp32 error bounds, before the final store.

    One fp32 term per row, then an M-term sum. Every path sums in per-program loops and
    then torch.sum over the partials; either way each term passes through at most M - 1
    roundings, so the any-order bound gamma(M - 1) applies.
    """
    m = dy64.shape[0]
    xhat = st.xc * st.rstd
    xhat_err = mul_err(st.xc.abs(), st.xc_err, st.rstd, st.rstd * st.rstd_rel)
    t = dy64 * xhat
    t_err = mul_err(dy64.abs(), 0.0, xhat.abs(), xhat_err)
    dw_err = t_err.sum(0) + gamma(m - 1) * (t.abs() + t_err).sum(0)
    db_err = gamma(m - 1) * dy64.abs().sum(0)
    return t.sum(0), dw_err, dy64.sum(0), db_err


def layer_norm_bwd_reference_with_bound(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = EPS
) -> tuple[Grads, Grads]:
    """fp64 reference gradients and per-element error bounds for the Triton backward.

    The bound follows the kernels' arithmetic: dy, x and w exact in fp32, mean and rstd as
    the forward stored them (error terms from row_stats_with_error), fp32 after that, and
    one final rounding to the output dtype. It covers every path, because they differ only
    in how sums are grouped and the sum bounds do not assume an order.

    Args:
        dy, x: shape (..., N), fp16 or bf16.
        weight: shape (N,).
        eps: same eps the forward used.

    Returns:
        ((dx, dw, db), (dx_bound, dw_bound, db_bound)), all fp64. dx and its bound have the
        shape of x; the rest have shape (N,). A correct kernel's gradient g satisfies
        |g - ref| <= bound elementwise.

    Raises:
        ValueError: see row_stats_with_error.
    """
    n = x.shape[-1]
    st = row_stats_with_error(x, eps)
    dy64 = dy.reshape(-1, n).double()
    dx, dx_err = _dx_with_bound(dy64, weight.double(), st, n)
    dw, dw_err, db, db_err = _dwdb_with_bound(dy64, st)
    refs = (dx.reshape(x.shape), dw, db)
    bounds = (
        store_err(dx, dx_err, x.dtype).reshape(x.shape),
        store_err(dw, dw_err, weight.dtype),
        store_err(db, db_err, weight.dtype),
    )
    return refs, bounds


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def eager_grads(dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> Grads:
    """Gradients from PyTorch eager autograd, in the input dtype."""
    xg = x.detach().requires_grad_(True)
    wg = weight.detach().requires_grad_(True)
    bg = bias.detach().requires_grad_(True)
    layer_norm_eager(xg, wg, bg, eps).backward(dy)
    return xg.grad, wg.grad, bg.grad


def _check_path(
    path: str,
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    want_eager: Grads,
    want_fp64: tuple[Grads, Grads],
) -> Grads:
    """Run the parity, bound, determinism and batch-invariance checks for one backward path.

    Returns:
        (dx, dw, db) from this path, for the autograd comparison.

    Raises:
        AssertionError: any check fails. The message names the path and the check.
    """
    _, mean, rstd = layer_norm_triton_fwd(x, weight, bias, eps)
    got = layer_norm_triton_bwd(dy, x, weight, mean, rstd, path=path)
    for name, t in zip(("dx", "dw", "db"), got, strict=True):
        assert torch.isfinite(t).all(), f"{path} {name} has non-finite values"

    # Fixed tolerance against eager autograd. dw and db are sums over M rows, so their
    # magnitude grows with M: rtol does the work, atol scales with M for values near zero.
    atol, rtol = parity_tolerances(x.dtype)
    m = x.reshape(-1, x.shape[-1]).shape[0]
    for name, g, r, a in zip(("dx", "dw", "db"), got, want_eager, (atol, atol * m, atol * m), strict=True):
        assert_close(f"{path} {name} vs eager", g, r, a, rtol)

    # Per-element bound against fp64. Tighter than allclose where gradients are small,
    # and catches a kernel that matches eager only because both share a bug.
    for name, g, r, b in zip(("dx", "dw", "db"), got, *want_fp64, strict=True):
        assert_within_bound(f"{path} {name} vs fp64", g, r, b)

    again = layer_norm_triton_bwd(dy, x, weight, mean, rstd, path=path)
    assert all(torch.equal(a, b) for a, b in zip(got, again, strict=True)), f"{path}: two launches differ"

    # Batch invariance of dx. dw and db are batch reductions by definition, so only dx
    # must match when a row is launched alone.
    n = x.shape[-1]
    rows, dy_rows, dx_rows = x.reshape(-1, n), dy.reshape(-1, n), got[0].reshape(-1, n)
    for i in sorted({0, m // 2, m - 1}):
        _, mean_i, rstd_i = layer_norm_triton_fwd(rows[i : i + 1], weight, bias, eps)
        dx_i, _, _ = layer_norm_triton_bwd(dy_rows[i : i + 1], rows[i : i + 1], weight, mean_i, rstd_i, path=path)
        assert torch.equal(dx_i, dx_rows[i : i + 1]), f"{path}: dx row {i} differs when launched alone"
    return got


def check_backward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> None:
    """Run every backward check for one input on every path.

    Raises:
        AssertionError: any check fails. The message names the path and the check.
    """
    g = torch.Generator(device=x.device).manual_seed(7)
    dy = torch.randn(x.shape, generator=g, device=x.device).to(x.dtype)
    want_eager = eager_grads(dy, x, weight, bias, eps)
    want_fp64 = layer_norm_bwd_reference_with_bound(dy, x, weight, eps)
    got = {path: _check_path(path, dy, x, weight, bias, eps, want_eager, want_fp64) for path in PATHS}

    # The autograd wrapper must route to the default path for this width.
    default = got[default_path(x.shape[-1])]
    xg = x.detach().requires_grad_(True)
    wg = weight.detach().requires_grad_(True)
    bg = bias.detach().requires_grad_(True)
    layer_norm_autograd(xg, wg, bg, eps).backward(dy)
    assert all(torch.equal(a, b) for a, b in zip((xg.grad, wg.grad, bg.grad), default, strict=True)), (
        "autograd wrapper gradients differ from the default path"
    )


@triton.jit
def _mutant_dx_drop_c1_kernel(dy_ptr, w_ptr, rstd_ptr, dx_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Deliberately wrong dx kernel: omits the xhat * mean(wdy * xhat) term. Test use only.

    Expects contiguous (M, N) tensors. Without that term dx no longer depends on x or mean.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    wdy = tl.where(mask, w * dy, 0.0)
    dx = (wdy - tl.sum(wdy, axis=0) / N) * rstd
    tl.store(dx_ptr + row * N + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def mutant_cases(device: str, shrink: float = 1.0) -> list[MutantCase]:
    """Planted faults: two realistic backward bugs, as three faulty outputs.

    Both run real kernels on gradients of size ~1e-3, typical late in training. At that
    scale a fixed atol of 1e-3 is as large as the gradients themselves, so allclose is
    expected to miss errors that the bound, which scales with the values, should catch.

    Mutants:
      - a dx kernel that drops the xhat * mean(wdy * xhat) term,
      - dw and db with an off-by-one that skips the last row.

    Args:
        device: device to run on.
        shrink: multiplies the upstream gradient, to plant the same faults at a smaller
            magnitude. The mutation test uses 1.

    Returns:
        One case each for dx, dw and db. Each carries every path's correct output for
        the same input.
    """
    m, n = 256, 1024
    x, w, b = make_inputs((m, n), torch.float16, device, seed=3)
    g = torch.Generator(device=device).manual_seed(9)
    dy = (1e-3 * shrink * torch.randn(x.shape, generator=g, device=device)).half()
    _, mean, rstd = layer_norm_triton_fwd(x, w, b)
    refs, bounds = layer_norm_bwd_reference_with_bound(dy, x, w)
    good = {path: layer_norm_triton_bwd(dy, x, w, mean, rstd, path=path) for path in PATHS}

    dx_bad = torch.empty_like(x)
    _mutant_dx_drop_c1_kernel[(m,)](dy, w, rstd, dx_bad, n, BLOCK_SIZE=triton.next_power_of_2(n))
    _, dw_bad, db_bad = layer_norm_triton_bwd(dy[:-1], x[:-1], w, mean[:-1], rstd[:-1], path="split")

    eager = eager_grads(dy, x, w, b, EPS)
    # dw and db are sums over the m rows, so the parity check scales atol by m for them.
    mutants = (
        ("dx drops the c1 term", dx_bad, 1),
        ("dw skips the last row", dw_bad, m),
        ("db skips the last row", db_bad, m),
    )
    return [
        MutantCase(label, {path: grads[i] for path, grads in good.items()}, bad, eager[i], refs[i], bounds[i], scale)
        for i, (label, bad, scale) in enumerate(mutants)
    ]


def check_bound_rejects_mutants(device: str) -> dict[str, bool]:
    """Mutation tests: every case from mutant_cases must fail the per-element bound.

    The correct kernels must pass on the same input first, so a rejection means the bound
    saw the bug, not that it is too tight for small gradients.

    Args:
        device: device to run on.

    Returns:
        {mutant name: True if allclose against eager also caught it}.

    Raises:
        AssertionError: the correct kernels fail the bound, or a mutant passes it.
    """
    return {case.label: bound_rejects_mutant(case) for case in mutant_cases(device)}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _backward_runner(
    forward: Callable[..., torch.Tensor], x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, dy: torch.Tensor
) -> Callable[[], None]:
    """Build a zero-arg function that runs one backward pass through `forward`'s graph.

    The forward runs once here; each call then replays only the backward. Grads are
    cleared before each call: accumulating into an existing .grad adds a read and a write
    of dx and would overstate the baseline's time.
    """
    xg = x.detach().requires_grad_(True)
    wg = w.detach().requires_grad_(True)
    bg = b.detach().requires_grad_(True)
    y = forward(xg, wg, bg)

    def run() -> None:
        xg.grad = wg.grad = bg.grad = None
        y.backward(dy, retain_graph=True)

    return run


def benchmark(device: str, dtype: torch.dtype = torch.float16, m: int = 4096, compile_baseline: bool = True) -> None:
    """Print mean backward time for every path, eager PyTorch and torch.compile.

    GB/s uses the same byte count for all columns: one read each of x and dy and one write
    of dx, the traffic any backward must move. The split path's extra reads show up as
    lower GB/s rather than as a different formula, so the columns compare directly.
    A second table gives the median of the same runs.
    """
    names = [*PATHS, "eager"] + (["compile"] if compile_baseline else [])
    title = f"backward benchmark: M={m}, dtype={dtype}"
    print(f"\n{title}, mean ms and GB/s")
    header = "  ".join(f"{nm + ' ms':>10} {'GB/s':>6}" for nm in names)
    print(f"{'N':>6}  {header}  default")
    rows = []
    for n in BENCH_WIDTHS:
        x, w, b = make_inputs((m, n), dtype, device)
        dy = torch.randn_like(x)
        _, mean, rstd = layer_norm_triton_fwd(x, w, b)
        bytes_moved = 3 * m * n * x.element_size()

        runs: list[Callable[[], object]] = [
            (lambda p=p: layer_norm_triton_bwd(dy, x, w, mean, rstd, path=p)) for p in PATHS
        ]
        runs.append(_backward_runner(layer_norm_eager, x, w, b, dy))
        if compile_baseline:
            runs.append(_backward_runner(compile_fresh(layer_norm_eager), x, w, b, dy))
        timings = [time_fn(fn) for fn in runs]
        print(f"{n:>6}  {'  '.join(mean_cell(t, bytes_moved) for t in timings)}  {default_path(n)}")
        rows.append((n, timings))
    print_median_table(title, names, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton LayerNorm backward: correctness checks and benchmark")
    parser.add_argument("--skip-bench", action="store_true", help="run correctness checks only")
    parser.add_argument("--no-compile", action="store_true", help="leave the torch.compile column out of the benchmark")
    args = parser.parse_args()

    device = require_cuda()
    if device is None:
        return 1

    for shape, dtype, offset, why in CASES:
        check_backward(*make_inputs(shape, dtype, device, offset))
        print(f"ok  {str(dtype):15} {str(shape):14} {why} ({', '.join(PATHS)})")

    for label, also in check_bound_rejects_mutants(device).items():
        print(
            f"ok  mutation test: per-element bound rejects '{label}' "
            f"(allclose {'also caught it' if also else 'missed it'})"
        )

    if not args.skip_bench:
        benchmark(device, compile_baseline=not args.no_compile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
