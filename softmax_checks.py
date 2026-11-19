#!/usr/bin/env python3
"""Correctness checks, mutation tests and benchmark for the Triton softmax.

Checks per input, on every path that takes the width:
  forward:  torch.allclose against eager, per-element bound against fp64, bitwise
            determinism, batch invariance.
  backward: the same four, computed from the kernel's own saved y.
Plus: the autograd wrapper matches the default paths, and two mutation tests on real
kernels:
  - an online forward that forgets to rescale the running sum when the max rises, on a
    row whose max rises in every tile;
  - a backward that drops the sum(dy * y) term, on gradients of size ~1e-3 with a nonzero
    mean, where the true dx is near zero and the mutant's is not.
The bound must reject both; the output reports whether allclose also caught them.

Each case also prints the largest |error| / bound per path. Rounding the result to fp16
or bf16 makes up almost all of the bound, so on correct kernels that ratio sits close to
1 and says little about the fp32 budgets. probe_math.py measures those directly.

Error model: common.py, with exp_err for the exponential. The online forward's
denominator terms pass through up to one rescale per later tile plus the lane merge, so
that bound takes the tile count; the numerator is always a single exp.

Usage: python softmax_checks.py [--skip-bench] [--no-compile]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import (
    DIV_ULPS,
    FP32_MIN_SUBNORMAL,
    U_FP32,
    ULP_REL_FP32,
    MutantCase,
    add_err,
    assert_close,
    assert_same_bits,
    assert_within_bound,
    bound_rejects_mutant,
    compile_fresh,
    exp_err,
    mean_cell,
    mul_err,
    parity_tolerances,
    print_median_table,
    require_cuda,
    store_err,
    strided_views,
    sum_err,
    time_fn,
)
from softmax_triton import (
    BWD_PATHS,
    FWD_PATHS,
    default_bwd_path,
    default_fwd_path,
    fits_block,
    online_tiles,
    softmax_autograd,
    softmax_triton,
    softmax_triton_bwd,
    tile_for,
)

BENCH_WIDTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def softmax_eager(x: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dimension using PyTorch eager mode."""
    return F.softmax(x, dim=-1)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def make_inputs(
    shape: tuple[int, ...], dtype: torch.dtype, device: str, kind: str = "normal", seed: int = 0
) -> torch.Tensor:
    """Build a seeded input of one of several kinds.

    Args:
        shape: tensor shape, softmax over the last dim.
        dtype: fp16 or bf16.
        device: device to build on.
        kind: "normal" (standard normal), "wide" (normal x 20: a spread of about 60, so
            most exps underflow), "masked" (second half of every row is -inf, like a
            causal attention mask), "constant" (uniform rows, y must be 1/N), or "ramp"
            (increasing along the row, so the online kernel's max rises in every tile).
        seed: generator seed.

    Raises:
        ValueError: unknown kind, or "masked" with N < 2 (that would be a whole -inf row).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    n = shape[-1]
    x = torch.randn(shape, generator=g, device=device)
    if kind == "wide":
        x = x * 20.0
    elif kind == "masked":
        if n < 2:
            raise ValueError("masked rows need N >= 2")
        x[..., n // 2 :] = float("-inf")
    elif kind == "constant":
        x = torch.full(shape, 0.5, device=device)
    elif kind == "ramp":
        x = torch.linspace(-4.0, 4.0, n, device=device).expand(shape).contiguous()
    elif kind != "normal":
        raise ValueError(f"unknown input kind {kind!r}")
    return x.to(dtype)


# (shape, dtype, kind, what the case protects)
CASES: list[tuple[tuple[int, ...], torch.dtype, str, str]] = [
    ((1, 1), torch.float16, "normal", "N=1: output must be exactly 1"),
    ((17, 7), torch.float16, "normal", "N below one warp, not a power of two"),
    ((64, 127), torch.bfloat16, "normal", "bf16, mask covers 1 lane"),
    ((128, 1000), torch.float16, "normal", "common non-power-of-two width"),
    ((2, 3, 768), torch.float16, "normal", "3-D input flattened to rows"),
    ((32, 4096), torch.bfloat16, "normal", "bf16 at a typical width"),
    ((8, 5000), torch.float16, "normal", "mask covers 3192 of 8192 lanes"),
    ((2, 20000), torch.float16, "normal", "near the single-block limit"),
    ((2, 40000), torch.float16, "normal", "above the single-block limit"),
    ((64, 1024), torch.float16, "wide", "spread of about 60: most exps underflow"),
    ((64, 1024), torch.float16, "masked", "second half of each row is -inf"),
    ((64, 1024), torch.float16, "constant", "uniform rows: y must be 1/N"),
    ((16, 8192), torch.float16, "ramp", "max rises in every tile of the online kernel"),
]


# ---------------------------------------------------------------------------
# fp64 references with per-element bounds
# ---------------------------------------------------------------------------


def softmax_reference_with_bound(x: torch.Tensor, tiles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 softmax and a per-element error bound for the Triton forward kernels.

    The bound follows the kernels' arithmetic: x exact in fp32, the max exact, one fp32
    subtraction per element, exp, an fp32 sum in any order, one division for 1/l, one
    multiply per element, one rounding to the output dtype. -inf inputs give exactly 0
    with no error.

    Args:
        x: input of shape (..., N), fp16 or bf16.
        tiles: 0 for the block kernel. For the online kernel, the number of column tiles:
            a denominator term from one tile is rescaled at each later tile and once more
            in the lane merge, so it passes through up to tiles + 1 exp calls and tiles
            multiplies. The numerator is always one exp.

    Returns:
        (y_ref, bound), both fp64 with the shape of x.

    Raises:
        ValueError: the denominator's error is not small relative to the denominator, so
            the first-order model for 1/l does not apply.
    """
    n = x.shape[-1]
    x64 = x.reshape(-1, n).double()
    m = x64.max(dim=-1, keepdim=True).values
    d = x64 - m
    finite = torch.isfinite(d)
    d_abs = torch.where(finite, -d, torch.zeros_like(d))
    e = torch.exp(d)
    denom = e.sum(dim=-1, keepdim=True)
    y_ref = e / denom

    # The subtraction x - m rounds once, relative to |d|. Below about exp(-87) the fp32
    # result is subnormal and the relative model breaks, so every term also gets an
    # absolute floor of one fp32 subnormal step.
    ed = U_FP32 * d_abs

    def term_err(calls: int, muls: int) -> torch.Tensor:
        rel = exp_err(d_abs, ed, calls, muls)
        return torch.where(finite, e * rel + FP32_MIN_SUBNORMAL, torch.zeros_like(e))

    err_l = sum_err(e, term_err(tiles + 1 if tiles else 1, tiles))
    rel_l = err_l / denom
    if (rel_l >= 0.5).any():
        raise ValueError("denominator error is too large for this error model")
    # inv = 1 / l': a relative input error r moves 1/l by at most r / (1 - r), then the
    # division rounds.
    rel_inv = (1.0 + rel_l / (1.0 - rel_l)) * (1.0 + DIV_ULPS * ULP_REL_FP32) - 1.0

    inv = 1.0 / denom
    err_y = mul_err(e, term_err(1, 0), inv, inv * rel_inv)
    bound = store_err(y_ref, err_y, x.dtype)
    return y_ref.reshape(x.shape), bound.reshape(x.shape)


def softmax_bwd_reference_with_bound(dy: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 softmax backward from the kernel's own y, with a per-element error bound.

    y is the saved fp16/bf16 output, exact in fp32, so the bound covers only the backward
    arithmetic: one multiply per element, an fp32 sum in any order, one subtraction and
    one multiply per element, one rounding to the output dtype.

    Args:
        dy, y: shape (..., N), same dtype.

    Returns:
        (dx_ref, bound), both fp64 with the shape of y.
    """
    n = y.shape[-1]
    y64 = y.reshape(-1, n).double()
    dy64 = dy.reshape(-1, n).double()
    p = dy64 * y64
    s = p.sum(dim=-1, keepdim=True)
    t = dy64 - s
    dx_ref = y64 * t

    err_s = sum_err(p, mul_err(dy64.abs(), 0.0, y64.abs(), 0.0))
    err_t = add_err(t.abs(), 0.0, err_s)
    err_dx = mul_err(y64.abs(), 0.0, t.abs(), err_t)
    bound = store_err(dx_ref, err_dx, y.dtype)
    return dx_ref.reshape(y.shape), bound.reshape(y.shape)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _row_indices(m: int) -> list[int]:
    return sorted({0, m // 2, m - 1})


def _format_ratios(ratios: dict[str, float]) -> str:
    """Render {path: ratio} as "block 0.912, online 0.913"."""
    return ", ".join(f"{path} {ratio:.3f}" for path, ratio in ratios.items())


def check_forward(x: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Run every forward check for one input on every path that takes its width.

    Returns:
        ({path: y}, {path: largest |y - fp64| / bound}) for the paths that ran.

    Raises:
        AssertionError: any check fails. The message names the path and the check.
    """
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    m = rows.shape[0]
    y_eager = softmax_eager(x)
    atol, rtol = parity_tolerances(x.dtype)
    got, ratios = {}, {}
    for path in FWD_PATHS:
        if path == "block" and not fits_block(n):
            continue
        y = softmax_triton(x, path=path)
        assert torch.isfinite(y).all(), f"{path}: non-finite output"
        assert_close(f"{path} y vs eager", y, y_eager, atol, rtol)
        y_ref, bound = softmax_reference_with_bound(x, 0 if path == "block" else online_tiles(n))
        ratios[path] = assert_within_bound(f"{path} y vs fp64", y, y_ref, bound)
        assert torch.equal(y, softmax_triton(x, path=path)), f"{path}: two launches differ"
        y_rows = y.reshape(-1, n)
        for i in _row_indices(m):
            single = softmax_triton(rows[i : i + 1], path=path)
            assert torch.equal(single, y_rows[i : i + 1]), f"{path}: row {i} differs when launched alone"
        got[path] = y
    return got, ratios


def check_backward(x: torch.Tensor, y: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Run every backward check for one input on every path that takes its width.

    Args:
        x: the forward input, used for the eager and autograd comparisons.
        y: the default forward path's output, what the autograd wrapper saves.

    Returns:
        ({path: dx}, {path: largest |dx - fp64| / bound}) for the paths that ran.

    Raises:
        AssertionError: any check fails. The message names the path and the check.
    """
    n = x.shape[-1]
    m = x.reshape(-1, n).shape[0]
    g = torch.Generator(device=x.device).manual_seed(7)
    dy = torch.randn(x.shape, generator=g, device=x.device).to(x.dtype)

    xg = x.detach().requires_grad_(True)
    softmax_eager(xg).backward(dy)
    dx_eager = xg.grad
    dx_ref, bound = softmax_bwd_reference_with_bound(dy, y)
    atol, rtol = parity_tolerances(x.dtype)

    got, ratios = {}, {}
    y_rows, dy_rows = y.reshape(-1, n), dy.reshape(-1, n)
    for path in BWD_PATHS:
        if path == "block" and not fits_block(n):
            continue
        dx = softmax_triton_bwd(dy, y, path=path)
        assert torch.isfinite(dx).all(), f"{path}: non-finite dx"
        assert_close(f"{path} dx vs eager", dx, dx_eager, atol, rtol)
        ratios[path] = assert_within_bound(f"{path} dx vs fp64", dx, dx_ref, bound)
        assert torch.equal(dx, softmax_triton_bwd(dy, y, path=path)), f"{path}: two launches differ"
        dx_rows = dx.reshape(-1, n)
        for i in _row_indices(m):
            single = softmax_triton_bwd(dy_rows[i : i + 1], y_rows[i : i + 1], path=path)
            assert torch.equal(single, dx_rows[i : i + 1]), f"{path}: dx row {i} differs when launched alone"
        got[path] = dx

    # The autograd wrapper must route to the default paths for this width.
    xa = x.detach().requires_grad_(True)
    y_auto = softmax_autograd(xa)
    y_auto.backward(dy)
    assert torch.equal(y_auto, y), "autograd forward differs from the default path"
    assert torch.equal(xa.grad, got[default_bwd_path(n)]), "autograd backward differs from the default path"
    return got, ratios


@triton.jit
def _mutant_online_no_rescale_kernel(x_ptr, y_ptr, N, BLOCK_N: tl.constexpr):
    """Deliberately wrong online forward: never rescales the running sum. Test use only.

    Expects contiguous (M, N) tensors. Correct only when a lane's max is set by its first
    tile; wrong whenever the max rises later, which is the whole point of the rescale.
    """
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * N
    y_row = y_ptr + row * N
    neg_inf = float("-inf")
    m_run = tl.full([BLOCK_N], neg_inf, dtype=tl.float32)
    l_run = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + cols, mask=cols < N, other=neg_inf).to(tl.float32)
        m_new = tl.maximum(m_run, x)
        p = tl.where(x == neg_inf, 0.0, tl.exp(x - m_new))
        l_run = l_run + p  # bug: should be l_run * exp(m_run - m_new) + p
        m_run = m_new
    m = tl.max(m_run, axis=0)
    denom = tl.sum(l_run * tl.where(m_run == neg_inf, 0.0, tl.exp(m_run - m)), axis=0)
    inv = 1.0 / denom
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=neg_inf).to(tl.float32)
        tl.store(y_row + cols, (tl.exp(x - m) * inv).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mutant_bwd_drop_sum_kernel(dy_ptr, y_ptr, dx_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Deliberately wrong backward: dx = y * dy, without the sum(dy * y) term. Test use only."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(dx_ptr + row * N + cols, (y * dy).to(dx_ptr.dtype.element_ty), mask=mask)


def no_rescale_case(device: str) -> MutantCase:
    """Planted fault: an online forward without the rescale, on rows whose max rises in every tile.

    Args:
        device: device to run on.
    """
    m, n = 16, 8192
    x = make_inputs((m, n), torch.float16, device, kind="ramp")
    y_ref, bound = softmax_reference_with_bound(x, online_tiles(n))
    y_bad = torch.empty_like(x)
    _mutant_online_no_rescale_kernel[(m,)](x, y_bad, n, BLOCK_N=tile_for(n))
    return MutantCase(
        label="online forward skips the rescale",
        good={"online": softmax_triton(x, path="online")},
        bad=y_bad,
        eager=softmax_eager(x),
        ref=y_ref,
        bound=bound,
        atol_scale=1,
    )


def dropped_sum_case(device: str, shrink: float = 1.0) -> MutantCase:
    """Planted fault: a backward without the sum(dy * y) term.

    dy has a nonzero mean and a small spread, so the true dx = y * (dy - s) is near zero
    while the mutant's y * dy is not. Both are far below atol = 1e-3, so allclose is
    expected to miss this.

    Args:
        device: device to run on.
        shrink: multiplies the upstream gradient, to plant the same fault at a smaller
            magnitude. The mutation test uses 1.
    """
    m, n = 256, 1024
    x = make_inputs((m, n), torch.float16, device, seed=3)
    y = softmax_triton(x)
    g = torch.Generator(device=device).manual_seed(9)
    dy = (1e-3 * shrink * (1.0 + 0.1 * torch.randn(x.shape, generator=g, device=device))).half()
    dx_ref, bound = softmax_bwd_reference_with_bound(dy, y)
    dx_bad = torch.empty_like(y)
    _mutant_bwd_drop_sum_kernel[(m,)](dy, y, dx_bad, n, BLOCK_SIZE=triton.next_power_of_2(n))
    xg = x.detach().requires_grad_(True)
    softmax_eager(xg).backward(dy)
    return MutantCase(
        label="backward drops the sum(dy * y) term",
        good={path: softmax_triton_bwd(dy, y, path=path) for path in BWD_PATHS},
        bad=dx_bad,
        eager=xg.grad,
        ref=dx_ref,
        bound=bound,
        atol_scale=1,
    )


def mutant_cases(device: str) -> list[MutantCase]:
    """Planted faults: two realistic softmax bugs, each on an input where it shows.

    Args:
        device: device to run on.

    Returns:
        The forward case, then the backward case. Each carries the correct kernels'
        output for the same input.
    """
    return [no_rescale_case(device), dropped_sum_case(device)]


def check_bound_rejects_mutants(device: str) -> dict[str, bool]:
    """Mutation tests: every case from mutant_cases must fail the per-element bound.

    The correct kernels must pass on the same inputs first, so a rejection means the
    bound saw the bug, not that it is too tight for the input.

    Args:
        device: device to run on.

    Returns:
        {mutant name: True if allclose against eager also caught it}.

    Raises:
        AssertionError: the correct kernels fail the bound, or a mutant passes it.
    """
    return {case.label: bound_rejects_mutant(case) for case in mutant_cases(device)}


def check_strided_inputs(device: str) -> None:
    """Layout test: strided inputs must give the same bits as contiguous copies, on every path.

    Every other case is contiguous, so nothing else exercises the row strides the
    launchers pass to the kernels, or the copy they make when a row is not unit-stride.
    Getting either wrong reads the view's neighbours in the larger tensor: no error,
    plausible numbers. The backward is compared view against copy, so its y does not have
    to be a softmax output.

    Args:
        device: device to run on.

    Raises:
        AssertionError: an output from the views differs from the contiguous copies'.
    """
    m, n = 6, 5000  # fits one block, and is two column tiles for the online and tiled kernels
    xs = strided_views(make_inputs((2 * m, 2 * n), torch.float16, device), m, n)
    # dy comes from a wider tensor than y, so the two "every other row" views have different
    # row strides and a launcher that passed one in place of the other is caught.
    dys = strided_views(make_inputs((2 * m, 3 * n), torch.float16, device, seed=7), m, n)
    for (name, x), dy in zip(xs.items(), dys.values(), strict=True):
        for path in FWD_PATHS:
            got, want = softmax_triton(x, path=path), softmax_triton(x.contiguous(), path=path)
            assert_same_bits(f"{path} y, {name} view against its contiguous copy", got, want)
        for path in BWD_PATHS:
            got = softmax_triton_bwd(dy, x, path=path)
            want = softmax_triton_bwd(dy.contiguous(), x.contiguous(), path=path)
            assert_same_bits(f"{path} dx, {name} view against its contiguous copy", got, want)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _backward_runner(
    forward: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, dy: torch.Tensor
) -> Callable[[], None]:
    """Zero-arg function that replays one backward through `forward`'s graph, grads cleared each time."""
    xg = x.detach().requires_grad_(True)
    y = forward(xg)

    def run() -> None:
        xg.grad = None
        y.backward(dy, retain_graph=True)

    return run


def _print_table(
    title: str, names: list[str], rows: Iterable[tuple[int, list[Callable[[], object] | None], int]]
) -> None:
    """Time each callable with do_bench and print one row per width. None prints a dash.

    `rows` is consumed one width at a time. Pass a generator, not a list: each width's
    tensors and its fresh torch.compile cache must exist only while that width is timed.
    A second table gives the median of the same runs.
    """
    print(f"\n{title}, mean ms and GB/s")
    print(f"{'N':>6}  " + "  ".join(f"{nm + ' ms':>10} {'GB/s':>6}" for nm in names))
    measured = []
    for n, fns, nbytes in rows:
        timings = [None if fn is None else time_fn(fn) for fn in fns]
        print(f"{n:>6}  {'  '.join(mean_cell(t, nbytes) for t in timings)}")
        measured.append((n, timings))
    print_median_table(title, names, measured)


def benchmark(device: str, dtype: torch.dtype = torch.float16, m: int = 4096, compile_baseline: bool = True) -> None:
    """Forward and backward benchmarks: every path, eager and torch.compile across widths.

    Forward GB/s counts x in and y out; backward counts dy and y in and dx out. Paths that
    cannot take a width print a dash. Rows are timed as they print, so each width's tensors
    are built just before use.
    """
    names = [*FWD_PATHS, "eager"] + (["compile"] if compile_baseline else [])
    esize = torch.empty((), dtype=dtype).element_size()

    def fwd_rows():
        for n in BENCH_WIDTHS:
            x = make_inputs((m, n), dtype, device)
            fns: list[Callable[[], object] | None] = [
                (lambda x=x: softmax_triton(x, path="block")) if fits_block(n) else None,
                lambda x=x: softmax_triton(x, path="online"),
                lambda x=x: softmax_eager(x),
            ]
            if compile_baseline:
                compiled = compile_fresh(softmax_eager)
                fns.append(lambda x=x, c=compiled: c(x))
            yield n, fns, 2 * m * n * esize

    _print_table(f"softmax forward: M={m}, dtype={dtype}", names, fwd_rows())

    names = [*BWD_PATHS, "eager"] + (["compile"] if compile_baseline else [])

    def bwd_rows():
        for n in BENCH_WIDTHS:
            x = make_inputs((m, n), dtype, device)
            y = softmax_triton(x)
            dy = torch.randn_like(x)
            fns: list[Callable[[], object] | None] = [
                (lambda dy=dy, y=y: softmax_triton_bwd(dy, y, path="block")) if fits_block(n) else None,
                lambda dy=dy, y=y: softmax_triton_bwd(dy, y, path="tiled"),
                _backward_runner(softmax_eager, x, dy),
            ]
            if compile_baseline:
                fns.append(_backward_runner(compile_fresh(softmax_eager), x, dy))
            yield n, fns, 3 * m * n * esize

    _print_table(f"softmax backward: M={m}, dtype={dtype}", names, bwd_rows())


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton softmax: correctness checks and benchmark")
    parser.add_argument("--skip-bench", action="store_true", help="run correctness checks only")
    parser.add_argument("--no-compile", action="store_true", help="leave the torch.compile column out of the benchmark")
    args = parser.parse_args()

    device = require_cuda()
    if device is None:
        return 1

    for shape, dtype, kind, why in CASES:
        x = make_inputs(shape, dtype, device, kind=kind)
        ys, y_ratios = check_forward(x)
        dxs, dx_ratios = check_backward(x, ys[default_fwd_path(shape[-1])])
        ran = ", ".join(sorted(set(ys) | set(dxs)))
        print(f"ok  {str(dtype):15} {str(shape):14} {why} ({ran})")
        print(f"      largest |error| / bound   y: {_format_ratios(y_ratios)}   dx: {_format_ratios(dx_ratios)}")

    check_strided_inputs(device)
    print("ok  layout: strided views give the same bits as their contiguous copies (every path)")

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
