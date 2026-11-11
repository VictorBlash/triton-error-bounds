#!/usr/bin/env python3
"""LayerNorm backward in Triton, plus an autograd wrapper around the forward kernel.

Contents:
  1. Math. With xhat = (x - mean) * rstd and wdy = w * dy, per row:
         dx = rstd * (wdy - mean(wdy) - xhat * mean(wdy * xhat))
     and across rows:
         dw = sum(dy * xhat),  db = sum(dy).
  2. Two backward paths, picked by row width:
       - fused: one kernel. Each program loops over a block of rows, computes dx per row,
         and keeps dw/db accumulators for the full row width in registers. Fastest while
         those fit. Measured on a T4: at N=16384 the live vectors need about 256 KB, the
         whole register file, so the kernel spills to local memory and runs slower than
         eager PyTorch.
       - split: two kernels. The dx kernel handles one row per program and holds only
         x, dy and w. The dw/db kernel tiles over columns, so each program holds small
         accumulators. It reads x and dy a second time (about 1.67x the traffic of the
         fused path) but does not spill.
     Neither path uses atomics. dw/db partial sums go to a fixed-shape fp32 buffer and a
     final torch.sum reduces it in a fixed order. Two launches are bitwise equal, and a
     row's dx does not depend on the other rows in the batch.
  3. Launchers and a torch.autograd.Function so the kernels plug into a model.
  4. Checks per input, run on both paths:
       - torch.allclose against eager PyTorch autograd (fixed tolerance),
       - per-element error bound against an fp64 reference, for dx, dw and db,
       - bitwise determinism across repeated launches,
       - batch invariance of dx.
     Plus: the autograd wrapper matches the default path, and mutation tests on real
     kernels with gradients of size ~1e-3: a dx kernel that drops the
     xhat * mean(wdy * xhat) term, and dw/db that skip the last row. The bound must reject
     both; the output reports whether allclose also caught them.
  5. Benchmark: fused vs split vs eager across widths, to locate the crossover.

The per-element bound uses the same error model as the forward file (mul_err, add_err,
mean_err, store_err, row_stats_with_error) and the same assumptions: no fp32 overflow or
underflow, approximate sqrt and division within APPROX_MATH_ULPS and DIV_ULPS.

Known gap: above N=16384 the split dx kernel holds more than 16 elements per thread and
spills again. Fixing that needs a column-tiled, two-pass row reduction for dx.

Requires a CUDA (or ROCm) GPU with PyTorch and Triton installed.
Usage: python layernorm_triton_bwd.py [--skip-bench]
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import triton
import triton.language as tl

from layernorm_triton import (
    CASES,
    EPS,
    U_FP32,
    RowStats,
    _gamma,
    _parity_tolerances,
    add_err,
    layer_norm_eager,
    layer_norm_triton_fwd,
    make_inputs,
    mean_err,
    mul_err,
    num_warps_for,
    row_stats_with_error,
    store_err,
    validate_inputs,
)

# Fused path: target program count. Rows per program is derived from M, so the partial-sum
# buffer stays near FUSED_TARGET_PROGRAMS x N regardless of batch size.
FUSED_TARGET_PROGRAMS = 256

# The split path is the default from this block size up. On a T4 the fused kernel runs
# cleanly at 8192 and spills at 16384. Re-measure with the benchmark on other GPUs.
SPLIT_MIN_BLOCK = 16384

# dx kernel: at most this many elements per thread per vector, so x, dy and w stay in
# registers. 16 elements x 3 vectors fits the 64-register cap that applies at 32 warps.
DX_ELEMENTS_PER_THREAD = 16
MAX_WARPS = 32

# dw/db kernel: column tile width, target number of row groups, and warps per program.
# 1024 columns over 4 warps is 8 elements per thread per vector.
DWDB_BLOCK_N = 1024
DWDB_ROW_GROUPS = 32
DWDB_NUM_WARPS = 4


# ---------------------------------------------------------------------------
# 2. Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _row_dx(x, dy, w, mean, rstd, mask, N):
    """dx for one row in fp32. Also returns xhat, which the fused kernel needs for dw.

    Shared by both paths so they run the same formula.
    """
    # Masked lanes are zero in x, so xhat there would be -mean * rstd. Zero them so
    # they contribute nothing to the two row means.
    xhat = tl.where(mask, (x - mean) * rstd, 0.0)
    wdy = tl.where(mask, w * dy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    return (wdy - (xhat * c1 + c2)) * rstd, xhat


@triton.jit
def _layer_norm_bwd_fused_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    dy_row_stride,
    x_row_stride,
    dx_row_stride,
    M,
    N,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    row_start = pid * ROWS_PER_PROGRAM
    for i in range(ROWS_PER_PROGRAM):
        row = row_start + i
        # Rows past M belong to no program. Guard the loads with a combined mask so the
        # trailing program adds zeros instead of reading out of bounds.
        row_ok = row < M
        rmask = mask & row_ok

        x = tl.load(x_ptr + row * x_row_stride + cols, mask=rmask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=rmask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=row_ok, other=0.0)
        rstd = tl.load(rstd_ptr + row, mask=row_ok, other=0.0)

        dx, xhat = _row_dx(x, dy, w, mean, rstd, rmask, N)
        tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=rmask)

        dw_acc += dy * xhat
        db_acc += dy

    tl.store(partial_dw_ptr + pid * N + cols, dw_acc, mask=mask)
    tl.store(partial_db_ptr + pid * N + cols, db_acc, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dy_row_stride,
    x_row_stride,
    dx_row_stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # One row per program, no cross-row state, so no accumulators to hold.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)

    dx, _ = _row_dx(x, dy, w, mean, rstd, mask, N)
    tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _layer_norm_bwd_dwdb_kernel(
    dy_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    dy_row_stride,
    x_row_stride,
    M,
    N,
    rows_per_program,
    BLOCK_N: tl.constexpr,
):
    # Grid is (row group, column tile). Each program sums dy * xhat and dy over its rows
    # for BLOCK_N columns, so the accumulators are BLOCK_N wide instead of N wide.
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N

    dw_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    db_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    row_start = pid_m * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        # Masked lanes have dy = 0, so they add nothing even though xhat is nonzero there.
        dw_acc += dy * ((x - mean) * rstd)
        db_acc += dy

    tl.store(partial_dw_ptr + pid_m * N + cols, dw_acc, mask=mask)
    tl.store(partial_db_ptr + pid_m * N + cols, db_acc, mask=mask)


# ---------------------------------------------------------------------------
# 3. Launchers and autograd wrapper
# ---------------------------------------------------------------------------


def uses_split_path(n: int) -> bool:
    """Return True if the default backward for row width n is the split path."""
    return triton.next_power_of_2(n) >= SPLIT_MIN_BLOCK


def _dx_num_warps(block_size: int) -> int:
    """Warps for the dx kernel: enough that each thread holds at most DX_ELEMENTS_PER_THREAD."""
    return min(max(block_size // (32 * DX_ELEMENTS_PER_THREAD), 1), MAX_WARPS)


def _prepare_rows(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Validate backward inputs and flatten x and dy to row-contiguous 2-D views.

    Returns:
        (block_size, x_rows, dy_rows).

    Raises:
        TypeError, ValueError: dtype, device or shape problems.
    """
    # Reuse the forward validation. bias is not an input here, so pass weight in its place.
    block_size = validate_inputs(x, weight, weight)
    if dy.shape != x.shape or dy.dtype != x.dtype:
        raise ValueError("dy must have the same shape and dtype as x")
    n = x.shape[-1]
    x_rows = x.reshape(-1, n)
    dy_rows = dy.reshape(-1, n)
    if x_rows.stride(-1) != 1:
        x_rows = x_rows.contiguous()
    if dy_rows.stride(-1) != 1:
        dy_rows = dy_rows.contiguous()
    m = x_rows.shape[0]
    if mean.shape != (m,) or rstd.shape != (m,):
        raise ValueError(f"mean and rstd must have shape ({m},)")
    return block_size, x_rows, dy_rows


def _bwd_fused(
    dy_rows: torch.Tensor,
    x_rows: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused path. Returns dx in the input dtype, dw and db in fp32."""
    m, n = x_rows.shape
    dx = torch.empty_like(x_rows)
    rows_per_program = max(1, math.ceil(m / FUSED_TARGET_PROGRAMS))
    num_programs = math.ceil(m / rows_per_program)
    partial_dw = torch.empty(num_programs, n, dtype=torch.float32, device=x_rows.device)
    partial_db = torch.empty_like(partial_dw)

    _layer_norm_bwd_fused_kernel[(num_programs,)](
        dy_rows,
        x_rows,
        weight,
        mean,
        rstd,
        dx,
        partial_dw,
        partial_db,
        dy_rows.stride(0),
        x_rows.stride(0),
        dx.stride(0),
        m,
        n,
        ROWS_PER_PROGRAM=rows_per_program,
        BLOCK_SIZE=block_size,
        num_warps=num_warps_for(block_size),
    )
    # Fixed-shape fp32 reduction: deterministic, no atomics.
    return dx, partial_dw.sum(dim=0), partial_db.sum(dim=0)


def _bwd_split(
    dy_rows: torch.Tensor,
    x_rows: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split path. Returns dx in the input dtype, dw and db in fp32."""
    m, n = x_rows.shape
    dx = torch.empty_like(x_rows)
    _layer_norm_bwd_dx_kernel[(m,)](
        dy_rows,
        x_rows,
        weight,
        mean,
        rstd,
        dx,
        dy_rows.stride(0),
        x_rows.stride(0),
        dx.stride(0),
        n,
        BLOCK_SIZE=block_size,
        num_warps=_dx_num_warps(block_size),
    )

    block_n = min(DWDB_BLOCK_N, block_size)
    rows_per_program = max(1, math.ceil(m / DWDB_ROW_GROUPS))
    row_groups = math.ceil(m / rows_per_program)
    partial_dw = torch.empty(row_groups, n, dtype=torch.float32, device=x_rows.device)
    partial_db = torch.empty_like(partial_dw)
    _layer_norm_bwd_dwdb_kernel[(row_groups, triton.cdiv(n, block_n))](
        dy_rows,
        x_rows,
        mean,
        rstd,
        partial_dw,
        partial_db,
        dy_rows.stride(0),
        x_rows.stride(0),
        m,
        n,
        rows_per_program,
        BLOCK_N=block_n,
        num_warps=DWDB_NUM_WARPS,
    )
    return dx, partial_dw.sum(dim=0), partial_db.sum(dim=0)


def layer_norm_triton_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    split: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm backward using the Triton kernels.

    Args:
        dy: gradient of the loss w.r.t. the forward output, same shape and dtype as x.
        x: forward input, fp16 or bf16 CUDA tensor of shape (..., N).
        weight: shape (N,), same dtype as x.
        mean: fp32 per-row mean from layer_norm_triton_fwd, shape (M,).
        rstd: fp32 per-row 1/sqrt(var + eps) from layer_norm_triton_fwd, shape (M,).
        split: force the split (True) or fused (False) path. None picks by row width,
            which depends only on N, so batch invariance holds.

    Returns:
        (dx, dweight, dbias). dx has the shape and dtype of x; dweight and dbias have the
        shape and dtype of weight.

    Raises:
        TypeError, ValueError: dtype, device or shape problems. See validate_inputs.
    """
    block_size, x_rows, dy_rows = _prepare_rows(dy, x, weight, mean, rstd)
    if x_rows.shape[0] == 0:
        zeros = torch.zeros_like(weight)
        return torch.empty_like(x), zeros, zeros.clone()

    use_split = uses_split_path(x.shape[-1]) if split is None else split
    impl = _bwd_split if use_split else _bwd_fused
    dx, dw, db = impl(dy_rows, x_rows, weight.contiguous(), mean, rstd, block_size)
    return dx.reshape(x.shape), dw.to(weight.dtype), db.to(weight.dtype)


class _LayerNormFunction(torch.autograd.Function):
    """Autograd glue: Triton forward saves (x, w, mean, rstd); Triton backward uses them."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        y, mean, rstd = layer_norm_triton_fwd(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dweight, dbias = layer_norm_triton_bwd(dy.contiguous(), x, weight, mean, rstd)
        return dx, dweight, dbias, None


def layer_norm_autograd(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Differentiable LayerNorm: Triton kernels in both directions.

    Args:
        x: fp16 or bf16 CUDA tensor of shape (..., N).
        weight: shape (N,), same dtype and device as x.
        bias: shape (N,), same dtype and device as x.
        eps: value added to the variance before the square root.

    Returns:
        Tensor with the shape and dtype of x, attached to the autograd graph.
    """
    return _LayerNormFunction.apply(x, weight, bias, eps)


# ---------------------------------------------------------------------------
# 4. Correctness checks
# ---------------------------------------------------------------------------


Grads = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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

    One fp32 term per row, then an M-term sum. Both paths sum in per-program loops and then
    torch.sum over the partials; either way each term passes through at most M - 1
    roundings, so the any-order bound gamma(M - 1) applies.
    """
    m = dy64.shape[0]
    xhat = st.xc * st.rstd
    xhat_err = mul_err(st.xc.abs(), st.xc_err, st.rstd, st.rstd * st.rstd_rel)
    t = dy64 * xhat
    t_err = mul_err(dy64.abs(), 0.0, xhat.abs(), xhat_err)
    dw_err = t_err.sum(0) + _gamma(m - 1) * (t.abs() + t_err).sum(0)
    db_err = _gamma(m - 1) * dy64.abs().sum(0)
    return t.sum(0), dw_err, dy64.sum(0), db_err


def layer_norm_bwd_reference_with_bound(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float = EPS
) -> tuple[Grads, Grads]:
    """fp64 reference gradients and per-element error bounds for the Triton backward.

    The bound follows the kernels' arithmetic: dy, x and w exact in fp32, mean and rstd as
    the forward stored them (error terms from row_stats_with_error), fp32 after that, and
    one final rounding to the output dtype. It covers both paths, because they differ only
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


def _eager_grads(
    dy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients from PyTorch eager autograd, in the input dtype."""
    xg = x.detach().requires_grad_(True)
    wg = weight.detach().requires_grad_(True)
    bg = bias.detach().requires_grad_(True)
    layer_norm_eager(xg, wg, bg, eps).backward(dy)
    return xg.grad, wg.grad, bg.grad


def _assert_close(name: str, got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (got.double() - want.double()).abs()
    assert torch.allclose(got.double(), want.double(), atol=atol, rtol=rtol), (
        f"{name}: max |diff| {diff.max().item():.3e} (atol={atol}, rtol={rtol})"
    )


def _assert_within_bound(name: str, got: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor) -> None:
    excess = (got.double() - ref).abs() - bound
    worst = int(excess.argmax())
    assert (excess <= 0).all(), (
        f"{name}: per-element bound exceeded at flat index {worst}: "
        f"got={got.flatten()[worst].item():.6g}, ref={ref.flatten()[worst].item():.6g}, "
        f"bound={bound.flatten()[worst].item():.3e}"
    )


def _check_path(
    split: bool,
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
    tag = "split" if split else "fused"
    _, mean, rstd = layer_norm_triton_fwd(x, weight, bias, eps)
    got = layer_norm_triton_bwd(dy, x, weight, mean, rstd, split=split)
    for name, t in zip(("dx", "dw", "db"), got):
        assert torch.isfinite(t).all(), f"{tag} {name} has non-finite values"

    # Fixed tolerance against eager autograd. dw and db are sums over M rows, so their
    # magnitude grows with M: rtol does the work, atol scales with M for values near zero.
    atol, rtol = _parity_tolerances(x.dtype)
    m = x.reshape(-1, x.shape[-1]).shape[0]
    for name, g, r, a in zip(("dx", "dw", "db"), got, want_eager, (atol, atol * m, atol * m)):
        _assert_close(f"{tag} {name} vs eager", g, r, a, rtol)

    # Per-element bound against fp64. Tighter than allclose where gradients are small,
    # and catches a kernel that matches eager only because both share a bug.
    for name, g, r, b in zip(("dx", "dw", "db"), got, *want_fp64):
        _assert_within_bound(f"{tag} {name} vs fp64", g, r, b)

    again = layer_norm_triton_bwd(dy, x, weight, mean, rstd, split=split)
    assert all(torch.equal(a, b) for a, b in zip(got, again)), f"{tag}: two launches differ"

    # Batch invariance of dx. dw and db are batch reductions by definition, so only dx
    # must match when a row is launched alone.
    n = x.shape[-1]
    rows, dy_rows, dx_rows = x.reshape(-1, n), dy.reshape(-1, n), got[0].reshape(-1, n)
    for i in sorted({0, m // 2, m - 1}):
        _, mean_i, rstd_i = layer_norm_triton_fwd(rows[i : i + 1], weight, bias, eps)
        dx_i, _, _ = layer_norm_triton_bwd(dy_rows[i : i + 1], rows[i : i + 1], weight, mean_i, rstd_i, split=split)
        assert torch.equal(dx_i, dx_rows[i : i + 1]), f"{tag}: dx row {i} differs when launched alone"
    return got


def check_backward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> None:
    """Run every backward check for one input on both paths.

    Raises:
        AssertionError: any check fails. The message names the path and the check.
    """
    g = torch.Generator(device=x.device).manual_seed(7)
    dy = torch.randn(x.shape, generator=g, device=x.device).to(x.dtype)
    want_eager = _eager_grads(dy, x, weight, bias, eps)
    want_fp64 = layer_norm_bwd_reference_with_bound(dy, x, weight, eps)
    got = {split: _check_path(split, dy, x, weight, bias, eps, want_eager, want_fp64) for split in (False, True)}

    # The autograd wrapper must route to the default path for this width.
    default = got[uses_split_path(x.shape[-1])]
    xg = x.detach().requires_grad_(True)
    wg = weight.detach().requires_grad_(True)
    bg = bias.detach().requires_grad_(True)
    layer_norm_autograd(xg, wg, bg, eps).backward(dy)
    assert all(torch.equal(a, b) for a, b in zip((xg.grad, wg.grad, bg.grad), default)), (
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


def check_bound_rejects_mutants(device: str) -> dict[str, bool]:
    """Mutation tests: two realistic backward bugs must fail the per-element bound.

    Both run real kernels on gradients of size ~1e-3, typical late in training. At that
    scale a fixed atol of 1e-3 is as large as the gradients themselves, so allclose is
    expected to miss errors that the bound, which scales with the values, should catch.

    Mutants:
      - a dx kernel that drops the xhat * mean(wdy * xhat) term,
      - dw and db with an off-by-one that skips the last row.

    The correct kernels must pass on the same input first, so a rejection means the bound
    saw the bug, not that it is too tight for small gradients.

    Args:
        device: device to run on.

    Returns:
        {mutant name: True if allclose against eager also caught it}.

    Raises:
        AssertionError: the correct kernels fail the bound, or a mutant passes it.
    """
    m, n = 256, 1024
    x, w, b = make_inputs((m, n), torch.float16, device, seed=3)
    g = torch.Generator(device=device).manual_seed(9)
    dy = (1e-3 * torch.randn(x.shape, generator=g, device=device)).half()
    _, mean, rstd = layer_norm_triton_fwd(x, w, b)
    refs, bounds = layer_norm_bwd_reference_with_bound(dy, x, w)

    for split in (False, True):
        got = layer_norm_triton_bwd(dy, x, w, mean, rstd, split=split)
        tag = "split" if split else "fused"
        for name, gt, r, bd in zip(("dx", "dw", "db"), got, refs, bounds):
            _assert_within_bound(f"{tag} {name} with small dy", gt, r, bd)

    dx_bad = torch.empty_like(x)
    _mutant_dx_drop_c1_kernel[(m,)](dy, w, rstd, dx_bad, n, BLOCK_SIZE=triton.next_power_of_2(n))
    _, dw_bad, db_bad = layer_norm_triton_bwd(dy[:-1], x[:-1], w, mean[:-1], rstd[:-1], split=True)

    eager = _eager_grads(dy, x, w, b, EPS)
    atol, rtol = _parity_tolerances(torch.float16)
    mutants = (
        ("dx drops the c1 term", dx_bad, 0, atol),
        ("dw skips the last row", dw_bad, 1, atol * m),
        ("db skips the last row", db_bad, 2, atol * m),
    )
    caught = {}
    for label, bad, i, a in mutants:
        assert ((bad.double() - refs[i]).abs() > bounds[i]).any(), f"per-element bound accepted mutant: {label}"
        caught[label] = not torch.allclose(bad.double(), eager[i].double(), atol=a, rtol=rtol)
    return caught


# ---------------------------------------------------------------------------
# 5. Benchmark
# ---------------------------------------------------------------------------


def benchmark(device: str, dtype: torch.dtype = torch.float16, m: int = 4096) -> None:
    """Print median backward time for the fused path, the split path and eager PyTorch.

    GB/s uses the same byte count for all three: one read each of x and dy and one write
    of dx, the traffic any backward must move. The split path's extra reads show up as
    lower GB/s rather than as a different formula, so the columns compare directly.
    """
    print(f"\nbackward benchmark: M={m}, dtype={dtype}, median ms and GB/s")
    print(
        f"{'N':>6}  {'fused ms':>9} {'GB/s':>6}  {'split ms':>9} {'GB/s':>6}  "
        f"{'eager ms':>9} {'GB/s':>6}  default"
    )
    for n in (512, 1024, 2048, 4096, 8192, 16384):
        x, w, b = make_inputs((m, n), dtype, device)
        dy = torch.randn_like(x)
        _, mean, rstd = layer_norm_triton_fwd(x, w, b)
        bytes_moved = 3 * m * n * x.element_size()

        xg = x.detach().requires_grad_(True)
        wg = w.detach().requires_grad_(True)
        bg = b.detach().requires_grad_(True)
        y_eager = layer_norm_eager(xg, wg, bg)

        def run_eager() -> None:
            # Clear grads so each call writes fresh ones. Accumulating into an existing
            # .grad adds a read and a write of dx and would overstate eager's time.
            xg.grad = wg.grad = bg.grad = None
            y_eager.backward(dy, retain_graph=True)

        runs = (
            lambda: layer_norm_triton_bwd(dy, x, w, mean, rstd, split=False),
            lambda: layer_norm_triton_bwd(dy, x, w, mean, rstd, split=True),
            run_eager,
        )
        cells = []
        for fn in runs:
            ms = triton.testing.do_bench(fn)
            cells.append(f"{ms:9.4f} {bytes_moved / (ms * 1e-3) / 1e9:6.1f}")
        default = "split" if uses_split_path(n) else "fused"
        print(f"{n:>6}  {'  '.join(cells)}  {default}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton LayerNorm backward: correctness checks and benchmark")
    parser.add_argument("--skip-bench", action="store_true", help="run correctness checks only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("A CUDA or ROCm GPU is required: Triton compiles kernels for the GPU.", file=sys.stderr)
        return 1
    device = "cuda"

    for shape, dtype, offset, why in CASES:
        check_backward(*make_inputs(shape, dtype, device, offset))
        print(f"ok  {str(dtype):15} {str(shape):14} {why} (fused + split)")

    for label, also in check_bound_rejects_mutants(device).items():
        print(
            f"ok  mutation test: per-element bound rejects '{label}' "
            f"(allclose {'also caught it' if also else 'missed it'})"
        )

    if not args.skip_bench:
        benchmark(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
