#!/usr/bin/env python3
"""LayerNorm backward in Triton, plus an autograd wrapper around the forward kernel.

Contents:
  1. Math. With xhat = (x - mean) * rstd and wdy = w * dy, per row:
         dx = rstd * (wdy - mean(wdy) - xhat * mean(wdy * xhat))
     and across rows:
         dw = sum(dy * xhat),  db = sum(dy).
  2. Three backward paths, picked by row width:
       - fused: one kernel. Each program loops over a block of rows, computes dx per row,
         and keeps dw/db accumulators for the full row width in registers. Fastest while
         those fit. Measured on a T4: at N=16384 the live vectors need about 256 KB, the
         whole register file, so the kernel spills to local memory and runs slower than
         eager PyTorch.
       - split: two kernels. The dx kernel handles one row per program and holds only
         x, dy and w. The dw/db kernel tiles over columns, so each program holds small
         accumulators. It reads x and dy a second time (about 1.67x the traffic of the
         fused path) but does not spill at N=16384.
       - tiled: the same dw/db kernel, but dx is computed in two passes over column tiles
         so that only tile-sized vectors are live. Pass 1 accumulates the two row sums,
         pass 2 recomputes xhat and wdy per tile and writes dx. The second pass re-reads
         x, dy and w. Meant for widths where even the one-pass dx kernel cannot keep
         three full-width vectors in registers.
     No path uses atomics. dw/db partial sums go to a fixed-shape fp32 buffer and a final
     torch.sum reduces it in a fixed order. Two launches are bitwise equal, and a row's dx
     does not depend on the other rows in the batch. The default path depends only on N,
     so batch invariance holds across the switch points.
  3. Launchers and a torch.autograd.Function so the kernels plug into a model.

Correctness checks, mutation tests and the benchmark live in layernorm_bwd_checks.py.
probe_spills.py reports registers and spills per kernel.

Requires a CUDA (or ROCm) GPU with PyTorch and Triton installed.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from layernorm_triton import EPS, layer_norm_triton_fwd, num_warps_for, validate_inputs

PATHS = ("fused", "split", "tiled")

# Fused path: target program count. Rows per program is derived from M, so the partial-sum
# buffer stays near FUSED_TARGET_PROGRAMS x N regardless of batch size.
FUSED_TARGET_PROGRAMS = 256

# Switch points by block size (next power of two at or above N). On a T4 the fused kernel
# runs cleanly at 8192 and spills at 16384 (RESULTS.md). TILED_MIN_BLOCK is provisional:
# the one-pass dx kernel needs 32 elements per thread at 32768, which exceeds the 64
# registers a 32-warp program gets, but that has not been measured yet. Re-measure with
# probe_spills.py and the benchmark on other GPUs.
SPLIT_MIN_BLOCK = 16384
TILED_MIN_BLOCK = 32768

# dx kernels: at most this many elements per thread per vector, so x, dy and w stay in
# registers. 16 elements x 3 vectors fits the 64-register cap that applies at 32 warps.
DX_ELEMENTS_PER_THREAD = 16
MAX_WARPS = 32

# Tiled dx kernel: column tile width. Pass 1 holds about seven tile-sized fp32 vectors.
DX_TILE_N = 4096

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

    Shared by the fused and split paths so they run the same formula. The tiled kernel
    runs the same arithmetic split across two passes.
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
def _layer_norm_bwd_dx_tiled_kernel(
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
    BLOCK_N: tl.constexpr,
):
    # One row per program, two passes over column tiles. Only tile-sized vectors are live,
    # so register use does not grow with N. Pass 2 re-reads x, dy and w; whether that hits
    # L2 depends on how many rows are in flight at once.
    row = tl.program_id(0).to(tl.int64)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_row = x_ptr + row * x_row_stride
    dy_row = dy_ptr + row * dy_row_stride
    dx_row = dx_ptr + row * dx_row_stride

    # Pass 1: elementwise partial sums per tile, reduced once at the end. Same any-order
    # sum the error bound already assumes.
    acc1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.where(mask, (x - mean) * rstd, 0.0)
        wdy = tl.where(mask, w * dy, 0.0)
        acc1 += xhat * wdy
        acc2 += wdy
    c1 = tl.sum(acc1, axis=0) / N
    c2 = tl.sum(acc2, axis=0) / N

    # Pass 2: recompute xhat and wdy per tile (bitwise the same values as pass 1) and
    # write dx. Masked lanes are not stored, so they need no zeroing here.
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        xhat = (x - mean) * rstd
        wdy = w * dy
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(dx_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


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


def default_path(n: int) -> str:
    """Return the backward path used for row width n when none is forced."""
    block = triton.next_power_of_2(n)
    if block >= TILED_MIN_BLOCK:
        return "tiled"
    if block >= SPLIT_MIN_BLOCK:
        return "split"
    return "fused"


def dx_num_warps(block: int) -> int:
    """Warps for a dx kernel of the given block or tile width.

    Enough that each thread holds at most DX_ELEMENTS_PER_THREAD elements per vector.
    """
    return min(max(block // (32 * DX_ELEMENTS_PER_THREAD), 1), MAX_WARPS)


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


def _dwdb(
    dy_rows: torch.Tensor, x_rows: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column-tiled dw/db reduction. Returns (dw, db) in fp32."""
    m, n = x_rows.shape
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
    # Fixed-shape fp32 reduction: deterministic, no atomics.
    return partial_dw.sum(dim=0), partial_db.sum(dim=0)


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
    return dx, partial_dw.sum(dim=0), partial_db.sum(dim=0)


def _bwd_split(
    dy_rows: torch.Tensor,
    x_rows: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split path: one-pass dx kernel plus the dw/db kernel."""
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
        num_warps=dx_num_warps(block_size),
    )
    return dx, *_dwdb(dy_rows, x_rows, mean, rstd, block_size)


def _bwd_tiled(
    dy_rows: torch.Tensor,
    x_rows: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiled path: two-pass dx kernel plus the dw/db kernel."""
    m, n = x_rows.shape
    dx = torch.empty_like(x_rows)
    tile = min(DX_TILE_N, block_size)
    _layer_norm_bwd_dx_tiled_kernel[(m,)](
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
        BLOCK_N=tile,
        num_warps=dx_num_warps(tile),
    )
    return dx, *_dwdb(dy_rows, x_rows, mean, rstd, block_size)


_IMPL = {"fused": _bwd_fused, "split": _bwd_split, "tiled": _bwd_tiled}


def layer_norm_triton_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm backward using the Triton kernels.

    Args:
        dy: gradient of the loss w.r.t. the forward output, same shape and dtype as x.
        x: forward input, fp16 or bf16 CUDA tensor of shape (..., N).
        weight: shape (N,), same dtype as x.
        mean: fp32 per-row mean from layer_norm_triton_fwd, shape (M,).
        rstd: fp32 per-row 1/sqrt(var + eps) from layer_norm_triton_fwd, shape (M,).
        path: one of PATHS to force a path, or None to pick by row width (default_path).

    Returns:
        (dx, dweight, dbias). dx has the shape and dtype of x; dweight and dbias have the
        shape and dtype of weight.

    Raises:
        TypeError, ValueError: dtype, device or shape problems, or an unknown path.
    """
    if path is not None and path not in _IMPL:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    block_size, x_rows, dy_rows = _prepare_rows(dy, x, weight, mean, rstd)
    if x_rows.shape[0] == 0:
        zeros = torch.zeros_like(weight)
        return torch.empty_like(x), zeros, zeros.clone()

    impl = _IMPL[path or default_path(x.shape[-1])]
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
