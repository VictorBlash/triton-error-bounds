#!/usr/bin/env python3
"""Row softmax in Triton, forward and backward, with an online (streaming) forward.

Contents:
  1. Math. Per row, with m = max(x):
         y_i = exp(x_i - m) / sum_j exp(x_j - m)
     and for the backward, with s = sum_j dy_j * y_j:
         dx_i = y_i * (dy_i - s).
  2. Forward kernels, picked by row width:
       - block: one program per row, the whole row in registers. Max, sum of exp, scale.
         Limited to N <= 32768 for fp16 and bf16, like the LayerNorm kernels.
       - online: one program per row, two passes over column tiles. Pass 1 keeps a running
         max and a running sum per lane and rescales the sum by exp(m_old - m_new) whenever
         the max rises; that recurrence is what fused attention is built on. Pass 2
         recomputes exp(x - m) per tile and scales it. No width limit. The second pass
         re-reads x, so expect it to lose to the block kernel wherever both fit; it is here
         for the widths the block kernel cannot take, and as the reference implementation
         of the recurrence.
     Backward kernels mirror them: block (one pass) and tiled (two passes: the row sum,
     then dx). No atomics anywhere. Two launches are bitwise equal, and a row's output
     depends only on that row.
  3. Launchers and a torch.autograd.Function. The backward uses the saved fp16/bf16 y.

-inf inputs are allowed (attention masks): they contribute exactly 0. A row that is
entirely -inf gives nan, as PyTorch does.

Checks, mutation tests and the benchmark live in softmax_checks.py.
Requires a CUDA (or ROCm) GPU with PyTorch and Triton installed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from common import UNIT_ROUNDOFF, num_warps_for

FWD_PATHS = ("block", "online")
BWD_PATHS = ("block", "tiled")

# Largest block for the single-block kernels: 64 KB of input per row. Both supported
# dtypes are 2 bytes, so this is 32768 elements.
BLOCK_LIMIT = 32768

# Column tile for the online and tiled kernels. 4096 at 8 warps is 16 elements per thread
# per vector, the same density the LayerNorm dx kernel runs at.
TILE_N = 4096

# The kernels below keep Triton's default specialization on their row-stride arguments.
# Triton compiles a separate variant when a stride is divisible by 16, with a different
# thread layout, and the two variants round a few elements differently: the same values
# in a tensor with another row stride can come out one unit in the last place apart.
# Passing do_not_specialize and do_not_specialize_on_alignment for the strides removes
# that. It was measured up to 1.7 times slower (block backward, N = 32768), so it was not
# kept. RESULTS.md, "Strided and non-finite inputs", has the numbers.


# ---------------------------------------------------------------------------
# 2. Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _softmax_fwd_block_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Masked lanes load -inf: they lose every max, and exp(-inf) is exactly 0 in the sum.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=float("-inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    denom = tl.sum(e, axis=0)
    y = e * (1.0 / denom)
    tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_fwd_online_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride
    neg_inf = float("-inf")

    # Pass 1: per-lane running max and running sum. When a lane's max rises, its sum is
    # rescaled by exp(m_old - m_new) so every accumulated term stays relative to the
    # current max. Lanes are independent streams; they merge after the loop.
    m_run = tl.full([BLOCK_N], neg_inf, dtype=tl.float32)
    l_run = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + cols, mask=cols < N, other=neg_inf).to(tl.float32)
        m_new = tl.maximum(m_run, x)
        # A lane whose max is still -inf has seen only -inf, masked or real. Its sum is 0
        # and must stay 0; without the guard, (-inf) - (-inf) = nan would poison it.
        alpha = tl.where(m_new == neg_inf, 1.0, tl.exp(m_run - m_new))
        p = tl.where(x == neg_inf, 0.0, tl.exp(x - m_new))
        l_run = l_run * alpha + p
        m_run = m_new

    # Merge the lanes: the row max, then each lane's sum rescaled to it.
    m = tl.max(m_run, axis=0)
    scale = tl.where(m_run == neg_inf, 0.0, tl.exp(m_run - m))
    denom = tl.sum(l_run * scale, axis=0)
    inv = 1.0 / denom

    # Pass 2: exp(x - m) per tile, scaled. Same arithmetic as the block kernel's numerator;
    # only the denominator took the longer road.
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=neg_inf).to(tl.float32)
        y = tl.exp(x - m) * inv
        tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_bwd_block_kernel(
    dy_ptr, y_ptr, dx_ptr, dy_row_stride, y_row_stride, dx_row_stride, N, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    dy = tl.load(dy_ptr + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * y_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(dy * y, axis=0)
    dx = y * (dy - s)
    tl.store(dx_ptr + row * dx_row_stride + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_bwd_tiled_kernel(
    dy_ptr, y_ptr, dx_ptr, dy_row_stride, y_row_stride, dx_row_stride, N, BLOCK_N: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    dy_row = dy_ptr + row * dy_row_stride
    y_row = y_ptr + row * y_row_stride
    dx_row = dx_ptr + row * dx_row_stride

    # Pass 1: the row sum of dy * y, accumulated per lane and reduced once.
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += dy * y
    s = tl.sum(acc, axis=0)

    # Pass 2: dx per tile.
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + cols, mask=mask, other=0.0).to(tl.float32)
        dx = y * (dy - s)
        tl.store(dx_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# 3. Launchers and autograd wrapper
# ---------------------------------------------------------------------------


def fits_block(n: int) -> bool:
    """Return True if a row of width n fits the single-block kernels."""
    return triton.next_power_of_2(n) <= BLOCK_LIMIT


def tile_for(n: int) -> int:
    """Column tile the online and tiled kernels use for row width n."""
    return min(TILE_N, triton.next_power_of_2(n))


def online_tiles(n: int) -> int:
    """Number of column tiles the online kernel walks for row width n. The error bound needs it."""
    return triton.cdiv(n, tile_for(n))


def default_fwd_path(n: int) -> str:
    """Forward path used for row width n when none is forced."""
    return "block" if fits_block(n) else "online"


def default_bwd_path(n: int) -> str:
    """Backward path used for row width n when none is forced."""
    return "block" if fits_block(n) else "tiled"


def _rows(x: torch.Tensor, name: str) -> torch.Tensor:
    """Validate dtype and device and flatten to a row-contiguous 2-D view.

    Raises:
        TypeError: dtype is not fp16/bf16.
        ValueError: not a CUDA tensor, or the last dimension is 0.
    """
    if x.dtype not in UNIT_ROUNDOFF:
        raise TypeError(f"{name} must be float16 or bfloat16, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    n = x.shape[-1]
    if n == 0:
        raise ValueError(f"{name} must have a non-empty last dimension")
    rows = x.reshape(-1, n)
    return rows if rows.stride(-1) == 1 else rows.contiguous()


def _resolve_path(path: str | None, paths: tuple[str, ...], default: str, n: int) -> str:
    if path is None:
        return default
    if path not in paths:
        raise ValueError(f"path must be one of {paths}, got {path!r}")
    if path == "block" and not fits_block(n):
        raise ValueError(f"N={n} exceeds the single-block limit of {BLOCK_LIMIT}; use the other path")
    return path


def softmax_triton(x: torch.Tensor, path: str | None = None) -> torch.Tensor:
    """Softmax over the last dimension using the Triton kernels. Forward only.

    Args:
        x: fp16 or bf16 CUDA tensor of shape (..., N). -inf entries are allowed.
        path: one of FWD_PATHS to force a kernel, or None to pick by width.

    Returns:
        Tensor with the shape and dtype of x.

    Raises:
        TypeError, ValueError: dtype, device or shape problems, or a path that cannot
            take this width.
    """
    rows = _rows(x, "x")
    m, n = rows.shape
    y = torch.empty_like(rows)
    if m == 0:
        return y.reshape(x.shape)
    chosen = _resolve_path(path, FWD_PATHS, default_fwd_path(n), n)
    if chosen == "block":
        block = triton.next_power_of_2(n)
        _softmax_fwd_block_kernel[(m,)](
            rows, y, rows.stride(0), y.stride(0), n, BLOCK_SIZE=block, num_warps=num_warps_for(block)
        )
    else:
        tile = tile_for(n)
        _softmax_fwd_online_kernel[(m,)](
            rows, y, rows.stride(0), y.stride(0), n, BLOCK_N=tile, num_warps=num_warps_for(tile)
        )
    return y.reshape(x.shape)


def softmax_triton_bwd(dy: torch.Tensor, y: torch.Tensor, path: str | None = None) -> torch.Tensor:
    """Softmax backward using the Triton kernels.

    Args:
        dy: gradient of the loss w.r.t. y, same shape and dtype as y.
        y: the softmax output, fp16 or bf16 CUDA tensor of shape (..., N).
        path: one of BWD_PATHS to force a kernel, or None to pick by width.

    Returns:
        dx with the shape and dtype of y.

    Raises:
        TypeError, ValueError: dtype, device or shape problems, or a path that cannot
            take this width.
    """
    if dy.shape != y.shape or dy.dtype != y.dtype:
        raise ValueError("dy must have the same shape and dtype as y")
    y_rows = _rows(y, "y")
    dy_rows = _rows(dy, "dy")
    m, n = y_rows.shape
    dx = torch.empty_like(y_rows)
    if m == 0:
        return dx.reshape(y.shape)
    chosen = _resolve_path(path, BWD_PATHS, default_bwd_path(n), n)
    strides = (dy_rows.stride(0), y_rows.stride(0), dx.stride(0))
    if chosen == "block":
        block = triton.next_power_of_2(n)
        _softmax_bwd_block_kernel[(m,)](
            dy_rows, y_rows, dx, *strides, n, BLOCK_SIZE=block, num_warps=num_warps_for(block)
        )
    else:
        tile = tile_for(n)
        _softmax_bwd_tiled_kernel[(m,)](
            dy_rows, y_rows, dx, *strides, n, BLOCK_N=tile, num_warps=num_warps_for(tile)
        )
    return dx.reshape(y.shape)


class _SoftmaxFunction(torch.autograd.Function):
    """Autograd glue: the forward saves y; the backward needs nothing else."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        y = softmax_triton(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (y,) = ctx.saved_tensors
        return softmax_triton_bwd(dy.contiguous(), y)


def softmax_autograd(x: torch.Tensor) -> torch.Tensor:
    """Differentiable softmax over the last dimension: Triton kernels in both directions.

    Args:
        x: fp16 or bf16 CUDA tensor of shape (..., N).

    Returns:
        Tensor with the shape and dtype of x, attached to the autograd graph.
    """
    return _SoftmaxFunction.apply(x)
