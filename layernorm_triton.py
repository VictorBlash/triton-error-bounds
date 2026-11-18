#!/usr/bin/env python3
"""Fused LayerNorm forward in Triton, checked with a per-element error bound.

Contents:
  1. PyTorch eager reference (torch.nn.functional.layer_norm).
  2. Triton kernel: one program per row, fp32 statistics, fp16/bf16 in and out.
  3. Launcher: single-block size limit and a warp-count heuristic.
  4. Checks per input:
       - torch.allclose against eager PyTorch (fixed tolerance),
       - per-element error bound against an fp64 reference,
       - bitwise determinism across repeated launches,
       - batch invariance (a row's output does not depend on the other rows).
     Plus a mutation test: a kernel that drops a small bias must be rejected.
  5. Benchmark with triton.testing.do_bench: Triton vs eager PyTorch vs torch.compile.

Why a per-element bound: a fixed atol/rtol is too loose where outputs are near zero
and too tight where they are large, so it both misses faults and raises false alarms.
"Beyond atol and rtol" (arXiv 2610.38983) derives an acceptance region per element from
a rounding-error model instead. This file does the same for this kernel's arithmetic.

Requires a CUDA (or ROCm) GPU with PyTorch and Triton installed.
Usage: python layernorm_triton.py [--skip-bench] [--no-compile]
"""

from __future__ import annotations

import argparse
import sys
from typing import NamedTuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import (
    APPROX_MATH_ULPS,
    U_FP32,
    ULP_REL_FP32,
    UNIT_ROUNDOFF,
    add_err,
    compile_fresh,
    mean_cell,
    mean_err,
    mul_err,
    num_warps_for,
    parity_tolerances,
    print_median_table,
    require_cuda,
    store_err,
    time_fn,
)

EPS = 1e-5
BENCH_WIDTHS = (512, 1024, 2048, 4096, 8192, 16384, 32768)


# ---------------------------------------------------------------------------
# 1. PyTorch eager reference
# ---------------------------------------------------------------------------


def layer_norm_eager(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """LayerNorm over the last dimension using PyTorch eager mode."""
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


# ---------------------------------------------------------------------------
# 2. Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _layer_norm_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    mean_ptr,
    rstd_ptr,
    x_row_stride,
    y_row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # int64 row offset: row * stride overflows int32 once a tensor passes 2**31 elements.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Masked lanes load 0 so they add nothing to the sum.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N

    # Two-pass variance in fp32. Masked lanes would hold -mean, so zero them.
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Saved for the backward pass (fp32, one scalar each per row).
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(y_ptr + row * y_row_stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# 3. Launcher
# ---------------------------------------------------------------------------


def validate_inputs(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> int:
    """Check dtypes, devices and shapes shared by the forward and backward launchers.

    Returns:
        The block size (next power of two at or above N).

    Raises:
        TypeError: dtype is not fp16/bf16, or dtypes differ.
        ValueError: shapes mismatch, device is not CUDA, or N exceeds the single-block limit.
    """
    if x.dtype not in UNIT_ROUNDOFF:
        raise TypeError(f"x must be float16 or bfloat16, got {x.dtype}")
    if weight.dtype != x.dtype or bias.dtype != x.dtype:
        raise TypeError("weight and bias must have the same dtype as x")
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        raise ValueError("x, weight and bias must be CUDA tensors")
    n = x.shape[-1]
    if n == 0 or weight.shape != (n,) or bias.shape != (n,):
        raise ValueError(f"expected weight and bias of shape ({n},), got {tuple(weight.shape)} and {tuple(bias.shape)}")

    # Each row must fit in one block so the statistics come from a single load.
    block_size = triton.next_power_of_2(n)
    max_fused = 65536 // x.element_size()
    if block_size > max_fused:
        raise ValueError(f"N={n} exceeds the single-block limit of {max_fused} for {x.dtype}")
    return block_size


def layer_norm_triton_fwd(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward LayerNorm that also returns the per-row statistics the backward pass needs.

    Args:
        x: fp16 or bf16 CUDA tensor of shape (..., N).
        weight: tensor of shape (N,), same dtype and device as x.
        bias: tensor of shape (N,), same dtype and device as x.
        eps: value added to the variance before the square root.

    Returns:
        (y, mean, rstd). y has the shape and dtype of x. mean and rstd are fp32 of shape
        (M,) where M is the number of rows in x.

    Raises:
        TypeError, ValueError: see validate_inputs.
    """
    block_size = validate_inputs(x, weight, bias)
    n = x.shape[-1]
    rows = x.reshape(-1, n)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    m = rows.shape[0]
    y = torch.empty_like(rows)
    mean = torch.empty(m, dtype=torch.float32, device=x.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x.device)
    if m == 0:
        return y.reshape(x.shape), mean, rstd

    _layer_norm_fwd_kernel[(m,)](
        rows,
        y,
        weight.contiguous(),
        bias.contiguous(),
        mean,
        rstd,
        rows.stride(0),
        y.stride(0),
        n,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps_for(block_size),
    )
    return y.reshape(x.shape), mean, rstd


def layer_norm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """LayerNorm over the last dimension using the Triton kernel. Forward only.

    See layer_norm_triton_fwd for arguments and errors. For a differentiable version use
    layer_norm_autograd in layernorm_triton_bwd.py.
    """
    return layer_norm_triton_fwd(x, weight, bias, eps)[0]


# ---------------------------------------------------------------------------
# 4. Correctness checks
# ---------------------------------------------------------------------------


class RowStats(NamedTuple):
    """Exact fp64 row statistics plus bounds on the kernel's fp32 versions of them."""

    xc: torch.Tensor  # x - mean, shape (M, N)
    rstd: torch.Tensor  # 1 / sqrt(var + eps), shape (M, 1)
    xc_err: torch.Tensor  # bound on |fl(x - mean') - xc|, shape (M, N)
    rstd_rel: torch.Tensor  # bound on the relative error of the kernel's rstd, shape (M, 1)


def row_stats_with_error(x: torch.Tensor, eps: float = EPS) -> RowStats:
    """Follow the forward kernel's statistics: two-pass fp32 mean and variance, then rstd.

    The backward kernels read the mean and rstd that the forward stored, so the backward
    bound starts from these same error terms.

    Args:
        x: input of shape (..., N), fp16 or bf16.
        eps: same eps the kernel uses.

    Returns:
        RowStats for x flattened to (M, N).

    Raises:
        ValueError: the variance error is not small relative to var + eps, so the
            first-order rstd model does not apply to this input.
    """
    n = x.shape[-1]
    x64 = x.reshape(-1, n).double()
    mean = x64.mean(dim=-1, keepdim=True)
    xc = x64 - mean
    var = (xc * xc).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)

    # Inputs are exact in fp32. Mean: n-term sum, then a division. Centered values inherit
    # the mean error plus one subtraction rounding. Variance: squares, sum, division.
    m_err = mean_err(x64, torch.zeros_like(x64), n)
    xc_err = add_err(xc.abs(), 0.0, m_err)
    var_err = mean_err(xc * xc, mul_err(xc.abs(), xc_err, xc.abs(), xc_err), n)

    # rstd = (var + eps)^-1/2. The add rounds once. A relative input error d < 1 then moves
    # rstd by at most d / (2(1 - d)), and sqrt plus division add APPROX_MATH_ULPS.
    rel_in = (1.0 + var_err / (var + eps)) * (1.0 + U_FP32) - 1.0
    if (rel_in >= 0.5).any():
        raise ValueError("variance error is too large relative to var + eps for this error model")
    rstd_rel = (1.0 + rel_in / (2.0 * (1.0 - rel_in))) * (1.0 + APPROX_MATH_ULPS * ULP_REL_FP32) - 1.0
    return RowStats(xc, rstd, xc_err, rstd_rel)


def layer_norm_reference_with_bound(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the fp64 reference and a per-element error bound for the Triton kernel.

    The bound follows the kernel's arithmetic: inputs exact in fp32, two-pass fp32 mean and
    variance summed in any order, fp32 normalize and affine, one final rounding to the
    output dtype. It holds if no fp32 intermediate overflows or underflows and sqrt and
    division stay within APPROX_MATH_ULPS and DIV_ULPS. Summation order is not assumed, so
    the bound grows with N. A known tree order would allow a log2(N) term instead.

    Args:
        x: input of shape (..., N), fp16 or bf16.
        weight: shape (N,).
        bias: shape (N,).
        eps: same eps the kernel uses.

    Returns:
        (y_ref, bound), both fp64 with the shape of x. A correct kernel output y satisfies
        |y - y_ref| <= bound elementwise.

    Raises:
        ValueError: see row_stats_with_error.
    """
    st = row_stats_with_error(x, eps)
    w64 = weight.double()
    xhat = st.xc * st.rstd
    y_ref = xhat * w64 + bias.double()

    # y = (xc' * rstd') * w + b: two products and one add, then the store.
    xhat_err = mul_err(st.xc.abs(), st.xc_err, st.rstd, st.rstd * st.rstd_rel)
    y_err = add_err(y_ref.abs(), mul_err(xhat.abs(), xhat_err, w64.abs(), 0.0), 0.0)
    bound = store_err(y_ref, y_err, x.dtype)
    return y_ref.reshape(x.shape), bound.reshape(x.shape)


def check_kernel(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> None:
    """Run every correctness check for one input.

    Raises:
        AssertionError: any check fails. The message names the check and the worst element.
    """
    y = layer_norm_triton(x, weight, bias, eps)
    assert torch.isfinite(y).all(), "kernel produced non-finite output"

    # Check 1: fixed tolerance against eager PyTorch.
    y_eager = layer_norm_eager(x, weight, bias, eps)
    atol, rtol = parity_tolerances(x.dtype)
    diff = (y.float() - y_eager.float()).abs()
    assert torch.allclose(y, y_eager, atol=atol, rtol=rtol), (
        f"allclose vs eager failed: max |diff| {diff.max().item():.3e} (atol={atol}, rtol={rtol})"
    )

    # Check 2: per-element bound against fp64. Tighter than check 1 near zero, where a
    # fixed atol hides small faults, and looser only where the input really allows it.
    y_ref, bound = layer_norm_reference_with_bound(x, weight, bias, eps)
    excess = (y.double() - y_ref).abs() - bound
    worst = int(excess.argmax())
    assert (excess <= 0).all(), (
        f"per-element bound exceeded at flat index {worst}: "
        f"y={y.flatten()[worst].item():.6g}, ref={y_ref.flatten()[worst].item():.6g}, "
        f"bound={bound.flatten()[worst].item():.3e}"
    )

    # Check 3: bitwise determinism. Atomics or races would make two launches disagree.
    assert torch.equal(y, layer_norm_triton(x, weight, bias, eps)), "two launches on the same input differ"

    # Check 4: batch invariance. In serving, a request's output must not depend on
    # which other requests share its batch.
    n = x.shape[-1]
    rows, y_rows = x.reshape(-1, n), y.reshape(-1, n)
    for i in sorted({0, rows.shape[0] // 2, rows.shape[0] - 1}):
        single = layer_norm_triton(rows[i : i + 1], weight, bias, eps)
        assert torch.equal(single, y_rows[i : i + 1]), f"row {i} differs when launched alone"


def check_oracle_rejects_dropped_bias(device: str) -> bool:
    """Mutation test: a kernel that ignores a small bias must fail the per-element bound.

    If this test passed for a broken kernel, check 2 would be worthless. Bias values stay
    under 1e-3, so the fixed allclose tolerance is expected to miss the fault.

    Args:
        device: device to run on.

    Returns:
        True if allclose also caught the fault, False if only the per-element bound did.

    Raises:
        AssertionError: the per-element bound accepted the faulty output.
    """
    g = torch.Generator(device=device).manual_seed(1)
    n = 1024
    x = torch.randn(256, n, generator=g, device=device).half()
    w = (1.0 + 0.1 * torch.randn(n, generator=g, device=device)).half()
    b = (3e-4 * torch.randn(n, generator=g, device=device)).clamp(-9e-4, 9e-4).half()

    mutant = layer_norm_triton(x, w, torch.zeros_like(b))
    y_ref, bound = layer_norm_reference_with_bound(x, w, b)
    assert ((mutant.double() - y_ref).abs() > bound).any(), "per-element bound accepted a kernel that drops the bias"

    atol, rtol = parity_tolerances(torch.float16)
    return not torch.allclose(mutant, layer_norm_eager(x, w, b), atol=atol, rtol=rtol)


def make_inputs(
    shape: tuple[int, ...], dtype: torch.dtype, device: str, offset: float = 0.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a seeded input, weight and bias. offset shifts the input mean."""
    g = torch.Generator(device=device).manual_seed(seed)
    n = shape[-1]
    x = torch.randn(shape, generator=g, device=device) + offset
    w = 1.0 + 0.1 * torch.randn(n, generator=g, device=device)
    b = 0.1 * torch.randn(n, generator=g, device=device)
    return x.to(dtype), w.to(dtype), b.to(dtype)


# (shape, dtype, input mean offset, what the case protects)
CASES: list[tuple[tuple[int, ...], torch.dtype, float, str]] = [
    ((1, 1), torch.float16, 0.0, "N=1: variance is 0, output must equal bias"),
    ((17, 7), torch.float16, 0.0, "N below one warp, not a power of two"),
    ((64, 127), torch.bfloat16, 0.0, "bf16, mask covers 1 lane"),
    ((128, 1000), torch.float16, 0.0, "common non-power-of-two width"),
    ((2, 3, 768), torch.float16, 0.0, "3-D input flattened to rows"),
    ((32, 4096), torch.bfloat16, 0.0, "bf16 at a typical hidden size"),
    ((8, 5000), torch.float16, 0.0, "mask covers 3192 of 8192 lanes"),
    ((2, 20000), torch.float16, 0.0, "near the single-block limit"),
    ((64, 1024), torch.float16, 50.0, "large mean: cancellation in x - mean"),
]


# ---------------------------------------------------------------------------
# 5. Benchmark
# ---------------------------------------------------------------------------


def benchmark(device: str, dtype: torch.dtype = torch.float16, m: int = 4096, compile_baseline: bool = True) -> None:
    """Print mean forward time and effective bandwidth for Triton, eager and torch.compile.

    do_bench handles warmup, cache flushing and repeats. Bandwidth counts one read of x,
    one write of y, and one read each of weight and bias. The compiled column is
    inductor's own generated kernel for the same shape, the baseline a custom kernel has
    to beat to be worth keeping. A second table gives the median of the same runs.
    """
    names = ["triton", "eager"] + (["compile"] if compile_baseline else [])
    title = f"forward benchmark: M={m}, dtype={dtype}"
    print(f"\n{title}, mean ms and GB/s")
    print(f"{'N':>6}  " + "  ".join(f"{nm + ' ms':>10} {'GB/s':>6}" for nm in names))
    rows = []
    for n in BENCH_WIDTHS:
        x, w, b = make_inputs((m, n), dtype, device)
        bytes_moved = (2 * m * n + 2 * n) * x.element_size()
        fns = [layer_norm_triton, layer_norm_eager]
        if compile_baseline:
            fns.append(compile_fresh(layer_norm_eager))
        timings = [time_fn(lambda fn=fn: fn(x, w, b)) for fn in fns]
        print(f"{n:>6}  {'  '.join(mean_cell(t, bytes_moved) for t in timings)}")
        rows.append((n, timings))
    print_median_table(title, names, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton LayerNorm: correctness checks and benchmark")
    parser.add_argument("--skip-bench", action="store_true", help="run correctness checks only")
    parser.add_argument("--no-compile", action="store_true", help="leave the torch.compile column out of the benchmark")
    args = parser.parse_args()

    device = require_cuda()
    if device is None:
        return 1

    for shape, dtype, offset, why in CASES:
        check_kernel(*make_inputs(shape, dtype, device, offset))
        print(f"ok  {str(dtype):15} {str(shape):14} {why}")

    allclose_caught = check_oracle_rejects_dropped_bias(device)
    print(
        "ok  mutation test: per-element bound rejects dropped bias "
        f"(allclose {'also caught it' if allclose_caught else 'missed it'})"
    )

    if not args.skip_bench:
        benchmark(device, compile_baseline=not args.no_compile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
