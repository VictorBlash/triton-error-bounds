#!/usr/bin/env python3
"""Register and spill probe for the LayerNorm kernels.

Launches each kernel directly, mirroring the launchers' settings, so the compiled handle
is available. Prints registers per thread, spill count, median time and bandwidth. A
kernel that spills shows up here before it shows up as a bandwidth cliff in the benchmark.
The N=16384 fused-backward cliff in RESULTS.md was found this way.

Sweeps the dx kernel over warp counts. Eager forward and backward are timed for reference.

GB/s counts the minimum traffic of each kernel: forward x in, y out, w and b in; dx
kernels x, dy in, dx out; dw/db x, dy in; fused x, dy in, dx out.

Usage: python probe_spills.py [--n 16384 32768] [--m 4096]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable

import torch
import triton

from common import num_warps_for
from layernorm_triton import (
    EPS,
    _layer_norm_fwd_kernel,
    layer_norm_eager,
    layer_norm_triton_fwd,
    make_inputs,
)
from layernorm_triton_bwd import (
    DWDB_BLOCK_N,
    DWDB_NUM_WARPS,
    DWDB_ROW_GROUPS,
    FUSED_TARGET_PROGRAMS,
    _layer_norm_bwd_dwdb_kernel,
    _layer_norm_bwd_dx_kernel,
    _layer_norm_bwd_fused_kernel,
    dx_num_warps,
)

DX_WARP_SWEEP = (4, 8, 16, 32)


def _report(name: str, launch: Callable[[], object], nbytes: int) -> None:
    """Launch once for the compiled handle, then time it."""
    handle = launch()
    ms = triton.testing.do_bench(launch)
    regs = getattr(handle, "n_regs", None)
    spills = getattr(handle, "n_spills", None)
    print(f"  {name:<30} regs={regs!s:>4}  spills={spills!s:>5}  {ms:9.4f} ms  {nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s")


def _time(name: str, fn: Callable[[], object], nbytes: int) -> None:
    ms = triton.testing.do_bench(fn)
    print(f"  {name:<30} {'':<12} {'':<13} {ms:9.4f} ms  {nbytes / (ms * 1e-3) / 1e9:6.1f} GB/s")


def probe(n: int, m: int, dtype: torch.dtype, device: str) -> None:
    """Probe every kernel at row width n."""
    x, w, b = make_inputs((m, n), dtype, device)
    dy = torch.randn_like(x)
    _, mean, rstd = layer_norm_triton_fwd(x, w, b)
    block = triton.next_power_of_2(n)
    esize = x.element_size()
    print(f"\nN={n} (block {block}), M={m}, {dtype}")

    y = torch.empty_like(x)
    _report(
        f"fwd warps={num_warps_for(block)}",
        lambda: _layer_norm_fwd_kernel[(m,)](
            x, y, w, b, mean, rstd, x.stride(0), y.stride(0), n, EPS, BLOCK_SIZE=block, num_warps=num_warps_for(block)
        ),
        (2 * m * n + 2 * n) * esize,
    )
    _time("eager fwd", lambda: layer_norm_eager(x, w, b), (2 * m * n + 2 * n) * esize)

    dx = torch.empty_like(x)
    rpp = max(1, math.ceil(m / FUSED_TARGET_PROGRAMS))
    progs = math.ceil(m / rpp)
    pdw = torch.empty(progs, n, dtype=torch.float32, device=device)
    pdb = torch.empty_like(pdw)
    _report(
        f"bwd fused warps={num_warps_for(block)}",
        lambda: _layer_norm_bwd_fused_kernel[(progs,)](
            dy, x, w, mean, rstd, dx, pdw, pdb, dy.stride(0), x.stride(0), dx.stride(0), m, n,
            ROWS_PER_PROGRAM=rpp, BLOCK_SIZE=block, num_warps=num_warps_for(block),
        ),
        3 * m * n * esize,
    )

    launcher_warps = dx_num_warps(block)
    for warps in sorted({launcher_warps, *DX_WARP_SWEEP}):
        if 32 * warps > max(block, 32):
            continue
        tag = " (launcher)" if warps == launcher_warps else ""
        _report(
            f"dx warps={warps}{tag}",
            lambda warps=warps: _layer_norm_bwd_dx_kernel[(m,)](
                dy, x, w, mean, rstd, dx, dy.stride(0), x.stride(0), dx.stride(0), n, BLOCK_SIZE=block, num_warps=warps
            ),
            3 * m * n * esize,
        )

    block_n = min(DWDB_BLOCK_N, block)
    rpp_g = max(1, math.ceil(m / DWDB_ROW_GROUPS))
    groups = math.ceil(m / rpp_g)
    gdw = torch.empty(groups, n, dtype=torch.float32, device=device)
    gdb = torch.empty_like(gdw)
    _report(
        f"dw/db tile={block_n} warps={DWDB_NUM_WARPS}",
        lambda: _layer_norm_bwd_dwdb_kernel[(groups, triton.cdiv(n, block_n))](
            dy, x, mean, rstd, gdw, gdb, dy.stride(0), x.stride(0), m, n, rpp_g,
            BLOCK_N=block_n, num_warps=DWDB_NUM_WARPS,
        ),
        2 * m * n * esize,
    )

    xg = x.detach().requires_grad_(True)
    wg = w.detach().requires_grad_(True)
    bg = b.detach().requires_grad_(True)
    y_eager = layer_norm_eager(xg, wg, bg)

    def run_eager() -> None:
        xg.grad = wg.grad = bg.grad = None
        y_eager.backward(dy, retain_graph=True)

    _time("eager bwd", run_eager, 3 * m * n * esize)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register and spill probe for the LayerNorm kernels")
    parser.add_argument("--n", type=int, nargs="+", default=[16384, 32768], help="row widths to probe")
    parser.add_argument("--m", type=int, default=4096, help="number of rows")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("A CUDA or ROCm GPU is required: Triton compiles kernels for the GPU.", file=sys.stderr)
        return 1
    print(f"torch {torch.__version__}, triton {triton.__version__}, {torch.cuda.get_device_name(0)}")
    for n in args.n:
        probe(n, args.m, torch.float16, "cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
