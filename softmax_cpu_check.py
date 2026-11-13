#!/usr/bin/env python3
"""CPU check of the softmax error model against an fp32 emulation of the kernels.

Runs without a GPU or Triton. It does not test the Triton kernels. It tests that the
per-element bounds in softmax_checks.py accept arithmetic that follows the kernels step
for step in fp32, and reject the two mutants. Use it before spending GPU time on a change
to the error model. Prints, per case and path, the largest |error| / bound seen: 1.0 or
more is a failure, below 1 is slack.

What it cannot check: the GPU's exp and division are approximate, and EXP_ULPS,
EXP_ARG_ULPS and DIV_ULPS budget for that from NVIDIA's documented error bounds. CPU math
is correctly rounded, so only the GPU run tests those budgets.

Usage: python softmax_cpu_check.py            (needs torch; triton optional)
   or: uv run --with torch python softmax_cpu_check.py
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


def _stub_triton() -> None:
    """Register a stand-in for triton so the kernel modules import on a machine without it.

    Only what those modules touch at import time: the jit decorator and two integer
    helpers. Kernels decorated this way are plain functions and are never launched here.
    """
    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **_: fn if fn is not None else (lambda f: f)  # type: ignore[attr-defined]
    triton.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (n - 1).bit_length()  # type: ignore[attr-defined]
    triton.cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
    language = types.ModuleType("triton.language")
    language.constexpr = int  # type: ignore[attr-defined]
    triton.language = language  # type: ignore[attr-defined]
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language


try:
    import triton  # noqa: F401
except ImportError:
    _stub_triton()

from common import parity_tolerances  # noqa: E402
from softmax_checks import (  # noqa: E402
    CASES,
    make_inputs,
    softmax_bwd_reference_with_bound,
    softmax_reference_with_bound,
)
from softmax_triton import fits_block, online_tiles, tile_for  # noqa: E402

NEG_INF = float("-inf")


def block_fp32(x: torch.Tensor) -> torch.Tensor:
    """The block kernel's arithmetic in fp32."""
    xf = x.float()
    m = xf.max(-1, keepdim=True).values
    e = torch.exp(xf - m)
    denom = e.sum(-1, keepdim=True)
    return (e * (1.0 / denom)).to(x.dtype)


def online_fp32(x: torch.Tensor, rescale: bool = True) -> torch.Tensor:
    """The online kernel's arithmetic in fp32. rescale=False is the mutant."""
    n = x.shape[-1]
    tile = tile_for(n)
    tiles = online_tiles(n)
    rows = x.reshape(-1, n).float()
    m_rows = rows.shape[0]
    padded = torch.full((m_rows, tiles * tile), NEG_INF)
    padded[:, :n] = rows
    m_run = torch.full((m_rows, tile), NEG_INF)
    l_run = torch.zeros((m_rows, tile))
    for t in range(tiles):
        xt = padded[:, t * tile : (t + 1) * tile]
        m_new = torch.maximum(m_run, xt)
        alpha = torch.where(m_new == NEG_INF, torch.ones_like(m_new), torch.exp(m_run - m_new))
        p = torch.where(xt == NEG_INF, torch.zeros_like(xt), torch.exp(xt - m_new))
        l_run = l_run * alpha + p if rescale else l_run + p
        m_run = m_new
    m = m_run.max(-1, keepdim=True).values
    scale = torch.where(m_run == NEG_INF, torch.zeros_like(m_run), torch.exp(m_run - m))
    denom = (l_run * scale).sum(-1, keepdim=True)
    inv = 1.0 / denom
    return (torch.exp(rows - m) * inv).to(x.dtype).reshape(x.shape)


def bwd_fp32(dy: torch.Tensor, y: torch.Tensor, drop_sum: bool = False) -> torch.Tensor:
    """The backward kernels' arithmetic in fp32. drop_sum=True is the mutant."""
    yf, dyf = y.float(), dy.float()
    if drop_sum:
        return (yf * dyf).to(y.dtype)
    s = (dyf * yf).sum(-1, keepdim=True)
    return (yf * (dyf - s)).to(y.dtype)


def ratio(got: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor) -> float:
    """Largest |got - ref| / bound over all elements."""
    return float(((got.double() - ref).abs() / bound).max())


def main() -> int:
    failures = 0
    print(f"{'case':<46} {'block':>7} {'online':>7} {'bwd':>7}  eager-ok")
    for shape, dtype, kind, why in CASES:
        x = make_inputs(shape, dtype, "cpu", kind=kind)
        n = shape[-1]
        cells = []
        y_default = None

        if fits_block(n):
            y_default = block_fp32(x)
            ref, bound = softmax_reference_with_bound(x, 0)
            r = ratio(y_default, ref, bound)
            cells.append(f"{r:7.3f}")
            failures += r >= 1.0
        else:
            cells.append(f"{'-':>7}")

        y = online_fp32(x)
        ref, bound = softmax_reference_with_bound(x, online_tiles(n))
        r = ratio(y, ref, bound)
        cells.append(f"{r:7.3f}")
        failures += r >= 1.0
        if y_default is None:
            y_default = y

        g = torch.Generator().manual_seed(7)
        dy = torch.randn(x.shape, generator=g).to(dtype)
        ref, bound = softmax_bwd_reference_with_bound(dy, y_default)
        r = ratio(bwd_fp32(dy, y_default), ref, bound)
        cells.append(f"{r:7.3f}")
        failures += r >= 1.0

        atol, rtol = parity_tolerances(dtype)
        eager_ok = torch.allclose(y_default.float(), F.softmax(x.float(), dim=-1), atol=atol, rtol=rtol)
        print(f"{why:<46} {' '.join(cells)}  {eager_ok}")

    x = make_inputs((16, 8192), torch.float16, "cpu", kind="ramp")
    ref, bound = softmax_reference_with_bound(x, online_tiles(8192))
    good = ratio(online_fp32(x), ref, bound)
    bad = ratio(online_fp32(x, rescale=False), ref, bound)
    verdict = "REJECTED" if bad >= 1.0 else "ACCEPTED (bad)"
    print(f"\nmutant: online without rescale   correct {good:.3f}  mutant {bad:.3g}  {verdict}")
    failures += bad < 1.0

    x = make_inputs((256, 1024), torch.float16, "cpu", seed=3)
    y = block_fp32(x)
    g = torch.Generator().manual_seed(9)
    dy = (1e-3 * (1.0 + 0.1 * torch.randn(x.shape, generator=g))).half()
    ref, bound = softmax_bwd_reference_with_bound(dy, y)
    good = ratio(bwd_fp32(dy, y), ref, bound)
    bad = ratio(bwd_fp32(dy, y, drop_sum=True), ref, bound)
    xg = x.float().requires_grad_(True)
    F.softmax(xg, dim=-1).backward(dy.float())
    atol, rtol = parity_tolerances(torch.float16)
    allclose_caught = not torch.allclose(bwd_fp32(dy, y, drop_sum=True).float(), xg.grad, atol=atol, rtol=rtol)
    verdict = "REJECTED" if bad >= 1.0 else "ACCEPTED (bad)"
    print(
        f"mutant: backward drops the sum   correct {good:.3f}  mutant {bad:.3g}  {verdict}  "
        f"allclose {'caught it' if allclose_caught else 'missed it'}"
    )
    failures += bad < 1.0

    print(f"\n{'FAIL' if failures else 'OK'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
