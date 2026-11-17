#!/usr/bin/env python3
"""Go / no-go probe: can a fused Triton attention forward beat PyTorch on this GPU?

Attention is softmax(Q K^T * scale) V. A fused kernel never writes the L x L score matrix
to memory, which is where PyTorch's unfused path spends most of its time. But the kernel
then has to do both matrix products itself, and on a GPU below compute capability 8.0
Triton lowers tl.dot to plain fp32 multiply-adds, not to tensor cores (Triton
AccelerateMatmul.cpp: "Dot op using MMA for compute capability < 80 has been deprecated.
It falls back to the FMA path"). PyTorch's matmul does use tensor cores there. So which
side wins on a T4 is a measurement, and this probe is that measurement, made before
building a checked kernel with an error bound.

The kernel here is a prototype: non-causal, one program per block of query rows, online
softmax over tiles of keys, as in softmax_triton.py's online path and Triton's tutorial
06-fused-attention.py. It is checked only against an fp32 reference with a fixed
tolerance, which is enough to trust its timing and not enough to ship it.

Baselines, all fp16 on (batch, heads, L, head_dim):
  sdpa     F.scaled_dot_product_attention, PyTorch's built-in
  math     softmax(q @ k^T * scale) @ v, written out
  compile  torch.compile of the written-out form
TFLOPS counts the two matrix products: 4 * groups * L * L * head_dim per call.

Usage: python probe_attention.py [--no-compile]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import compile_fresh, require_cuda

BATCH, HEADS, HEAD_DIM = 4, 8, 64
LENGTHS = (256, 512, 1024, 2048)
# (query rows per program, keys per tile, warps). Small on purpose: this asks whether any
# reasonable setting is competitive, not which one is best.
CONFIGS = ((32, 32, 4), (64, 32, 4), (64, 64, 4), (64, 64, 8), (128, 32, 8), (128, 64, 8))
SANITY_ATOL = 2e-2


@triton.jit
def _attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, scale, L, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """o = softmax(q k^T * scale) v for one block of query rows of one group. Prototype.

    Expects contiguous (groups, L, HEAD_DIM) fp16 tensors with L a multiple of BLOCK_M and
    BLOCK_N. Grid: (L // BLOCK_M, groups).
    """
    start_m = tl.program_id(0)
    base = tl.program_id(1).to(tl.int64) * L * HEAD_DIM
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_offs = base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    q = tl.load(q_ptr + q_offs)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, L, BLOCK_N):
        kv_offs = base + (start_n + offs_n[:, None]) * HEAD_DIM + offs_d[None, :]
        k = tl.load(k_ptr + kv_offs)
        qk = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptr + kv_offs)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + q_offs, acc.to(tl.float16))


def attention_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_m: int, block_n: int, num_warps: int
) -> torch.Tensor:
    """Fused attention forward with the prototype kernel.

    Args:
        q, k, v: fp16 CUDA tensors of shape (batch, heads, L, head_dim), contiguous.
        block_m: query rows per program.
        block_n: keys per tile.
        num_warps: warps per program.

    Returns:
        Output of the same shape and dtype as q.

    Raises:
        ValueError: L is not a multiple of both block sizes.
    """
    b, h, length, d = q.shape
    if length % block_m or length % block_n:
        raise ValueError(f"L={length} must be a multiple of block_m={block_m} and block_n={block_n}")
    out = torch.empty_like(q)
    _attn_fwd_kernel[(length // block_m, b * h)](
        q, k, v, out, 1.0 / math.sqrt(d), length, HEAD_DIM=d, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps
    )
    return out


def attention_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention written out: two matmuls around a softmax."""
    return torch.softmax(q @ k.transpose(-2, -1) * (1.0 / math.sqrt(q.shape[-1])), dim=-1) @ v


def _ms(fn: Callable[[], object]) -> float:
    return triton.testing.do_bench(fn)


def probe(length: int, device: str, compile_baseline: bool) -> float:
    """Check and time every variant at one sequence length.

    Returns:
        Best PyTorch time / best Triton time: above 1 means the fused kernel is faster.

    Raises:
        AssertionError: the prototype's output is not close to the fp32 reference.
    """
    g = torch.Generator(device=device).manual_seed(0)
    shape = (BATCH, HEADS, length, HEAD_DIM)
    q, k, v = (torch.randn(shape, generator=g, device=device).half() for _ in range(3))
    reference = attention_math(q.float(), k.float(), v.float())
    flops = 4 * BATCH * HEADS * length * length * HEAD_DIM

    def line(name: str, ms: float) -> None:
        print(f"  {name:<28} {ms:9.4f} ms  {flops / (ms * 1e-3) / 1e12:7.2f} TFLOPS")

    print(f"\nL={length}, batch={BATCH}, heads={HEADS}, head_dim={HEAD_DIM}, fp16")
    baselines = {
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v),
        "math": lambda: attention_math(q, k, v),
    }
    if compile_baseline:
        compiled = compile_fresh(attention_math)
        baselines["compile"] = lambda: compiled(q, k, v)
    best_torch = math.inf
    for name, fn in baselines.items():
        diff = (fn().float() - reference).abs().max().item()
        assert diff < SANITY_ATOL, f"{name} differs from the fp32 reference by {diff:.3e}"
        ms = _ms(fn)
        best_torch = min(best_torch, ms)
        line(name, ms)

    best_triton, best_config = math.inf, None
    for block_m, block_n, warps in CONFIGS:
        if length % block_m or length % block_n:
            continue
        out = attention_triton(q, k, v, block_m, block_n, warps)
        diff = (out.float() - reference).abs().max().item()
        assert diff < SANITY_ATOL, f"triton {block_m}x{block_n} differs from the fp32 reference by {diff:.3e}"
        ms = _ms(lambda bm=block_m, bn=block_n, w=warps: attention_triton(q, k, v, bm, bn, w))
        line(f"triton {block_m}x{block_n} warps={warps}", ms)
        if ms < best_triton:
            best_triton, best_config = ms, (block_m, block_n, warps)
    ratio = best_torch / best_triton
    print(f"  best PyTorch / best Triton {best_config}: {ratio:.2f}x")
    return ratio


def main() -> int:
    """Run the probe at every length and print the verdict."""
    parser = argparse.ArgumentParser(description="Go / no-go probe for a fused Triton attention forward")
    parser.add_argument("--no-compile", action="store_true", help="leave the torch.compile baseline out")
    args = parser.parse_args()

    device = require_cuda()
    if device is None:
        return 1
    capability = torch.cuda.get_device_capability(0)
    print(
        f"torch {torch.__version__}, triton {triton.__version__}, {torch.cuda.get_device_name(0)}, "
        f"compute capability {capability[0]}.{capability[1]}"
    )
    backends = torch.backends.cuda
    print(
        "sdpa backends enabled: "
        f"flash={backends.flash_sdp_enabled()}, mem_efficient={backends.mem_efficient_sdp_enabled()}, "
        f"math={backends.math_sdp_enabled()}"
    )

    ratios = {length: probe(length, device, not args.no_compile) for length in LENGTHS}
    wins = [length for length, ratio in ratios.items() if ratio > 1.0]
    print("\nfused Triton forward against the best PyTorch baseline:")
    for length, ratio in ratios.items():
        print(f"  L={length:<5} {ratio:.2f}x")
    print(f"VERDICT: {'faster at L = ' + ', '.join(map(str, wins)) if wins else 'slower at every length'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
