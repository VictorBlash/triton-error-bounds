#!/usr/bin/env python3
"""Do a kernel's output bits depend on the row stride of its input, and what does removing that cost?

softmax_checks.py found the softmax block forward giving 2 of 30000 elements one fp16 unit
in the last place apart between a view of every other row (row stride 20000) and its
contiguous copy (row stride 5000). The first run of this probe showed why: Triton compiles a
separate variant of a kernel when an integer argument is divisible by 16, with a different
thread layout, and all four softmax kernels gave different bits in the two variants
(RESULTS.md, "Strided and non-finite inputs"). The softmax kernels now leave their stride
arguments out of that specialization. This probe measures both states.

1. With the kernels as they are, it holds the same rows at six row strides, three
   divisible by 16 and three not, and groups the strides by the bits of the output. One
   group means the output does not depend on the stride.
2. It does the same with each softmax kernel recompiled the way Triton compiles by
   default, specializing on the strides: the state the failure was found in.
3. It times each softmax kernel both ways on contiguous inputs, to price the change.
4. It prints, for every compiled variant of the block forward in Triton's cache, which
   arguments were marked divisible by 16 and which thread layouts Triton chose.

The exit status is 1 if step 1 finds a kernel whose bits depend on the stride.

--emulate runs step 1 on the CPU with fp32 torch arithmetic in place of every kernel
(emulation.py), where every stride must give the same bits. That tests this script, not
the kernels. Steps 2 to 4 need Triton and are skipped.

Usage: python probe_layout.py [--emulate]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

# Set before anything imports triton: a fresh cache, so step 4 sees exactly the variants
# this run compiled.
CACHE_DIR = tempfile.mkdtemp(prefix="probe_layout_cache_")
os.environ["TRITON_CACHE_DIR"] = CACHE_DIR

import torch  # noqa: E402

from common import require_cuda, time_fn  # noqa: E402

M, N = 64, 5000
STRIDES = (5000, 5008, 10000, 10004, 20000, 20008)  # the first is the contiguous layout
BENCH_M = 4096  # the row count of every benchmark table in RESULTS.md
BENCH_WIDTHS = (1024, 4096, 16384, 32768)
SAME_BITS = "same bits at every stride"
Outputs = tuple[torch.Tensor, ...]


def held_at(rows: torch.Tensor, stride: int) -> torch.Tensor:
    """The same (M, N) values as the first N columns of a tensor `stride` wide, so with that row stride.

    The other columns hold random numbers: reading them would change most of the output.
    """
    g = torch.Generator(device=rows.device).manual_seed(stride)
    base = torch.randn((rows.shape[0], stride), generator=g, device=rows.device).to(rows.dtype)
    base[:, : rows.shape[1]] = rows
    held = base[:, : rows.shape[1]]
    assert held.stride() == (stride, 1) and torch.equal(held, rows)
    return held


def group_by_bits(outputs: dict[int, Outputs]) -> list[list[int]]:
    """Group the strides whose outputs are bit-identical, in the order the strides were given."""
    groups: list[list[int]] = []
    for stride, out in outputs.items():
        for group in groups:
            if all(torch.equal(a, b) for a, b in zip(out, outputs[group[0]], strict=True)):
                group.append(stride)
                break
        else:
            groups.append([stride])
    return groups


def describe(groups: list[list[int]]) -> str:
    """Name the grouping: one group, the two divisibility classes, or something else."""
    if len(groups) == 1:
        return SAME_BITS
    by_16 = [[s for s in STRIDES if s % 16 == 0], [s for s in STRIDES if s % 16 != 0]]
    if sorted(map(sorted, groups)) == sorted(by_16):
        return "SPLITS EXACTLY BY DIVISIBILITY BY 16"
    return "SPLITS SOME OTHER WAY"


def report(name: str, run: Callable[[int], Outputs]) -> str:
    """Run one kernel at every stride, print the groups and how far apart they are, return the verdict."""
    outputs = {stride: run(stride) for stride in STRIDES}
    groups = group_by_bits(outputs)
    verdict = describe(groups)
    print(f"{name}: {verdict}")
    for group in groups:
        print(f"    {group}")
    first = outputs[groups[0][0]][0]
    for group in groups[1:]:
        other = outputs[group[0]][0]
        diff = (other.double() - first.double()).abs()
        print(
            f"    stride {group[0]} against {groups[0][0]}: {int((other != first).sum())} of {first.numel()} "
            f"elements differ, largest |difference| {diff.max().item():.3e}"
        )
    return verdict


@contextlib.contextmanager
def default_specialization(module: object, name: str) -> Iterator[None]:
    """Swap one kernel in `module` for the same source compiled the way Triton compiles by default."""
    import triton

    current = getattr(module, name)
    setattr(module, name, triton.jit(current.fn))
    try:
        yield
    finally:
        setattr(module, name, current)


def compiled_variants(kernel_name: str) -> None:
    """Print every compiled variant of a kernel in this run's cache: divisibility marks and thread layouts."""
    paths = sorted(Path(CACHE_DIR).rglob(f"{kernel_name}.ttgir"))
    print(f"\n{len(paths)} compiled variants of {kernel_name} in the cache; distinct ones:")
    seen = set()
    for path in paths:
        text = path.read_text()
        signature = next((line for line in text.splitlines() if "tt.func" in line), "")
        args = signature[signature.find("(") + 1 :].split(", %")
        marks = "".join("D" if "tt.divisibility" in arg else "-" for arg in args)
        layouts = tuple(sorted(set(re.findall(r"#ttg\.blocked<\{[^}]*\}>", text))))
        if (marks, layouts) in seen:
            continue
        seen.add((marks, layouts))
        print(f"  arguments marked divisible by 16: {marks}")
        for layout in layouts:
            print(f"      {layout}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Do a kernel's output bits depend on its input's row stride?")
    parser.add_argument("--emulate", action="store_true", help="CPU only: fp32 torch arithmetic for every kernel")
    args = parser.parse_args()
    if args.emulate:
        import emulation

        emulation.install()
        device = "cpu"
    else:
        cuda = require_cuda()
        if cuda is None:
            return 1
        device = cuda

    # Imported here so that --emulate can register its stand-ins first.
    import layernorm_triton as ln
    import softmax_checks as sm
    import softmax_triton as st
    from layernorm_triton_bwd import PATHS, layer_norm_triton_bwd
    from softmax_triton import softmax_triton, softmax_triton_bwd

    # name: (the kernel's name in softmax_triton, how to run it on an input and an upstream gradient)
    softmax_kernels: dict[str, tuple[str, Callable[[torch.Tensor, torch.Tensor], Outputs]]] = {
        "softmax forward, block": ("_softmax_fwd_block_kernel", lambda a, g: (softmax_triton(a, path="block"),)),
        "softmax forward, online": ("_softmax_fwd_online_kernel", lambda a, g: (softmax_triton(a, path="online"),)),
        "softmax backward, block": (
            "_softmax_bwd_block_kernel",
            lambda a, g: (softmax_triton_bwd(g, a, path="block"),),
        ),
        "softmax backward, tiled": (
            "_softmax_bwd_tiled_kernel",
            lambda a, g: (softmax_triton_bwd(g, a, path="tiled"),),
        ),
    }

    x = sm.make_inputs((M, N), torch.float16, device)
    dy = sm.make_inputs((M, N), torch.float16, device, seed=7)
    lx, w, b = ln.make_inputs((M, N), torch.float16, device)
    _, mean, rstd = ln.layer_norm_triton_fwd(lx, w, b)

    print(f"step 1: the kernels as they are, {M} rows of {N} fp16 values held at row strides {STRIDES}\n")
    verdicts = {}
    for name, (_, run) in softmax_kernels.items():
        verdicts[name] = report(name, lambda s, run=run: run(held_at(x, s), held_at(dy, s)))
    verdicts["LayerNorm forward"] = report("LayerNorm forward", lambda s: (ln.layer_norm_triton(held_at(lx, s), w, b),))
    for path in PATHS:
        verdicts[f"LayerNorm backward, {path}"] = report(
            f"LayerNorm backward, {path}",
            lambda s, path=path: layer_norm_triton_bwd(held_at(dy, s), held_at(lx, s), w, mean, rstd, path=path),
        )
    failures = sum(verdict != SAME_BITS for verdict in verdicts.values())

    if args.emulate:
        print("\nsteps 2 to 4 need Triton: skipped under --emulate")
        print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} failures")
        return 1 if failures else 0

    print("\nstep 2: each softmax kernel recompiled with Triton's default specialization on its strides\n")
    for name, (kernel, run) in softmax_kernels.items():
        with default_specialization(st, kernel):
            report(name, lambda s, run=run: run(held_at(x, s), held_at(dy, s)))

    print(f"\nstep 3: mean ms on contiguous inputs, {BENCH_M} rows. off = strides left out of specialization")
    print("(the kernels as they are), on = Triton's default. off / on above 1 means the change costs time.\n")
    print(f"{'kernel':<24} {'N':>6} {'off ms':>9} {'on ms':>9} {'off / on':>9} {'by median':>10}")
    worst = (0.0, "")
    for n in BENCH_WIDTHS:
        xb = sm.make_inputs((BENCH_M, n), torch.float16, device)
        yb = softmax_triton(xb)
        dyb = torch.randn_like(xb)
        for name, (kernel, run) in softmax_kernels.items():
            first = xb if "forward" in name else yb
            off = time_fn(lambda: run(first, dyb))
            with default_specialization(st, kernel):
                on = time_fn(lambda: run(first, dyb))
            ratio = off.mean / on.mean
            worst = max(worst, (ratio, f"{name} at N = {n}"))
            print(f"{name:<24} {n:>6} {off.mean:9.4f} {on.mean:9.4f} {ratio:9.3f} {off.median / on.median:10.3f}")
    print(f"\nlargest off / on: {worst[0]:.3f} ({worst[1]})")

    try:
        compiled_variants("_softmax_fwd_block_kernel")
    except Exception as error:  # reading Triton's cache is the one step that depends on its internals
        print(f"\nstep 4 could not run: {type(error).__name__}: {error}")

    print("\nsummary of step 1")
    for name, verdict in verdicts.items():
        print(f"  {name}: {verdict}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} kernels whose bits depend on the row stride")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
