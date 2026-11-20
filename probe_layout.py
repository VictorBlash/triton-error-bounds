#!/usr/bin/env python3
"""Does a kernel's output depend on the row stride of its input, and on what about that stride?

softmax_checks.py found the softmax block forward giving 2 of 30000 elements one fp16 unit
in the last place apart between a view of every other row (row stride 20000) and its
contiguous copy (row stride 5000). Triton compiles a separate variant of a kernel when an
integer argument is divisible by 16, and those two strides differ in exactly that. This
probe tests whether that is the cause.

1. For each kernel it holds the same rows at six row strides, three divisible by 16 and
   three not, and groups the strides by the bits of the output. If divisibility is the
   cause the groups are those two halves. One group means the kernel does not depend on
   the stride. Any other grouping means the cause is something else.
2. It repeats the softmax block forward with a second compilation of the same kernel that
   is told not to specialize on its stride arguments.
3. It prints, for every compiled variant of that kernel in Triton's cache, which arguments
   were marked divisible by 16 and which thread layouts Triton chose.

--emulate runs step 1 on the CPU with fp32 torch arithmetic in place of every kernel
(emulation.py), where every stride must give the same bits. That tests this script, not
the kernels. Steps 2 and 3 need Triton and are skipped.

Usage: python probe_layout.py [--emulate]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

# Set before anything imports triton: a fresh cache, so step 3 sees exactly the variants
# this run compiled.
CACHE_DIR = tempfile.mkdtemp(prefix="probe_layout_cache_")
os.environ["TRITON_CACHE_DIR"] = CACHE_DIR

import torch  # noqa: E402

from common import require_cuda  # noqa: E402

M, N = 64, 5000
STRIDES = (5000, 5008, 10000, 10004, 20000, 20008)  # the first is the contiguous layout
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
        return "same bits at every stride"
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


def compiled_variants(kernel_name: str) -> None:
    """Print every compiled variant of a kernel in this run's cache: divisibility marks and thread layouts."""
    paths = sorted(Path(CACHE_DIR).rglob(f"{kernel_name}.ttgir"))
    print(f"\n{len(paths)} compiled variants of {kernel_name} in the cache")
    for path in paths:
        text = path.read_text()
        signature = next((line for line in text.splitlines() if "tt.func" in line), "")
        args = signature[signature.find("(") + 1 :].split(", %")
        marks = "".join("D" if "tt.divisibility" in arg else "-" for arg in args)
        layouts = sorted(set(re.findall(r"#ttg\.blocked<\{[^}]*\}>", text)))
        print(f"  {path.parent.name[:8]}  arguments marked divisible by 16: {marks}")
        for layout in layouts:
            print(f"      {layout}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Does a kernel's output depend on its input's row stride?")
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
    from layernorm_triton_bwd import PATHS, layer_norm_triton_bwd
    from softmax_triton import BWD_PATHS, FWD_PATHS, softmax_triton, softmax_triton_bwd

    x = sm.make_inputs((M, N), torch.float16, device)
    dy = sm.make_inputs((M, N), torch.float16, device, seed=7)
    lx, w, b = ln.make_inputs((M, N), torch.float16, device)
    _, mean, rstd = ln.layer_norm_triton_fwd(lx, w, b)

    print(f"{M} rows of {N} fp16 values held at row strides {STRIDES}\n")
    verdicts = {}
    for path in FWD_PATHS:
        verdicts[f"softmax forward, {path}"] = report(
            f"softmax forward, {path}", lambda s, path=path: (softmax_triton(held_at(x, s), path=path),)
        )
    for path in BWD_PATHS:
        verdicts[f"softmax backward, {path}"] = report(
            f"softmax backward, {path}",
            lambda s, path=path: (softmax_triton_bwd(held_at(dy, s), held_at(x, s), path=path),),
        )
    verdicts["LayerNorm forward"] = report("LayerNorm forward", lambda s: (ln.layer_norm_triton(held_at(lx, s), w, b),))
    for path in PATHS:
        verdicts[f"LayerNorm backward, {path}"] = report(
            f"LayerNorm backward, {path}",
            lambda s, path=path: layer_norm_triton_bwd(held_at(dy, s), held_at(lx, s), w, mean, rstd, path=path),
        )

    if args.emulate:
        print("\nsteps 2 and 3 need Triton: skipped under --emulate")
        failures = sum(v != "same bits at every stride" for v in verdicts.values())
        print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} failures")
        return 1 if failures else 0

    import triton

    import softmax_triton as st
    from common import num_warps_for

    strides = ["x_row_stride", "y_row_stride"]
    try:
        unspecialized = triton.jit(do_not_specialize=strides, do_not_specialize_on_alignment=strides)(
            st._softmax_fwd_block_kernel.fn
        )

        def block_forward_unspecialized(stride: int) -> Outputs:
            rows = held_at(x, stride)
            y = torch.empty_like(rows)
            block = triton.next_power_of_2(N)
            unspecialized[(M,)](
                rows, y, rows.stride(0), y.stride(0), N, BLOCK_SIZE=block, num_warps=num_warps_for(block)
            )
            return (y,)

        print()
        report("softmax forward, block, strides not specialized", block_forward_unspecialized)
    except Exception as error:  # the probe must still print step 3 and its summary
        print(f"\nstep 2 could not run: {type(error).__name__}: {error}")

    try:
        compiled_variants("_softmax_fwd_block_kernel")
    except Exception as error:
        print(f"\nstep 3 could not run: {type(error).__name__}: {error}")

    print("\nsummary")
    for name, verdict in verdicts.items():
        print(f"  {name}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
