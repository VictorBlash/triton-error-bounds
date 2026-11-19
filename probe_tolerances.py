#!/usr/bin/env python3
"""Does a fixed tolerance do the job of the per-element bound?

The mutation tests show the bound rejecting planted faults that torch.allclose accepts at
atol = rtol = 1e-3. That tolerance is this repo's own choice and the faults are planted on
small values, so the result says nothing about a tighter tolerance. This probe measures
it. A check does the job only if it does two things at once: accept every correct kernel
output and reject every planted fault. Each check below is run on
  - every output of the correct kernels on the regular cases of the check scripts, on
    every path that takes the width,
  - the correct kernels on the mutation-test inputs, and
  - the six faulty outputs of the mutation tests.

Fixed tolerances, each accepting an element when |got - want| <= atol + rtol * |want|:
  repo          what the check scripts run: atol = rtol = 1e-3 for fp16 and 8e-3 for bf16,
                with atol times the row count for dw and db.
  assert_close  the defaults of torch.testing.assert_close for the dtype: rtol 1e-3 for
                fp16 and 1.6e-2 for bf16, atol 1e-5.
  rtol_only     the same rtol with atol = 0: nothing is hidden near zero.
  allclose      the defaults of torch.allclose: rtol 1e-5, atol 1e-8.
Each is measured against two references: PyTorch eager in the kernel's dtype, which the
scripts' parity check compares with, and the fp64 reference the bound uses. The last
column is the per-element bound against fp64.

A fixed atol is a statement about one magnitude, so the faults whose size can be set are
then planted again at 0.1 and 0.01 of the mutation tests' magnitude (gradients of 1e-4 and
1e-5 where the tests use 1e-3), with the correct kernels run on the same inputs.

Every number printed is the largest error / allowed error over an output's elements, so
at most 1 means the check accepts that output. The comparison is done in fp64.

--emulate runs the whole script on the CPU with fp32 torch arithmetic in place of every
kernel (emulation.py). It tests this script and needs neither a GPU nor Triton; its
numbers say nothing about the kernels.

Usage: python probe_tolerances.py [--emulate]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch

from common import MutantCase, bound_ratio, parity_tolerances, require_cuda

# (atol, rtol), the order parity_tolerances uses. From the table in the documentation of
# torch.testing.assert_close; check_helpers tests them against the installed torch.
ASSERT_CLOSE_DEFAULTS = {torch.float16: (1e-5, 1e-3), torch.bfloat16: (1e-5, 1.6e-2)}
ALLCLOSE_DEFAULTS = (1e-8, 1e-5)

# Fractions of the mutation tests' magnitude at which the faults are planted again.
SMALLER_MAGNITUDES = (0.1, 0.01)

REFERENCES = ("eager", "fp64")
BOUND = "bound/fp64"
MUTATION_INPUTS = "mutation inputs"
DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16"}
LABEL_WIDTH = 52
COLUMN_WIDTH = 13


class Policy(NamedTuple):
    """A fixed tolerance: how it picks atol and rtol for an output."""

    name: str
    tolerances: Callable[[torch.dtype], tuple[float, float]]  # dtype -> (atol, rtol)
    scales_sums: bool  # multiply atol by the row count for dw and db, as the check scripts do


POLICIES = (
    Policy("repo", parity_tolerances, True),
    Policy("assert_close", ASSERT_CLOSE_DEFAULTS.__getitem__, False),
    Policy("rtol_only", lambda dtype: (0.0, ASSERT_CLOSE_DEFAULTS[dtype][1]), False),
    Policy("allclose", lambda _dtype: ALLCLOSE_DEFAULTS, False),
)
CHECKS = [f"{policy.name}/{reference}" for policy in POLICIES for reference in REFERENCES] + [BOUND]

Ratios = dict[str, float]  # check name -> largest error / allowed error


class Sample(NamedTuple):
    """One output of a correct kernel, with the references and the bound it is judged against."""

    group: str  # which output of which kernel, e.g. "LayerNorm dw"
    label: str  # the case and the path
    got: torch.Tensor
    eager: torch.Tensor  # PyTorch eager's result, in the kernel's dtype
    ref: torch.Tensor  # fp64 reference
    bound: torch.Tensor  # per-element bound on |got - ref|
    atol_scale: int  # the row count for dw and db, else 1


# ---------------------------------------------------------------------------
# Judging one output
# ---------------------------------------------------------------------------


def tolerance_ratio(got: torch.Tensor, want: torch.Tensor, atol: float, rtol: float) -> float:
    """Largest |got - want| / (atol + rtol * |want|) over the elements, computed in fp64.

    torch.allclose and torch.testing.assert_close accept an output exactly when no
    element's error exceeds atol + rtol * |want|, so at most 1 is a pass. An exact element
    counts as 0. A NaN, or any error where nothing is allowed, counts as infinity.
    """
    want64 = want.double()
    return bound_ratio(got, want64, atol + rtol * want64.abs())


def judge(got: torch.Tensor, eager: torch.Tensor, ref: torch.Tensor, bound: torch.Tensor, atol_scale: int) -> Ratios:
    """Largest error / allowed error for one output under every check in CHECKS.

    Args:
        got: the output to judge, fp16 or bf16.
        eager: PyTorch eager's result for the same input, same dtype.
        ref: fp64 reference.
        bound: per-element bound on |correct output - ref|.
        atol_scale: what a policy with scales_sums multiplies atol by.
    """
    ratios = {}
    for policy in POLICIES:
        atol, rtol = policy.tolerances(got.dtype)
        if policy.scales_sums:
            atol *= atol_scale
        for reference, want in zip(REFERENCES, (eager, ref), strict=True):
            ratios[f"{policy.name}/{reference}"] = tolerance_ratio(got, want, atol, rtol)
    ratios[BOUND] = bound_ratio(got, ref, bound)
    return ratios


def check_helpers() -> int:
    """Check tolerance_ratio on hand-made values and the assert_close table against torch itself.

    The table is only worth using if torch.testing.assert_close, called with no tolerance
    arguments, accepts and rejects where the table says it does. Each entry is tested at
    half and at twice its allowance, on a value of 1 for rtol and of 0 for atol.

    Returns:
        Number of failed expectations.
    """
    failures = 0
    want = torch.tensor([2.0, 0.0], dtype=torch.float64)  # allowance with atol 0.5, rtol 0.25: 1.0 and 0.5
    inf = float("inf")
    cases = [
        ("exact result", [0.0, 0.0], 0.5, 0.0),
        ("half the allowance through rtol", [0.5, 0.0], 0.5, 0.5),
        ("twice the allowance through atol", [0.0, 1.0], 0.5, 2.0),
        ("NaN output", [float("nan"), 0.0], 0.5, inf),
        ("exact zero with atol 0", [0.0, 0.0], 0.0, 0.0),
        ("error at zero with atol 0", [0.0, 1e-9], 0.0, inf),
    ]
    for name, offset, atol, expected in cases:
        ratio = tolerance_ratio(want + torch.tensor(offset, dtype=torch.float64), want, atol=atol, rtol=0.25)
        if ratio != expected:
            print(f"ratio helper: {name}: got {ratio}, want {expected}")
            failures += 1

    for dtype, (atol, rtol) in ASSERT_CLOSE_DEFAULTS.items():
        for base in (1.0, 0.0):
            got = torch.tensor([base], dtype=dtype)
            for factor in (0.5, 2.0):
                want = got.double() + factor * (atol + rtol * base)
                try:
                    torch.testing.assert_close(got, want, check_dtype=False)
                    accepted = True
                except AssertionError:
                    accepted = False
                ours = tolerance_ratio(got, want, atol, rtol) <= 1.0
                if accepted != (factor < 1.0) or ours != accepted:
                    print(
                        f"assert_close table: {DTYPE_NAMES[dtype]} at {base}, {factor} x the allowance: "
                        f"torch {'accepts' if accepted else 'rejects'}, the table {'accepts' if ours else 'rejects'}"
                    )
                    failures += 1
    return failures


# ---------------------------------------------------------------------------
# Collecting outputs
# ---------------------------------------------------------------------------


def _seeded_like(x: torch.Tensor) -> torch.Tensor:
    """The upstream gradient the check scripts use for x: standard normal, seed 7, in the dtype of x."""
    g = torch.Generator(device=x.device).manual_seed(7)
    return torch.randn(x.shape, generator=g, device=x.device).to(x.dtype)


def regular_samples(device: str) -> Iterator[Sample]:
    """Yield every output of the correct kernels on the regular cases of the check scripts.

    Follows check_kernel and check_backward in the three check scripts: same inputs, same
    upstream gradients, every path that takes the width.

    Args:
        device: "cuda", or "cpu" after emulation.install().
    """
    # Imported here so that --emulate can register its stand-ins first.
    import layernorm_bwd_checks as ln_bwd
    import layernorm_triton as ln
    import layernorm_triton_bwd as ln_kernels
    import softmax_checks as sm
    import softmax_triton as sm_kernels

    for shape, dtype, offset, _ in ln.CASES:
        x, w, b = ln.make_inputs(shape, dtype, device, offset)
        case = f"{str(shape):<15}{DTYPE_NAMES[dtype]:<5}"
        y, mean, rstd = ln.layer_norm_triton_fwd(x, w, b)
        y_ref, y_bound = ln.layer_norm_reference_with_bound(x, w, b)
        yield Sample("LayerNorm y", case, y, ln.layer_norm_eager(x, w, b), y_ref, y_bound, 1)

        dy = _seeded_like(x)
        eager = ln_bwd.eager_grads(dy, x, w, b, ln.EPS)
        refs, bounds = ln_bwd.layer_norm_bwd_reference_with_bound(dy, x, w)
        rows = x.reshape(-1, shape[-1]).shape[0]
        for path in ln_kernels.PATHS:
            got = ln_kernels.layer_norm_triton_bwd(dy, x, w, mean, rstd, path=path)
            parts = zip(("dx", "dw", "db"), (1, rows, rows), got, eager, refs, bounds, strict=True)
            for name, scale, grad, grad_eager, grad_ref, grad_bound in parts:
                yield Sample(f"LayerNorm {name}", f"{case}{path}", grad, grad_eager, grad_ref, grad_bound, scale)

    for shape, dtype, kind, _ in sm.CASES:
        x = sm.make_inputs(shape, dtype, device, kind=kind)
        n = shape[-1]
        case = f"{str(shape):<15}{DTYPE_NAMES[dtype]:<5}{'' if kind == 'normal' else kind + ' '}"
        y_eager = sm.softmax_eager(x)
        ys = {}
        for path in sm_kernels.FWD_PATHS:
            if path == "block" and not sm_kernels.fits_block(n):
                continue
            ys[path] = sm_kernels.softmax_triton(x, path=path)
            y_ref, y_bound = sm.softmax_reference_with_bound(x, 0 if path == "block" else sm_kernels.online_tiles(n))
            yield Sample("softmax y", f"{case}{path}", ys[path], y_eager, y_ref, y_bound, 1)

        y = ys[sm_kernels.default_fwd_path(n)]
        dy = _seeded_like(x)
        xg = x.detach().requires_grad_(True)
        sm.softmax_eager(xg).backward(dy)
        dx_ref, dx_bound = sm.softmax_bwd_reference_with_bound(dy, y)
        for path in sm_kernels.BWD_PATHS:
            if path == "block" and not sm_kernels.fits_block(n):
                continue
            dx = sm_kernels.softmax_triton_bwd(dy, y, path=path)
            yield Sample("softmax dx", f"{case}{path}", dx, xg.grad, dx_ref, dx_bound, 1)


def planted_faults(device: str, shrink: float = 1.0) -> list[tuple[str, MutantCase]]:
    """The mutation-test cases of the three check scripts, each with the kernel it belongs to.

    Args:
        device: "cuda", or "cpu" after emulation.install().
        shrink: 1 gives the six cases as the mutation tests run them. Any other value
            gives the five whose magnitude can be set, planted at that fraction of it. The
            online softmax forward has no magnitude to set and is left out.
    """
    import layernorm_bwd_checks as ln_bwd
    import layernorm_triton as ln
    import softmax_checks as sm

    layer_norm = [ln.dropped_bias_case(device, shrink), *ln_bwd.mutant_cases(device, shrink)]
    softmax = [sm.dropped_sum_case(device, shrink)]
    if shrink == 1.0:
        softmax.insert(0, sm.no_rescale_case(device))
    return [("LayerNorm", case) for case in layer_norm] + [("softmax", case) for case in softmax]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    names = [check.split("/") for check in CHECKS]
    print(f"\n{title}")
    print(f"{'':<{LABEL_WIDTH}}" + "".join(f"{policy:>{COLUMN_WIDTH}}" for policy, _ in names))
    print(f"{'':<{LABEL_WIDTH}}" + "".join(f"{'vs ' + reference:>{COLUMN_WIDTH}}" for _, reference in names))


def _print_row(label: str, ratios: Ratios) -> None:
    print(f"{label:<{LABEL_WIDTH}}" + "".join(f"{ratios[check]:>{COLUMN_WIDTH}.3g}" for check in CHECKS))


def report_faults(device: str, shrink: float) -> tuple[list[Ratios], list[Ratios]]:
    """Judge and print every planted fault and the correct kernels on the same inputs.

    The faulty row also says how many elements differ from the first path's correct
    output. A fault that changes no element, or changes one only within its rounding, is
    one that no check of the output can see.

    Returns:
        (ratios of each correct output, ratios of each faulty output).
    """
    correct, faults = [], []
    for kernel, case in planted_faults(device, shrink):
        print(f"{kernel}: {case.label}")
        for path, good in case.good.items():
            ratios = judge(good, case.eager, case.ref, case.bound, case.atol_scale)
            _print_row(f"  correct kernel, {path}", ratios)
            correct.append(ratios)
        ratios = judge(case.bad, case.eager, case.ref, case.bound, case.atol_scale)
        changed = (case.bad != next(iter(case.good.values()))).double().mean().item()
        _print_row(f"  faulty kernel, {changed:.1%} of elements changed", ratios)
        faults.append(ratios)
    return correct, faults


def print_counts_by_output(correct: list[tuple[str, Ratios]]) -> None:
    """Print, per kernel output and check, how many correct outputs the check accepts."""
    _print_header("correct outputs accepted / checked, by output")
    for group in dict.fromkeys(group for group, _ in correct):
        rows = [ratios for g, ratios in correct if g == group]
        cells = [f"{sum(r[check] <= 1.0 for r in rows)}/{len(rows)}" for check in CHECKS]
        print(f"{group:<{LABEL_WIDTH}}" + "".join(f"{cell:>{COLUMN_WIDTH}}" for cell in cells))


def print_verdicts(title: str, correct: list[Ratios], faults: list[Ratios]) -> bool:
    """Print, per check, how many correct outputs it accepts and how many faults it rejects.

    Returns:
        True if the per-element bound accepted every correct output and rejected every fault.
    """
    print(f"\n{title}")
    print(f"{'check':<20}{'correct accepted':>18}{'faults rejected':>18}   does both")
    does_both = {}
    for check in CHECKS:
        accepted = sum(ratios[check] <= 1.0 for ratios in correct)
        rejected = sum(ratios[check] > 1.0 for ratios in faults)
        does_both[check] = accepted == len(correct) and rejected == len(faults)
        print(
            f"{check:<20}{f'{accepted}/{len(correct)}':>18}{f'{rejected}/{len(faults)}':>18}   "
            f"{'yes' if does_both[check] else 'no'}"
        )
    return does_both[BOUND]


def main() -> int:
    """Run the probe.

    Returns:
        0 if the helper checks passed and, at the magnitudes the check scripts use, the
        per-element bound accepted every correct output and rejected every fault, as those
        scripts assert it does. 1 otherwise. What the fixed tolerances do, and what any
        check does at the smaller magnitudes, is the measurement and does not change the
        status.
    """
    parser = argparse.ArgumentParser(description="Fixed tolerances against the per-element bound")
    parser.add_argument("--emulate", action="store_true", help="CPU only: fp32 torch arithmetic for every kernel")
    args = parser.parse_args()

    if args.emulate:
        import emulation

        emulation.install()
        device = "cpu"
        print("EMULATED on the CPU: fp32 torch arithmetic stands in for every kernel.")
        print("This tests the script, not the kernels.")
    else:
        cuda = require_cuda()
        if cuda is None:
            return 1
        device = cuda

    failures = check_helpers()

    regular: list[tuple[str, Ratios]] = []
    _print_header("correct kernels, regular check cases: largest error / allowed error (at most 1 is accepted)")
    for sample in regular_samples(device):
        ratios = judge(sample.got, sample.eager, sample.ref, sample.bound, sample.atol_scale)
        _print_row(f"{sample.group:<13}{sample.label}", ratios)
        regular.append((sample.group, ratios))

    _print_header("mutation-test inputs: largest error / allowed error (a fault is rejected above 1)")
    good, faults = report_faults(device, 1.0)
    print_counts_by_output(regular + [(MUTATION_INPUTS, ratios) for ratios in good])
    title = "summary: a check does the job only if it accepts every correct output and rejects every fault"
    if not print_verdicts(title, [ratios for _, ratios in regular] + good, faults):
        print("the per-element bound rejected a correct output or accepted a fault: see the bound column")
        failures += 1

    for shrink in SMALLER_MAGNITUDES:
        _print_header(f"the same faults at {shrink:g} of the magnitude: largest error / allowed error")
        good, faults = report_faults(device, shrink)
        if not print_verdicts(f"summary at {shrink:g} of the magnitude, on these inputs only", good, faults):
            print("note: here the per-element bound rejected a correct output or accepted a fault")

    print(f"\n{'FAIL' if failures else 'OK'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
