#!/usr/bin/env python3
"""The layout and non-finite checks, rehearsed on a CPU: do they run, and can they fail?

check_strided_inputs (in the three check scripts) and check_non_finite_rows test the
launchers and the kernels, which need a GPU. emulation.py replaces those launchers with
torch arithmetic, so on a CPU a pass says only that the check code runs. This script
therefore also swaps in deliberately broken launchers and requires every check to reject
them: launchers that ignore a stride, that read one tensor with another's row stride, that
hide a NaN or an infinity, or that let a NaN reach other rows. A check that cannot fail
is not a check.

It tests the checks, not the kernels. Only a GPU run tests the kernels.

Usage: python layout_cpu_check.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import torch

import emulation

Launcher = Callable[..., object]
Breaker = Callable[[Launcher], Launcher]


def packed(t: torch.Tensor) -> torch.Tensor:
    """The tensor's storage read as a packed array: what a kernel sees if the launcher ignores strides."""
    strides, step = [], 1
    for size in reversed(t.shape):
        strides.append(step)
        step *= size
    return t.as_strided(t.shape, tuple(reversed(strides)))


def ignores_stride_of(position: int) -> Breaker:
    """A launcher that passes its argument at `position` on without its strides, and without copying it."""

    def breaker(fn: Launcher) -> Launcher:
        def broken(*args: torch.Tensor, **kwargs: object) -> object:
            changed = list(args)
            changed[position] = packed(changed[position])
            return fn(*changed, **kwargs)

        return broken

    return breaker


def swaps_row_strides(fn: Launcher) -> Launcher:
    """A backward launcher that reads its first tensor with the row stride of its second."""

    def broken(first: torch.Tensor, second: torch.Tensor, *rest: object, **kwargs: object) -> object:
        return fn(first.as_strided(first.shape, second.stride()), second, *rest, **kwargs)

    return broken


def hides_bad_values(fn: Launcher) -> Launcher:
    """A forward that swallows NaN and infinity, as a clamp or a max that drops NaN would."""

    def broken(x: torch.Tensor, *rest: object, **kwargs: object) -> object:
        cleaned = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).to(x.dtype)
        return fn(cleaned, *rest, **kwargs)

    return broken


def leaks_across_rows(fn: Launcher) -> Launcher:
    """A forward whose rows are not independent: one NaN reaches every row."""

    def broken(x: torch.Tensor, *rest: object, **kwargs: object) -> object:
        y, mean, rstd = fn(x, *rest, **kwargs)
        return y + 0 * y.float().sum().to(y.dtype), mean, rstd

    return broken


def rejects(module: object, name: str, breaker: Breaker, check: Callable[[str], None]) -> str | None:
    """Run `check` with the launcher `module.name` replaced by a broken one.

    Returns:
        The check's failure message, or None if the check passed, which is the failure here.
    """
    original = getattr(module, name)
    setattr(module, name, breaker(original))
    try:
        check("cpu")
    except AssertionError as error:
        return str(error)
    finally:
        setattr(module, name, original)
    return None


def main() -> int:
    emulation.install()
    # Imported after install(): these modules import triton, which emulation stands in for.
    import layernorm_bwd_checks as ln_bwd
    import layernorm_triton as ln_fwd
    import softmax_checks as sm

    failures = 0
    for check in (
        ln_fwd.check_strided_inputs,
        ln_fwd.check_non_finite_rows,
        ln_bwd.check_strided_inputs,
        sm.check_strided_inputs,
    ):
        check("cpu")
        print(f"ok  runs and passes against the stand-ins: {check.__module__}.{check.__name__}")

    # (what is broken, module, launcher, how it is broken, the check that must reject it)
    broken_launchers: list[tuple[str, object, str, Breaker, Callable[[str], None]]] = [
        (
            "LayerNorm forward ignores x's strides",
            ln_fwd,
            "layer_norm_triton_fwd",
            ignores_stride_of(0),
            ln_fwd.check_strided_inputs,
        ),
        (
            "LayerNorm forward ignores weight's stride",
            ln_fwd,
            "layer_norm_triton_fwd",
            ignores_stride_of(1),
            ln_fwd.check_strided_inputs,
        ),
        (
            "LayerNorm backward ignores dy's strides",
            ln_bwd,
            "layer_norm_triton_bwd",
            ignores_stride_of(0),
            ln_bwd.check_strided_inputs,
        ),
        (
            "LayerNorm backward ignores x's strides",
            ln_bwd,
            "layer_norm_triton_bwd",
            ignores_stride_of(1),
            ln_bwd.check_strided_inputs,
        ),
        (
            "LayerNorm backward ignores weight's stride",
            ln_bwd,
            "layer_norm_triton_bwd",
            ignores_stride_of(2),
            ln_bwd.check_strided_inputs,
        ),
        (
            "LayerNorm backward ignores mean's stride",
            ln_bwd,
            "layer_norm_triton_bwd",
            ignores_stride_of(3),
            ln_bwd.check_strided_inputs,
        ),
        (
            "LayerNorm backward ignores rstd's stride",
            ln_bwd,
            "layer_norm_triton_bwd",
            ignores_stride_of(4),
            ln_bwd.check_strided_inputs,
        ),
        (
            "LayerNorm backward reads dy with x's row stride",
            ln_bwd,
            "layer_norm_triton_bwd",
            swaps_row_strides,
            ln_bwd.check_strided_inputs,
        ),
        ("softmax forward ignores x's strides", sm, "softmax_triton", ignores_stride_of(0), sm.check_strided_inputs),
        (
            "softmax backward ignores dy's strides",
            sm,
            "softmax_triton_bwd",
            ignores_stride_of(0),
            sm.check_strided_inputs,
        ),
        (
            "softmax backward ignores y's strides",
            sm,
            "softmax_triton_bwd",
            ignores_stride_of(1),
            sm.check_strided_inputs,
        ),
        (
            "softmax backward reads dy with y's row stride",
            sm,
            "softmax_triton_bwd",
            swaps_row_strides,
            sm.check_strided_inputs,
        ),
        (
            "LayerNorm forward hides NaN and infinity",
            ln_fwd,
            "layer_norm_triton_fwd",
            hides_bad_values,
            ln_fwd.check_non_finite_rows,
        ),
        (
            "LayerNorm forward lets a NaN reach other rows",
            ln_fwd,
            "layer_norm_triton_fwd",
            leaks_across_rows,
            ln_fwd.check_non_finite_rows,
        ),
    ]
    for what, module, name, breaker, check in broken_launchers:
        message = rejects(module, name, breaker, check)
        if message is None:
            failures += 1
            print(f"FAIL  not rejected: {what}")
        else:
            print(f"ok  rejected: {what}\n      {message}")

    print(f"\n{'OK' if failures == 0 else 'FAILED'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
