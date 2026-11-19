"""fp32 torch stand-ins for the Triton kernels, to rehearse the GPU scripts on the CPU.

install() replaces every kernel launch the check scripts make, and the eager results they
compare against, with torch arithmetic that follows the kernels step for step in fp32.
The scripts' own code then runs on a machine with no GPU and no Triton: the inputs, the
fp64 references, the bounds, the mutation tests and anything built on them.

This tests the scripts, not the kernels. CPU math is correctly rounded, the stand-ins sum
in torch's order, and the eager stand-ins compute in fp32 and round once, which is not
what PyTorch does with fp16 on a GPU. Only a GPU run says anything about the kernels.

Used by probe_tolerances.py --emulate and layout_cpu_check.py.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

import softmax_cpu_check as cpu  # also registers a stand-in for triton where it is not installed

EPS = 1e-5  # the kernels' eps; install() checks it against layernorm_triton.EPS

Grads = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def layer_norm_fwd_fp32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> Grads:
    """The forward kernel's arithmetic in fp32.

    Returns:
        (y, mean, rstd) as layer_norm_triton_fwd returns them: y in the dtype and shape of
        x, mean and rstd in fp32 with one value per row.
    """
    n = x.shape[-1]
    rows = x.reshape(-1, n).float()
    mean = rows.sum(-1, keepdim=True) / n
    xc = rows - mean
    var = (xc * xc).sum(-1, keepdim=True) / n
    rstd = 1.0 / torch.sqrt(var + eps)
    y = xc * rstd * weight.float() + bias.float()
    return y.to(x.dtype).reshape(x.shape), mean[:, 0], rstd[:, 0]


def layer_norm_bwd_fp32(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    path: str | None = None,
) -> Grads:
    """The backward kernels' arithmetic in fp32.

    Every path computes this; they differ only in how the sums are grouped, so `path` is
    accepted and ignored.

    Returns:
        (dx, dweight, dbias) as layer_norm_triton_bwd returns them.
    """
    n = x.shape[-1]
    xf, dyf = x.reshape(-1, n).float(), dy.reshape(-1, n).float()
    rstd_col = rstd[:, None]
    xhat = (xf - mean[:, None]) * rstd_col
    wdy = weight.float() * dyf
    c1 = (xhat * wdy).sum(-1, keepdim=True) / n
    c2 = wdy.sum(-1, keepdim=True) / n
    dx = (wdy - (xhat * c1 + c2)) * rstd_col
    return dx.to(x.dtype).reshape(x.shape), (dyf * xhat).sum(0).to(weight.dtype), dyf.sum(0).to(weight.dtype)


def layer_norm_eager_fp32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Stand-in for eager LayerNorm: fp32 arithmetic, rounded once to the dtype of x."""
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)


def softmax_fp32(x: torch.Tensor, path: str | None = None) -> torch.Tensor:
    """The softmax forward kernels' arithmetic in fp32, on the path softmax_triton would take."""
    if path == "online" or not cpu.fits_block(x.shape[-1]):
        return cpu.online_fp32(x)
    return cpu.block_fp32(x)


def softmax_bwd_fp32(dy: torch.Tensor, y: torch.Tensor, path: str | None = None) -> torch.Tensor:
    """The softmax backward kernels' arithmetic in fp32. Both paths compute this, so `path` is ignored."""
    return cpu.bwd_fp32(dy, y)


def softmax_eager_fp32(x: torch.Tensor) -> torch.Tensor:
    """Stand-in for eager softmax: fp32 arithmetic, rounded once to the dtype of x."""
    return F.softmax(x.float(), dim=-1).to(x.dtype)


def _dx_drop_c1(dy: torch.Tensor, w: torch.Tensor, rstd: torch.Tensor, dx: torch.Tensor, n: int) -> None:
    """Stand-in for _mutant_dx_drop_c1_kernel: writes the faulty dx into `dx`."""
    wdy = w.float() * dy.float()
    dx.copy_((wdy - wdy.sum(-1, keepdim=True) / n) * rstd[:, None])


def _online_no_rescale(x: torch.Tensor, y: torch.Tensor, n: int) -> None:
    """Stand-in for _mutant_online_no_rescale_kernel: writes the faulty y into `y`."""
    y.copy_(cpu.online_fp32(x, rescale=False))


def _bwd_drop_sum(dy: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, n: int) -> None:
    """Stand-in for _mutant_bwd_drop_sum_kernel: writes the faulty dx into `dx`."""
    dx.copy_(cpu.bwd_fp32(dy, y, drop_sum=True))


class _Launch:
    """Stands in for a jitted kernel: `kernel[grid](*args, **constexprs)` calls `fn(*args)`."""

    def __init__(self, fn: Callable[..., None]) -> None:
        self._fn = fn

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        return lambda *args, **_constexprs: self._fn(*args)


STAND_INS: dict[str, object] = {
    "layer_norm_triton_fwd": layer_norm_fwd_fp32,
    "layer_norm_triton_bwd": layer_norm_bwd_fp32,
    "layer_norm_eager": layer_norm_eager_fp32,
    "softmax_triton": softmax_fp32,
    "softmax_triton_bwd": softmax_bwd_fp32,
    "softmax_eager": softmax_eager_fp32,
    "_mutant_dx_drop_c1_kernel": _Launch(_dx_drop_c1),
    "_mutant_online_no_rescale_kernel": _Launch(_online_no_rescale),
    "_mutant_bwd_drop_sum_kernel": _Launch(_bwd_drop_sum),
}


def install() -> None:
    """Point every kernel launch and eager result in the kernel and check modules at the stand-ins.

    Each module holds its own reference to a launcher it imported, so every module that
    has one of the names in STAND_INS gets it replaced. Call this before using the
    modules. It cannot be undone within a process.

    Raises:
        AssertionError: the kernels' eps is not the one the stand-ins assume, or a name in
            STAND_INS exists in no module, which means it was renamed and is no longer
            being replaced.
    """
    # Imported here: these modules import triton, and the stand-in for it is registered by
    # the softmax_cpu_check import at the top of this file.
    import layernorm_bwd_checks
    import layernorm_triton
    import layernorm_triton_bwd
    import softmax_checks
    import softmax_triton

    assert layernorm_triton.EPS == EPS, f"the kernel's eps is {layernorm_triton.EPS}, the stand-ins assume {EPS}"
    modules = (layernorm_triton, layernorm_triton_bwd, layernorm_bwd_checks, softmax_triton, softmax_checks)
    for name, stand_in in STAND_INS.items():
        holders = [module for module in modules if hasattr(module, name)]
        assert holders, f"no module defines {name}; the stand-in for it replaces nothing"
        for module in holders:
            setattr(module, name, stand_in)
