# Triton LayerNorm

Fused LayerNorm forward and backward in [Triton](https://github.com/triton-lang/triton), with a correctness check that is stricter than `torch.allclose` and every performance claim measured and recorded.

Measured on a Tesla T4 (Google Colab), M = 4096 rows, fp16, against eager `F.layer_norm` and `torch.compile`:

| | vs eager PyTorch | vs torch.compile | Bandwidth |
|:---|:---|:---|:---|
| Forward | 1.3–1.45x at N ≥ 4096 | a wash: 0.93–1.05x up to N = 8192, 1.1–1.3x at 16384–32768 | about 230 GB/s, 72% of the T4's 320 GB/s peak |
| Backward | 1.3–2.0x from N = 1024 | 1.35–1.8x at N = 1024 and N ≥ 4096; tie at 512, 0.96x at 2048 | up to 200 GB/s |

The eager comparison flatters the forward: inductor's generated forward kernel matches or beats this one at common hidden sizes. The backward is where the custom kernel earns its place. Full tables, the register-spill investigation, a retracted measurement and the reasons behind every tuning constant are in [RESULTS.md](RESULTS.md).

## What is different here

LayerNorm kernels are a Triton tutorial. The verification is the point of this repo.

- **Per-element error bound instead of `atol`/`rtol`.** A fixed tolerance is too loose near zero and too tight for large values, so it both misses bugs and raises false alarms. Following "Beyond atol and rtol" (arXiv 2610.38983), the checks derive an acceptance region for every output element from a rounding-error model of the kernel's own arithmetic, and compare against an fp64 reference. One error model covers the forward and both backward paths.
- **Mutation tests prove the bound has teeth.** Deliberately broken kernels (a dropped bias, a dropped gradient term, an off-by-one in the reduction) run against the same checks. The bound rejects all of them. `allclose` misses all of them.
- **Determinism and batch invariance.** No atomics. Two launches are bitwise equal, and a row's output does not depend on which other rows share its batch. The checks assert both.
- **Measured, not assumed.** The backward switches between two kernel layouts at N = 16384 because the fused kernel was measured spilling registers there. A third layout designed to fix a predicted spill at N = 32768 was measured slower and removed; its numbers stay in RESULTS.md. `probe_spills.py` reports registers, spills and time per kernel so the next change can be measured the same way.

## Files

| File | Contents |
|:---|:---|
| `layernorm_triton.py` | Forward kernel, launcher, the rounding-error model, forward checks and benchmark |
| `layernorm_triton_bwd.py` | Backward kernels (fused and split paths), launchers, `torch.autograd.Function` wrapper |
| `layernorm_bwd_checks.py` | Backward checks, mutation tests, benchmark |
| `probe_spills.py` | Registers, spills, time and bandwidth per kernel, with a warp sweep |
| `RESULTS.md` | Measurements and decisions |

## Running

Needs a CUDA GPU with PyTorch and Triton. The results here were produced on a free Colab T4 (Runtime → Change runtime type → T4 GPU) with the four `.py` files uploaded through the Files panel.

```
python layernorm_triton.py --skip-bench       # forward checks, about a minute
python layernorm_bwd_checks.py --skip-bench   # backward checks on both paths
python layernorm_triton.py                    # checks plus forward benchmark
python layernorm_bwd_checks.py                # checks plus backward benchmark
python probe_spills.py --n 16384 32768        # registers and spills per kernel
```

Every check prints an `ok` line; a failure raises with the check name and, for the bound, the worst element. The benchmarks include eager PyTorch and `torch.compile` columns; `--no-compile` leaves the compiled one out.

Use in a model:

```python
from layernorm_triton_bwd import layer_norm_autograd
y = layer_norm_autograd(x, weight, bias)   # x: fp16 or bf16 CUDA tensor of shape (..., N)
```

## Limits

- fp16 and bf16 inputs only, with fp32 statistics. No fp32 input path.
- Each row must fit one block: N ≤ 32768 for fp16 and bf16.
- Tuning constants (the layout switch at N = 16384, the 8-warp cap on the dx kernel) are T4 measurements. Rerun the probe and benchmarks before trusting them on another GPU.
- The autograd wrapper is tested in isolation, not inside a training loop.
- The error bound assumes no fp32 overflow or underflow and that approximate sqrt and division stay within a stated ulp budget; the constants are at the top of `layernorm_triton.py`.
