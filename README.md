# Triton LayerNorm

Fused LayerNorm forward and backward in [Triton](https://github.com/triton-lang/triton), with a correctness check that is stricter than `torch.allclose` and every performance claim measured and recorded.

Measured on a Tesla T4 (Google Colab), M = 4096 rows, fp16, against eager `F.layer_norm` and `torch.compile`:

| | vs eager PyTorch | vs torch.compile | Bandwidth |
|:---|:---|:---|:---|
| Forward | 1.3–1.45x at N ≥ 4096 | a wash: 0.93–1.05x up to N = 8192; 1.1–1.3x at 16384–32768 on torch 2.14.1 but 0.97–0.98x on torch 2.11.0 | about 230 GB/s, 72% of the T4's 320 GB/s peak |
| Backward | 1.3–2.0x from N = 1024 | 1.35–1.8x at N = 1024 and N ≥ 4096 (1.2x at 32768 on the older versions); tie at 512 and 2048 | up to 200 GB/s |

The eager comparison flatters the forward: inductor's generated forward kernel matches or beats this one at common hidden sizes. The backward is where the custom kernel earns its place. Full tables, the register-spill investigation, a retracted measurement and the reasons behind every tuning constant are in [RESULTS.md](RESULTS.md).

The repo also has a softmax built on the same error model, with an online forward and a tiled backward for rows too wide for one block. Measured on a Tesla T4 (Kaggle, older torch and triton than the Colab runs), same M and dtype, two runs that agree:

| | vs eager PyTorch | vs torch.compile | Bandwidth |
|:---|:---|:---|:---|
| Forward | 1.1–1.4x, except 0.96x at N = 16384 | a tie up to N = 8192 (0.96–1.03x); 1.1x at 16384, 1.34x at 32768 | about 230 GB/s up to N = 32768 |
| Backward | 1.9–2.1x up to N = 32768, 1.4x at 65536 | a tie: 0.97–1.01x, except 1.22x at N = 32768 | 220–240 GB/s up to N = 32768 |

Softmax is not a speed win over inductor. It is here because its error model has to budget for the GPU's approximate exp and division, and the GPU run is the only test of that budget: every element of every case stayed inside its bound. `probe_math.py` then measures the budgets directly: on the T4 the worst exp, division and 1 / sqrt results use 0.50, 0.41 and 0.31 of what the model allows. It also shows Triton's exp slightly outside the bound NVIDIA documents for `__expf`, so the budget rests on that measurement and not on the documentation.

## What is different here

LayerNorm kernels are a Triton tutorial. The verification is the point of this repo.

- **Per-element error bound instead of `atol`/`rtol`.** A fixed tolerance is too loose near zero and too tight for large values, so it both misses bugs and raises false alarms. Following "Beyond atol and rtol" (arXiv 2610.38983), the checks derive an acceptance region for every output element from a rounding-error model of the kernel's own arithmetic, and compare against an fp64 reference. One error model covers the forward and both backward paths.
- **Mutation tests prove the bound has teeth.** Deliberately broken kernels (a dropped bias, a dropped gradient term, an off-by-one in the reduction) run against the same checks. The bound rejects all of them. `allclose` misses all of them.
- **Determinism and batch invariance.** No atomics. Two launches are bitwise equal, and a row's output does not depend on which other rows share its batch. The checks assert both.
- **Measured, not assumed.** The backward switches between two kernel layouts at N = 16384 because the fused kernel was measured spilling registers there. A third layout designed to fix a predicted spill at N = 32768 was measured slower and removed; its numbers stay in RESULTS.md. `probe_spills.py` reports registers, spills and time per kernel so the next change can be measured the same way.

## Files

| File | Contents |
|:---|:---|
| `common.py` | Shared by every script: the rounding-error primitives and ulp budgets, tolerances, assertion and benchmark helpers |
| `layernorm_triton.py` | Forward kernel, launcher, the LayerNorm error bound, forward checks and benchmark |
| `layernorm_triton_bwd.py` | Backward kernels (fused and split paths), launchers, `torch.autograd.Function` wrapper |
| `layernorm_bwd_checks.py` | Backward checks, mutation tests, benchmark |
| `probe_spills.py` | Registers, spills, time and bandwidth per kernel, with a warp sweep |
| `softmax_triton.py` | Softmax forward (block and online) and backward (block and tiled) kernels, launchers, autograd wrapper |
| `softmax_checks.py` | Softmax error bound, checks, mutation tests, benchmark |
| `probe_math.py` | The GPU's fp32 exp, division and 1 / sqrt error in ulps, against the budgets the bounds assume |
| `softmax_cpu_check.py` | The softmax error bound against an fp32 emulation of the kernels; runs without a GPU or Triton |
| `kaggle_run.py` | Runs the checks or benchmarks on a Kaggle T4 from the command line and returns the result |
| `RESULTS.md` | Measurements and decisions |

## Running

Needs a CUDA GPU with PyTorch and Triton. The LayerNorm results were produced on a free Colab T4 and the softmax results on a free Kaggle T4. Upload every `.py` file: all the scripts import `common.py`.

```
python layernorm_triton.py --skip-bench       # forward checks, about a minute
python layernorm_bwd_checks.py --skip-bench   # backward checks on both paths
python softmax_checks.py --skip-bench         # softmax checks on every path
python layernorm_triton.py                    # checks plus forward benchmark
python layernorm_bwd_checks.py                # checks plus backward benchmark
python softmax_checks.py                      # checks plus both softmax benchmarks
python probe_spills.py --n 16384 32768        # registers and spills per kernel
python probe_math.py                          # exp, division and 1/sqrt error against the budgets
python softmax_cpu_check.py                   # error-model check, no GPU needed
```

Without a local GPU, `kaggle_run.py` does the upload and the run. It needs the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) (`uv tool install kaggle --with six`, then `kaggle auth login`) and the kernel owner in `KERNEL_ID` at the top of the script set to your Kaggle username.

```
python kaggle_run.py run            # math probe and every check on a Kaggle T4, a few minutes
python kaggle_run.py run --bench    # checks plus every benchmark
```

It packs every module into one private Kaggle script, so a file cannot be left out of an upload, and it prints the git commit the run was built from. The exit status is 0 only if every command passed.

Every check prints an `ok` line; a failure raises with the check name and, for the bound, the worst element. The benchmarks include eager PyTorch and `torch.compile` columns; `--no-compile` leaves the compiled one out.

Use in a model:

```python
from layernorm_triton_bwd import layer_norm_autograd
y = layer_norm_autograd(x, weight, bias)   # x: fp16 or bf16 CUDA tensor of shape (..., N)
```

## Limits

- fp16 and bf16 inputs only, with fp32 statistics. No fp32 input path.
- LayerNorm rows must fit one block: N ≤ 32768 for fp16 and bf16. Softmax has no such limit, but above it uses the slower two-pass paths.
- Tuning constants (the layout switch at N = 16384, the 8-warp cap on the dx kernel, the softmax block limit and tile width) are T4 measurements. Rerun the probe and benchmarks before trusting them on another GPU.
- The Colab and Kaggle numbers come from different torch and triton versions. RESULTS.md has LayerNorm on both; the two disagree on whether the forward beats `torch.compile` at N ≥ 16384.
- The approximate-math budgets were measured on one GPU model and one Triton version. Run `probe_math.py` before trusting the bounds on another.
- The autograd wrappers are tested in isolation, not inside a training loop.
- The LayerNorm bound assumes no fp32 overflow or underflow. Both bounds assume approximate sqrt, division and exp stay within stated ulp budgets; the constants are in `common.py`.
