# Triton error bounds

[![CPU checks](https://github.com/VictorBlash/triton-error-bounds/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VictorBlash/triton-error-bounds/actions/workflows/ci.yml)

LayerNorm and softmax kernels, forward and backward, in [Triton](https://github.com/triton-lang/triton), with a correctness check that gives every output element its own error bound, and every performance claim measured and recorded.

LayerNorm, measured on a Tesla T4 (Google Colab), M = 4096 rows, fp16, against eager `F.layer_norm` and `torch.compile`:

| | vs eager PyTorch | vs torch.compile | Bandwidth |
|:---|:---|:---|:---|
| Forward | 1.3–1.45x at N ≥ 4096 | a wash: 0.93–1.05x up to N = 8192; 1.1–1.3x at 16384–32768 on torch 2.14.1 but 0.97–0.98x on torch 2.11.0 | about 230 GB/s, 72% of the T4's 320 GB/s peak |
| Backward | 1.3–2.0x from N = 1024 | 1.35–1.8x at N = 1024 and N ≥ 4096 (1.2x at 32768 on Triton 3.6.0); tie at 512 and 2048 | up to 200 GB/s |

The eager comparison flatters the forward: inductor's generated forward kernel matches or beats this one at common hidden sizes. The backward is where the custom kernel earns its place. Full tables, the register-spill investigation, a retracted measurement and the reasons behind every tuning constant are in [RESULTS.md](RESULTS.md).

The repo also has a softmax built on the same error model, with an online forward and a tiled backward for rows too wide for one block. Measured on a Tesla T4 (Kaggle, older torch and triton than the Colab runs), same M and dtype, three runs; two baseline columns moved between them (see Limits):

| | vs eager PyTorch | vs torch.compile | Bandwidth |
|:---|:---|:---|:---|
| Forward | 1.1–1.4x, except 0.96–0.99x at N = 16384 | a tie up to N = 8192 (0.96–1.07x across three runs); 1.1–1.2x at 16384, 1.3–1.4x at 32768 | about 230 GB/s up to N = 32768 |
| Backward | 1.9–2.1x up to N = 32768, 1.4x at 65536 | a tie: 0.97–1.01x, except 1.22x at N = 32768 | 220–240 GB/s up to N = 32768 |

Softmax is not a speed win over inductor. It is here because its error model has to budget for the GPU's approximate exp and division, and the GPU run is the only test of that budget: every element of every case stayed inside its bound. `probe_math.py` then measures the budgets directly: on the T4 the worst exp, division and 1 / sqrt results use 0.50, 0.41 and 0.31 of what the model allows. It also shows Triton's exp slightly outside the bound NVIDIA documents for `__expf`, so the budget rests on that measurement and not on the documentation.

Both autograd wrappers were then put inside a model. `train_check.py` trains a small transformer on the T4 three times from the same weights and batches: with PyTorch's LayerNorm and softmax, with PyTorch operations held to the kernels' fp16 contract, and with the Triton wrappers. Every parameter's gradient agrees within 0.1%, at initialization and partly trained; the Triton loss stays within 3e-4 of the reference at every one of 300 steps and all three runs end at the same loss and 100% held-out accuracy; and two deliberately wrong backwards are rejected by 20 to 450 times the limit.

## What is different here

LayerNorm kernels are a Triton tutorial. The verification is the point of this repo.

- **Per-element error bound instead of `atol`/`rtol`.** A fixed tolerance fits one magnitude: an `atol` large enough to accept correct results near zero hides faults smaller than itself, and without one, correct results near zero are rejected. Following "Beyond atol and rtol" (arXiv 2610.38983), the checks derive an acceptance region for every output element from a rounding-error model of the kernel's own arithmetic, and compare against an fp64 reference. One error model covers the forward and both backward paths.
- **Mutation tests, and what they do and do not show.** Deliberately broken kernels (a dropped bias, a dropped gradient term, a reduction that skips a row, an online softmax that never rescales its running sum) run on inputs chosen so that the damage is small, about 1e-3 or less. The bound rejects all six faulty outputs. The fixed-tolerance check these scripts also run, `torch.allclose` with atol = rtol = 1e-3, accepts all six. That is this repo's tolerance being too loose, not a limit of fixed tolerances: `probe_tolerances.py` measured the defaults of `torch.testing.assert_close` (rtol 1e-3, atol 1e-5 for fp16) against an fp64 reference rejecting all six too, while accepting every correct output. The bound pulls ahead when the same faults are planted on smaller values. At a tenth of the magnitude the default tolerance accepts one fault of five and at a hundredth three of five; the bound accepts none and one. Dropping `atol` instead rejects about a third of the correct outputs. The table and the bound's own blind spot are in RESULTS.md under "Fixed tolerances against the bound".
- **Determinism and batch invariance.** No atomics. Two launches are bitwise equal, and a row's output does not depend on which other rows share its batch. The checks assert both.
- **Measured, not assumed.** The backward switches between two kernel layouts at N = 16384 because the fused kernel was measured spilling registers there. A third layout designed to fix a predicted spill at N = 32768 was measured slower and removed; its numbers stay in RESULTS.md. `probe_spills.py` reports registers, spills and time per kernel so the next change can be measured the same way.
- **Not built, on evidence.** A fused attention kernel was the obvious next step. A go/no-go probe measured a prototype at a tenth of PyTorch's speed on the T4, because Triton cannot use tensor cores below compute capability 8.0. The kernel was not built; the probe and its numbers are kept.

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
| `probe_attention.py` | Go / no-go probe for a fused attention forward: a prototype against PyTorch's attention |
| `probe_tolerances.py` | Fixed tolerances against the per-element bound: which accept every correct output and reject every planted fault |
| `emulation.py` | fp32 torch stand-ins for every kernel launch, so the probe and the mutation tests can be rehearsed on a CPU |
| `softmax_cpu_check.py` | The softmax error bound against an fp32 emulation of the kernels; runs without a GPU or Triton |
| `train_check.py` | Training-loop check: a small transformer trained through both autograd wrappers, against PyTorch's operations, with mutation tests |
| `kaggle_run.py` | Runs the checks or benchmarks on a Kaggle T4 from the command line and returns the result |
| `requirements.txt` | The torch and triton versions every check last passed on |
| `.github/workflows/ci.yml` | Runs the linter and the three CPU scripts on every push. No GPU, so no kernel runs there |
| `RESULTS.md` | Measurements and decisions |

## Running

Needs a CUDA GPU with PyTorch and Triton, on Linux. `requirements.txt` pins the versions every check last passed on: torch 2.11.0 and triton 3.6.0, the ones on the free Kaggle T4 that produced the softmax results. The LayerNorm results were produced on a free Colab T4 with torch 2.14.1 and triton 3.8.0. Upload every `.py` file: all the scripts import `common.py`.

```
python layernorm_triton.py --skip-bench       # forward checks, about a minute
python layernorm_bwd_checks.py --skip-bench   # backward checks on both paths
python softmax_checks.py --skip-bench         # softmax checks on every path
python layernorm_triton.py                    # checks plus forward benchmark
python layernorm_bwd_checks.py                # checks plus backward benchmark
python softmax_checks.py                      # checks plus both softmax benchmarks
python probe_spills.py --n 16384 32768        # registers and spills per kernel
python probe_math.py                          # exp, division and 1/sqrt error against the budgets
python probe_attention.py                     # is a fused attention kernel worth building on this GPU?
python probe_tolerances.py                    # which fixed tolerances do the job of the per-element bound?
python probe_tolerances.py --emulate          # the same script on the CPU, without Triton: tests the script
python softmax_cpu_check.py                   # error-model check, no GPU needed
python train_check.py                         # train a small model through both autograd wrappers
python train_check.py --emulate               # the same script on the CPU, without Triton: tests the script
```

Without a local GPU, `kaggle_run.py` does the upload and the run. It needs the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) (`uv tool install kaggle --with six`, then `kaggle auth login`) and the kernel owner in `KERNEL_ID` at the top of the script set to your Kaggle username.

```
python kaggle_run.py run            # math probe, every check and the training-loop check on a Kaggle T4, a few minutes
python kaggle_run.py run --bench    # checks plus every benchmark
```

It packs every module into one private Kaggle script, so a file cannot be left out of an upload, and it prints the git commit the run was built from. The exit status is 0 only if every command passed.

Every check prints an `ok` line; a failure raises with the check name and, for the bound, the worst element. The benchmarks include eager PyTorch and `torch.compile` columns; `--no-compile` leaves the compiled one out.

Every push also runs `ruff check` and the three CPU scripts (`softmax_cpu_check.py`, `probe_tolerances.py --emulate`, `train_check.py --emulate`) on GitHub Actions. The hosted runners have no GPU and the emulation replaces every kernel launch, so the badge at the top says the error model and the scripts hold together. It says nothing about the kernels; only a GPU run does.

Use in a model:

```python
from layernorm_triton_bwd import layer_norm_autograd
y = layer_norm_autograd(x, weight, bias)   # x: fp16 or bf16 CUDA tensor of shape (..., N)
```

## Limits

- Every number and every passed check is from a Tesla T4, compute capability 7.5. Triton's README lists NVIDIA GPUs of compute capability 8.0 or higher as supported hardware, so the T4 is below Triton's stated support. Nothing here has run on a supported GPU.
- The strided-input checks and the non-finite-input check for the LayerNorm forward were added after the last GPU run. On a CPU they have only been rehearsed against stand-ins for the launchers they test, so they are unproven until the next GPU run.
- fp16 and bf16 inputs only, with fp32 statistics. No fp32 input path.
- LayerNorm rows must fit one block: N ≤ 32768 for fp16 and bf16. Softmax has no such limit, but above it uses the slower two-pass paths.
- Tuning constants (the layout switch at N = 16384, the 8-warp cap on the dx kernel, the softmax block limit and tile width) are T4 measurements. Rerun the probe and benchmarks before trusting them on another GPU.
- The Colab and Kaggle numbers come from different torch and triton versions. RESULTS.md has LayerNorm on both; the two disagree on whether the forward beats `torch.compile` at N ≥ 16384.
- Timings are means over `do_bench`'s timed runs; the benchmarks print the median beside them. For the Triton kernels the two agree within 0.5%. Two baseline columns move between runs on the Kaggle T4 (eager LayerNorm forward at N ≤ 1024, inductor's softmax forward at N ≥ 4096), so comparisons against them are ranges. No table was measured on torch 2.15.0 with triton 3.9.0.
- The backward numbers at N = 32768 assume Triton 3.8.0 or later. On 3.6.0 and 3.7.1 Triton generates code that spills more registers there, and the backward is 9 to 12% slower at that width only. Measured on one GPU with only Triton changed; see RESULTS.md.
- The approximate-math budgets were measured on one GPU model and one Triton version. Run `probe_math.py` before trusting the bounds on another.
- The training-loop check is one small model: LayerNorm at width 256, softmax at width 64, fp16, 300 steps, one seed, contiguous tensors. It shows the wrappers are drop-in there. It does not time them, and it does not cover bf16 or the wide rows the benchmarks use.
- The LayerNorm bound assumes no fp32 overflow or underflow. Both bounds assume approximate sqrt, division and exp stay within stated ulp budgets; the constants are in `common.py`.

## License

MIT. See [LICENSE](LICENSE).
