# Triton LayerNorm: GPU results

Measured on Google Colab with an NVIDIA T4 (16 GB, about 320 GB/s peak memory bandwidth), November 2026. The PyTorch and Triton versions were not recorded. Capture them on the next run with:

```python
import torch, triton; print(torch.__version__, triton.__version__)
```

All timings use M = 4096 rows, fp16, and the median from `triton.testing.do_bench`.

## Correctness

Both scripts pass every check on the T4.

- `layernorm_triton.py`: 9 shapes plus the mutation test. A fixed-tolerance `allclose` missed the dropped-bias mutant; the per-element bound caught it.
- `layernorm_triton_bwd.py`: 9 shapes on both the fused and the split path, each checked against eager with `allclose` and against fp64 with the per-element bound for dx, dw and db.
- Backward mutation tests, run on real kernels with gradients of size about 1e-3. The correct kernels passed the bound on the same input first. The bound rejected all three mutants and `allclose` missed all three:

| Mutant | Per-element bound | `allclose` |
|:---|:---|:---|
| dx kernel drops the `xhat * mean(wdy * xhat)` term | rejected | missed |
| dw skips the last row | rejected | missed |
| db skips the last row | rejected | missed |

At gradients near 1e-3, a fixed `atol` of 1e-3 is as large as the values it checks, so it cannot see errors of that size. The per-element bound scales with the values.

The bound runs use the corrected error model: an earlier version counted the 4-ulp sqrt and division allowance as 4 × 2^-24, which is 2 ulps. The forward checks were rerun with the fix and still pass.

## Forward

| N | Triton ms | Triton GB/s | Eager ms | Eager GB/s | Speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.0416 | 201.8 | 0.1034 | 81.2 | 2.49x |
| 1024 | 0.0754 | 222.6 | 0.0761 | 220.6 | 1.01x |
| 2048 | 0.1475 | 227.6 | 0.1475 | 227.5 | 1.00x |
| 4096 | 0.2942 | 228.1 | 0.3830 | 175.2 | 1.30x |
| 8192 | 0.5885 | 228.1 | 0.8253 | 162.7 | 1.40x |
| 16384 | 1.1551 | 232.5 | 1.6688 | 160.9 | 1.44x |

GB/s counts one read of x, one write of y, and one read each of weight and bias. From N = 1024 up, Triton holds about 71% of peak bandwidth. Eager at N = 512 was noisy (its 20th percentile was 0.0758 ms), so treat that speedup as approximate.

## Backward

The final version uses the fused path below N = 16384 and the split path from N = 16384.

| N | Fused ms | Fused GB/s | Split ms | Split GB/s | Eager ms | Eager GB/s | Default | Speedup vs eager |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 512 | 0.1101 | 114.2 | 0.1454 | 86.6 | 0.1210 | 104.0 | fused | 1.10x |
| 1024 | 0.1487 | 169.2 | 0.2055 | 122.5 | 0.1887 | 133.4 | fused | 1.27x |
| 2048 | 0.2743 | 183.5 | 0.3711 | 135.6 | 0.4117 | 122.3 | fused | 1.50x |
| 4096 | 0.5187 | 194.1 | 0.7057 | 142.6 | 0.9787 | 102.9 | fused | 1.89x |
| 8192 | 1.0039 | 200.5 | 1.3747 | 146.5 | 1.9652 | 102.4 | fused | 1.96x |
| 16384 | 13.1731 | 30.6 | 2.7087 | 148.7 | 4.2218 | 95.4 | split | 1.56x |

GB/s counts only the traffic any backward must move: one read each of x and dy and one write of dx. The split path also reads x and dy a second time, so its real traffic is about 1.67 times this. At N = 16384 that is about 248 GB/s, close to the best this kernel reaches on the T4.

The fused path beats the split path at every N up to 8192, by 1.3 to 1.4 times, so the switch at N = 16384 (`SPLIT_MIN_BLOCK`) stays.

### The N = 16384 cliff

The first backward had only the fused path. At N = 16384 it took 13.3 ms, 2.25 times slower than eager. A sweep of that kernel at N = 16384:

| num_warps | Registers per thread | Spills | ms |
|---:|---:|---:|---:|
| 4 | 64 | 902 | 24.414 |
| 8 | 128 | 310 | 13.009 |
| 16 | 64 | 166 | 14.612 |
| 32 | 64 | 40 | 6.520 |

Every setting spills. A T4 SM has 256 KB of registers. At N = 16384 each fp32 row vector is 64 KB, and the fused kernel keeps four of them live: the dw and db accumulators across the whole row loop, plus xhat and wdy for each row. More warps only divide the same data among more threads. The split path moves dw and db into a column-tiled kernel, so no full-width accumulator stays live.

### Retracted numbers

The first backward benchmark reported Triton as 2 to 2.8 times faster than eager. That eager timing was wrong. Each repeated `.backward()` call added into the existing `.grad`, which costs an extra read and write of dx per call. The benchmark now clears the gradients before each call, and eager at N = 16384 went from 5.89 ms to 4.22 ms. Use the table above, not the first run.

## Open items

- Above N = 16384 the split path's dx kernel holds more than 16 elements per thread and spills again.
- The benchmarks above predate the error-model change. That change touched only the checks, not the kernels, so the timings should still hold; rerun them to confirm.
- The PyTorch and Triton versions are still unrecorded.
