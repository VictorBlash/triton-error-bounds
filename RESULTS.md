# Triton LayerNorm: GPU results

Measured on Google Colab, November 2026:

- GPU: NVIDIA Tesla T4, 16 GB, about 320 GB/s peak memory bandwidth. The best any kernel here reaches is about 250 GB/s, and that is treated as the practical ceiling.
- torch 2.14.1+cu130, triton 3.8.0.

All timings use M = 4096 rows, fp16, and the median from `triton.testing.do_bench`. Tables marked "run 3" are from the run that added N = 32768 and the tiled path; they supersede the earlier runs, whose numbers agreed within noise at every width they shared.

## Correctness

Both scripts pass every check on the T4.

- `layernorm_triton.py`: 9 shapes plus the mutation test. A fixed-tolerance `allclose` missed the dropped-bias mutant; the per-element bound caught it.
- `layernorm_bwd_checks.py`: 9 shapes on every backward path (fused, split, tiled), each checked against eager with `allclose` and against fp64 with the per-element bound for dx, dw and db, plus determinism and batch invariance of dx.
- Backward mutation tests, run on real kernels with gradients of size about 1e-3. The correct kernels passed the bound on the same input first. The bound rejected all three mutants and `allclose` missed all three:

| Mutant | Per-element bound | `allclose` |
|:---|:---|:---|
| dx kernel drops the `xhat * mean(wdy * xhat)` term | rejected | missed |
| dw skips the last row | rejected | missed |
| db skips the last row | rejected | missed |

At gradients near 1e-3, a fixed `atol` of 1e-3 is as large as the values it checks, so it cannot see errors of that size. The per-element bound scales with the values.

The bound runs use the corrected error model: an earlier version counted the 4-ulp sqrt and division allowance as 4 × 2^-24, which is 2 ulps.

## Forward (run 3)

| N | Triton ms | Triton GB/s | Eager ms | Eager GB/s | Speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.0415 | 202.2 | 0.0588 | 142.7 | 1.42x |
| 1024 | 0.0758 | 221.5 | 0.0764 | 219.7 | 1.01x |
| 2048 | 0.1454 | 230.8 | 0.1484 | 226.2 | 1.02x |
| 4096 | 0.2938 | 228.5 | 0.3834 | 175.1 | 1.30x |
| 8192 | 0.5882 | 228.2 | 0.8218 | 163.4 | 1.40x |
| 16384 | 1.1479 | 233.9 | 1.6653 | 161.2 | 1.45x |
| 32768 | 2.4318 | 220.8 | 3.3488 | 160.4 | 1.38x |

GB/s counts one read of x, one write of y, and one read each of weight and bias. Eager at N = 512 is noisy between runs (0.0588 ms here, 0.1034 ms in run 1, 80th percentile 0.1043 ms both times), so treat that speedup as somewhere between 1.4x and 2.5x.

At N = 32768, the top of the fp16 single-block range, the forward kernel uses 233 of 255 registers with no spills and drops to 221 GB/s.

## Backward (run 4, current code)

Run 4 is after the dx kernel's warp cap was lowered from 32 to 8 and the tiled path was removed. See "The tiled experiment" for why.

| N | Fused ms | GB/s | Split ms | GB/s | Eager ms | GB/s | Default | Speedup, default vs eager |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 512 | 0.1106 | 113.8 | 0.1359 | 92.6 | 0.1200 | 104.9 | fused | 1.08x |
| 1024 | 0.1414 | 177.9 | 0.2066 | 121.8 | 0.1881 | 133.8 | fused | 1.33x |
| 2048 | 0.2743 | 183.5 | 0.3714 | 135.5 | 0.4140 | 121.6 | fused | 1.51x |
| 4096 | 0.5196 | 193.7 | 0.7069 | 142.4 | 0.9784 | 102.9 | fused | 1.88x |
| 8192 | 1.0008 | 201.2 | 1.3768 | 146.2 | 1.9761 | 101.9 | fused | 1.97x |
| 16384 | 13.1703 | 30.6 | 2.6904 | 149.7 | 4.2494 | 94.8 | split | 1.58x |
| 32768 | 50.2820 | 16.0 | 6.0267 | 133.6 | 8.8866 | 90.6 | split | 1.47x |

GB/s counts only the traffic any backward must move: one read each of x and dy and one write of dx. The split path reads more than that, so its real bandwidth is higher than the column shows.

At N = 512 the fused path and eager trade places between runs (1.10x, 0.95x, 1.08x across runs 2 to 4); call it even.

The fused path beats the split path at every N up to 8192, by 1.3 to 1.4 times, so the switch at N = 16384 (`SPLIT_MIN_BLOCK`) stays.

The warp cap changed only the N = 32768 row: split went from 6.66 ms in run 3 to 6.03 ms, matching the 6.0 ms projected from the probe. All other rows are within noise of run 3.

### The N = 16384 cliff

The first backward had only the fused path. At N = 16384 it took 13.3 ms, 2.25 times slower than eager. A sweep of that kernel at N = 16384:

| num_warps | Registers per thread | Spills | ms |
|---:|---:|---:|---:|
| 4 | 64 | 902 | 24.414 |
| 8 | 128 | 310 | 13.009 |
| 16 | 64 | 166 | 14.612 |
| 32 | 64 | 40 | 6.520 |

Every setting spills. A T4 SM has 256 KB of registers. At N = 16384 each fp32 row vector is 64 KB, and the fused kernel keeps four of them live: the dw and db accumulators across the whole row loop, plus xhat and wdy for each row. More warps only divide the same data among more threads. The split path moves dw and db into a column-tiled kernel, so no full-width accumulator stays live.

### The tiled experiment (N = 32768)

Prediction before measuring: the one-pass dx kernel would spill at N = 32768, because at 32 warps each thread holds 32 elements per vector and three vectors exceed the 64-register cap. Proposed fix: a two-pass dx kernel over 4096-column tiles, so only tile-sized vectors stay live. That kernel was committed as the `tiled` path in `d3321b0` and removed in the next commit.

Run 3 benchmark, with the tiled path as a third column:

| N | Fused ms | Split ms | Tiled ms | Eager ms | Default in run 3 |
|---:|---:|---:|---:|---:|:---|
| 512 | 0.1096 | 0.1357 | 0.1243 | 0.1046 | fused |
| 1024 | 0.1422 | 0.2064 | 0.2211 | 0.1881 | fused |
| 2048 | 0.2736 | 0.3706 | 0.4026 | 0.4154 | fused |
| 4096 | 0.5196 | 0.7066 | 0.7873 | 0.9786 | fused |
| 8192 | 1.0009 | 1.3763 | 1.8376 | 1.9753 | fused |
| 16384 | 13.1564 | 2.7043 | 3.7281 | 4.2496 | split |
| 32768 | 50.2999 | 6.6595 | 7.4710 | 8.8857 | tiled |

Tiled lost to split at every width, including N = 32768 where it was the default. In run 3 the split path still used 32 warps for dx.

`probe_spills.py` output, M = 4096, fp16:

| N = 16384 | Registers | Spills | ms | GB/s |
|:---|---:|---:|---:|---:|
| forward, 8 warps | 127 | 0 | 1.0940 | 245.4 |
| eager forward | | | 1.6773 | 160.1 |
| backward fused, 8 warps | 128 | 310 | 12.9700 | 31.0 |
| dx one-pass, 8 warps | 162 | 0 | 1.6283 | 247.3 |
| dx one-pass, 16 warps | 84 | 0 | 1.6478 | 244.4 |
| dx one-pass, 32 warps (launcher) | 60 | 0 | 1.6503 | 244.0 |
| dx tiled, tile 2048, 4 warps | 64 | 0 | 2.6823 | 150.1 |
| dx tiled, tile 4096, 8 warps (launcher) | 64 | 0 | 2.6670 | 151.0 |
| dx tiled, tile 8192, 16 warps | 64 | 0 | 2.4790 | 162.4 |
| dw/db, tile 1024, 4 warps | 62 | 0 | 1.0369 | 258.9 |
| eager backward | | | 4.1879 | 96.1 |

| N = 32768 | Registers | Spills | ms | GB/s |
|:---|---:|---:|---:|---:|
| forward, 8 warps | 233 | 0 | 2.4757 | 216.9 |
| eager forward | | | 3.3479 | 160.4 |
| backward fused, 8 warps | 64 | 910 | 49.9505 | 16.1 |
| dx one-pass, 8 warps | 255 | 26 | 3.8928 | 206.9 |
| dx one-pass, 16 warps | 128 | 20 | 4.2447 | 189.7 |
| dx one-pass, 32 warps (launcher) | 64 | 12 | 4.5112 | 178.5 |
| dx tiled, tile 2048, 4 warps | 64 | 0 | 5.3604 | 150.2 |
| dx tiled, tile 4096, 8 warps (launcher) | 64 | 0 | 5.3434 | 150.7 |
| dx tiled, tile 8192, 16 warps | 64 | 0 | 5.2902 | 152.2 |
| dw/db, tile 1024, 4 warps | 62 | 0 | 2.0732 | 259.0 |
| eager backward | | | 8.7651 | 91.9 |

GB/s here counts each kernel's minimum traffic: x, dy in and dx out for the dx kernels.

What the numbers say:

1. The one-pass kernel does spill at 32768, but mildly: 12 spill instructions at 32 warps, not the hundreds the fused kernel shows. It still runs at 178 GB/s.
2. The tiled kernel has zero spills and is slower at every tile width and at both N. Counting its real traffic (x and dy read twice, dx written once), it moves 1.34 GB in 5.34 ms at N = 32768, which is 251 GB/s: it is at the bandwidth ceiling. It is slow because it moves 5/3 the bytes, and the second pass does not hit L2. With one program per SM and 40 SMs, the rows in flight total about 5 MB of x and dy against a 4 MB L2.
3. The warp sweep found a better setting than the launcher's: 8 warps runs 3.89 ms at 32768, 14% faster than 32 warps, and ties at 16384. 8 warps shows more spill instructions per thread (26 against 12) but runs 4 times fewer threads, so the total spill traffic is about half. Fewer threads also means a cheaper cross-warp reduction, though that part is not measured separately.

Decision: drop the tiled path and cap the dx kernel at 8 warps. Projected split time at N = 32768: about 6.0 ms (3.89 + 2.07 plus the final sums). Run 4 measured 6.03 ms.

### Retracted numbers

The first backward benchmark reported Triton as 2 to 2.8 times faster than eager. That eager timing was wrong. Each repeated `.backward()` call added into the existing `.grad`, which costs an extra read and write of dx per call. The benchmark now clears the gradients before each call, and eager at N = 16384 went from 5.89 ms to 4.22 ms. Use the tables above, not the first run.

## Open items

- At N = 32768 the dx kernel keeps 26 spill instructions at 8 warps and runs at 207 GB/s against 247 GB/s at N = 16384. Accepted: the measured alternative was slower, and 32768 is the top of the fp16 single-block range.
- All numbers are from one GPU model. The switch point and the warp cap are T4 measurements; rerun `probe_spills.py` and both benchmarks before trusting them on another GPU.
