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

## Backward (run 3)

| N | Fused ms | GB/s | Split ms | GB/s | Tiled ms | GB/s | Eager ms | GB/s | Default (run 3) | Speedup, default vs eager |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 512 | 0.1096 | 114.8 | 0.1357 | 92.7 | 0.1243 | 101.2 | 0.1046 | 120.3 | fused | 0.95x |
| 1024 | 0.1422 | 177.0 | 0.2064 | 121.9 | 0.2211 | 113.8 | 0.1881 | 133.8 | fused | 1.32x |
| 2048 | 0.2736 | 183.9 | 0.3706 | 135.8 | 0.4026 | 125.0 | 0.4154 | 121.2 | fused | 1.52x |
| 4096 | 0.5196 | 193.7 | 0.7066 | 142.5 | 0.7873 | 127.9 | 0.9786 | 102.9 | fused | 1.88x |
| 8192 | 1.0009 | 201.1 | 1.3763 | 146.3 | 1.8376 | 109.6 | 1.9753 | 101.9 | fused | 1.97x |
| 16384 | 13.1564 | 30.6 | 2.7043 | 148.9 | 3.7281 | 108.0 | 4.2496 | 94.8 | split | 1.57x |
| 32768 | 50.2999 | 16.0 | 6.6595 | 120.9 | 7.4710 | 107.8 | 8.8857 | 90.6 | tiled | 1.19x |

GB/s counts only the traffic any backward must move: one read each of x and dy and one write of dx. The split and tiled paths read more than that, so their real bandwidth is higher than the column shows.

At N = 512 the fused path and eager trade places between runs (1.10x in run 2, 0.95x here); call it even.

The fused path beats the split path at every N up to 8192, by 1.3 to 1.4 times, so the switch at N = 16384 (`SPLIT_MIN_BLOCK`) stays.

The tiled path is slower than split at every width, including N = 32768 where it was the default in run 3. See "The tiled experiment" below. The default at 32768 is being changed to split.

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

Prediction before measuring: the one-pass dx kernel would spill at N = 32768, because at 32 warps each thread holds 32 elements per vector and three vectors exceed the 64-register cap. Proposed fix: a two-pass dx kernel over 4096-column tiles, so only tile-sized vectors stay live.

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

Decision: drop the tiled path and cap the dx kernel at 8 warps. With that cap the split path at N = 32768 should take about 6.0 ms (3.89 + 2.07 plus the final sums), about 1.48x eager. That number is a projection until the benchmark is rerun.

### Retracted numbers

The first backward benchmark reported Triton as 2 to 2.8 times faster than eager. That eager timing was wrong. Each repeated `.backward()` call added into the existing `.grad`, which costs an extra read and write of dx per call. The benchmark now clears the gradients before each call, and eager at N = 16384 went from 5.89 ms to 4.22 ms. Use the tables above, not the first run.

## Open items

- Rerun `layernorm_bwd_checks.py` after the 8-warp cap to confirm the checks and the projected 6.0 ms at N = 32768.
- At N = 32768 the one-pass dx kernel keeps 26 spill instructions at 8 warps. Accepted: it costs about 15% against the 16384 bandwidth, and the measured alternative was slower.
