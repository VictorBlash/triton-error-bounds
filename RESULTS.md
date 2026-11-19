# Triton LayerNorm and softmax: GPU results

The first LayerNorm sections were measured on Google Colab, November 2026. Everything from "LayerNorm on the Kaggle versions" down was measured on Kaggle with older library versions; the environment is stated there. Colab:

- GPU: NVIDIA Tesla T4, 16 GB, about 320 GB/s peak memory bandwidth. The best any kernel here reaches is about 250 GB/s, and that is treated as the practical ceiling.
- torch 2.14.1+cu130, triton 3.8.0.

All timings use M = 4096 rows, fp16, and the mean over the timed runs of `triton.testing.do_bench`, which is what it returns by default. Until 2026-11-17 this file and the scripts called these figures medians. That label was wrong; the numbers are unchanged. Runs are numbered in the order they happened. The current tables are from run 5, which added the `torch.compile` column (default inductor mode, `dynamic=False`, one compile per width). Earlier runs agreed within noise at every width they shared, except eager at N ≤ 1024, which swings by up to 2x between runs.

## Correctness

Both scripts pass every check on the T4.

- `layernorm_triton.py`: 9 shapes plus the mutation test. A fixed-tolerance `allclose` missed the dropped-bias mutant; the per-element bound caught it.
- `layernorm_bwd_checks.py`: 9 shapes on every backward path (fused and split in the current code; the tiled path of run 3 has been removed, see "The tiled experiment"), each checked against eager with `allclose` and against fp64 with the per-element bound for dx, dw and db, plus determinism and batch invariance of dx.
- Backward mutation tests, run on real kernels with gradients of size about 1e-3. The correct kernels passed the bound on the same input first. The bound rejected all three mutants. The fixed-tolerance `allclose` (atol = rtol = 1e-3, with atol multiplied by the row count, 256, for dw and db) missed all three:

| Mutant | Per-element bound | `allclose` |
|:---|:---|:---|
| dx kernel drops the `xhat * mean(wdy * xhat)` term | rejected | missed |
| dw skips the last row | rejected | missed |
| db skips the last row | rejected | missed |

At gradients near 1e-3, a fixed `atol` of 1e-3 is as large as the values it checks, so it cannot see errors of that size. The per-element bound scales with the values. A tighter fixed tolerance does catch these faults at this magnitude; "Fixed tolerances against the bound" below measures which ones and where they stop.

The bound runs use the corrected error model: an earlier version counted the 4-ulp sqrt and division allowance as 4 × 2^-24, which is 2 ulps.

## Forward (run 5, current code)

| N | Triton ms | GB/s | Eager ms | GB/s | Compile ms | GB/s | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0418 | 200.9 | 0.1041 | 80.6 | 0.0430 | 195.3 | 2.49x | 1.03x |
| 1024 | 0.0742 | 226.2 | 0.1073 | 156.3 | 0.0777 | 216.0 | 1.45x | 1.05x |
| 2048 | 0.1445 | 232.3 | 0.1489 | 225.5 | 0.1421 | 236.1 | 1.03x | 0.98x |
| 4096 | 0.2912 | 230.5 | 0.3825 | 175.5 | 0.2745 | 244.5 | 1.31x | 0.94x |
| 8192 | 0.5810 | 231.1 | 0.8158 | 164.6 | 0.5429 | 247.3 | 1.40x | 0.93x |
| 16384 | 1.1291 | 237.8 | 1.6538 | 162.4 | 1.2561 | 213.8 | 1.46x | 1.11x |
| 32768 | 2.3668 | 226.9 | 3.3431 | 160.6 | 3.0133 | 178.2 | 1.41x | 1.27x |

GB/s counts one read of x, one write of y, and one read each of weight and bias.

Against `torch.compile` the forward is a wash. Inductor's generated kernel is 2 to 7% faster at N = 2048 to 8192 and reaches 247 GB/s there, the highest bandwidth any kernel in this repo has measured on the T4; this kernel plateaus near 231 GB/s. The Triton kernel wins only at N = 16384 and 32768, where inductor's kernel drops to 214 and 178 GB/s. The eager comparison flatters the forward: eager `F.layer_norm` runs at about 160 GB/s from N = 4096 up.

Eager at N ≤ 1024 is noisy between runs (0.0588 to 0.1041 ms at 512 across runs 1 to 5), so the small-N eager speedups are not stable numbers.

At N = 32768, the top of the fp16 single-block range, the forward kernel uses 233 of 255 registers with no spills and drops to 227 GB/s.

## Backward (run 5, current code)

Runs 4 and 5 are after the dx kernel's warp cap was lowered from 32 to 8 and the tiled path was removed. See "The tiled experiment" for why. Run 5 adds the compile column; the other columns agree with run 4 within noise.

| N | Fused ms | GB/s | Split ms | GB/s | Eager ms | GB/s | Compile ms | GB/s | Default | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|
| 512 | 0.1107 | 113.6 | 0.1451 | 86.7 | 0.1165 | 108.0 | 0.1120 | 112.3 | fused | 1.05x | 1.01x |
| 1024 | 0.1451 | 173.5 | 0.2081 | 120.9 | 0.1881 | 133.8 | 0.1966 | 128.0 | fused | 1.30x | 1.35x |
| 2048 | 0.2754 | 182.8 | 0.3721 | 135.2 | 0.4152 | 121.2 | 0.2640 | 190.7 | fused | 1.51x | 0.96x |
| 4096 | 0.5178 | 194.4 | 0.7055 | 142.7 | 0.9823 | 102.5 | 0.7527 | 133.7 | fused | 1.90x | 1.45x |
| 8192 | 0.9948 | 202.4 | 1.3700 | 147.0 | 1.9794 | 101.7 | 1.7740 | 113.5 | fused | 1.99x | 1.78x |
| 16384 | 12.9467 | 31.1 | 2.7768 | 145.0 | 4.1998 | 95.9 | 3.9416 | 102.2 | split | 1.51x | 1.42x |
| 32768 | 49.7185 | 16.2 | 5.9427 | 135.5 | 8.8432 | 91.1 | 8.0206 | 100.4 | split | 1.49x | 1.35x |

GB/s counts only the traffic any backward must move: one read each of x and dy and one write of dx. The split path reads more than that, so its real bandwidth is higher than the column shows.

Against `torch.compile` the backward is a real win: 1.35x to 1.78x at N = 1024 and N ≥ 4096. The exceptions are N = 512, a tie, and N = 2048, where inductor's backward is 4% faster and reaches 191 GB/s, its best width by far. Inductor's backward bandwidth is not monotonic in N (128, 191, 134, 114 GB/s at 1024 to 8192), which suggests its reduction heuristics pick a different strategy at 2048 than on either side; not investigated.

At N = 512 the fused path and eager trade places between runs (1.10x, 0.95x, 1.08x, 1.05x across runs 2 to 5); call it even.

The fused path beats the split path at every N up to 8192, by 1.3 to 1.4 times, so the switch at N = 16384 (`SPLIT_MIN_BLOCK`) stays.

The warp cap changed only the N = 32768 row: split went from 6.66 ms in run 3 to 6.03 ms in run 4 and 5.94 ms in run 5, matching the 6.0 ms projected from the probe.

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

## Kaggle environment (2026-11-16)

Everything below ran on Kaggle's "GPU T4 x2" accelerator; the scripts use the first GPU.

- GPU: NVIDIA Tesla T4.
- torch 2.11.0+cu130, triton 3.6.0. Both are older than the Colab versions above (torch 2.14.1, triton 3.8.0).
- Timings use M = 4096 rows, fp16, the mean from `do_bench`, and one fresh `torch.compile` per width, as in the Colab tables.
- The tables are from one job, pushed by `kaggle_run.py run --bench` from a clean tree at commit `7b0c7f8`. The job prints the commit it was built from.

## LayerNorm on the Kaggle versions

One run. A second run on 2026-11-17 reproduced 45 of the 49 cells within 3%; "Mean against median" below has the four that moved. The checks pass, as they did on Colab.

| N | Triton ms | GB/s | Eager ms | GB/s | Compile ms | GB/s | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.0437 | 192.0 | 0.1038 | 80.8 | 0.0434 | 193.4 | 2.38x | 0.99x |
| 1024 | 0.0758 | 221.5 | 0.0873 | 192.2 | 0.0762 | 220.1 | 1.15x | 1.01x |
| 2048 | 0.1476 | 227.4 | 0.1477 | 227.3 | 0.1425 | 235.5 | 1.00x | 0.97x |
| 4096 | 0.2939 | 228.4 | 0.3839 | 174.8 | 0.2794 | 240.3 | 1.31x | 0.95x |
| 8192 | 0.5879 | 228.4 | 0.8254 | 162.7 | 0.5616 | 239.0 | 1.40x | 0.96x |
| 16384 | 1.1544 | 232.6 | 1.6675 | 161.0 | 1.1211 | 239.5 | 1.44x | 0.97x |
| 32768 | 2.5138 | 213.6 | 3.3574 | 159.9 | 2.4566 | 218.6 | 1.34x | 0.98x |

| N | Fused ms | GB/s | Split ms | GB/s | Eager ms | GB/s | Compile ms | GB/s | Default | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|
| 512 | 0.1122 | 112.2 | 0.1474 | 85.4 | 0.1343 | 93.7 | 0.1062 | 118.5 | fused | 1.20x | 0.95x |
| 1024 | 0.1413 | 178.1 | 0.2056 | 122.4 | 0.1880 | 133.9 | 0.1938 | 129.8 | fused | 1.33x | 1.37x |
| 2048 | 0.2665 | 188.9 | 0.3711 | 135.6 | 0.4151 | 121.3 | 0.2630 | 191.3 | fused | 1.56x | 0.99x |
| 4096 | 0.5187 | 194.1 | 0.7054 | 142.7 | 0.9796 | 102.8 | 0.7474 | 134.7 | fused | 1.89x | 1.44x |
| 8192 | 1.0027 | 200.8 | 1.3784 | 146.1 | 1.9784 | 101.8 | 1.7668 | 114.0 | fused | 1.97x | 1.76x |
| 16384 | 12.4660 | 32.3 | 2.7047 | 148.9 | 4.2251 | 95.3 | 3.9207 | 102.7 | split | 1.56x | 1.45x |
| 32768 | 65.0506 | 12.4 | 6.5837 | 122.3 | 8.8166 | 91.3 | 7.9166 | 101.7 | split | 1.34x | 1.20x |

Against the Colab run 5 tables:

- The forward no longer beats `torch.compile` at any width: 0.95x to 1.01x. On Colab it won at N = 16384 and 32768 (1.11x and 1.27x). The difference is inductor, not this kernel. Inductor's forward on torch 2.11.0 takes 1.12 ms at 16384 and 2.46 ms at 32768, against 1.26 and 3.01 ms on torch 2.14.1, and it holds 219 to 240 GB/s where the newer version dropped to 214 and 178. So the forward's only wins over inductor depend on the torch version.
- The backward conclusion holds: 1.37x to 1.76x over inductor at N = 1024, 4096, 8192 and 16384, a tie at 512 and 2048, and 1.20x at 32768 (1.35x on Colab).
- The Triton kernels themselves are within 3% of Colab up to N = 16384, apart from the forward at 512 (4.5% slower). At N = 32768 the split backward is 11% slower (6.58 against 5.94 ms) and the forward reads 6% slower (214 against 227 GB/s). The next section finds the cause of the first and shows the second is noise.

### Triton version and the N = 32768 backward

Question: is the slower N = 32768 backward caused by the GPU, by torch, or by Triton? Method: `probe_spills.py` on one Kaggle T4, in one session, with torch 2.11.0 throughout and only Triton changed. Each other version was installed into a side folder and put first on the path for that one command (`kaggle_run.py run --internet`). Every probe prints the version it imported. Tree at commit `a96b524`.

The dx kernel at N = 32768, registers per thread / spill instructions, then ms:

| Triton | 8 warps (launcher) | 16 warps | 32 warps | dx + dw/db |
|:---|:---|:---|:---|---:|
| 3.6.0 (Kaggle's) | 255 / 46, 4.44 | 128 / 24, 4.42 | 64 / 18, 5.08 | 6.52 |
| 3.7.1 | 255 / 48, 4.57 | 128 / 24, 4.36 | 64 / 24, 5.90 | 6.64 |
| 3.8.0 (Colab's) | 255 / 26, 3.85 | 128 / 20, 4.25 | 64 / 12, 4.48 | 5.92 |
| 3.9.0 | 255 / 26, 3.93 | 128 / 18, 4.13 | 64 / 16, 4.77 | 6.01 |

"dx + dw/db" adds the dw/db kernel, which takes 2.07 ms with 62 registers and no spills on every version. It predicts the split path: the benchmarks measured 6.58 ms on 3.6.0 and 5.94 ms on 3.8.0.

1. The cause is the Triton version. Up to 3.7.1 the dx kernel has 46 to 48 spill instructions at the launcher's 8 warps and takes 4.44 to 4.57 ms; from 3.8.0 it has 26 and takes 3.85 to 3.93 ms. That is 13 to 19% on the dx kernel and 9 to 12% on the split backward, at N = 32768 only.
2. It is not the GPU. With Triton 3.8.0 the Kaggle T4 reproduces the Colab probe table above: registers and spills are identical in all 12 kernel rows and every time is within 1.2%.
3. It is not noise. Triton 3.6.0 was probed six times in three sessions: 4.42 to 4.48 ms, with the same register and spill counts every time.
4. It is Triton's generated code, not NVIDIA's assembler. Triton 3.6.0 bundles ptxas 12.8.93 and 3.8.0 bundles 12.9.86. With `TRITON_PTXAS_PATH` pointing Triton 3.8.0 at the older ptxas, the dx kernel still has 26 spills and takes 3.90 ms; the forward's register count moved from 233 to 229, which shows the override took effect. Pointing Triton 3.6.0 at the newer ptxas left it at 46 spills and 4.46 ms. So the PTX that Triton hands to the assembler differs. Which change between 3.7.1 and 3.8.0 is responsible was not identified; the release notes list LLVM updates in that window.
5. Below the spill threshold the versions differ in registers but not in time. At N = 16384 the dx kernel at 8 warps needs 218 registers on 3.6.0, 202 on 3.7.1 and 162 on 3.8.0 and 3.9.0, with no spills and 1.63 to 1.65 ms on all four.
6. The 8-warp cap stays. It is the fastest setting on 3.8.0 and 3.9.0, ties 16 warps on 3.6.0 (4.44 against 4.42 ms), and loses to 16 warps by about 5% on 3.7.1.

The forward at N = 32768 is a different story. It has no spills on any version (255 registers on 3.6.0 and 3.7.1, 233 on 3.8.0, 239 on 3.9.0), and its time does not follow the version: six samples on 3.6.0 span 2.51 to 2.64 ms, two on 3.8.0 are 2.50 and 2.51 ms, 3.7.1 gave 2.49 and 3.9.0 gave 2.47. In the one session that ran 3.6.0 and 3.8.0 twice each (once with each assembler), they measured 2.51 and 2.52 against 2.51 and 2.50. So no version effect is shown for the forward. The 6% above compares two benchmark figures, and the Colab one (2.37 ms) is lower than every probe sample on either machine, including Colab's own probe (2.48 ms). Treat the forward's N = 32768 row as good to about 5%.

Not on the default path, but large: the fused backward at N = 32768 takes 64.7 ms on 3.6.0, 50.0 on 3.8.0 and 39.1 on 3.9.0, and the dx kernel at 4 warps takes 38.0 ms on 3.6.0 and 16.6 on 3.8.0. Heavily spilling kernels are the most sensitive to the Triton version.

## Softmax

The softmax kernels, checks and benchmarks are in `softmax_triton.py` and `softmax_checks.py`.

The tables are the second of two runs on this environment. The first, a day earlier in a hand-driven notebook at commit `1bfef9e`, agrees within 0.4% in every backward cell and within 3.2% in every forward cell except three: online and compile at N = 512 (9 to 10%) and eager at N = 4096 (6%). The softmax block forward plateaus at 230 GB/s and the LayerNorm forward at 228 GB/s on the same versions.

### Correctness

`softmax_checks.py` passes every check on the T4.

- 13 shapes, each on every path that fits: block and online forward, block and tiled backward. N = 40000 is above the single-block limit and runs on the online and tiled paths only. N = 1 passes on the online path, where the tile is one element wide.
- Each forward output and each dx is checked for finite values, against eager with `allclose`, and against fp64 with the per-element bound. Two launches must be bitwise equal, and a row launched alone must equal the same row from the batch.
- Mutation tests on real kernels. The bound rejected both mutants and `allclose` at atol = rtol = 1e-3 missed both:

| Mutant | Per-element bound | `allclose` |
|:---|:---|:---|
| Online forward skips the rescale of the running sum | rejected | missed |
| Backward drops the `sum(dy * y)` term | rejected | missed |

These runs are the first test of the exp and division budgets (`EXP_ULPS`, `EXP_ARG_ULPS`, `DIV_ULPS`). On the CPU those operations are correctly rounded, so `softmax_cpu_check.py` could not exercise them. They held on every element of every case.

The checks also print the largest |error| / bound per path. Leaving out the results that are exact (N = 1, and y for uniform rows), it is 0.862 to 0.996 for y and 0.902 to 0.997 for dx, and the block and online paths agree to within 0.001. That ratio is close to 1 by construction: rounding the result to fp16 or bf16 is almost the whole bound, and some element always lands near a rounding boundary. It shows the bound is not loose. It does not show how much of the fp32 budgets is used; "Approximate-math budgets, measured" below does.

### Forward

| N | Block ms | GB/s | Online ms | GB/s | Eager ms | GB/s | Compile ms | GB/s | Default | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|
| 512 | 0.0396 | 212.0 | 0.0447 | 187.7 | 0.0496 | 169.1 | 0.0390 | 215.0 | block | 1.25x | 0.98x |
| 1024 | 0.0753 | 222.8 | 0.0728 | 230.4 | 0.0805 | 208.4 | 0.0772 | 217.3 | block | 1.07x | 1.03x |
| 2048 | 0.1448 | 231.7 | 0.1416 | 237.0 | 0.1707 | 196.5 | 0.1427 | 235.1 | block | 1.18x | 0.99x |
| 4096 | 0.2925 | 229.5 | 0.2941 | 228.2 | 0.4050 | 165.7 | 0.2954 | 227.2 | block | 1.38x | 1.01x |
| 8192 | 0.5847 | 229.5 | 0.7672 | 174.9 | 0.6882 | 195.0 | 0.5634 | 238.2 | block | 1.18x | 0.96x |
| 16384 | 1.1652 | 230.4 | 1.6815 | 159.6 | 1.1135 | 241.1 | 1.3109 | 204.8 | block | 0.96x | 1.13x |
| 32768 | 2.3025 | 233.2 | 3.3277 | 161.3 | 3.1410 | 170.9 | 3.0833 | 174.1 | block | 1.36x | 1.34x |
| 65536 | - | - | 6.6515 | 161.4 | 8.4716 | 126.7 | 6.6028 | 162.6 | online | 1.27x | 0.99x |

GB/s counts one read of x and one write of y. The online path reads x twice, so its real bandwidth is higher than the column shows. The two "vs" columns compare the default path.

- Against `torch.compile` the forward is a tie up to N = 8192 (0.96x to 1.03x) and at 65536 (0.99x). The block kernel wins at N = 16384 (1.13x; 1.09x in the first run) and at N = 32768 (1.34x), where inductor's softmax drops to 205 and 174 GB/s. In a third run inductor's kernel was 5 to 9% slower from N = 4096 up, which moves these figures to 1.07x, 1.06x, 1.24x and 1.41x at N = 4096 to 32768; see "Mean against median".
- The block kernel holds 223 to 233 GB/s from N = 1024 to 32768.
- Eager `F.softmax` is not monotonic: 166 to 241 GB/s up to N = 16384, then 171 and 127 GB/s. At N = 16384 eager is 4% faster than the block kernel in both runs, and 0.7% faster in a third. That is the one width where this forward loses to eager.
- Above the block limit the online kernel ties inductor (0.99x) and beats eager by 1.27x.
- The online kernel matches the block kernel where the row is one tile and wide enough to amortise its extra loop (N = 1024 to 4096, within 3%), is 13% slower at N = 512 (24% in the first run), and is 1.3 to 1.45 times slower from N = 8192 up. Counting its real traffic (x read twice, y written once) it moves 240 GB/s at N = 16384 and above. It is at the bandwidth ceiling and slow only because it moves 3/2 the bytes. This is the same result as the LayerNorm tiled experiment.

Decision: keep `BLOCK_LIMIT = 32768`. The block path is the default wherever a row fits one block, and the online path is used only above that.

### Backward

| N | Block ms | GB/s | Tiled ms | GB/s | Eager ms | GB/s | Compile ms | GB/s | Default | vs eager | vs compile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|
| 512 | 0.0572 | 220.1 | 0.0568 | 221.3 | 0.1144 | 110.0 | 0.0565 | 222.5 | block | 2.00x | 0.99x |
| 1024 | 0.1098 | 229.2 | 0.1098 | 229.3 | 0.2188 | 115.0 | 0.1098 | 229.3 | block | 1.99x | 1.00x |
| 2048 | 0.2112 | 238.4 | 0.2068 | 243.4 | 0.4165 | 120.8 | 0.2121 | 237.3 | block | 1.97x | 1.00x |
| 4096 | 0.4266 | 236.0 | 0.4924 | 204.4 | 0.8237 | 122.2 | 0.4145 | 242.8 | block | 1.93x | 0.97x |
| 8192 | 0.8385 | 240.1 | 1.3281 | 151.6 | 1.6337 | 123.2 | 0.8465 | 237.8 | block | 1.95x | 1.01x |
| 16384 | 1.6691 | 241.2 | 2.6878 | 149.8 | 3.2543 | 123.7 | 1.6813 | 239.5 | block | 1.95x | 1.01x |
| 32768 | 3.5260 | 228.4 | 5.4153 | 148.7 | 7.5304 | 106.9 | 4.3165 | 186.6 | block | 2.14x | 1.22x |
| 65536 | - | - | 10.8436 | 148.5 | 15.1162 | 106.5 | 10.8258 | 148.8 | tiled | 1.39x | 1.00x |

GB/s counts one read each of dy and y and one write of dx. The tiled path reads dy and y twice.

- Against `torch.compile` the backward is also a tie: 0.97x to 1.01x, except 1.22x at N = 32768 where inductor drops to 187 GB/s. Unlike LayerNorm, the custom softmax backward does not beat inductor. The softmax backward is one reduction and one elementwise pass, and inductor already generates that at 222 to 243 GB/s up to N = 16384.
- Against eager the block path is 1.9 to 2.1 times faster at every width it runs. Eager autograd for softmax runs at 106 to 124 GB/s.
- The tiled path ties the block path up to N = 2048, is 15% slower at 4096 and 1.5 to 1.6 times slower above. Counting its real traffic (5/3 of the minimum) it moves 247 to 250 GB/s from N = 16384 up: at the ceiling, and slow only because it moves more bytes.
- At N = 65536 the tiled kernel and inductor take the same time (10.84 and 10.83 ms).

Decision: the block path stays the default up to N = 32768 and the tiled path is used only above it.

What softmax adds to this repo is therefore not speed over inductor. It is the error model exercised on approximate GPU exp and division, and kernels that are checked per element for rows wider than one block.

## Mean against median

Question: every table above is a mean (see the note at the top of this file). Do a few slow runs pull any figure away from the typical run?

Method: `do_bench(fn, return_mode="all")` returns every timed run, so each benchmark now prints a second table from the same runs: the median, median / mean, and the number of runs. One job, `kaggle_run.py run --bench` from a clean tree at commit `d82bc47` on 2026-11-17, in the Kaggle environment above (T4, torch 2.11.0, triton 3.6.0). All four benchmarks, 111 cells. The checks in the same job passed.

| Cells | Count | Median / mean |
|:---|---:|:---|
| Triton kernels, every path and width | 51 | 0.995 to 1.003 |
| Eager and `torch.compile`, all but the seven below | 53 | 0.993 to 1.006 |
| Softmax forward, eager, N = 4096 and 8192 | 2 | 0.992 and 1.019 |
| Softmax forward, `torch.compile`, N = 4096, 8192, 16384 | 3 | 0.957, 0.972, 0.959 |
| LayerNorm forward, eager, N = 512 and 1024 | 2 | 1.217 and 0.794 |

1. For every Triton kernel the mean and the median agree within 0.5%. The old "median" label was wrong, but no kernel figure in this file moves by more than that under the other statistic.
2. Eager LayerNorm forward is not stable at N = 512 and 1024. This run measured means of 0.0848 and 0.0981 ms and medians of 0.1032 and 0.0779 ms; the Kaggle table has 0.1038 and 0.0873. "vs eager" at those two widths is therefore somewhere between 1.0x and 2.4x: 1.93x and 1.28x by this run's mean, 2.35x and 1.01x by its median, 2.38x and 1.15x in the table. The individual run times were not saved, so the shape of the spread is not known.
3. Inductor's softmax forward is not stable from N = 4096 to 32768. Its mean is 3 to 4% above its median at N = 4096 to 16384, and both are above the table: the mean by 5 to 9% at all four widths, the median by 1 to 6%. The block kernel is within 0.8% of the table at every width. So against `torch.compile` the softmax forward reads 1.07x, 1.06x, 1.24x and 1.41x at N = 4096, 8192, 16384 and 32768 by mean, and 1.03x, 1.03x, 1.19x and 1.41x by median, where the table has 1.01x, 0.96x, 1.13x and 1.34x. The wins at 16384 and 32768 appear in all three runs; their size depends on the run.
4. The means of this run are within 3% of the Kaggle tables in 98 of 111 cells, and within 0.8% in all 31 softmax backward cells. Of the 13 outside, 11 are baselines: the six cells in findings 2 and 3, eager softmax forward at N = 512, 4096, 8192 and 16384 (3 to 6% slower), and eager LayerNorm backward at N = 512 (11% faster). Two are Triton kernels at N = 512: the split backward (7.5% faster) and the online softmax forward (14% slower). In both, mean and median agree within 0.1%, so each run was steady and the difference is between runs. Every other kernel cell is within 1.5%.
5. Run counts are small. `do_bench` sets the count to 100 ms over its estimate of one iteration, and the estimate includes the cache flush it does before each run, about 1.2 ms on this GPU (inferred from the counts: 80 runs for a 0.04 ms kernel). So no cell has more than 83 runs, and slow ones have few: 27 at 2.5 ms, 12 at 6.6 ms, 7 at 12 ms, and 1 for the fused backward at N = 32768 (65 ms). Five cells have fewer than 10 runs: the fused backward at N = 16384 and 32768, and the three softmax backward columns that run at N = 65536. By the same arithmetic the 38 to 65 ms figures under "Triton version and the N = 32768 backward" are one or two runs each.

Decision: the tables stay as they are, as means, and both statistics are printed from now on. Switching to the median would change no kernel figure by more than 0.5%, and it would not steady the unstable baselines: this run's medians are also 5 to 6% above the table for inductor's softmax forward at N = 8192 to 32768, and 11% below it for eager LayerNorm forward at N = 1024. Read "vs eager" for the LayerNorm forward at N ≤ 1024 and "vs compile" for the softmax forward at N ≥ 4096 as ranges, not single figures.

Limits: one run. Why those two baselines move was not investigated.

## Approximate-math budgets, measured

The bounds charge the GPU's approximate math a budget: `EXP_ULPS` and `EXP_ARG_ULPS` for `tl.exp`, `DIV_ULPS` for a division, `APPROX_MATH_ULPS` for LayerNorm's 1 / sqrt. The kernel checks cannot show how much of a budget is used, because the fp16 or bf16 store rounding is hundreds of times larger. `probe_math.py` runs each operation alone in fp32, written as the kernels write it, and compares every element with fp64. Same Kaggle job as above; an earlier job gave identical numbers, as it should with seeded inputs.

| Operation | Inputs | Max error, true ulps | Budget, model units | Used by the worst element |
|:---|:---|---:|:---|---:|
| `tl.exp(x)` | 8.4 million x, uniform in [-80, 0] | 61.4 | 2 + 1.173 \|x\| | 0.502 |
| `1.0 / x` | 4.2 million x, log-uniform in [2^-20, 2^20] | 0.97 | 2 | 0.409 |
| `1.0 / tl.sqrt(x)` | 4.2 million x, same range | 2.47 | 4 | 0.314 |

A model unit is 2^-23 relative to the result, which is between 1 and 2 true fp32 ulps. "Used" is the largest error / budget over the elements, each against its own budget.

Every budget holds with about a factor of 2 to spare, so no constant changes.

The exponential by argument size:

| \|x\| in | Max error, true ulps | vs documented bound | Used | Worst x |
|:---|---:|---:|---:|---:|
| [0, 1) | 2.18 | 1.091 | 0.478 | -0.246 |
| [1, 2) | 2.72 | 0.908 | 0.388 | -1.441 |
| [2, 4) | 3.73 | 0.746 | 0.364 | -3.017 |
| [4, 8) | 6.62 | 0.777 | 0.401 | -6.230 |
| [8, 16) | 13.89 | 0.863 | 0.453 | -11.724 |
| [16, 32) | 28.56 | 0.947 | 0.486 | -22.868 |
| [32, 64) | 57.81 | 0.995 | 0.502 | -44.757 |
| [64, 80) | 61.39 | 0.752 | 0.382 | -64.383 |

"vs documented bound" is the largest error in true ulps / (2 + floor(1.173 |x|)), the bound NVIDIA documents for `__expf`. "Worst x" is the argument of the element that uses most of its budget.

1. Triton's exp is not inside the documented `__expf` bound. For |x| below 1 it reaches 2.18 true ulps where the documented figure is 2, and at |x| between 32 and 64 it sits at 0.995 of it. So that documentation cannot be cited as a guarantee for this lowering. The budget still holds because it applies the same two numbers in model units, each of which is up to 2 true ulps. The margin comes from that unit and from this measurement, on one GPU and one Triton version.
2. The error grows with |x| as the model assumes. In model units the worst elements are at 1.1 near x = 0 and 27.4 at x = -44.8, about 0.6 per unit of |x| at the worst arguments, against the 1.173 the budget allows. The worst arguments sit where x * log2(e) is a little above a power of two (64.6, 33.0, 16.9, 9.0), which is where rounding that product to fp32 costs most relative to |x|. That fits the lowering; the parts were not measured separately.
3. The first model (4 units plus 0.5 per unit of |x|) was too small in fact, not only against the documentation: at x = -44.757 the measured error is 27.4 units and that model allowed 26.4. At the other rows' worst elements the first model would have held. A softmax output at such an argument is far below the smallest fp16 value, so the kernel checks would probably not have shown this. The probe does.

The probe samples random arguments. It is a measurement, not a proof, and it covers x in [-80, 0] only, which is the range softmax uses.

### The exp error budget

fp32 `tl.exp` lowers to `ex2.approx.f32` after an fp32 multiply by log2(e) (Triton, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`, `ExpOpConversionApprox`). CUDA's `__expf` has the same form, and the CUDA C++ Programming Guide's Mathematical Functions appendix documents it at 2 + floor(|1.173 x|) ulps. The scaled exponent x * log2(e) is rounded, so the error grows with |x|. An earlier version of this section called the two "the same instruction sequence"; the measurement above shows Triton's exp slightly outside that documented bound, so the two are not known to be identical.

The first model charged 4 ulps per call plus half an ulp per unit of |argument|. That is above the documented bound for |x| below 3 and below it beyond: 1.2x short at |x| = 6, 1.36x at 10, 1.46x at 20. In a softmax row most arguments x - max sit between -3 and -6, and the fp16 store term uses almost the whole bound (CPU ratios up to 0.998), so a GPU run could have failed on a few elements for a reason that had nothing to do with the kernel. The model now follows the documentation: `EXP_ULPS = 2` fixed per call and `EXP_ARG_ULPS = 1.173` per unit of |argument|, both in `common.py`. The CPU ratios did not move, because the store term dominates there. The GPU runs passed with this model. The earlier model was never run in a kernel check on a GPU; the probe shows the exp error exceeding it at one large argument (item 3 above).

## In a training loop

Question: are the autograd wrappers drop-in inside a real model? Every check above tests one kernel alone, on inputs made for the check.

`train_check.py` trains a small transformer: 2 pre-norm blocks, d_model 256, 4 heads, sequence 64, batch 64. LayerNorm runs at width 256, five times per forward pass, and softmax at width 64, twice, on 16384 rows each. fp16 autocast, `GradScaler`, AdamW at 1e-3, 300 steps. The task is to output the next token of a random sequence, which cannot be done without attention. Three variants start from the same weights, see the same batches, and differ only in LayerNorm and the attention softmax:

- torch: `F.layer_norm` and `torch.softmax` as autocast runs them.
- emulate: PyTorch operations held to the kernels' contract (fp16 in, fp32 arithmetic, fp16 out, fp16 weight and bias). What the kernels should compute, without Triton.
- triton: `layer_norm_autograd` and `softmax_autograd`.

Kaggle T4, torch 2.11.0, triton 3.6.0, commit `0a4645c`, in the same job as the kernel checks.

Gradients. For each of the 32 parameter tensors the check computes ||a - b|| / ||b|| on a held-out batch; the table gives the largest. It does so at two states of the model: at initialization, where attention is close to uniform, and partly trained, which is the torch variant's weights after step 33, the first step whose loss was below half the starting loss. The limit is 1e-2. The last two columns are the mutation tests: the Triton forward with a deliberately wrong backward, compared with emulate. The wrong softmax backward returns y * dy, without the sum(dy * y) term. The wrong LayerNorm backward returns w * rstd * dy for x, without the two mean terms.

| State | Triton vs emulate | Triton vs torch | Wrong softmax backward | Wrong LayerNorm backward |
|:---|---:|---:|---:|---:|
| At initialization | 7.17e-04 | 9.97e-04 | 0.208 | 0.408 |
| Partly trained | 4.44e-04 | 7.18e-04 | 4.49 | 1.91 |

Loss before each step:

| Step | torch | emulate | triton |
|---:|---:|---:|---:|
| 1 | 4.3215 | 4.3215 | 4.3215 |
| 10 | 4.2109 | 4.2109 | 4.2109 |
| 20 | 4.0920 | 4.0920 | 4.0920 |
| 30 | 2.9398 | 2.9400 | 2.9399 |
| 40 | 0.2561 | 0.2562 | 0.2562 |
| 50 | 0.0240 | 0.0240 | 0.0240 |
| 100 | 0.0028 | 0.0028 | 0.0028 |
| 200 | 0.0013 | 0.0013 | 0.0013 |
| 300 | 0.0008 | 0.0008 | 0.0008 |

1. Every parameter's gradient agrees within 0.1% at both states, against both references. That is 10 times inside the limit or better.
2. Both wrong backwards are rejected at both states, by 21 to 449 times the limit. They are hardest to see at initialization (0.21 and 0.41), which is why the comparison is repeated on a partly trained model.
3. The Triton and emulate losses differ by at most 1.0e-05 over the first 10 steps (limit 1e-3) and by at most 2.4e-04 over all 300. The largest difference is at step 33, in the steepest part of the curve, where the loss falls from 2.9 to 0.26 in ten steps.
4. All three variants end at a loss of 0.0008 and a held-out accuracy of 1.0000.
5. An earlier run of the same script, from an uncommitted tree, gave the same four numbers at initialization and the same mutant numbers at both states, to three digits. The numbers that depend on training moved: 4.37e-04 and 7.69e-04 partly trained, 5.3e-06 and 1.9e-04 for the two loss differences. Training on the GPU did not repeat exactly between the two sessions. Nothing came near a limit in either.
6. Triton against emulate (4e-4 to 7e-4) is not much closer than Triton against torch (7e-4 to 1e-3), although only the second pair differs by the fp16 contract. The likely reason, not investigated: two fp32 computations that differ in their last bits round to different fp16 values on a small share of elements, and each such element is off by a whole fp16 unit, about 5e-4 relative. The per-element bounds in the kernel checks are where rounding is held to account. This check asks a coarser question, whether training sees the right gradients.

How the limit was set. `python train_check.py --emulate` runs emulate against torch on a CPU, with the mutants built on emulate, and needs no kernel. There the fp16 contract alone moved gradients by 5.8e-04 at initialization and 7.6e-04 partly trained, and the two wrong backwards by 0.21 to 4.5. The limit of 1e-2 sits between and was fixed before the kernels were run. That CPU run also caught three faults in the first draft of the check. The model did not learn in 300 steps (6% accuracy). The wrong softmax backward moved no parameter's gradient by more than 9%. And the loss was not scaled on the CPU, so the fp16 gradients lost their digits: with the loss unscaled, emulate against torch at initialization is 5.2e-02, about 90 times the scaled figure. The fixes were position embeddings initialized like the token embeddings, separate query, key and value weights so that the gradients that exist only through the softmax backward are compared on their own, and a fixed loss scale of 65536 on the CPU, which is where `GradScaler` starts on the GPU.

Limits of this check: one small model, one seed, 300 steps, fp16 only, widths 256 and 64, and every tensor the wrappers received was contiguous. It does not time anything, so it says nothing about training speed. bf16 and the wide rows of the benchmarks are covered by the kernel checks only.

## Fixed tolerances against the bound

Question: the mutation tests show the bound rejecting faults that `allclose` accepts at atol = rtol = 1e-3. That tolerance is this repo's own choice, and the faults are planted on small values. Would another fixed tolerance do the bound's job?

Method: `probe_tolerances.py` on the Kaggle T4, torch 2.11.0, triton 3.6.0, commit `024a1bc`, 2026-11-19. A check does the job only if it accepts every correct kernel output and rejects every planted fault. The correct outputs are the 113 that the kernels produce on the regular cases of the three check scripts (every shape, dtype and path) and the 10 they produce on the mutation-test inputs. The faults are the six faulty outputs of the mutation tests. Each fixed tolerance accepts an element when |got - want| ≤ atol + rtol × |want|, and is measured against two references: PyTorch eager in the kernel's dtype, which the scripts' parity check compares with, and the fp64 reference the bound uses. Values are for fp16; bf16 uses 8e-3 in the first row and rtol 1.6e-2 in the next two.

| Check | atol | rtol | Reference | Correct outputs accepted | Faults rejected |
|:---|---:|---:|:---|---:|---:|
| This repo's parity check | 1e-3 (× rows for dw, db) | 1e-3 | eager | 123 of 123 | 0 of 6 |
| | | | fp64 | 123 of 123 | 0 of 6 |
| `torch.testing.assert_close` defaults | 1e-5 | 1e-3 | eager | 117 of 123 | 6 of 6 |
| | | | fp64 | 123 of 123 | 6 of 6 |
| The same rtol without atol | 0 | 1e-3 | eager | 88 of 123 | 6 of 6 |
| | | | fp64 | 80 of 123 | 6 of 6 |
| `torch.allclose` defaults | 1e-8 | 1e-5 | eager | 63 of 123 | 6 of 6 |
| | | | fp64 | 13 of 123 | 6 of 6 |
| Per-element bound | | | fp64 | 123 of 123 | 6 of 6 |

A fixed atol is a statement about one magnitude, so the probe then plants the faults again at 0.1 and 0.01 of the mutation tests' magnitude (upstream gradients of 1e-4 and 1e-5 where the tests use 1e-3, bias values up to 9e-5 and 9e-6 where the tests use 9e-4), with the correct kernels run on the same inputs. The online softmax forward fault has no magnitude to set and is left out there. The table gives the faulty output's largest error over what the check allows, for the `assert_close` defaults against fp64 and for the bound. Above 1 the fault is rejected; **bold** marks a fault the check accepted.

| Fault | Magnitude 1: tolerance | bound | 0.1: tolerance | bound | 0.01: tolerance | bound |
|:---|---:|---:|---:|---:|---:|---:|
| LayerNorm forward drops the bias | 84.5 | 21.4 | 8.22 | 2.16 | **0.82** | **0.84** |
| LayerNorm dx drops the `xhat * mean(wdy * xhat)` term | 29.9 | 546 | 3.12 | 329 | **0.31** | 82 |
| LayerNorm dw skips the last row | 300 | 145 | 40.8 | 144 | 4.77 | 133 |
| LayerNorm db skips the last row | 256 | 821 | 29.7 | 765 | 3.02 | 456 |
| Softmax online forward skips the rescale | 43.7 | 486 | - | - | - | - |
| Softmax backward drops the `sum(dy * y)` term | 5.92 | 1620 | **0.59** | 194 | **0.06** | 20.2 |
| Faults rejected | 6 of 6 | 6 of 6 | 4 of 5 | 5 of 5 | 2 of 5 | 4 of 5 |

At both smaller magnitudes the `assert_close` defaults and the bound accepted all 9 correct outputs on those inputs. Against eager the tolerance's verdicts on the faults are the same as against fp64.

1. The earlier claim was about this repo's tolerance. At atol = rtol = 1e-3 `allclose` accepts all six faults against either reference; the largest faulty error is 0.90 of what it allows. That is a tolerance chosen too loose, not a limit of fixed tolerances.
2. At the magnitude the mutation tests use, the defaults of `torch.testing.assert_close` against the fp64 reference do the whole job: every correct output accepted, the worst using 0.49 of the allowance, and every fault rejected, by 5.9 to 300 times the allowance. On these six faults the bound shows no advantage over that check. For two of them, the dropped bias and dw, the fixed tolerance rejects by the wider margin.
3. Against eager in the kernel's dtype the same tolerance rejects 6 of the 123 correct outputs: softmax dx on (17, 7) fp16, (64, 127) bf16 and the wide-spread (64, 1024) case, on both paths, by up to 2.2 times the allowance. The kernels are inside their bound against fp64 on those cases, so these are false alarms. A tolerance this tight needs a reference more accurate than the fp16 eager result.
4. Dropping atol rejects every fault and 35 to 43 of the 123 correct outputs. Near zero a relative tolerance allows almost nothing, and fp16 rounding or fp32 cancellation is larger than that. The `allclose` defaults are tighter than fp16 rounding and reject 60 to 110.
5. The bound's advantage appears when the values shrink. At 0.1 of the magnitude the `assert_close` defaults accept one fault of five, the softmax backward, at 0.59 of the allowance. At 0.01 they accept three of five. The bound rejects all five at 0.1 and four of five at 0.01, with margins of 20 to 765 on the ones it rejects apart from the bias.
6. The bound is not the stronger check everywhere. It accepts the dropped bias at 0.01: the fault changed 1.8% of the elements and every one stayed inside its bound, and at 0.1 it rejects that fault by only 2.2. The cause is the bound's own slack. Computed from the error model on the CPU for the same input recipe, the LayerNorm forward bound for outputs with |y| < 1e-2 at N = 1024 has a median of 5.1e-5, about 20 times fp16 rounding there (2.5e-6), because it allows any summation order for the row statistics. A bias below that floor cannot be seen. A bound that modeled the kernel's actual reduction order would be tighter.

Limits: six faults, chosen by the author, on inputs chosen to make them small; they are not a sample of real bugs and the counts are not rates. One GPU, one torch and Triton version, one seed. The fp64 reference for softmax dx is computed from the kernel's own y, so the eager column there also contains the difference between the two forwards. At 0.01 the planted gradients are about 1e-5, which is subnormal in fp16; mixed-precision training normally scales the loss to stay out of that range, so the 0.1 row is the more realistic one. The eager columns describe PyTorch's fp16 implementation on this GPU and version.

The mutation tests were restructured for this probe so that both judge the same tensors (`MutantCase` in `common.py`). The forward test now also requires the correct kernel to pass the bound on its own input, which the other five already did; it does, at 0.83 of the bound. `probe_tolerances.py --emulate` runs the probe on a CPU with fp32 torch arithmetic in place of the kernels (`emulation.py`). It tests the script, not the kernels, and none of the numbers above come from it except the two CPU figures in finding 6.

## Fused attention: measured, not built

Question: can a fused Triton attention forward beat PyTorch on this GPU? A fused kernel never writes the L x L score matrix to memory. But it has to do both matrix products itself, and Triton lowers `tl.dot` to plain fp32 multiply-adds below compute capability 8.0 (`lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`: "Dot op using MMA for compute capability ... has been deprecated. It falls back to the FMA path"; the same text in 3.6.0 and 3.8.0). The T4 is 7.5. PyTorch's matmul uses the T4's tensor cores.

`probe_attention.py` measures it before anything larger is built. It times a prototype fused forward (online softmax over key tiles, six block and warp settings) against `F.scaled_dot_product_attention`, the written-out `softmax(q @ k^T * scale) @ v`, and `torch.compile` of the written-out form. Batch 4, 8 heads, head dimension 64, fp16, non-causal. Kaggle T4, torch 2.11.0, triton 3.6.0, commit `81f6476`. TFLOPS counts the two matrix products.

| L | Built-in ms | TFLOPS | Written-out ms | TFLOPS | Compile ms | TFLOPS | Triton best ms | TFLOPS | Best setting | Triton vs best PyTorch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 256 | 0.1345 | 3.99 | 0.2033 | 2.64 | 0.1346 | 3.99 | 0.8841 | 0.61 | 64x32, 4 warps | 0.15x |
| 512 | 0.3174 | 6.77 | 0.5383 | 3.99 | 0.1592 | 13.49 | 1.5754 | 1.36 | 32x32, 4 warps | 0.10x |
| 1024 | 0.6172 | 13.92 | 2.0184 | 4.26 | 0.5569 | 15.43 | 6.3193 | 1.36 | 32x32, 4 warps | 0.09x |
| 2048 | 2.5841 | 13.30 | 8.0322 | 4.28 | 2.5975 | 13.23 | 24.8423 | 1.38 | 32x32, 4 warps | 0.10x |

1. The fused kernel is slower at every length, by 7 to 11 times. An earlier run of the same probe gave 0.09x to 0.16x.
2. No setting of the prototype exceeds 1.4 TFLOPS in either run. PyTorch reaches 13 to 15 TFLOPS from L = 1024.
3. Tuning cannot close that. NVIDIA publishes 8.1 TFLOPS single precision and 65 TFLOPS fp16 on tensor cores for the T4. A kernel limited to fp32 multiply-adds is capped by the first figure, and PyTorch already measures above it. That cap is NVIDIA's number, not a measurement made here.
4. The memory traffic that fusing saves is not the bottleneck on this GPU. The written-out form pays for the whole score matrix, runs at 4.3 TFLOPS, and is still 3 times faster than the prototype.
5. `torch.compile` of the written-out form matches the built-in from L = 1024 and is twice as fast at L = 512 (0.16 against 0.32 ms).

Decision: stop. No checked attention kernel is built for this GPU. The rule was set before the measurement: build it only if the prototype beats PyTorch somewhere.

Limits of the verdict: one head dimension, non-causal, six hand-picked settings without autotuning, and a prototype checked only against an fp32 reference with a fixed tolerance. Those could move the result by tens of percent, not by 7 times. On a GPU with compute capability 8.0 or higher Triton does use tensor cores, and the answer may be different; `probe_attention.py` is kept so it can be asked there first.

## Open items

- At the magnitude the mutation tests use, the per-element bound shows no advantage over the defaults of `torch.testing.assert_close` against an fp64 reference: both accept every correct output and reject all six planted faults. The bound pulls ahead only when the same faults are planted on smaller values. That rests on six faults chosen by the author, one GPU and one seed. No fault that a well-chosen fixed tolerance misses at ordinary magnitudes has been shown here.
- The LayerNorm forward bound allows any summation order for the row statistics, which leaves a floor of about 5e-5 for outputs near zero at N = 1024, 20 times fp16 rounding. It accepted a dropped bias of up to 9e-6. Modeling the kernel's actual reduction order would tighten it; that has not been done.
- The approximate-math budgets rest on a measurement on one GPU model and one Triton version, not on a documented guarantee: Triton's exp is slightly outside NVIDIA's documented `__expf` bound. `kaggle_run.py run` runs `probe_math.py` before the checks, so a new Triton version or GPU that uses more of a budget shows up there first. Run it before trusting the bounds anywhere else.
- The backward numbers at N = 32768 assume Triton 3.8.0 or later. On 3.6.0 and 3.7.1 the dx kernel spills more and the split backward is 9 to 12% slower there. The Triton change responsible, somewhere between 3.7.1 and 3.8.0 in the code it generates, was not identified.
- The LayerNorm benchmark tables on the Kaggle versions are from one run, reproduced once (45 of 49 cells within 3%). The forward at N = 32768 varies by about 5% between samples on one GPU.
- Two baselines move between runs on the Kaggle T4 by more than the kernels do: eager LayerNorm forward at N ≤ 1024 (up to 18%) and inductor's softmax forward at N = 4096 to 32768 (5 to 9%). The cause was not investigated and the individual run times were not saved. Comparisons against them are ranges.
- No table was measured on the releases current on 2026-11-17, torch 2.15.0 with triton 3.9.0. The `torch.compile` columns depend on the torch version, as the two LayerNorm forward results show.
- Whether the LayerNorm forward beats `torch.compile` at N = 16384 and 32768 depends on the torch version: it did on torch 2.14.1 and does not on torch 2.11.0. Only the newer result has been repeated.
- At N = 32768 the dx kernel keeps 26 spill instructions at 8 warps and runs at 207 GB/s against 247 GB/s at N = 16384. Accepted: the measured alternative was slower, and 32768 is the top of the fp16 single-block range.
- The attention verdict is for the T4 only. On compute capability 8.0 or higher, rerun `probe_attention.py` before deciding anything. Kaggle would not provide such a GPU on this account (2026-11-16): asked for `NvidiaL4X1` it ran two T4s, and asked for `NvidiaL4` it ran a session with no GPU, both without an error. `kaggle_run.py` now fails a run whose GPU is not the one requested.
- The training-loop check is one small model at LayerNorm width 256 and softmax width 64, fp16, one seed. Nothing was trained in bf16 or at the wide rows, and no training speed was measured.
- All numbers are from one GPU model. The switch point and the warp cap are T4 measurements; rerun `probe_spills.py` and both benchmarks before trusting them on another GPU.
