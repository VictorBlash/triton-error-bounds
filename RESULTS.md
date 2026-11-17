# Triton LayerNorm and softmax: GPU results

The first LayerNorm sections were measured on Google Colab, November 2026. Everything from "LayerNorm on the Kaggle versions" down was measured on Kaggle with older library versions; the environment is stated there. Colab:

- GPU: NVIDIA Tesla T4, 16 GB, about 320 GB/s peak memory bandwidth. The best any kernel here reaches is about 250 GB/s, and that is treated as the practical ceiling.
- torch 2.14.1+cu130, triton 3.8.0.

All timings use M = 4096 rows, fp16, and the median from `triton.testing.do_bench`. Runs are numbered in the order they happened. The current tables are from run 5, which added the `torch.compile` column (default inductor mode, `dynamic=False`, one compile per width). Earlier runs agreed within noise at every width they shared, except eager at N ≤ 1024, which swings by up to 2x between runs.

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
- Timings use M = 4096 rows, fp16, the median from `do_bench`, and one fresh `torch.compile` per width, as in the Colab tables.
- The tables are from one job, pushed by `kaggle_run.py run --bench` from a clean tree at commit `7b0c7f8`. The job prints the commit it was built from.

## LayerNorm on the Kaggle versions

One run. The checks pass, as they did on Colab.

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
- The Triton kernels themselves are within 3% of Colab up to N = 16384, apart from the forward at 512 (4.5% slower). At N = 32768 both are slower on triton 3.6.0: the forward by 6% (214 against 227 GB/s) and the split backward by 11% (6.58 against 5.94 ms). `probe_spills.py` was not run on this version, so the cause is not known.

## Softmax

The softmax kernels, checks and benchmarks are in `softmax_triton.py` and `softmax_checks.py`.

The tables are the second of two runs on this environment. The first, a day earlier in a hand-driven notebook at commit `1bfef9e`, agrees within 0.4% in every backward cell and within 3.2% in every forward cell except three: online and compile at N = 512 (9 to 10%) and eager at N = 4096 (6%). The softmax block forward plateaus at 230 GB/s and the LayerNorm forward at 228 GB/s on the same versions.

### Correctness

`softmax_checks.py` passes every check on the T4.

- 13 shapes, each on every path that fits: block and online forward, block and tiled backward. N = 40000 is above the single-block limit and runs on the online and tiled paths only. N = 1 passes on the online path, where the tile is one element wide.
- Each forward output and each dx is checked for finite values, against eager with `allclose`, and against fp64 with the per-element bound. Two launches must be bitwise equal, and a row launched alone must equal the same row from the batch.
- Mutation tests on real kernels. The bound rejected both mutants and `allclose` missed both:

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

- Against `torch.compile` the forward is a tie up to N = 8192 (0.96x to 1.03x) and at 65536 (0.99x). The block kernel wins at N = 16384 (1.13x; 1.09x in the first run) and at N = 32768 (1.34x), where inductor's softmax drops to 205 and 174 GB/s.
- The block kernel holds 223 to 233 GB/s from N = 1024 to 32768.
- Eager `F.softmax` is not monotonic: 166 to 241 GB/s up to N = 16384, then 171 and 127 GB/s. At N = 16384 eager is 4% faster than the block kernel, in both runs. That is the one width where this forward loses to eager.
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

## Open items

- The approximate-math budgets rest on a measurement on one GPU model and one Triton version, not on a documented guarantee: Triton's exp is slightly outside NVIDIA's documented `__expf` bound. `kaggle_run.py run` runs `probe_math.py` before the checks, so a new Triton version or GPU that uses more of a budget shows up there first. Run it before trusting the bounds anywhere else.
- The LayerNorm tables on the Kaggle versions are from one run. On those versions the forward and the split backward are 6% and 11% slower at N = 32768 than on Colab, and `probe_spills.py` has not been run there to say why.
- Whether the LayerNorm forward beats `torch.compile` at N = 16384 and 32768 depends on the torch version: it did on torch 2.14.1 and does not on torch 2.11.0. Only the newer result has been repeated.
- At N = 32768 the dx kernel keeps 26 spill instructions at 8 warps and runs at 207 GB/s against 247 GB/s at N = 16384. Accepted: the measured alternative was slower, and 32768 is the top of the fp16 single-block range.
- All numbers are from one GPU model. The switch point and the warp cap are T4 measurements; rerun `probe_spills.py` and both benchmarks before trusting them on another GPU.
