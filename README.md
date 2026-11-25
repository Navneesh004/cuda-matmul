# cuda-matmul: SGEMM from naive to near-cuBLAS

Matrix multiplication (C = A × B, FP32, square N × N) written as a series of CUDA kernels. Each one moves
data into faster memory and reuses it more. Every kernel is checked against cuBLAS, timed with CUDA
events, profiled with Nsight Compute, and placed on a roofline chart.

The point of the project: a GPU's cores are rarely limited by arithmetic. They are limited by how fast
you can **feed** them. Each step below is a memory-hierarchy change, not a maths change.

| # | kernel | idea | file |
|---|---|---|---|
| 1 | naive | one thread per output cell; neighbouring threads read rows N floats apart (uncoalesced) | [01_naive.cuh](src/kernels/01_naive.cuh) |
| 2 | coalesced | swap row/col so a warp reads 32 neighbouring floats in one trip | [02_coalesced.cuh](src/kernels/02_coalesced.cuh) |
| 3 | shared | 32 × 32 tiles of A and B in shared memory, each value reused 32 times | [03_shared.cuh](src/kernels/03_shared.cuh) |
| 4 | reg1d | each thread computes 8 outputs in a column; B value held in a register | [04_reg1d.cuh](src/kernels/04_reg1d.cuh) |
| 5 | reg2d | each thread computes an 8 × 8 square: 16 shared loads per 64 FMAs | [05_reg2d.cuh](src/kernels/05_reg2d.cuh) |
| 6 | vectorized | `float4` global loads/stores, A tile stored transposed for `float4` shared reads | [06_vectorized.cuh](src/kernels/06_vectorized.cuh) |
| 7 | tuned | kernel 6 + padding to remove a shared-memory bank conflict + tile-size sweep | [07_tuned.cuh](src/kernels/07_tuned.cuh) |
| 0 | cuBLAS | `cublasSgemm`, the reference | [main.cu](src/main.cu) |
| – | cpu | single-threaded CPU loop, as a baseline (small N only) | [main.cu](src/main.cu) |

All kernels handle any N (edges are zero-padded), including sizes like 1000 and 1023 that are not
multiples of the tile size or of 4.

## Results

Benchmarks not yet run; results will be added after running on a GPU.

## Running it

You need an NVIDIA GPU. No local card? Use Google Colab (free T4) or Kaggle (T4 / P100).

### Colab or Kaggle (easiest)

1. Open [notebooks/colab.ipynb](notebooks/colab.ipynb) in Colab (File → Open notebook → GitHub) or upload it
   to Kaggle. Pick a GPU runtime.
2. Run all cells. It builds, tests, tunes, benchmarks, profiles and
   draws the charts, then gives you `results.zip` to commit.

### Linux / WSL2 with the CUDA Toolkit

```bash
make                      # detects your GPU's arch; or: make ARCH=sm_75 (T4), sm_86 (RTX 30), sm_89 (RTX 40)
make test                 # every kernel vs cuBLAS at 15 awkward sizes
./matmul --kernel 3 --n 4096
scripts/run_all.sh        # test, tune, bench, profile, charts -> results/
pip install -r requirements.txt   # for the Python chart scripts
```

Without `make`:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo src/main.cu -o matmul -lcublas
```

### Harness options

```
./matmul --list                          kernels and kernel-7 tile configs
./matmul --info                          GPU name, compute capability, SMs, memory
./matmul --kernel 5 --n 4096             human-readable report
./matmul --kernel tuned --cfg 1 --n 4096 kernels can be given by id or name
./matmul --kernel 7 --n 4096 --csv       one CSV row (header: --csv-header)
./matmul --kernel 3 --n 2048 --no-check  skip the cuBLAS reference (for profiling)
```

Example report:

```
GPU          : Tesla T4 (sm_75)
kernel       : 3 (shared)
N            : 4096
time         : ... ms median, ... ms min over 10 runs
throughput   : ... GFLOPS
cuBLAS       : ... ms, ... GFLOPS -> ...% of cuBLAS
H2D copy     : ... ms (A and B)   D2H copy: ... ms (C)
correct      : yes (normalised max error 2.1e-07, tolerance 1.0e-03)
```

The exit code is 1 if the result does not match cuBLAS, so scripts stop on a wrong kernel.

## How the measurements are made

- **Timing:** `cudaEvent`s around each launch (`cudaEventSynchronize` before reading), never a CPU clock:
  launches return immediately while the GPU works in the background.
- **Warm-up:** the first launch (which also serves as the correctness run) is not timed. The reported time
  is the **median** of 10 runs (3 for the naive kernels at N ≥ 2048, which take seconds each).
- **Copies vs kernel:** host-to-device and device-to-host copies are timed separately (pinned memory) and
  never included in kernel time.
- **GFLOPS** = 2·N³ / seconds / 10⁹ (one multiply and one add per inner step).
- **Correctness:** C is pre-filled with NaN, so any cell a kernel forgets to write fails. The check is
  `max|C − C_cuBLAS| / max|C_cuBLAS| < 1e-3`. Exact equality would fail for correct code, because adding
  floats in a different order changes the last bits. Real bugs give errors of order 1.
- **Error checks:** every CUDA call goes through `CUDA_CHECK`, and `cudaGetLastError()` runs after every
  launch, so a bad launch configuration fails loudly instead of leaving garbage in C.
- cuBLAS runs in its default FP32 math mode (no TF32), so the comparison is like for like.

## What each step changes, and how to see it in the profiler

`scripts/profile.sh` collects these per kernel into `results/ncu_metrics.csv` and
`results/ncu_summary.md`. `FULL=1 scripts/profile.sh` also saves full reports
(`results/ncu/kernel*.ncu-rep`) for the Nsight Compute GUI.

| step | expected evidence |
|---|---|
| 1 → 2 coalescing | **sectors/request** for global loads drops towards 4 (a warp's 32 floats in four 32-byte sectors). This is the modern form of nvprof's "global load efficiency". |
| 2 → 3 shared tiling | DRAM bytes fall by roughly the tile width; arithmetic intensity rises from ~0.25 to ~8 FLOP/byte. |
| 3 → 4 → 5 register tiling | shared-memory loads per FMA fall (1 → 1/8 → 16/64); SM throughput % rises; the dot moves right on the roofline, past the ridge point. |
| 5 → 6 vectorised | fewer load instructions (`float4`); kernel 5's strided shared reads had 2–4-way bank conflicts. |
| 6 → 7 padding | **shared store bank conflicts** fall to ~0. With `As[k][m]` and BM = 128, threads storing k and k+4 for the same m hit the same bank; 4 floats of padding per row shift them 16 banks apart. Kernel 7 `--cfg 0` has the same tiles as kernel 6, so it isolates this. |
| tile sweep | `scripts/tune.sh` tries 7 configs; the best is saved to `results/best_cfg.txt` and used by `bench.sh`. |

Also look at **occupancy** (big register tiles use more registers per thread, so fewer warps fit) and
**warp stall reasons** in the full report. `build.log` lists registers and shared memory per kernel
(`-Xptxas=-v`).

If `ncu` fails with `ERR_NVGPUCTRPERM`, the machine blocks hardware counters (common on cloud notebooks).
Use Kaggle or a local GPU, or run `SKIP_PROFILE=1 scripts/run_all.sh`; the roofline then uses theoretical
intensities.

## The roofline

Arithmetic intensity (AI) = FLOPs per byte read from global memory. A kernel is capped at
`min(peak FLOPS, AI × memory bandwidth)`. On a T4 (≈ 8.1 TFLOPS FP32, ≈ 320 GB/s) the ridge point is
≈ 25 FLOP/byte: below it a kernel is memory-bound, above it compute-bound.

A block that computes a BM × BN tile of C loads (BM + BN) floats per k step and does 2·BM·BN FLOPs, so
**AI = BM·BN / (2(BM + BN))**:

| kernel | tile | theoretical AI |
|---|---|---:|
| naive, coalesced | 1 × 1 | 0.25 |
| shared | 32 × 32 | 8 |
| reg1d | 64 × 64 | 16 |
| reg2d, vectorized, tuned | 128 × 128 | 32 |

So the tiling steps carry the kernel from the memory-bound side of the ridge to the compute-bound side.
When `results/ncu_metrics.csv` exists, `roofline.py` uses the **measured** AI (2N³ / DRAM bytes) instead.
Measured AI is higher than theoretical for the naive kernels, because L1/L2 caches give some reuse the
tile model does not count. Peak values for common GPUs are built in (approximate); override with
`--peak-gflops` and `--peak-bw`.

## Repo layout

```
cuda-matmul/
├── src/
│   ├── main.cu            harness: kernel selection, timing, correctness check, CSV output
│   ├── utils.cuh          CUDA_CHECK / CUBLAS_CHECK, random fill, compare
│   └── kernels/           01_naive.cuh ... 07_tuned.cuh
├── scripts/
│   ├── run_all.sh         build, test, tune, bench, profile, plot
│   ├── test.sh            correctness at sizes 1, 7, 31, ..., 1023, 1025, 2048
│   ├── tune.sh            kernel-7 tile-size sweep -> results/best_cfg.txt
│   ├── bench.sh           all kernels, N = 256 ... 8192 -> results/bench.csv
│   ├── profile.sh         Nsight Compute metrics -> results/ncu_metrics.csv, ncu_summary.md
│   ├── plot.py            timing.png, sizes.png, summary.md
│   └── roofline.py        roofline.png
├── notebooks/colab.ipynb  one-click run on Colab / Kaggle
├── results/               CSVs, charts, profiler summaries (committed)
├── Makefile
└── README.md
```

## Ideas not done yet

- Double buffering: load the next tile from global memory while computing on the current one.
- Warp tiling (an extra level between block and thread tiles), as cuBLAS/CUTLASS do.
- Tensor Cores through the WMMA API (FP16/TF32 inputs), which is how modern AI workloads get their speed.

## References

- Hwu, Kirk, El Hajj, *Programming Massively Parallel Processors* (4th ed.).
- NVIDIA, *CUDA C++ Programming Guide* and *Nsight Compute* documentation.
- Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog". This
  project follows a similar path; the code here was written independently.
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore
  Architectures", CACM 2009.
