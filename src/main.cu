// main.cu - test harness: kernel selection, timing, correctness check vs cuBLAS.
//
//   ./matmul --kernel 3 --n 4096            human-readable report
//   ./matmul --kernel 3 --n 4096 --csv      one CSV row (see --csv-header)
//   ./matmul --list                         list kernels
//
// Exit codes: 0 = ok, 1 = result did not match cuBLAS, 2 = bad arguments.
#include "utils.cuh"

#include "kernels/01_naive.cuh"
#include "kernels/02_coalesced.cuh"
#include "kernels/03_shared.cuh"
#include "kernels/04_reg1d.cuh"
#include "kernels/05_reg2d.cuh"
#include "kernels/06_vectorized.cuh"
#include "kernels/07_tuned.cuh"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

using LaunchFn = void (*)(const float*, const float*, float*, int, int);

struct KernelInfo {
    int id;
    const char* name;
    LaunchFn launch;   // nullptr for cuBLAS / CPU, which are handled specially
    const char* desc;
};

constexpr int KERNEL_CPU = -1;
constexpr int KERNEL_CUBLAS = 0;

static const KernelInfo KERNELS[] = {
    {KERNEL_CPU,    "cpu",        nullptr,          "single-threaded CPU loop (i-k-j order); baseline only, slow"},
    {KERNEL_CUBLAS, "cublas",     nullptr,          "cublasSgemm, the reference"},
    {1,             "naive",      launch_naive,     "one thread per output, uncoalesced"},
    {2,             "coalesced",  launch_coalesced, "one thread per output, coalesced"},
    {3,             "shared",     launch_shared,    "32x32 shared-memory tiling"},
    {4,             "reg1d",      launch_reg1d,     "1D register tiling, 8 outputs per thread"},
    {5,             "reg2d",      launch_reg2d,     "2D register tiling, 8x8 outputs per thread"},
    {6,             "vectorized", launch_vectorized,"2D tiling + float4 loads, transposed A tile"},
    {7,             "tuned",      launch_tuned,     "kernel 6 + bank-conflict padding, tile size via --cfg"},
};

static const KernelInfo* find_kernel(const std::string& s) {
    for (const auto& k : KERNELS)
        if (s == k.name || s == std::to_string(k.id)) return &k;
    return nullptr;
}

struct Options {
    std::string kernel = "1";
    int n = 1024;
    int reps = 10;
    int warmup = 1;
    int cfg = DEFAULT_TUNED_CONFIG;
    int device = 0;
    unsigned seed = 42;
    double tol = 1e-3;
    bool check = true;
    bool csv = false;
    bool csv_header = false;
    bool list = false;
    bool info = false;
    bool force = false;
};

static void usage(const char* prog) {
    std::printf(
        "usage: %s [options]\n"
        "  --kernel K     kernel id or name (default 1); see --list\n"
        "  --n N          matrix size N (C = A x B, all N x N; default 1024)\n"
        "  --reps R       timed runs, median reported (default 10)\n"
        "  --warmup W     untimed runs before timing (default 1, min 1)\n"
        "  --cfg I        tile config for kernel 7 (0..%d, default %d)\n"
        "  --tol T        max allowed normalised error vs cuBLAS (default 1e-3)\n"
        "  --seed S       RNG seed for A and B (default 42)\n"
        "  --device D     CUDA device index (default 0)\n"
        "  --no-check     skip the cuBLAS comparison (use when profiling)\n"
        "  --force        allow the CPU kernel for N > 2048\n"
        "  --csv          print one CSV row instead of the report\n"
        "  --csv-header   print the CSV header and exit\n"
        "  --list         list kernels and kernel-7 configs, then exit\n"
        "  --info         print GPU info and exit\n",
        prog, NUM_TUNED_CONFIGS - 1, DEFAULT_TUNED_CONFIG);
}

static Options parse_args(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(2); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--kernel" || a == "-k") o.kernel = need(i);
        else if (a == "--n" || a == "-n") o.n = std::atoi(need(i));
        else if (a == "--reps") o.reps = std::atoi(need(i));
        else if (a == "--warmup") o.warmup = std::atoi(need(i));
        else if (a == "--cfg") o.cfg = std::atoi(need(i));
        else if (a == "--tol") o.tol = std::atof(need(i));
        else if (a == "--seed") o.seed = (unsigned)std::strtoul(need(i), nullptr, 10);
        else if (a == "--device") o.device = std::atoi(need(i));
        else if (a == "--no-check") o.check = false;
        else if (a == "--force") o.force = true;
        else if (a == "--csv") o.csv = true;
        else if (a == "--csv-header") o.csv_header = true;
        else if (a == "--list") o.list = true;
        else if (a == "--info") o.info = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(argv[0]); std::exit(2); }
    }
    if (o.n < 1 || o.n > 16384) { std::fprintf(stderr, "--n must be in 1..16384\n"); std::exit(2); }
    if (o.reps < 1) o.reps = 1;
    if (o.warmup < 1) o.warmup = 1;
    return o;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5f * (v[m - 1] + v[m]);
}

// Time `reps` runs of fn() with CUDA events (the GPU clock, not the CPU clock:
// launches return immediately, so a CPU timer would measure almost nothing).
template <typename F>
static std::vector<float> time_gpu(F fn, int reps) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> ms(reps);
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms[r], start, stop));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// Row-major C = A * B via column-major cuBLAS: compute C^T = B^T * A^T,
// i.e. pass B first. No transposes are actually performed.
static void cublas_matmul(cublasHandle_t h, const float* A, const float* B, float* C, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N));
}

static void cpu_matmul(const float* A, const float* B, float* C, int N) {
    std::memset(C, 0, sizeof(float) * (size_t)N * N);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k) {
            const float a = A[(size_t)i * N + k];
            const float* b = &B[(size_t)k * N];
            float* c = &C[(size_t)i * N];
            for (int j = 0; j < N; ++j) c[j] += a * b[j];
        }
}

static double gflops(int n, double ms) { return 2.0 * (double)n * n * n / (ms * 1e-3) / 1e9; }

static const char* CSV_HEADER =
    "gpu,kernel_id,kernel,cfg,n,ms_median,ms_min,gflops,cublas_ms,cublas_gflops,pct_cublas,"
    "rel_err,pass,h2d_ms,d2h_ms";

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);

    if (opt.csv_header) { std::printf("%s\n", CSV_HEADER); return 0; }
    if (opt.list) {
        std::printf("kernels:\n");
        for (const auto& k : KERNELS) std::printf("  %3d  %-11s %s\n", k.id, k.name, k.desc);
        std::printf("kernel 7 configs (--cfg):\n");
        for (int i = 0; i < NUM_TUNED_CONFIGS; ++i) {
            const auto& c = TUNED_CONFIGS[i];
            std::printf("  %d  BM=%-3d BN=%-3d BK=%-2d TM=%d TN=%d  (%d threads)%s\n", i, c.BM, c.BN, c.BK,
                        c.TM, c.TN, (c.BM / c.TM) * (c.BN / c.TN), i == DEFAULT_TUNED_CONFIG ? "  [default]" : "");
        }
        return 0;
    }

    const KernelInfo* kinfo = find_kernel(opt.kernel);
    if (!kinfo) { std::fprintf(stderr, "unknown kernel '%s' (try --list)\n", opt.kernel.c_str()); return 2; }
    if (kinfo->id == 7 && (opt.cfg < 0 || opt.cfg >= NUM_TUNED_CONFIGS)) {
        std::fprintf(stderr, "--cfg must be in 0..%d\n", NUM_TUNED_CONFIGS - 1);
        return 2;
    }
    if (kinfo->id == KERNEL_CPU && opt.n > 2048 && !opt.force) {
        std::fprintf(stderr, "CPU kernel at N=%d would take very long; pass --force to run it anyway\n", opt.n);
        return 2;
    }

    CUDA_CHECK(cudaSetDevice(opt.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
    if (opt.info) {
        std::printf("device %d: %s\n", opt.device, prop.name);
        std::printf("  compute capability : %d.%d (build with -arch=sm_%d%d)\n", prop.major, prop.minor,
                    prop.major, prop.minor);
        std::printf("  SMs                : %d\n", prop.multiProcessorCount);
        std::printf("  global memory      : %.1f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("  shared mem / block : %zu KiB\n", prop.sharedMemPerBlock / 1024);
        std::printf("  memory bus width   : %d bits\n", prop.memoryBusWidth);
        std::printf("  L2 cache           : %d KiB\n", prop.l2CacheSize / 1024);
        return 0;
    }

    const int N = opt.n;
    const size_t elems = (size_t)N * N;
    const size_t bytes = elems * sizeof(float);
    const bool is_cpu = kinfo->id == KERNEL_CPU;
    const bool is_cublas = kinfo->id == KERNEL_CUBLAS;
    const bool check = opt.check;
    const int cfg = kinfo->id == 7 ? opt.cfg : -1;

    // Host buffers (pinned, so the copy timings reflect real PCIe bandwidth).
    float *hA, *hB, *hC;
    CUDA_CHECK(cudaMallocHost(&hA, bytes));
    CUDA_CHECK(cudaMallocHost(&hB, bytes));
    CUDA_CHECK(cudaMallocHost(&hC, bytes));
    std::vector<float> hRef;
    fill_random(hA, elems, opt.seed);
    fill_random(hB, elems, opt.seed + 1);

    float *dA, *dB, *dC, *dRef = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    if (check) CUDA_CHECK(cudaMalloc(&dRef, bytes));

    // Host -> device copy, timed separately from the kernel.
    float h2d_ms = 0.0f;
    {
        auto t = time_gpu([&] {
            CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
        }, 1);
        h2d_ms = t[0];
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Plain FP32 maths for the reference (no TF32 shortcuts on Ampere+).
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // What we run and time.
    auto run = [&] {
        if (is_cublas) {
            cublas_matmul(handle, dA, dB, dC, N);
        } else {
            kinfo->launch(dA, dB, dC, N, cfg);
            CUDA_CHECK_LAUNCH();
        }
    };

    // Reference result and cuBLAS timing (for "% of cuBLAS").
    float cublas_ms = 0.0f;
    if (check) {
        auto ref = [&] { cublas_matmul(handle, dA, dB, dRef, N); };
        ref();   // warm-up: first call includes cuBLAS/handle setup
        CUDA_CHECK(cudaDeviceSynchronize());
        cublas_ms = median(time_gpu(ref, opt.reps));
        hRef.resize(elems);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));
    }

    std::vector<float> times;
    float d2h_ms = 0.0f;
    CompareResult cmp;
    bool compared = false;

    if (is_cpu) {
        for (int r = 0; r < std::min(opt.reps, 3); ++r) {   // the CPU is slow: at most 3 runs
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_matmul(hA, hB, hC, N);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<float, std::milli>(t1 - t0).count());
        }
        if (check) { cmp = compare(hC, hRef.data(), elems, opt.tol); compared = true; }
    } else {
        // Fill C with NaN (0xFF bytes) so cells the kernel never writes fail the check.
        CUDA_CHECK(cudaMemset(dC, 0xFF, bytes));
        // The first run is the correctness run and also the first warm-up.
        run();
        CUDA_CHECK(cudaDeviceSynchronize());
        d2h_ms = time_gpu([&] { CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost)); }, 1)[0];
        if (check && !is_cublas) { cmp = compare(hC, hRef.data(), elems, opt.tol); compared = true; }

        for (int w = 1; w < opt.warmup; ++w) run();
        CUDA_CHECK(cudaDeviceSynchronize());
        times = time_gpu(run, opt.reps);
    }

    const float ms_med = median(times);
    const float ms_min = *std::min_element(times.begin(), times.end());
    const double gf = gflops(N, ms_med);
    const double cublas_gf = check ? gflops(N, cublas_ms) : 0.0;
    const double pct = check ? 100.0 * gf / cublas_gf : 0.0;
    const bool pass = !compared || cmp.pass;
    const char* pass_str = compared ? (cmp.pass ? "yes" : "NO") : (is_cublas ? "ref" : "skipped");

    if (opt.csv) {
        std::printf("\"%s\",%d,%s,%d,%d,%.4f,%.4f,%.2f,%.4f,%.2f,%.2f,%.3e,%s,%.3f,%.3f\n", prop.name, kinfo->id,
                    kinfo->name, cfg, N, ms_med, ms_min, gf, cublas_ms, cublas_gf, pct, compared ? cmp.rel_err : 0.0,
                    pass_str, h2d_ms, d2h_ms);
    } else {
        std::printf("GPU          : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
        std::printf("kernel       : %d (%s)%s\n", kinfo->id, kinfo->name,
                    cfg >= 0 ? (" cfg " + std::to_string(cfg)).c_str() : "");
        std::printf("N            : %d\n", N);
        std::printf("time         : %.4f ms median, %.4f ms min over %zu runs\n", ms_med, ms_min, times.size());
        std::printf("throughput   : %.1f GFLOPS\n", gf);
        if (check) std::printf("cuBLAS       : %.4f ms, %.1f GFLOPS -> %.1f%% of cuBLAS\n", cublas_ms, cublas_gf, pct);
        std::printf("H2D copy     : %.3f ms (A and B)   D2H copy: %.3f ms (C)\n", h2d_ms, d2h_ms);
        if (compared) {
            std::printf("correct      : %s (normalised max error %.3e, tolerance %.1e", cmp.pass ? "yes" : "NO",
                        cmp.rel_err, opt.tol);
            if (cmp.nan_count) std::printf(", %zu non-finite outputs", cmp.nan_count);
            std::printf(")\n");
            if (!cmp.pass && cmp.nan_count == 0) {
                size_t i = cmp.worst_index;
                std::printf("worst cell   : C[%zu][%zu] = %g, expected %g\n", i / N, i % N, hC[i], hRef[i]);
            }
        } else {
            std::printf("correct      : %s\n", pass_str);
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    if (dRef) CUDA_CHECK(cudaFree(dRef));
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
    return pass ? 0 : 1;
}
