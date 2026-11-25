// utils.cuh - error-checking macros, random fill, result comparison.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

// Every CUDA runtime call goes through this. GPU errors are silent otherwise:
// a failed launch just leaves C full of garbage.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         cudaGetErrorName(err_), __FILE__, __LINE__,            \
                         cudaGetErrorString(err_));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_,        \
                         __FILE__, __LINE__);                                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Call right after every kernel launch: catches bad launch configurations
// (too many threads, too much shared memory, ...) that would otherwise be silent.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

__host__ __device__ constexpr inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Uniform random values in [-1, 1], reproducible from a seed.
inline void fill_random(float* x, size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) x[i] = dist(gen);
}

struct CompareResult {
    double max_abs_err = 0.0;   // max |c - ref|
    double max_abs_ref = 0.0;   // max |ref|
    double rel_err = 0.0;       // max_abs_err / max_abs_ref
    size_t worst_index = 0;     // where max_abs_err happened
    size_t nan_count = 0;       // non-finite outputs (e.g. never written)
    bool pass = false;
};

// Normalised max error: max|c - ref| / max|ref|.
// Summing in a different order changes the last bits, so an exact match would
// fail even for correct code. Real bugs give errors of order 1 (or NaN, because
// the harness pre-fills C with NaN, so unwritten cells are caught).
inline CompareResult compare(const float* c, const float* ref, size_t n, double tol) {
    CompareResult r;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(c[i])) { ++r.nan_count; continue; }
        double e = std::fabs((double)c[i] - (double)ref[i]);
        if (e > r.max_abs_err) { r.max_abs_err = e; r.worst_index = i; }
        r.max_abs_ref = std::max(r.max_abs_ref, std::fabs((double)ref[i]));
    }
    r.rel_err = r.max_abs_err / std::max(r.max_abs_ref, 1e-30);
    r.pass = (r.nan_count == 0) && (r.rel_err < tol);
    return r;
}
