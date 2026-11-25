// 07_tuned.cuh - kernel 6 with bank-conflict padding and a choice of tile sizes.
//
// Pick a configuration with --cfg <i>; scripts/tune.sh sweeps all of them.
// cfg 0 has exactly kernel 6's tile shape, so kernel 6 vs kernel 7 --cfg 0
// isolates the effect of the padding alone.
#pragma once
#include "06_vectorized.cuh"

struct TunedConfig { int BM, BN, BK, TM, TN; };

// Keep this table in sync with the switch below.
static const TunedConfig TUNED_CONFIGS[] = {
    {128, 128,  8, 8, 8},   // 0: 256 threads (kernel 6 shape)
    {128, 128, 16, 8, 8},   // 1: 256 threads, deeper K step
    {128,  64,  8, 8, 8},   // 2: 128 threads
    { 64, 128,  8, 8, 8},   // 3: 128 threads
    { 64,  64,  8, 8, 8},   // 4:  64 threads, more blocks for small N
    { 64,  64, 16, 4, 4},   // 5: 256 threads, smaller per-thread tile
    {128, 128, 32, 8, 8},   // 6: 256 threads, 33 KB shared memory
};
constexpr int NUM_TUNED_CONFIGS = sizeof(TUNED_CONFIGS) / sizeof(TUNED_CONFIGS[0]);
constexpr int DEFAULT_TUNED_CONFIG = 0;

inline void launch_tuned(const float* A, const float* B, float* C, int N, int cfg) {
    constexpr int P = 4;   // padding that removes the transposed-store bank conflict
    switch (cfg) {
        case 0: launch_vec_cfg<128, 128,  8, 8, 8, P>(A, B, C, N); break;
        case 1: launch_vec_cfg<128, 128, 16, 8, 8, P>(A, B, C, N); break;
        case 2: launch_vec_cfg<128,  64,  8, 8, 8, P>(A, B, C, N); break;
        case 3: launch_vec_cfg< 64, 128,  8, 8, 8, P>(A, B, C, N); break;
        case 4: launch_vec_cfg< 64,  64,  8, 8, 8, P>(A, B, C, N); break;
        case 5: launch_vec_cfg< 64,  64, 16, 4, 4, P>(A, B, C, N); break;
        case 6: launch_vec_cfg<128, 128, 32, 8, 8, P>(A, B, C, N); break;
        default:
            std::fprintf(stderr, "unknown --cfg %d (valid: 0..%d)\n", cfg, NUM_TUNED_CONFIGS - 1);
            std::exit(2);
    }
}
