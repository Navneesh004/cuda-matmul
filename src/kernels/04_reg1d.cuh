// 04_reg1d.cuh - 1D register tiling: each thread computes a column of TM outputs.
//
// Block tile: BM x BN outputs, stepping through K in chunks of BK.
// Threads: (BM * BN) / TM, laid out so each thread owns TM consecutive rows of
// one column. In the inner loop the B value is read from shared memory once
// into a register and reused TM times.
//
//          BN                 thread (threadRow, threadCol) owns:
//   +---------------+         rows  threadRow*TM .. threadRow*TM + TM-1
//   |   |   |   |   | BM      col   threadCol
//   |   | x |   |   |
//   |   | x |   |   |         x = the TM accumulators held in registers
//   +---------------+
#pragma once
#include "../utils.cuh"

template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
matmul_reg1d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    constexpr int NT = (BM * BN) / TM;
    // One element of each tile per thread keeps the loaders simple.
    static_assert(BM * BK == NT && BK * BN == NT, "tile shape must give one load per thread");

    __shared__ float As[BM * BK];   // row-major BM x BK
    __shared__ float Bs[BK * BN];   // row-major BK x BN

    const int tid = threadIdx.x;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    const int threadCol = tid % BN;
    const int threadRow = tid / BN;

    const int aRow = tid / BK, aCol = tid % BK;   // which A element this thread loads
    const int bRow = tid / BN, bCol = tid % BN;   // which B element this thread loads

    float acc[TM] = {0.0f};

    for (int k0 = 0; k0 < N; k0 += BK) {
        const int ar = rowBase + aRow, ac = k0 + aCol;
        const int br = k0 + bRow,      bc = colBase + bCol;
        As[aRow * BK + aCol] = (ar < N && ac < N) ? A[ar * N + ac] : 0.0f;
        Bs[bRow * BN + bCol] = (br < N && bc < N) ? B[br * N + bc] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int d = 0; d < BK; ++d) {
            const float b = Bs[d * BN + threadCol];   // loaded once, used TM times
#pragma unroll
            for (int r = 0; r < TM; ++r) acc[r] += As[(threadRow * TM + r) * BK + d] * b;
        }
        __syncthreads();
    }

    const int col = colBase + threadCol;
#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const int row = rowBase + threadRow * TM + r;
        if (row < N && col < N) C[row * N + col] = acc[r];
    }
}

inline void launch_reg1d(const float* A, const float* B, float* C, int N, int /*cfg*/) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);
    dim3 grid(ceil_div(N, BN), ceil_div(N, BM));
    matmul_reg1d<BM, BN, BK, TM><<<grid, block>>>(A, B, C, N);
}
