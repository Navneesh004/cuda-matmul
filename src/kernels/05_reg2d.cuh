// 05_reg2d.cuh - 2D register tiling: each thread computes a TM x TN square.
//
// Per inner step d, a thread loads TM values of A's column d and TN values of
// B's row d into registers, then does the TM x TN outer product. That is
// TM + TN shared-memory loads for TM * TN fused multiply-adds (16 loads for
// 64 FMAs with 8 x 8), versus 1 load per FMA in the shared-memory kernel.
#pragma once
#include "../utils.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmul_reg2d(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    constexpr int NT = (BM / TM) * (BN / TN);

    __shared__ float As[BM * BK];   // row-major BM x BK
    __shared__ float Bs[BK * BN];   // row-major BK x BN

    const int tid = threadIdx.x;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);

    float acc[TM][TN] = {{0.0f}};
    float regM[TM];
    float regN[TN];

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Each thread loads several elements of each tile (strided by NT).
        for (int i = tid; i < BM * BK; i += NT) {
            const int r = rowBase + i / BK, c = k0 + i % BK;
            As[i] = (r < N && c < N) ? A[r * N + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += NT) {
            const int r = k0 + i / BN, c = colBase + i % BN;
            Bs[i] = (r < N && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int d = 0; d < BK; ++d) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + d];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = rowBase + threadRow * TM + i;
        if (row >= N) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = colBase + threadCol * TN + j;
            if (col < N) C[row * N + col] = acc[i][j];
        }
    }
}

inline void launch_reg2d(const float* A, const float* B, float* C, int N, int /*cfg*/) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM / TM) * (BN / TN));
    dim3 grid(ceil_div(N, BN), ceil_div(N, BM));
    matmul_reg2d<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, N);
}
