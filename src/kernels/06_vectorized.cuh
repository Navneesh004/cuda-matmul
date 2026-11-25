// 06_vectorized.cuh - 2D register tiling + float4 loads/stores (+ optional padding).
//
// Changes vs 05_reg2d:
//  * Global loads/stores use float4: one instruction moves 16 bytes.
//  * A's tile is stored TRANSPOSED in shared memory (As[k][m]), so the TM values
//    a thread needs for step k are contiguous and can be read as float4 too.
//  * PAD extra floats per As row. With PAD = 0 the transposed store gives a 2-way
//    shared-memory bank conflict (two threads, same bank, different address);
//    PAD = 4 shifts every other row by 16 banks and removes it (kernel 7).
//
// float4 global access needs 16-byte alignment, i.e. N % 4 == 0. For other N the
// loads fall back to guarded scalar accesses, so odd sizes still work.
#pragma once
#include "../utils.cuh"

// Load 4 consecutive floats M[r][c..c+3] (row-major, N x N), zero-padding the edges.
__device__ __forceinline__ float4 load4(const float* __restrict__ M, int r, int c, int N, bool vec_ok) {
    if (vec_ok && r < N && c + 3 < N) return *reinterpret_cast<const float4*>(&M[r * N + c]);
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (r < N) {
        const float* p = &M[r * N];
        if (c + 0 < N) v.x = p[c + 0];
        if (c + 1 < N) v.y = p[c + 1];
        if (c + 2 < N) v.z = p[c + 2];
        if (c + 3 < N) v.w = p[c + 3];
    }
    return v;
}

template <int BM, int BN, int BK, int TM, int TN, int PAD>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
matmul_vec(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int LDA = BM + PAD;   // leading dimension of the transposed A tile
    static_assert(BK % 4 == 0 && BN % 4 == 0 && TM % 4 == 0 && TN % 4 == 0, "need multiples of 4 for float4");
    static_assert(LDA % 4 == 0, "padding must keep float4 alignment");

    __shared__ __align__(16) float As[BK * LDA];   // As[k * LDA + m]  (transposed)
    __shared__ __align__(16) float Bs[BK * BN];    // Bs[k * BN  + n]

    const int tid = threadIdx.x;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;
    const int threadCol = tid % (BN / TN);
    const int threadRow = tid / (BN / TN);
    const bool vec_ok = (N % 4) == 0;   // uniform across the grid: no divergence

    float acc[TM][TN] = {{0.0f}};
    float regM[TM];
    float regN[TN];

    for (int k0 = 0; k0 < N; k0 += BK) {
        // A tile: BM x BK, read as float4 along k, written transposed.
        for (int i = tid; i < BM * BK / 4; i += NT) {
            const int r = i / (BK / 4);
            const int c4 = (i % (BK / 4)) * 4;
            const float4 v = load4(A, rowBase + r, k0 + c4, N, vec_ok);
            As[(c4 + 0) * LDA + r] = v.x;
            As[(c4 + 1) * LDA + r] = v.y;
            As[(c4 + 2) * LDA + r] = v.z;
            As[(c4 + 3) * LDA + r] = v.w;
        }
        // B tile: BK x BN, float4 in, float4 out.
        for (int i = tid; i < BK * BN / 4; i += NT) {
            const int r = i / (BN / 4);
            const int c4 = (i % (BN / 4)) * 4;
            *reinterpret_cast<float4*>(&Bs[r * BN + c4]) = load4(B, k0 + r, colBase + c4, N, vec_ok);
        }
        __syncthreads();

#pragma unroll
        for (int d = 0; d < BK; ++d) {
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&As[d * LDA + threadRow * TM + i]);
                regM[i] = v.x; regM[i + 1] = v.y; regM[i + 2] = v.z; regM[i + 3] = v.w;
            }
#pragma unroll
            for (int j = 0; j < TN; j += 4) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[d * BN + threadCol * TN + j]);
                regN[j] = v.x; regN[j + 1] = v.y; regN[j + 2] = v.z; regN[j + 3] = v.w;
            }
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
        for (int j = 0; j < TN; j += 4) {
            const int col = colBase + threadCol * TN + j;
            float* dst = &C[row * N + col];
            if (vec_ok && col + 3 < N) {
                *reinterpret_cast<float4*>(dst) = make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
            } else {
                if (col + 0 < N) dst[0] = acc[i][j + 0];
                if (col + 1 < N) dst[1] = acc[i][j + 1];
                if (col + 2 < N) dst[2] = acc[i][j + 2];
                if (col + 3 < N) dst[3] = acc[i][j + 3];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN, int PAD>
inline void launch_vec_cfg(const float* A, const float* B, float* C, int N) {
    dim3 block((BM / TM) * (BN / TN));
    dim3 grid(ceil_div(N, BN), ceil_div(N, BM));
    matmul_vec<BM, BN, BK, TM, TN, PAD><<<grid, block>>>(A, B, C, N);
}

// Kernel 6: vectorised, no padding (same tile shape as kernel 5).
inline void launch_vectorized(const float* A, const float* B, float* C, int N, int /*cfg*/) {
    launch_vec_cfg<128, 128, 8, 8, 8, 0>(A, B, C, N);
}
