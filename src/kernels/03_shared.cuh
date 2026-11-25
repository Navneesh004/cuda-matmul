// 03_shared.cuh - shared-memory tiling.
//
// The block cooperatively copies a TILE x TILE tile of A and of B into shared
// memory, then every thread computes from the tile. Each value fetched from
// global memory is reused TILE times, so arithmetic intensity goes from ~0.25
// to ~TILE/4 FLOP/byte.
//
// Out-of-range elements are loaded as zeros, so any N works (e.g. 1000, 1023).
#pragma once
#include "../utils.cuh"

template <int TILE>
__global__ void matmul_shared(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < N; t += TILE) {
        // Each thread loads one value of A's tile and one value of B's tile.
        As[ty][tx] = (row < N && t + tx < N) ? A[row * N + (t + tx)] : 0.0f;
        Bs[ty][tx] = (t + ty < N && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();   // wait until the whole tile is loaded

#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
        __syncthreads();   // wait before the tile is overwritten
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

inline void launch_shared(const float* A, const float* B, float* C, int N, int /*cfg*/) {
    constexpr int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid(ceil_div(N, TILE), ceil_div(N, TILE));
    matmul_shared<TILE><<<grid, block>>>(A, B, C, N);
}
