// 02_coalesced.cuh - same as naive, with row/col swapped.
//
// Now consecutive threadIdx.x values get consecutive *columns*: a warp reads 32
// neighbouring floats of B and writes 32 neighbouring floats of C (one 128-byte
// trip each), and every thread in the warp reads the *same* A element
// (a broadcast). Only the index mapping changed.
#pragma once
#include "../utils.cuh"

__global__ void matmul_coalesced(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

inline void launch_coalesced(const float* A, const float* B, float* C, int N, int /*cfg*/) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(N, 32), ceil_div(N, 32));
    matmul_coalesced<<<grid, block>>>(A, B, C, N);
}
