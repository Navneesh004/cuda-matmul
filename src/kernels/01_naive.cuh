// 01_naive.cuh - one thread per output cell, uncoalesced.
//
// Consecutive threadIdx.x values get consecutive *rows*. Rows sit N floats
// apart, so the 32 threads of a warp read 32 scattered addresses of A and write
// 32 scattered addresses of C: up to 32 memory transactions per warp access.
#pragma once
#include "../utils.cuh"

__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;   // which output row this thread owns
    const int col = blockIdx.y * blockDim.y + threadIdx.y;   // which output column
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

inline void launch_naive(const float* A, const float* B, float* C, int N, int /*cfg*/) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(N, 32), ceil_div(N, 32));
    matmul_naive<<<grid, block>>>(A, B, C, N);
}
