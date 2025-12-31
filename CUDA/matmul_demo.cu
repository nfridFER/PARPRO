/*
 * MATMUL - GLOBAL vs. SCRATCHPAD DEMO
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define M 1024
#define BLOCKSIZE 32


__global__ void matmul_naive(const int* A, const int* B, int* C, int N) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x; //col
    int gy = blockIdx.y * blockDim.y + threadIdx.y; //row

    if (gx >= N || gy >= N) return;

    int sum = 0;
    for (int i = 0; i < N; i++) {
        int tempA = A[gy * N + i];
        int tempB = B[i * N + gx];
        sum += tempA * tempB;
    }
    C[gy * N + gx] = sum;
}

__global__ void matmul_tiled(const int* A, const int* B, int* C, int N)
{
    int lx = threadIdx.x;    // col in block
    int ly = threadIdx.y;    // row in block

    int col = blockIdx.x * blockDim.x + lx; 
    int row = blockIdx.y * blockDim.y + ly; 

    __shared__ int tA[BLOCKSIZE][BLOCKSIZE];
    __shared__ int tB[BLOCKSIZE][BLOCKSIZE];

    int sum = 0;
    int nTiles = (N + BLOCKSIZE - 1) / BLOCKSIZE;

    for (int tile = 0; tile < nTiles; tile++) {
        int tileAOffset = tile * BLOCKSIZE + lx;
        int tileBOffset = tile * BLOCKSIZE + ly;

        // guarded loads
        tA[ly][lx] = (row < N && tileAOffset < N)
            ? A[row * N + tileAOffset] : 0;
        tB[ly][lx] = (tileBOffset < N && col < N)
            ? B[tileBOffset * N + col] : 0;

        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k++)
            sum += tA[ly][k] * tB[k][lx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}


int main() {
    int n = M;
    size_t size = (size_t)n * (size_t)n * sizeof(int);

    int *a = (int*)malloc(size);
    int *b = (int*)malloc(size);
    int *c_naive = (int*)malloc(size);
    int *c_tiled = (int*)malloc(size);

    
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            a[i * n + j] = i * j;
            b[i * n + j] = i + j;
        }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 numBlocks((n + BLOCKSIZE - 1) / BLOCKSIZE,
                   (n + BLOCKSIZE - 1) / BLOCKSIZE);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int WARMUP  = 10;
    const int REPEATS = 50;

    cudaError_t err;

    // -------- warmup naive --------
    for (int i = 0; i < WARMUP; i++)
        matmul_naive<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // -------- time naive --------
    cudaEventRecord(start);
    for (int i = 0; i < REPEATS; i++)
        matmul_naive<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "matrix_mul launch failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaEventSynchronize(stop);

    float ms_naive_total = 0.0f;
    cudaEventElapsedTime(&ms_naive_total, start, stop);
    float ms_naive = ms_naive_total / REPEATS;

    cudaMemcpy(c_naive, d_c, size, cudaMemcpyDeviceToHost);
    std::cout << "Naive avg time: " << ms_naive << " ms\n";

    // -------- warmup tiled --------
    for (int i = 0; i < WARMUP; i++)
        matmul_tiled<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // -------- time tiled --------
    cudaEventRecord(start);
    for (int i = 0; i < REPEATS; i++)
        matmul_tiled<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "matrix_mul2 launch failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaEventSynchronize(stop);

    float ms_tiled_total = 0.0f;
    cudaEventElapsedTime(&ms_tiled_total, start, stop);
    float ms_tiled = ms_tiled_total / REPEATS;

    cudaMemcpy(c_tiled, d_c, size, cudaMemcpyDeviceToHost);
    std::cout << "Tiled avg time: " << ms_tiled << " ms\n";
    std::cout << "Speedup: " << (ms_naive / ms_tiled) << "x\n";

    // -------- correctness check --------
    bool ok = true;
    for (int i = 0; i < 20; i++) {
        int r = rand() % n;
        int c = rand() % n;
        if (c_naive[r * n + c] != c_tiled[r * n + c]) {
            ok = false;
            break;
        }
    }
    std::cout << "Naive vs tiled match (sampled): "
              << (ok ? "YES" : "NO") << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c_naive);
    free(c_tiled);

    return 0;
}
