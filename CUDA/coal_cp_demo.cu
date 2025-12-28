/*
 *   MEMORY COALESCING DEMO

 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>

#define W 1024
#define H 1024
#define N ((size_t)W * (size_t)H)

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void init_kernel(float* A) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) {
        A[i] = (float)(i%1024);
    }
}

// Coalesced: row-major
__global__ void copy_row_traversal(const float* __restrict__ A, float* __restrict__ B) {
    size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t gstride = (size_t)gridDim.x * blockDim.x;

    for (size_t t = gtid; t < N; t += gstride) {
        size_t y = t / W;
        size_t x = t - y * W;     // == t % W, modulo is "expensive"
        size_t idx = y * W + x;   // == t
        B[idx] = A[idx];
    }
}

// Non-coalesced: column-major
__global__ void copy_col_traversal(const float* __restrict__ A, float* __restrict__ B) {
    size_t gtid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t gstride = (size_t)gridDim.x * blockDim.x;

    for (size_t t = gtid; t < N; t += gstride) {
        size_t x = t / H;
        size_t y = t - x * H;     
        size_t idx = y * W + x;   
        B[idx] = A[idx];
    }
}

static float time_kernel_row(const float* A, float* B, dim3 grid, dim3 block, int iters) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop),  "event create stop");

    // warmup
    copy_row_traversal<<<grid, block>>>(A, B);
    check(cudaGetLastError(), "row kernel launch");
    check(cudaDeviceSynchronize(), "row warmup sync");

    check(cudaEventRecord(start), "row record start");
    for (int i = 0; i < iters; i++) {
        copy_row_traversal<<<grid, block>>>(A, B);
    }
    check(cudaEventRecord(stop), "row record stop");
    check(cudaEventSynchronize(stop), "row sync stop");

    float ms = 0.f;
    check(cudaEventElapsedTime(&ms, start, stop), "row elapsed");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static float time_kernel_col(const float* A, float* B, dim3 grid, dim3 block, int iters) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop),  "event create stop");

    // warmup
    copy_col_traversal<<<grid, block>>>(A, B);
    check(cudaGetLastError(), "col kernel launch");
    check(cudaDeviceSynchronize(), "col warmup sync");

    check(cudaEventRecord(start), "col record start");
    for (int i = 0; i < iters; i++) {
        copy_col_traversal<<<grid, block>>>(A, B);
    }
    check(cudaEventRecord(stop), "col record stop");
    check(cudaEventSynchronize(stop), "col sync stop");

    float ms = 0.f;
    check(cudaEventElapsedTime(&ms, start, stop), "col elapsed");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    int device = 0;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int maxWarpsPerSM =
    prop.maxThreadsPerMultiProcessor / prop.warpSize;

    printf("Device name: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max active warps per SM: %d\n", maxWarpsPerSM);
    printf("=====================\n");



    printf("Matrix: %dx%d \n", W, H);

    float *A, *B;
    check(cudaMalloc(&A, N * sizeof(float)), "malloc A");
    check(cudaMalloc(&B, N * sizeof(float)), "malloc B");

    int block = 64;
    int grid  = 16384; 
    
    dim3 dblock(block);
    dim3 dgrid(grid);

    printf("Block: %d, grid:%d \n", block, grid);

    // init A
    init_kernel<<<dgrid, dblock>>>(A);
    check(cudaGetLastError(), "init kernel launch");
    check(cudaDeviceSynchronize(), "init sync");

    // clear B 
    check(cudaMemset(B, 0, N * sizeof(float)), "memset B");

    int iters = 50;

    float t_row = time_kernel_row(A, B, dgrid, dblock, iters);
    float t_col = time_kernel_col(A, B, dgrid, dblock, iters);

    printf("Copy coalesced:    %.2f ms \n", t_row);
    printf("Copy non-coalesced: %.2f ms  \n", t_col);
    printf("Speedup: %.2fx\n", t_col / t_row);

    cudaFree(A);
    cudaFree(B);
    return 0;
}
