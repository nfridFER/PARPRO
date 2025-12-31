/*
 *  CUDA ATOMIC DEMO
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

__global__ void vectSumRace(int* d_vect, size_t size, int* result) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < size) {
        *result += d_vect[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void vectSumAtomic(int* d_vect, size_t size, int* result) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < size) {
        atomicAdd(result, d_vect[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    const size_t N = 1 << 20;
    int* vect = new int[N];

    int result = 0, result2 = 0;

    const int threads = 256;
    const int blocks  = 256;

    const int iters = 500;

    for (int i = 0; i < N; i++) vect[i] = 1;

    int* dev_vect = nullptr;
    int* dev_res = nullptr;
    int* dev_res2 = nullptr;

    cudaSetDevice(0);

    cudaMalloc(&dev_vect, N * sizeof(int));
    cudaMalloc(&dev_res, sizeof(int));
    cudaMalloc(&dev_res2, sizeof(int));

    cudaMemcpy(dev_vect, vect, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res, &result, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res2, &result2, sizeof(int), cudaMemcpyHostToDevice);

   
    cudaEvent_t start, stop;
    float timeRace = 0.0f, timeAtomic = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    vectSumAtomic<<<blocks, threads>>>(dev_vect, N, dev_res2);
    cudaDeviceSynchronize();

    // --- Race kernel ---
    cudaMemset(dev_res, 0, sizeof(int));
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cudaMemset(dev_res, 0, sizeof(int));
        vectSumRace<<<blocks, threads>>>(dev_vect, N, dev_res);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeRace, start, stop);
    timeRace /= iters;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "vectSumRace launch failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // --- Atomic kernel ---
    cudaMemset(dev_res2, 0, sizeof(int));
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cudaMemset(dev_res2, 0, sizeof(int));
        vectSumAtomic<<<blocks, threads>>>(dev_vect, N, dev_res2);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeAtomic, start, stop);
    timeAtomic /= iters;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "vectSumAtomic launch failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    
    cudaDeviceSynchronize();

    cudaMemcpy(&result, dev_res, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result2, dev_res2, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum-race:   " << result << "\n";
    std::cout << "Time-race:  " << timeRace << " ms\n\n";

    std::cout << "Sum-atomic: " << result2 << "\n";
    std::cout << "Time-atomic:" << timeAtomic << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_vect);
    cudaFree(dev_res);
    cudaFree(dev_res2);

    return 0;
}
