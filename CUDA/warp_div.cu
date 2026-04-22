#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// MACRO: force CUDA error report immediately (kernel launches are async and otherwise fail silently)
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":"            \
                      << __LINE__ << " -> " << cudaGetErrorString(err)  \
                      << std::endl;                                     \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)


// DEMO1
__global__ void warp_demo1(int* branch_taken, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;   // lane inside the warp

    int value = tid;

    if (lane < 16) {
        branch_taken[tid] = 0;

        value = value + 100;
        value = value * 2;
    }
    else {
        branch_taken[tid] = 1;

        value = value + 200;
        value = value * 3;
    }

    result[tid] = value;
}



//DEMO 2
__global__ void warp_demo2(int* branch_taken, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;   // lane inside the warp

    int value = tid;

    if (lane%2==0) {
        branch_taken[tid] = 0;

        value = value + 100;
        value = value * 2;
    }
    else {
        branch_taken[tid] = 1;

        value = value + 200;
        value = value * 3;
    }

    result[tid] = value;
}


//DEMO 3
__global__ void warp_demo3(int* branch_taken, int* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;   // lane inside the warp

    int value = tid;

    if (lane % 3 == 0) {
        branch_taken[tid] = 0;

        value = value + 100;
        value = value * 2;
    }
    else if (lane % 3 == 1) {
        branch_taken[tid] = 1;

        value = value + 200;
        value = value * 3;
    }
    else {
        branch_taken[tid] = 2;

        value = value + 300;
        value = value * 4;
    }

    result[tid] = value;
}



void print_results(const char* title,
    const std::vector<int>& h_branch,
    const std::vector<int>& h_result,
    int N) {
    std::cout << "\n=== " << title << " ===\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "thread " << i
            << " lane " << (i & 31)
            << " branch " << h_branch[i]
            << " rez " << h_result[i]
            << "\n";
    }
}


int main() {
    const int numBlocks = 1;
    const int numThreads = 32;   // exactly one warp
    const int N = numBlocks * numThreads;

    std::vector<int> h_branch(N);
    std::vector<int> h_result(N);

    int* d_branch = nullptr;
    int* d_result = nullptr;

    

    CUDA_CHECK(cudaMalloc(&d_branch, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(int)));

    /* DEMO 1, prvih 16 i drugih 16*/

    warp_demo1 << <numBlocks, numThreads >> > (d_branch, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_branch.data(), d_branch, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, N * sizeof(int), cudaMemcpyDeviceToHost));

    print_results("DEMO 1: prvih 16 vs drugih 16", h_branch, h_result, N);


    /* DEMO 2, parni i neparni*/

    warp_demo2 << <numBlocks, numThreads >> > (d_branch, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_branch.data(), d_branch, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, N * sizeof(int), cudaMemcpyDeviceToHost));

    print_results("DEMO 2: parni i neparni", h_branch, h_result, N);


    /* DEMO 3, svaki treći*/

    warp_demo3 << <numBlocks, numThreads >> > (d_branch, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_branch.data(), d_branch, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, N * sizeof(int), cudaMemcpyDeviceToHost));

    print_results("DEMO 3: svaki treci", h_branch, h_result, N);

    CUDA_CHECK(cudaFree(d_branch));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}