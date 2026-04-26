#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <chrono>


__global__ void sumsq_cuda(const double* A, double* partial, int rows, int cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int total = rows * cols;

    double s = 0.0;

    for (int idx = tid; idx < total; idx += stride)
    {
        int row = idx / cols;
        int col = idx % cols;

        double x = A[row * cols + col];
        s += x * x;
    }

   
    partial[tid] = s;
}


// slijedna impl za usporedbu
double sumsq_cpu(const double* A, int rows, int cols)
{
    double s = 0.0;

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            s += A[i * cols + j] * A[i * cols + j];

    return s;
}



int main()
{
    int N = 2048;
    int rows = N;
    int cols = N;
    int total = rows * cols;

    size_t bytes = total * sizeof(double);

    double* A = (double*)malloc(bytes);

    // init
    for (int i = 0; i < total; i++)
        A[i] = (double)rand() / RAND_MAX;

    // CPU
    clock_t t0 = clock();
    double s_cpu = sumsq_cpu(A, rows, cols);
    clock_t t1 = clock();


    // CUDA
    int threads_per_block = 256;
    int blocks_per_grid = 32;
    int num_threads = threads_per_block * blocks_per_grid;

    double* d_A;
    double* d_partial;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_partial, num_threads * sizeof(double));

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);

    // warm-up
    sumsq_cuda << <blocks_per_grid, threads_per_block >> > (d_A, d_partial, rows, cols);
    cudaDeviceSynchronize();

    //mjerenje
    auto t2 = std::chrono::high_resolution_clock::now();
    sumsq_cuda << <blocks_per_grid, threads_per_block >> > (d_A, d_partial, rows, cols);
    cudaDeviceSynchronize();

    double* partial = (double*)malloc(num_threads * sizeof(double));
    cudaMemcpy(partial, d_partial, num_threads * sizeof(double), cudaMemcpyDeviceToHost);

    double s_gpu = 0.0;
    for (int i = 0; i < num_threads; i++)
        s_gpu += partial[i];    //konačni zbroj parcijalnih suma na CPU

    auto t3 = std::chrono::high_resolution_clock::now();

    double gpu_time =
        std::chrono::duration<double>(t3 - t2).count();

    
    printf("CPU rez: %f, vrijeme:  %.8f s\n",s_cpu, (double)(t1 - t0) / CLOCKS_PER_SEC);
    printf("GPU rez: %f, vrijeme:   %.8f s\n",s_gpu, gpu_time);
    

    cudaFree(d_A);
    cudaFree(d_partial);
    free(A);
    free(partial);

    return 0;
}