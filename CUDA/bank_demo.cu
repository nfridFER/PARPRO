/*
 * BANKING CONFLICTS DEMO
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#define ITR 4096


// helper func: prevents overoptimization of variables that seem unused
__device__ __forceinline__ void sink(float x) { asm volatile("" :: "f"(x)); }



// --------------------------- Demo 1: 1D stride -------------------------
__global__ void shmem_stride_1d(float* out, int stride) {
  constexpr int SIZE = 1024;        // must be >= 32*stride 

  __shared__ volatile int indices[SIZE];    // shared MEM
                                            // volatile forces the load

  int tid  = threadIdx.x;    // thread index in block
  int offset = tid & 31;     // start position based on warp (0–31)
                      

 for (int i = tid; i < SIZE; i += blockDim.x) {
    indices[i] = (i + 1) & (SIZE - 1);    //cycle 0-1023
  }
  
  __syncthreads();  //ensures all writes are done before any thread starts reading

  
  int idx = (offset * stride) & (SIZE - 1); // start index
                                            // determines access pattern (source of conflict)

  int acc = 0;
  #pragma unroll 1
  for (int it = 0; it < ITR; ++it) {
    idx = indices[idx];         // next index  
    acc += idx;
    sink((float) acc);  // helper func
  }

  out[blockIdx.x * blockDim.x + tid] = (float)acc;
}



// -------------------------- Demo 2: 2D row-wise read ------------------------------
__global__ void shmem_2d_rowwise(float* out) {
  __shared__ float tile[32][32];

  int tid  = threadIdx.x;
  int offset = tid & 31;   // 0..31
  int warp = tid >> 5;   // 32 threads per warp (default)

  tile[warp][offset] = (float)(warp * 32 + offset);
  __syncthreads();

  float acc = 0.0f;
  #pragma unroll 1
  for (int it = 0; it < ITR; ++it) {  
    int col = (offset + it) & 31;   
    acc += tile[warp][col];       
    sink(acc);
  }

  out[blockIdx.x * blockDim.x + tid] = acc;
}





// ---------------------- Demo 3: column-wise read bad vs good ------------------------
__global__ void shmem_column_bad(float* out) {
  __shared__ float tile[32][32];

  int tid  = threadIdx.x;
  int offset = tid & 31;
  int warp = tid >> 5;

  tile[warp][offset] = (float)(warp * 32 + offset);
  __syncthreads();

  float acc = 0.0f;
  #pragma unroll 1
  for (int it = 0; it < ITR; ++it) {
    int row = (offset + it) & 31;     
    acc += tile[row][warp];         // threads hit same column - bank conflict
    sink(acc);
  }

  out[blockIdx.x * blockDim.x + tid] = acc;
}


__global__ void shmem_column_good(float* out) {
  __shared__ float tile[32][33];    // +1 padding 

  int tid  = threadIdx.x;
  int offset = tid & 31;
  int warp = tid >> 5;

  tile[warp][offset] = (float)(warp * 32 + offset);
  __syncthreads();

  float acc = 0.0f;
  #pragma unroll 1
  for (int it = 0; it < ITR; ++it) {
    int row = (offset + it) & 31;
    acc += tile[row][warp];         // pitch=33 => banks spread out => conflicts largely gone
    sink(acc);
  }

  out[blockIdx.x * blockDim.x + tid] = acc;
}


// ------------------------------ Timing helper --------------------------------

template<typename LaunchFn>
float time_kernel_ms(LaunchFn launch, int warmup, int reps) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // warmup
  for (int i = 0; i < warmup; ++i) launch();
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for (int i = 0; i < reps; ++i) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return ms / reps;
}



int main() {
  cudaSetDevice(0);

  const int block_size = 256;    
  const int grid_size  = 128;   
  dim3 block(block_size);
  dim3 grid(grid_size);

  float* d_out = nullptr;
  cudaMalloc(&d_out, block_size * grid_size * sizeof(float));

  const int warmup = 5;
  const int reps   = 30;


  std::printf("[Demo 1] 1D shared-memory stride \n");
  std::printf("Stride\tTime (ms)\n");

  std::vector<int> strides = {1, 2, 4, 8, 16, 32};
  for (int s : strides) {
    float ms = time_kernel_ms([&](){
      shmem_stride_1d<<<grid, block>>>(d_out, s);
    }, warmup, reps);

    std::printf("%d\t%8.4f\n", s, ms);
  }
  

  std::printf("[Demo 2] 2D tile row-wise read\n");
  float ms_row = time_kernel_ms([&](){
    shmem_2d_rowwise<<<grid, block>>>(d_out);
  }, warmup, reps);

  std::printf("Time (ms): %8.4f\n\n", ms_row);

  

  std::printf("[Demo 3] 2D column-wise read: bad (32x32) vs good (32x33)\n");
  float ms_bad = time_kernel_ms([&](){
    shmem_column_bad<<<grid, block>>>(d_out);
  }, warmup, reps);


  float ms_good = time_kernel_ms([&](){
    shmem_column_good<<<grid, block>>>(d_out);
  }, warmup, reps);


  std::printf("Bad  tile[32][32] : %8.4f ms\n", ms_bad);
  std::printf("Good tile[32][33] : %8.4f ms\n", ms_good);
  std::printf("\n");

  cudaFree(d_out);
  cudaDeviceSynchronize();
  return 0;
}
