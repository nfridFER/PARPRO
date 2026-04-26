import numpy as np
import time
from numba import cuda


# slijedna impl za usporedbu
def sumsq_loop(A):
    N, M = A.shape
    s = 0.0

    for i in range(N):
        for j in range(M):
            s += A[i, j] * A[i, j]

    return s



@cuda.jit
def sumsq_cuda(A, partial):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    rows = A.shape[0]
    cols = A.shape[1]
    total = rows * cols

    s = 0.0


    for idx in range(tid, total, stride):
        row = idx // cols
        col = idx % cols

        x = A[row, col]
        s += x * x

    partial[tid] = s   # parcijalna suma



N = 2048
A = np.random.rand(N, N).astype(np.float64)


# slijedno
t0 = time.perf_counter()
s1 = sumsq_loop(A)
t1 = time.perf_counter()


# CUDA
threads_per_block = 256
blocks_per_grid = 32
num_threads = threads_per_block * blocks_per_grid

partial = np.zeros(num_threads, dtype=np.float64)

# warm-up / compilation
sumsq_cuda[blocks_per_grid, threads_per_block](A, partial)
cuda.synchronize()

# mjerenje
t2 = time.perf_counter()
sumsq_cuda[blocks_per_grid, threads_per_block](A, partial)
cuda.synchronize()
s2 = partial.sum()
t3 = time.perf_counter()


print(f"Slijedno rez: {s1}, vrijeme: {t1 - t0:.6f} s")
print(f"cuda.jit rez: {s2}, vrijeme: {t3 - t2:.6f} s")
