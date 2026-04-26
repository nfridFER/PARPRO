import numpy as np
import time
from numba import vectorize, float64
from numba import cuda  # samo ako ima nvidia

print(cuda.is_available())


#slijedno (za usporedbu)
def sumsq_loop(A):
    N, M = A.shape
    s = 0.0

    for i in range(N):
        for j in range(M):
            s += A[i, j] * A[i, j]

    return s


# -------- VECTORIZED (CPU) --------
@vectorize([float64(float64)],
           target='parallel')
def square_cpu(x):
    return x * x


# -------- VECTORIZED (GPU) --------
@vectorize([float64(float64)],
           target='cuda')
def square_gpu(x):
    return x * x



N = 2048
A = np.random.rand(N, N).astype(np.float64)


# slijedno
t0 = time.perf_counter()
s1 = sumsq_loop(A)
t1 = time.perf_counter()


# -------- CPU VECTORIZE --------
# warm-up
tmp = square_cpu(A)

t2 = time.perf_counter()
tmp = square_cpu(A)
s2 = tmp.sum()
t3 = time.perf_counter()


# -------- GPU VECTORIZE --------
# warm-up
tmp = square_gpu(A)

t4 = time.perf_counter()
tmp = square_gpu(A)
s3 = tmp.sum()
t5 = time.perf_counter()



print(f"Slijedno rez: {s1}, vrijeme: {t1 - t0:.6f} s")
print(f"vectorize CPU rez: {s2}, vrijeme: {t3 - t2:.6f} s")
print(f"vectorize GPU rez: {s3}, vrijeme: {t5 - t4:.6f} s")

