import numpy as np
import time
from numba import guvectorize, float64
from numba import cuda  # samo ako ima nvidia


# slijedna impl za usporedbu
def sumsq_loop(A):
    N, M = A.shape
    s = 0.0

    for i in range(N):
        for j in range(M):
            s += A[i, j] * A[i, j]

    return s


# -------- GUVECTORIZE (CPU) --------
# (n)->() : jedan chunk/vektor -> jedan parcijalni zbroj
@guvectorize([(float64[:], float64[:])],
             '(n)->()',
             target='parallel')
def sumsq_chunk_cpu(x, out):
    s = 0.0

    for i in range(x.shape[0]):
        s += x[i] * x[i]

    out[0] = s


# -------- GUVECTORIZE (GPU) --------
@guvectorize([(float64[:], float64[:])],
             '(n)->()',
             target='cuda')
def sumsq_chunk_gpu(x, out):
    s = 0.0

    for i in range(x.shape[0]):
        s += x[i] * x[i]

    out[0] = s


N = 2048
A = np.random.rand(N, N).astype(np.float64)
# 2048 redaka  -> 2048 paralelnih poziva za guvect
# 2048 stupaca -> svaki poziv obrađuje jedan redak


# -------- LOOP --------
t0 = time.perf_counter()
s1 = sumsq_loop(A)
t1 = time.perf_counter()


# -------- CPU GUVECTORIZE --------
# warm-up
partial = sumsq_chunk_cpu(A)

t2 = time.perf_counter()
partial = sumsq_chunk_cpu(A)
s2 = partial.sum()
t3 = time.perf_counter()


# -------- GPU GUVECTORIZE --------
# warm-up
partial = sumsq_chunk_gpu(A)

t4 = time.perf_counter()
partial = sumsq_chunk_gpu(A)
s3 = partial.sum()
t5 = time.perf_counter()


print(f"Slijedno rez: {s1}, vrijeme: {t1 - t0:.6f} s")
print(f"guvectorize CPU rez:{s2}, vrijeme: {t3 - t2:.6f} s")
print(f"guvectorize GPU rez:{s3}, vrijeme: {t5 - t4:.6f} s")



#RESHAPED
num_chunks = 8192
chunk_size = 512
A_reshaped = A.reshape((num_chunks, chunk_size))
# 8192 redaka  -> 8192 paralelnih poziva guvect
# 512 stupaca -> svaki poziv obrađuje jedan redak


partial = sumsq_chunk_cpu(A_reshaped)

t2 = time.perf_counter()
partial = sumsq_chunk_cpu(A_reshaped)
s2 = partial.sum()
t3 = time.perf_counter()


partial = sumsq_chunk_gpu(A_reshaped)

t4 = time.perf_counter()
partial = sumsq_chunk_gpu(A_reshaped)
s3 = partial.sum()
t5 = time.perf_counter()


print(f"RESHAPED: {A_reshaped.shape} ")
print(f"guvectorize CPU rez {s2}, vrijeme: {t3 - t2:.6f} s")
print(f"guvectorize GPU rez {s3}, vrijeme: {t5 - t4:.6f} s")
