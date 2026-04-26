import numpy as np
import time
from numba import vectorize, float64

#samo ako postoji nvidia GPU, također treba instalirati paket numba-cuda
from numba import cuda

print(cuda.is_available()) # provjera da CUDA radi

#slijedno zbrajanje (za usporedbu)
def add_loop(A, B):
    N, M = A.shape
    C = np.empty_like(A)

    for i in range(N):
        for j in range(M):
            C[i, j] = A[i, j] + B[i, j]

    return C


# paralelizirano zbrajanje - multicore CPU
@vectorize([float64(float64, float64)],
           target='parallel')
def add_vec_cpu(a, b):
    return a + b


# paralelizirano zbrajanje - GPU
@vectorize([float64(float64, float64)],
           target='cuda')
def add_vec_gpu(a, b):
    return a + b




N = 2048
A = np.random.rand(N, N).astype(np.float64)
B = np.random.rand(N, N).astype(np.float64)


#slijedno
t0 = time.perf_counter()
C1 = add_loop(A, B)
t1 = time.perf_counter()


#vectorize - target='paralel' ==> izvodenje na CPU
#prvo zagrijavanje (numba kompajliranje) pa onda mjerenje
C = add_vec_cpu(A,B)

t2 = time.perf_counter()
C2 = add_vec_cpu(A, B)
t3 = time.perf_counter()


#vectorize - target='cuda' ==> izvodenje na GPU
#prvo zagrijavanje (numba kompajliranje) pa onda mjerenje
C = add_vec_gpu(A,B)

t4 = time.perf_counter()
C3 = add_vec_gpu(A, B)
t5 = time.perf_counter()


print(f"loop: {t1 - t0:.6f} s")
print(f"vectorize  CPU: {t3 - t2:.6f} s")
print(f"vectorize GPU: {t5 - t4:.6f} s")
print(f"provjera loop, CPU - isti rez: {np.allclose(C1, C2)}")
print(f"provjera loop, GPU - isti rez: {np.allclose(C1, C3)}")