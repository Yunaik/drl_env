import numpy as np
from numba import jit, njit, int32, prange
import time


# d_in = np.random.randn(5000, 1000)

def bench():
    d_out  = np.empty([50000, 1000], dtype = np.float32)
    idy = np.arange(1000)
    for i in range(1000):
        idx = np.random.choice(50000,1000, replace = False)
        d_out[idx, idy] = i


@njit(parallel = True, nogil = True)
def bench_1(obs):
    idx = np.random.choice(3000,512, replace = False)
    obs[idx] = np.random.randn(512, 21)



start = time.time()
obs = np.zeros((3000, 512, 21), dtype=np.float_)
bench_1(obs)
end = time.time()