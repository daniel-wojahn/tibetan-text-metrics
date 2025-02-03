# fast_lcs.pyx
import numpy as np

cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_lcs_fast(list words1, list words2):
    cdef int m = len(words1)
    cdef int n = len(words2)
    cdef np.ndarray[np.int32_t, ndim=2] dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    cdef int i, j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    
    return int(dp[m, n])