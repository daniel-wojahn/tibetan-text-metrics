import numpy as np

cimport cython
cimport numpy as np

# Use memory views for better performance
ctypedef np.int32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_lcs_fast(list words1, list words2):
    """
    Computes the Longest Common Subsequence (LCS) of two lists of words.

    This implementation is memory-optimized and uses O(min(m, n)) space, where
    m and n are the lengths of the word lists.

    Args:
        words1 (list): The first list of words.
        words2 (list): The second list of words.

    Returns:
        int: The length of the Longest Common Subsequence.
    """
    cdef int m = len(words1)
    cdef int n = len(words2)

    # Ensure words2 is the shorter sequence to optimize memory usage
    if m < n:
        return compute_lcs_fast(words2, words1)

    # We only need two rows for the DP table
    cdef np.ndarray[DTYPE_t, ndim=1] prev_row = np.zeros(n + 1, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] curr_row = np.zeros(n + 1, dtype=np.int32)
    
    # Use memory views for better access performance
    cdef DTYPE_t[:] prev_view = prev_row
    cdef DTYPE_t[:] curr_view = curr_row
    
    cdef int i, j
    cdef DTYPE_t val1, val2

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                curr_view[j] = prev_view[j - 1] + 1
            else:
                val1 = prev_view[j]
                val2 = curr_view[j - 1]
                curr_view[j] = val1 if val1 > val2 else val2
        
        # Swap views instead of copying for better performance
        prev_view, curr_view = curr_view, prev_view

    return <int>prev_view[n]
