# fast_patterns.pyx
# cython: language_level=3
# distutils: language=c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libc.math cimport sqrt

ctypedef vector[string] string_vec
ctypedef unordered_map[string, double] str_double_map

cdef class FastPatternAnalyzer:
    cdef str_double_map patterns1, patterns2

    def __cinit__(self):
        pass

    def __init__(self):
        self.patterns1.clear()
        self.patterns2.clear()

    cpdef dict extract_ngrams(self, list tokens, int n):
        cdef dict patterns = {}
        cdef int i
        cdef str ngram
        
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            if ngram in patterns:
                patterns[ngram] += 1
            else:
                patterns[ngram] = 1
        return patterns

    cpdef double compute_cosine_similarity(self, dict patterns1, dict patterns2) except -1:
        cdef double norm1 = 0.0
        cdef double norm2 = 0.0
        cdef double dot_product = 0.0
        cdef str pattern
        cdef double freq1, freq2

        self.patterns1.clear()
        self.patterns2.clear()

        for pattern, freq in patterns1.items():
            self.patterns1[pattern.encode('utf-8')] = freq
            norm1 += freq * freq

        for pattern, freq in patterns2.items():
            self.patterns2[pattern.encode('utf-8')] = freq
            norm2 += freq * freq

        for pattern, freq1 in patterns1.items():
            pattern_bytes = pattern.encode('utf-8')
            if self.patterns2.find(pattern_bytes) != self.patterns2.end():
                freq2 = self.patterns2[pattern_bytes]
                dot_product += freq1 * freq2

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (sqrt(norm1) * sqrt(norm2))

    def analyze_chapter_pair(self, words1, pos1, words2, pos2, n_gram_size=3):
        """Instance method to analyze a pair of chapters.
        
        Args:
            words1: List of words from first text
            pos1: List of POS tags from first text
            words2: List of words from second text
            pos2: List of POS tags from second text
            n_gram_size: Size of n-grams to use for pattern analysis (default: 3)
            
        Returns:
            Dict containing similarity scores and pattern counts
        """
        word_patterns1 = self.extract_ngrams(words1, n_gram_size)
        word_patterns2 = self.extract_ngrams(words2, n_gram_size)
        pos_patterns1 = self.extract_ngrams(pos1, n_gram_size)
        pos_patterns2 = self.extract_ngrams(pos2, n_gram_size)

        word_sim = self.compute_cosine_similarity(word_patterns1, word_patterns2)
        pos_sim = self.compute_cosine_similarity(pos_patterns1, pos_patterns2)

        return {
            "POS Pattern Similarity": pos_sim,
            "Word Pattern Similarity": word_sim,
            "Unique POS Patterns Text 1": len(pos_patterns1),
            "Unique POS Patterns Text 2": len(pos_patterns2),
            "Unique Word Patterns Text 1": len(word_patterns1),
            "Unique Word Patterns Text 2": len(word_patterns2)
        }
