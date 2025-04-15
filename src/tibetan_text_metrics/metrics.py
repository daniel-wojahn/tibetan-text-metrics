"""Functions for computing text similarity metrics."""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

try:
    from .fast_lcs import compute_lcs_fast

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


def compute_normalized_syntactic_distance(pos_seq1: List[str], pos_seq2: List[str]) -> float:
    """Compute normalized syntactic distance using Levenshtein distance.
    
    Normalized by the maximum length of the two sequences to create a value between 0 and 1.
    
    Args:
        pos_seq1: First list of POS tags
        pos_seq2: Second list of POS tags

    Returns:
        Normalized Levenshtein distance between tag sequences as a float between 0 and 1
    """
    if not pos_seq1 and not pos_seq2:
        return 0.0
        
    distance = float(Levenshtein.distance(tuple(pos_seq1), tuple(pos_seq2)))
    max_length = max(len(pos_seq1), len(pos_seq2))
    
    if max_length == 0:
        return 0.0
        
    return distance / max_length


def get_pos_weights() -> Dict[str, float]:
    """Get predefined weights for POS tags.

    Returns:
        Dictionary mapping POS tags to their weights
    """
    return {
        "n.count": 2.0,
        "n.prop": 2.0,
        "n.mass": 2.0,
        "n.rel": 2.0,
        "v.pres": 1.5,
        "v.fut": 1.5,
        "cv.fin": 1.5,
        "cv.sem": 1.5,
        "cv.impf": 1.5,
        "adj": 1.5,
        "num.ord": 1.0,
        "num.card": 1.0,
        "case.gen": 0.3,
        "case.all": 0.3,
        "case.agn": 0.3,
        "case.ass": 0.3,
        "case.term": 0.3,
        "case.fin": 0.3,
        "cl.top": 0.3,
        "cl.focus": 0.3,
        "cl.quot": 0.3,
        "cl.ques": 0.3,
        "p.pers": 0.2,
        "p.refl": 0.2,
        "p.indef": 0.2,
        "p.interrog": 0.2,
        "p.dem": 0.2,
        "d.quant": 0.2,
        "d.plural": 0.2,
        "d.dem": 0.2,
        "d.indef": 0.2,
        "adv.intense": 0.2,
    }


def compute_weighted_jaccard(
    words1: List[str], pos1: List[str], words2: List[str], pos2: List[str]
) -> float:
    """Compute weighted Jaccard similarity with POS-based weighting.

    Args:
        words1: Words from first text
        pos1: POS tags from first text
        words2: Words from second text
        pos2: POS tags from second text

    Returns:
        Weighted Jaccard similarity (0-1)
    """
    pos_weights = get_pos_weights()
    # Create word-POS pairs as tuples for faster set operations
    pairs1 = set(zip(words1, pos1))
    pairs2 = set(zip(words2, pos2))

    # Calculate intersection and union weights
    intersection_weight = sum(
        pos_weights.get(p, 1.0) for _, p in pairs1.intersection(pairs2)
    )
    union_weight = sum(pos_weights.get(p, 1.0) for _, p in pairs1.union(pairs2))

    return intersection_weight / union_weight if union_weight > 0 else 0.0


def compute_normalized_lcs(
    words1: List[str], pos_tags1: List[str], words2: List[str], pos_tags2: List[str]
) -> float:
    """Compute normalized longest common subsequence between two texts.
    
    Normalized by the average length of the two texts to account for chapter size.
    
    Args:
        words1: Words from first text
        pos_tags1: POS tags from first text
        words2: Words from second text
        pos_tags2: POS tags from second text
        
    Returns:
        float: Normalized LCS value between 0 and 1
    """
    if USE_CYTHON:
        lcs_length = compute_lcs_fast(words1, words2)
    else:
        # Fallback to pure Python implementation
        m, n = len(words1), len(words2)
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

        lcs_length = int(dp[m, n])
    
    avg_length = (len(words1) + len(words2)) / 2
    
    if avg_length == 0:
        return 0.0
        
    return lcs_length / avg_length
