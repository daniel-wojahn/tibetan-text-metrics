import pytest
from tibetan_text_metrics.metrics import (
    compute_lcs,
    compute_syntactic_distance,
    compute_weighted_jaccard,
    compute_normalized_syntactic_distance,
    compute_normalized_lcs,
    get_pos_weights,
)

def test_compute_lcs():
    words1 = ["a", "b", "c"]
    words2 = ["a", "d", "c"]
    pos_tags1 = ["n", "v", "n"]
    pos_tags2 = ["n", "v", "n"]
    result = compute_lcs(words1, pos_tags1, words2, pos_tags2)
    assert isinstance(result, int)
    assert result >= 0

def test_syntactic_distance():
    pos1 = ["n", "v", "n"]
    pos2 = ["n", "v", "adj"]
    distance = compute_syntactic_distance(pos1, pos2)
    assert isinstance(distance, float)
    assert distance >= 0

def test_weighted_jaccard():
    words1 = ["word1", "word2"]
    words2 = ["word1", "word3"]
    pos1 = ["n", "v"]
    pos2 = ["n", "v"]
    similarity = compute_weighted_jaccard(words1, pos1, words2, pos2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

def test_normalized_syntactic_distance():
    # Test normal case
    pos1 = ["n", "v", "n"]
    pos2 = ["n", "v", "adj"]
    distance = compute_normalized_syntactic_distance(pos1, pos2)
    assert isinstance(distance, float)
    assert 0 <= distance <= 1
    
    # Test empty lists
    empty_distance = compute_normalized_syntactic_distance([], [])
    assert empty_distance == 0.0
    
    # Test one empty list
    one_empty = compute_normalized_syntactic_distance(["n", "v"], [])
    assert one_empty == 1.0

def test_normalized_lcs():
    # Test normal case
    words1 = ["a", "b", "c"]
    words2 = ["a", "d", "c"]
    pos_tags1 = ["n", "v", "n"]
    pos_tags2 = ["n", "v", "n"]
    result = compute_normalized_lcs(words1, pos_tags1, words2, pos_tags2)
    assert isinstance(result, float)
    assert 0 <= result <= 1
    
    # Test with empty sequences
    empty_result = compute_normalized_lcs([], [], [], [])
    assert empty_result == 0.0
    
    # Test with identical sequences
    identical = compute_normalized_lcs(["a", "b"], ["n", "v"], ["a", "b"], ["n", "v"])
    assert identical == 1.0

def test_get_pos_weights():
    weights = get_pos_weights()
    assert isinstance(weights, dict)
    assert len(weights) > 0
    
    # Test a few specific weights
    assert weights.get("n.count") > weights.get("p.pers")
    assert weights.get("v.pres") > 0
