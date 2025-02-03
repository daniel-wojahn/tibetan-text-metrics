import pytest
from tibetan_text_metrics.metrics import (
    compute_lcs,
    compute_syntactic_distance,
    compute_weighted_jaccard,
)

def test_compute_lcs():
    words1 = ["a", "b", "c"]
    words2 = ["a", "d", "c"]
    pos1 = ["n", "v", "n"]
    pos2 = ["n", "v", "n"]
    assert compute_lcs(words1, pos1, words2, pos2) == 2

def test_syntactic_distance():
    pos1 = ["n", "v", "n"]
    pos2 = ["n", "v", "adj"]
    # Should be 1 because only one substitution is needed (n -> adj)
    assert compute_syntactic_distance(pos1, pos2) == 1

def test_weighted_jaccard():
    words1 = ["word1", "word2"]
    words2 = ["word1", "word3"]
    pos1 = ["n", "v"]
    pos2 = ["n", "v"]
    similarity = compute_weighted_jaccard(words1, pos1, words2, pos2)
    assert 0 <= similarity <= 1

# Skip WMD tests in CI since we don't have the actual word2vec files
@pytest.mark.skip(reason="Word2Vec model not available in CI")
def test_word_mover_distance():
    pass