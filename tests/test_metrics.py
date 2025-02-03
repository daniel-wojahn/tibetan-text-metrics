import pytest
from tibetan_text_metrics.metrics import (
    compute_lcs,
    compute_syntactic_distance,
    compute_weighted_jaccard,
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

# Skip WMD tests in CI since we don't have the actual word2vec files
@pytest.mark.skip(reason="Word2Vec model not available in CI")
def test_word_mover_distance():
    pass