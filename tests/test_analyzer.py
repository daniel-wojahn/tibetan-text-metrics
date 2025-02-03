"""Tests for text analysis functions."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from gensim.models import KeyedVectors

from tibetan_text_metrics.analyzer import compute_pairwise_analysis_pos


class MockWord2Vec:
    def __init__(self):
        self.vectors = np.array([[1, 0], [0, 1], [1, 1]])
        self.key_to_index = {"word1": 0, "word2": 1, "word3": 2}

    def get_vector(self, word):
        return self.vectors[self.key_to_index[word]]

    def wmdistance(self, words1, words2):
        return 0.5  # Mock distance


@pytest.fixture
def mock_model():
    return MockWord2Vec()


def test_compute_pairwise_analysis_pos(mock_model):
    # Prepare test data
    texts = {
        "file1.txt": ["word1/n word2/v", "word3/adj"],
        "file2.txt": ["word2/v word3/adj", "word1/n"]
    }
    file_names = ["file1.txt", "file2.txt"]

    # Run analysis
    result = compute_pairwise_analysis_pos(texts, mock_model, file_names)

    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert "Text Pair" in result.columns
    assert "Chapter" in result.columns
    assert "Syntactic Distance (POS Level)" in result.columns
    assert "Weighted Jaccard Similarity (%)" in result.columns
    assert "LCS Length" in result.columns
    assert "Word Mover's Distance" in result.columns

    # Check data types
    assert result["LCS Length"].dtype in [np.int32, np.int64]
    assert result["Syntactic Distance (POS Level)"].dtype == np.float64
    assert result["Weighted Jaccard Similarity (%)"].dtype == np.float64
    assert result["Word Mover's Distance"].dtype == np.float64


def test_compute_pairwise_analysis_pos_empty():
    # Test with empty texts
    texts = {}
    file_names = []
    mock_model = MockWord2Vec()

    result = compute_pairwise_analysis_pos(texts, mock_model, file_names)
    assert len(result) == 0


def test_compute_pairwise_analysis_pos_single_file():
    # Test with single file
    texts = {
        "file1.txt": ["word1/n word2/v"]
    }
    file_names = ["file1.txt"]
    mock_model = MockWord2Vec()

    result = compute_pairwise_analysis_pos(texts, mock_model, file_names)
    assert len(result) == 0  # No pairs to compare
