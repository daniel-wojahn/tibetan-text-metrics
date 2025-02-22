"""Tests for text processing functions."""

import os
import tempfile
from pathlib import Path

import pytest

from tibetan_text_metrics.text_processor import extract_words_and_pos, read_text_files


def test_read_text_files():
    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file 1
        file1_path = Path(tmpdir) / "test1.txt"
        with open(file1_path, "w", encoding="utf-8") as f:
            f.write("Chapter 1༈Chapter 2༈Chapter 3")
            
        # Create test file 2 with empty chapters
        file2_path = Path(tmpdir) / "test2.txt"
        with open(file2_path, "w", encoding="utf-8") as f:
            f.write("Chapter 1༈༈Chapter 2")
        
        # Test reading files
        result = read_text_files([str(file1_path), str(file2_path)])
        
        # Check results
        assert len(result) == 2
        assert "test1.txt" in result
        assert "test2.txt" in result
        assert len(result["test1.txt"]) == 3
        assert len(result["test2.txt"]) == 2
        assert result["test1.txt"] == ["Chapter 1", "Chapter 2", "Chapter 3"]
        assert result["test2.txt"] == ["Chapter 1", "Chapter 2"]


def test_read_text_files_empty():
    # Test with empty file list
    result = read_text_files([])
    assert result == {}


def test_read_text_files_invalid_path():
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        read_text_files(["nonexistent.txt"])


def test_extract_words_and_pos():
    # Test normal case
    text = "word1/n word2/v word3/adj"
    words, pos = extract_words_and_pos(text)
    assert words == ["word1", "word2", "word3"]
    assert pos == ["n", "v", "adj"]


def test_extract_words_and_pos_empty():
    # Test empty text
    words, pos = extract_words_and_pos("")
    assert words == []
    assert pos == []


def test_extract_words_and_pos_no_tags():
    # Test text without POS tags
    text = "word1 word2 word3"
    # Enable strict mode
    extract_words_and_pos._strict_mode = True
    with pytest.raises(ValueError):
        extract_words_and_pos(text)
    # Disable strict mode
    extract_words_and_pos._strict_mode = False


def test_extract_words_and_pos_mixed():
    # Test text with some words missing tags
    text = "word1/n word2 word3/adj"
    # Enable strict mode
    extract_words_and_pos._strict_mode = True
    with pytest.raises(ValueError):
        extract_words_and_pos(text)
    # Disable strict mode
    extract_words_and_pos._strict_mode = False
    
    # Test normal behavior (skipping untagged words)
    words, pos_tags = extract_words_and_pos(text)
    assert words == ["word1", "word3"]
    assert pos_tags == ["n", "adj"]
