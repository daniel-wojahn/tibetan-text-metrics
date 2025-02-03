"""Tests for main script functionality."""

import os
import tempfile
from pathlib import Path

import pytest
from gensim.models import KeyedVectors

from src.main import load_word2vec_model, main


def create_mock_word2vec_file(model_dir: Path):
    """Create a minimal word2vec file for testing."""
    vec_file = model_dir / "word2vec_zang_yinjie.vec"
    with open(vec_file, "w", encoding="utf-8") as f:
        f.write("3 2\n")  # 3 words, 2 dimensions
        f.write("word1 1.0 0.0\n")
        f.write("word2 0.0 1.0\n")
        f.write("word3 1.0 1.0\n")


def test_load_word2vec_model(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up mock directory structure
        tmp_path = Path(tmpdir)
        model_dir = tmp_path / "word2vec" / "藏文-音节"
        model_dir.mkdir(parents=True)
        
        # Create mock word2vec file
        create_mock_word2vec_file(model_dir)
        
        # Patch __file__ to use our temporary directory
        monkeypatch.setattr(Path, "parent", property(lambda _: tmp_path))
        
        # Test loading model
        model = load_word2vec_model()
        assert isinstance(model, KeyedVectors)
        assert "word1" in model.key_to_index
        assert "word2" in model.key_to_index
        assert "word3" in model.key_to_index


def test_load_word2vec_model_cached(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up mock directory structure
        tmp_path = Path(tmpdir)
        model_dir = tmp_path / "word2vec" / "藏文-音节"
        model_dir.mkdir(parents=True)
        
        # Create mock word2vec file
        create_mock_word2vec_file(model_dir)
        
        # Patch __file__ to use our temporary directory
        monkeypatch.setattr(Path, "parent", property(lambda _: tmp_path))
        
        # Load model first time (creates cache)
        model1 = load_word2vec_model()
        
        # Load model second time (should use cache)
        model2 = load_word2vec_model()
        
        assert isinstance(model2, KeyedVectors)
        assert model2.key_to_index == model1.key_to_index


def test_load_word2vec_model_missing_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up mock directory structure without the vec file
        tmp_path = Path(tmpdir)
        model_dir = tmp_path / "word2vec" / "藏文-音节"
        model_dir.mkdir(parents=True)
        
        # Patch __file__ to use our temporary directory
        monkeypatch.setattr(Path, "parent", property(lambda _: tmp_path))
        
        # Should raise FileNotFoundError when vec file is missing
        with pytest.raises(FileNotFoundError):
            load_word2vec_model()


@pytest.mark.skip(reason="Integration test requiring actual input files")
def test_main():
    # This is an integration test that would need actual input files
    # We'll skip it for now, but it's good to have the structure
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
