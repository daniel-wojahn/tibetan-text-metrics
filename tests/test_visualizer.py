"""Tests for visualization functions."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.visualizer import save_results_and_visualize


@pytest.fixture
def sample_results_df():
    # Create a sample DataFrame that matches the expected structure
    data = {
        "file1": ["text1.txt", "text1.txt"],
        "file2": ["text2.txt", "text3.txt"],
        "chapter": [1, 2],
        "lcs": [5, 3],
        "syntactic_distance": [0.8, 0.6],
        "weighted_jaccard": [0.7, 0.4],
        "wmd": [0.3, 0.5]
    }
    return pd.DataFrame(data)


def test_save_results_and_visualize(sample_results_df, monkeypatch):
    # Create a temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the output directory to use our temporary directory
        def mock_parent():
            return Path(tmpdir)
        
        monkeypatch.setattr(Path, "parent", property(lambda _: mock_parent()))
        
        # Run the visualization function
        save_results_and_visualize(sample_results_df)
        
        # Check if CSV file was created
        csv_path = Path(tmpdir) / "output" / "pos_tagged_analysis.csv"
        assert csv_path.exists()
        
        # Read back the CSV and verify contents
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == len(sample_results_df)
        assert all(col in saved_df.columns for col in sample_results_df.columns)
        
        # Check if visualization files were created
        for metric in ["lcs", "syntactic_distance", "weighted_jaccard", "wmd"]:
            plot_path = Path(tmpdir) / "output" / f"{metric}_heatmap.png"
            assert plot_path.exists()


def test_save_results_and_visualize_empty(monkeypatch):
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        def mock_parent():
            return Path(tmpdir)
        
        monkeypatch.setattr(Path, "parent", property(lambda _: mock_parent()))
        
        # Should handle empty DataFrame without errors
        save_results_and_visualize(empty_df)
        
        # Check if CSV file was created even if empty
        csv_path = Path(tmpdir) / "output" / "pos_tagged_analysis.csv"
        assert csv_path.exists()


def test_save_results_and_visualize_single_row(monkeypatch):
    # Test with single row DataFrame
    data = {
        "file1": ["text1.txt"],
        "file2": ["text2.txt"],
        "chapter": [1],
        "lcs": [5],
        "syntactic_distance": [0.8],
        "weighted_jaccard": [0.7],
        "wmd": [0.3]
    }
    single_row_df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        def mock_parent():
            return Path(tmpdir)
        
        monkeypatch.setattr(Path, "parent", property(lambda _: mock_parent()))
        
        # Should handle single row DataFrame
        save_results_and_visualize(single_row_df)
        
        # Check if files were created
        csv_path = Path(tmpdir) / "output" / "pos_tagged_analysis.csv"
        assert csv_path.exists()
