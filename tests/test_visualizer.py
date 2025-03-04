"""Tests for visualization functions."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tibetan_text_metrics.visualizer import save_results_and_visualize


@pytest.fixture
def sample_results_df():
    # Create a sample DataFrame that matches the expected structure
    data = {
        "Text Pair": ["text1.txt vs text2.txt", "text1.txt vs text3.txt"],
        "Chapter": [1, 2],
        "Syntactic Distance (POS Level)": [0.8, 0.6],
        "Normalized Syntactic Distance": [0.4, 0.3],
        "Weighted Jaccard Similarity (%)": [70.0, 40.0],
        "LCS Length": [5, 3],
        "Normalized LCS (%)": [60.0, 35.0],
        "Chapter Length 1": [100, 120],
        "Chapter Length 2": [90, 130]
    }
    return pd.DataFrame(data)


def test_save_results_and_visualize(sample_results_df, monkeypatch):
    # Create a temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create output directories
        tmp_path = Path(tmpdir)
        metrics_dir = tmp_path / "metrics"
        heatmaps_dir = tmp_path / "heatmaps"
        metrics_dir.mkdir(parents=True)
        heatmaps_dir.mkdir(parents=True)
        
        # Run the visualization function
        save_results_and_visualize(sample_results_df, metrics_dir, heatmaps_dir)
        
        # Check if CSV file was created
        csv_path = metrics_dir / "pos_tagged_analysis.csv"
        assert csv_path.exists()
        
        # Read back the CSV and verify contents
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == len(sample_results_df)
        assert all(col in saved_df.columns for col in sample_results_df.columns)
        
        # Check if visualization files were created
        expected_plots = [
            "heatmap_syntactic_distance.png",
            "heatmap_normalized_syntactic_distance.png",
            "heatmap_weighted_jaccard.png",
            "heatmap_lcs.png",
            "heatmap_normalized_lcs.png"
        ]
        for plot_name in expected_plots:
            plot_path = heatmaps_dir / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"


def test_save_results_and_visualize_empty(monkeypatch):
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=[
        "Text Pair", "Chapter", "Syntactic Distance (POS Level)", "Normalized Syntactic Distance",
        "Weighted Jaccard Similarity (%)", "LCS Length", "Normalized LCS (%)", 
        "Chapter Length 1", "Chapter Length 2"
    ])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create output directories
        tmp_path = Path(tmpdir)
        metrics_dir = tmp_path / "metrics"
        heatmaps_dir = tmp_path / "heatmaps"
        metrics_dir.mkdir(parents=True)
        heatmaps_dir.mkdir(parents=True)
        
        # Should handle empty DataFrame without errors
        save_results_and_visualize(empty_df, metrics_dir, heatmaps_dir)
        
        # Check if CSV file was created even if empty
        csv_path = metrics_dir / "pos_tagged_analysis.csv"
        assert csv_path.exists()
        
        # No plots should be created for empty DataFrame
        plot_path = heatmaps_dir / "heatmap_syntactic_distance.png"
        assert not plot_path.exists()


def test_save_results_and_visualize_single_row(monkeypatch):
    # Test with single row DataFrame
    data = {
        "Text Pair": ["text1.txt vs text2.txt"],
        "Chapter": [1],
        "Syntactic Distance (POS Level)": [0.8],
        "Normalized Syntactic Distance": [0.4],
        "Weighted Jaccard Similarity (%)": [70.0],
        "LCS Length": [5],
        "Normalized LCS (%)": [60.0],
        "Chapter Length 1": [100],
        "Chapter Length 2": [90]
    }
    single_row_df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create output directories
        tmp_path = Path(tmpdir)
        metrics_dir = tmp_path / "metrics"
        heatmaps_dir = tmp_path / "heatmaps"
        metrics_dir.mkdir(parents=True)
        heatmaps_dir.mkdir(parents=True)
        
        # Should handle single row DataFrame
        save_results_and_visualize(single_row_df, metrics_dir, heatmaps_dir)
        
        # Check if files were created
        csv_path = metrics_dir / "pos_tagged_analysis.csv"
        assert csv_path.exists()
        
        # Check if visualization files were created
        expected_plots = [
            "heatmap_syntactic_distance.png",
            "heatmap_normalized_syntactic_distance.png",
            "heatmap_weighted_jaccard.png",
            "heatmap_lcs.png",
            "heatmap_normalized_lcs.png"
        ]
        for plot_name in expected_plots:
            plot_path = heatmaps_dir / plot_name
            assert plot_path.exists(), f"Missing plot: {plot_name}"
