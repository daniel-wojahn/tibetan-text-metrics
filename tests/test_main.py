"""Tests for main script functionality."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from tibetan_text_metrics.main import main

def test_main_with_mocks(monkeypatch):
    """Test the main function with mocked dependencies."""
    from unittest.mock import patch, MagicMock
    import tibetan_text_metrics.main as main_module
    from tibetan_text_metrics.main import main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Create mock input/output directories
        input_dir = temp_path / "input_files"
        input_dir.mkdir()
        output_dir = temp_path / "output"
        output_dir.mkdir(exist_ok=True)
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        pca_dir = output_dir / "pca"
        pca_dir.mkdir(exist_ok=True)
        heatmaps_dir = output_dir / "heatmaps"
        heatmaps_dir.mkdir(exist_ok=True)
        
        # Create a mock text file
        test_file = input_dir / "test_file.txt"
        test_file.write_text("This is a test.")
        
        # Patch the project root to use our temp directory
        def mock_project_root(*args, **kwargs):
            if hasattr(args[0], "name") and args[0].name == "__file__":
                return temp_path
            else:
                return temp_path
        
        monkeypatch.setattr(Path, "parent", property(mock_project_root))
        
        # Create mock objects and results
        mock_texts = {"test_file": ("test", ["POS"])}
        mock_results_df = pd.DataFrame([
            {
                'Text Pair': 'Test vs Test', 
                'Chapter': '1',
                'Syntactic Distance (POS Level)': 10, 
                'Normalized Syntactic Distance': 0.5,
                'Weighted Jaccard Similarity (%)': 70,
                'LCS Length': 80,
                'Normalized LCS (%)': 80,
                'Text Pair Category': 'Category A', 
                'Chapter Length 1': 100, 
                'Chapter Length 2': 100
            }
        ])
        
        # Create mock implementations for all functions
        mock_read = MagicMock(return_value=mock_texts)
        mock_analyze = MagicMock(return_value=mock_results_df)
        mock_save = MagicMock()
        mock_pca = MagicMock()
        
        # Directly patch the functions used in the main module
        main_module.read_text_files = mock_read
        main_module.compute_pairwise_analysis_pos = mock_analyze
        main_module.save_results_and_visualize = mock_save
        main_module.perform_pca_analysis = mock_pca
        
        # Call the main function
        main()
        
        # Verify all the important functions were called
        mock_read.assert_called_once()
        mock_analyze.assert_called_once()
        mock_save.assert_called_once()
        mock_pca.assert_called_once()


def test_main_no_input_files(monkeypatch):
    """Test the main function when no input files are found."""
    import tibetan_text_metrics.main as main_module
    
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Create input directory but without any files
        input_dir = temp_path / "input_files"
        input_dir.mkdir()
        
        # Patch the project root to use our temp directory
        def mock_project_root(*args, **kwargs):
            if hasattr(args[0], "name") and args[0].name == "__file__":
                return temp_path
            else:
                return temp_path
        
        monkeypatch.setattr(Path, "parent", property(mock_project_root))
        
        # Should raise FileNotFoundError when no input files are found
        with pytest.raises(FileNotFoundError):
            main_module.main()


@pytest.mark.skip(reason="Integration test requiring actual input files")
def test_main():
    # This is an integration test that would need actual input files
    # We'll skip it for now, but it's good to have the structure
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
