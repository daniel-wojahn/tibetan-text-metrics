"""Main script for running text analysis."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gensim.models import KeyedVectors

from .analyzer import compute_pairwise_analysis_pos
from .pca_visualizer import perform_pca_analysis
from .pca_core import perform_direct_text_pca
from .text_processor import read_text_files
from .visualizer import save_results_and_visualize


def main() -> None:
    """Main function to run the analysis pipeline."""
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Create output directory structure
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different types of outputs
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    pca_dir = output_dir / "pca"
    pca_dir.mkdir(exist_ok=True)
    
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)
    
    # Text comparison directory no longer needed

    # Automatically load all text files from the input directory
    input_dir = project_root / "input_files"
    file_paths = [str(f) for f in input_dir.glob("*.txt") if f.is_file()]
    
    if not file_paths:
        raise FileNotFoundError(
            f"No text files found in {input_dir}. "
            "Please ensure input files are present."
        )
        
    print(f"Found {len(file_paths)} text files in the input directory:")
    for path in file_paths:
        print(f"  - {Path(path).stem}")

    # Read and process input files
    print("Reading input files...")
    texts = read_text_files(file_paths)
    print(f"Processed {len(texts)} input files.")

    # Compute pairwise analysis
    print("Computing text similarity metrics...")
    results_df = compute_pairwise_analysis_pos(texts, file_paths)
    print("Analysis complete.")

    # Save results and generate visualizations
    print("\nGenerating visualizations...")
    save_results_and_visualize(results_df, metrics_dir, heatmaps_dir)
    
    # Perform PCA analysis based on metrics
    print("\nPerforming metrics-based PCA analysis...")
    perform_pca_analysis(results_df, pca_dir)
    
    # Direct text-based PCA analysis is skipped as requested by the user
    
    # Text comparison visualizations have been removed as requested
    
    print("Results saved to CSV and visualizations generated in output/")


def load_word2vec_model() -> KeyedVectors:
    """Load word2vec model for Tibetan syllables.
    
    This function loads a pre-trained word2vec model for Tibetan syllables
    from the word2vec directory.
    
    Returns:
        KeyedVectors: Loaded word2vec model
        
    Raises:
        FileNotFoundError: If the word2vec model file is not found
    """
    # Static variable to cache the model
    if not hasattr(load_word2vec_model, "_cached_model"):
        load_word2vec_model._cached_model = None
    
    # Return cached model if available
    if load_word2vec_model._cached_model is not None:
        return load_word2vec_model._cached_model
    
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    # Set the path to the word2vec model
    model_dir = project_root / "word2vec" / "藏文-音节"
    vec_file = model_dir / "word2vec_zang_yinjie.vec"
    
    if not vec_file.exists():
        raise FileNotFoundError(
            f"Word2vec model file not found at {vec_file}. "
            "Please ensure the model file is present."
        )
    
    # Load the model
    model = KeyedVectors.load_word2vec_format(str(vec_file), binary=False)
    
    # Cache the model
    load_word2vec_model._cached_model = model
    
    return model


if __name__ == "__main__":
    main()
