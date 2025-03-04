"""Main script for running text analysis."""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from gensim.models import KeyedVectors

from .analyzer import compute_pairwise_analysis_pos
from .text_processor import read_text_files
from .visualizer import save_results_and_visualize


def load_word2vec_model() -> KeyedVectors:
    """Load Word2Vec model with caching.

    Returns:
        KeyedVectors: Loaded word2vec model
    """
    # Get the path to the package directory
    package_dir = Path(__file__).parent
    model_dir = package_dir / "word2vec" / "藏文-音节"
    cache_file = model_dir / "model_cache.pkl"
    vec_file = model_dir / "word2vec_zang_yinjie.vec"

    print("Loading Word2Vec model...")
    if cache_file.exists():
        model = KeyedVectors.load(str(cache_file))
    else:
        if not vec_file.exists():
            raise FileNotFoundError(
                f"Word2Vec model not found at {vec_file}. "
                "Please ensure the model file is in the correct location."
            )
        model = KeyedVectors.load_word2vec_format(str(vec_file), binary=False)
        model.fill_norms()  # Precompute normalized vectors for faster WMD
        os.makedirs(model_dir, exist_ok=True)
        model.save(str(cache_file))

    print("Model loaded successfully.")
    return model


def main() -> None:
    """Main function to run the analysis pipeline."""
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent

    # Configuration - explicit list of files in specific order
    file_paths = [
        str(project_root / "input_files" / "Bailey.txt"),
        str(project_root / "input_files" / "Bhutan.txt"),
        str(project_root / "input_files" / "Dolanji.txt"),
        str(project_root / "input_files" / "LTWA.txt"),
        str(project_root / "input_files" / "Japan13.txt"),
    ]

    # Verify all files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Input file not found: {file_path}. "
                "Please ensure all required input files are present."
            )

    # Load model with caching
    model = load_word2vec_model()

    # Read and process input files
    print("Reading input files...")
    texts = read_text_files(file_paths)
    print(f"Processed {len(texts)} input files.")

    # Compute pairwise analysis
    print("Computing text similarity metrics...")
    results_df = compute_pairwise_analysis_pos(texts, model, file_paths)
    print("Analysis complete.")

    # Save results and generate visualizations
    print("\nGenerating visualizations...")
    save_results_and_visualize(results_df)
    print("Results saved to CSV and heatmaps generated in output/")


if __name__ == "__main__":
    main()
