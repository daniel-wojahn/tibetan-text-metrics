"""Main script for running text analysis."""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import argparse

from .analyzer import compute_pairwise_analysis_pos
from .pattern_analyzer import compute_pattern_metrics, visualize_pattern_analysis
from .pca_visualizer import perform_pca_analysis
from .pca_core import perform_direct_text_pca
from .text_processor import read_text_files
from .visualizer import save_results_and_visualize

def main(n_gram_size: int = 3) -> None:
    """Main function to run the analysis pipeline.
    
    Args:
        n_gram_size: Size of n-grams to use for pattern analysis (default: 3)
    """
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

    # Compute pattern analysis
    print(f"\nComputing pattern-based metrics with n-gram size {n_gram_size}...")
    pattern_results = compute_pattern_metrics(texts, file_paths, n_gram_size=n_gram_size)
    pattern_csv_path = metrics_dir / f"pattern_analysis_n{n_gram_size}.csv"
    pattern_results.to_csv(pattern_csv_path, index=False, float_format="%.4f")
    print(f"Pattern analysis results saved to {pattern_csv_path}")

    # Save results and generate visualizations
    print("\nGenerating visualizations...")
    save_results_and_visualize(results_df, metrics_dir, heatmaps_dir)
    
    # Generate pattern visualizations
    print("\nGenerating pattern visualizations...")
    visualize_pattern_analysis(pattern_results, heatmaps_dir, n_gram_size=n_gram_size)
    
    # Merge pattern metrics with main metrics for PCA
    print("\nMerging pattern metrics with main metrics for PCA...")
    # Ensure the key columns match for merging
    merge_keys = ['Text Pair', 'Chapter']
    
    # Verify columns exist in both dataframes
    if all(key in results_df.columns for key in merge_keys) and all(key in pattern_results.columns for key in merge_keys):
        # Create a copy to avoid modifying the original
        combined_df = results_df.copy()
        
        # Merge with pattern results
        pattern_metrics = ['POS Pattern Similarity', 'Word Pattern Similarity']
        available_pattern_metrics = [m for m in pattern_metrics if m in pattern_results.columns]
        
        # If pattern metrics are available, add them to the combined dataframe
        if available_pattern_metrics:
            pattern_subset = pattern_results[merge_keys + available_pattern_metrics]
            combined_df = pd.merge(combined_df, pattern_subset, on=merge_keys, how='left')
            print(f"Added pattern metrics to analysis: {', '.join(available_pattern_metrics)}")
        else:
            print("No pattern metrics columns found in pattern analysis results")
    else:
        print("Warning: Could not merge pattern metrics due to missing key columns")
        combined_df = results_df
        
    # Save the combined metrics
    combined_csv_path = metrics_dir / f"combined_analysis_n{n_gram_size}.csv"
    combined_df.to_csv(combined_csv_path, index=False, float_format="%.4f")
    print(f"Combined metrics saved to {combined_csv_path}")
    
    # Perform PCA analysis based on combined metrics
    print("\nPerforming metrics-based PCA analysis...")
    # Create a subdirectory for this n-gram size
    ngram_pca_dir = pca_dir / f"n{n_gram_size}"
    ngram_pca_dir.mkdir(exist_ok=True)
    perform_pca_analysis(combined_df, ngram_pca_dir)
    
    print("\nResults saved to CSV and visualizations generated in output/")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Tibetan text metrics analysis")
    parser.add_argument("--ngram", type=int, default=3, help="Size of n-grams for pattern analysis (default: 3)")
    args = parser.parse_args()
    
    print(f"Running analysis with n-gram size: {args.ngram}")
    main(n_gram_size=args.ngram)
