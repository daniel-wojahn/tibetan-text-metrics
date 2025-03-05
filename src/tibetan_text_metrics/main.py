"""Main script for running text analysis."""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

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





if __name__ == "__main__":
    main()
