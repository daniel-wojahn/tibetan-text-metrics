"""Functions for visualizing text analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_results_and_visualize(results_df: pd.DataFrame) -> None:
    """Save results to CSV and generate heatmap visualizations.

    Args:
        results_df: DataFrame containing pairwise analysis results.
    """
    # Create output directory in project root if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    csv_path = output_dir / "pos_tagged_analysis.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.15f")
    print(f"\nResults saved to {csv_path}")

    if len(results_df) == 0:
        print("No data to visualize")
        return

    # Convert columns to numeric
    numeric_columns = [
        "Syntactic Distance (POS Level)",
        "Weighted Jaccard Similarity (%)",
        "LCS Length",
        "Word Mover's Distance",
    ]
    for col in numeric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    print("Generating visualizations...")

    # Create visualizations for each metric
    metrics = {
        "syntactic_distance": "Syntactic Distance (POS Level)",
        "weighted_jaccard": "Weighted Jaccard Similarity (%)",
        "lcs": "LCS Length",
        "wmd": "Word Mover's Distance"
    }

    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10, 8))
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            values=metric_name,
            index="Text Pair",
            columns="Chapter",
            aggfunc="first"
        )
        
        # Generate heatmap
        sns.heatmap(
            pivot_data,
            annot=True,
            cmap="YlOrRd",
            fmt=".2f",
            cbar_kws={"label": metric_name}
        )
        
        plt.title(f"{metric_name} Heatmap")
        plt.xlabel("Chapter")
        plt.ylabel("Text Pair")
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"heatmap_{metric_key}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved {plot_path}")

    print("All visualizations have been saved in the output/ directory")
