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
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    csv_path = output_dir / "pos_tagged_analysis.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.15f")
    print(f"\nResults saved to {csv_path}")

    # Convert columns to numeric
    numeric_columns = [
        "Syntactic Distance (POS Level)",
        "Weighted Jaccard Similarity (%)",
        "LCS Length",
        "Word Mover's Distance",
    ]

    for col in numeric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    # Create pivot tables for heatmaps
    pivot_data = {
        "Syntactic Distance": (
            results_df.pivot(
                index="Chapter",
                columns="Text Pair",
                values="Syntactic Distance (POS Level)",
            ).fillna(0).astype(int),
            "Reds",
            "Syntactic Distance: Shows the number of operations (insertions, deletions, substitutions) needed to transform one POS tag sequence into another.\nHigher values indicate more structural differences.",
            "d",
        ),
        "Weighted Jaccard": (
            results_df.pivot(
                index="Chapter",
                columns="Text Pair",
                values="Weighted Jaccard Similarity (%)",
            ).fillna(0),
            "Blues",
            "Weighted Jaccard Similarity: Measures unique vocabulary overlap with POS-based weighting.",
            ".1f",
        ),
        "LCS": (
            results_df.pivot(index="Chapter", columns="Text Pair", values="LCS Length").fillna(0).astype(int),
            "Greens",
            "LCS Length: Measures the length of the longest common subsequence of words between text pairs.",
            ".0f",
        ),
        "WMD": (
            results_df.pivot(
                index="Chapter", columns="Text Pair", values="Word Mover's Distance"
            ).fillna(0),
            "Purples",
            "Word Mover's Distance: Measures the semantic distance between texts based on word embeddings.",
            ".2f",
        ),
    }

    print("\nGenerating visualizations...")
    # Generate heatmaps
    for metric_name, (data, cmap, description, fmt) in pivot_data.items():
        if data.empty:
            print(f"Skipping {metric_name} visualization: no data available")
            continue

        # All plots use the same figure size for consistent cell proportions
        plt.figure(figsize=(12, 8))

        # Create heatmap with styling
        sns.heatmap(
            data, 
            annot=True, 
            fmt=fmt, 
            cmap=cmap, 
            cbar_kws={"label": metric_name},
            annot_kws={"size": 10}
        )

        # Set title and labels
        plt.title(f"Heatmap of {metric_name} by Chapter", fontsize=16)
        plt.xlabel("Text Pair", fontsize=12)
        plt.ylabel("Chapter", fontsize=12)

        # Add description below title
        plt.text(0, -1.5, description, fontsize=10, color="black", ha="left")

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = (
            output_dir / f'heatmap_{metric_name.lower().replace(" ", "_")}.png'
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved {output_path}")
