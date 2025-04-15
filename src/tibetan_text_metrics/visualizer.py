"""Functions for visualizing text analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_results_and_visualize(results_df: pd.DataFrame, metrics_dir: Path, heatmaps_dir: Path) -> None:
    """Save results to CSV and generate heatmap visualizations.

    Args:
        results_df: DataFrame containing pairwise analysis results.
        metrics_dir: Directory to save metrics CSV files.
        heatmaps_dir: Directory to save heatmap visualizations.
    """
    # Save results to CSV
    csv_path = metrics_dir / "pos_tagged_analysis.csv"
    results_df.to_csv(csv_path, index=False, float_format="%.15f")
    print(f"\nResults saved to {csv_path}")

    # Convert columns to numeric
    numeric_columns = [
        "Normalized Syntactic Distance",
        "Weighted Jaccard Similarity (%)",
        "Normalized LCS (%)",
        "Chapter Length 1",
        "Chapter Length 2",
    ]

    for col in numeric_columns:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    # Create pivot tables for heatmaps
    pivot_data = {
        "Normalized Syntactic Distance": (
            results_df.pivot(
                index="Chapter",
                columns="Text Pair",
                values="Normalized Syntactic Distance",
            ).fillna(0),
            "Reds",
            "Normalized Syntactic Distance: Shows the proportion of POS tags that differ between texts (0-1).\nNormalized by text length to allow fair comparison between chapters of different sizes.\nHigher values (darker red) indicate greater differences.",
            ".2f",
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
        "Normalized LCS": (
            results_df.pivot(index="Chapter", columns="Text Pair", values="Normalized LCS (%)").fillna(0),
            "YlGn",
            "Normalized LCS: Measures the length of the longest common subsequence divided by average chapter length.\nHigher percentages indicate greater sequential similarity, normalized for chapter size.",
            ".1f",
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
            heatmaps_dir / f'heatmap_{metric_name.lower().replace(" ", "_")}.png'
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved {output_path}")
