"""Functions for analyzing text patterns and visualizing results."""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from .text_processor import extract_words_and_pos
from .fast_patterns import FastPatternAnalyzer


def process_single_pair(args):
    """
    Helper function for multiprocessing that analyzes a single chapter pair.
    Creates a new FastPatternAnalyzer instance for each task.
    
    Args:
        args: A tuple containing:
            file1_stem, file2_stem, chapter_idx, chapter1, chapter2
        
    Returns:
        dict: The results of similarity calculations, including metadata.
    """
    file1_stem, file2_stem, chapter_idx, chapter1, chapter2 = args

    # Create a new analyzer instance
    analyzer = FastPatternAnalyzer()

    # Extract words and POS tags
    words1, pos1 = extract_words_and_pos(chapter1)
    words2, pos2 = extract_words_and_pos(chapter2)

    # Get the analysis results
    results = analyzer.analyze_chapter_pair(words1, pos1, words2, pos2)

    # Add metadata to results
    results["Text Pair"] = f"{file1_stem} vs {file2_stem}"
    results["Chapter"] = chapter_idx + 1

    return results


class PatternAnalyzer:
    """
    Class for handling pattern extraction and similarity calculations.
    
    Attributes:
        n_gram_size (int): Size of n-grams to extract for patterns.
        fast_analyzer (FastPatternAnalyzer): An instance of the pattern analyzer for processing.
        pattern_cache (dict): Stores pre-calculated patterns for chapters (reduces redundant computation).
    """
    def __init__(self, n_gram_size: int = 3):
        self.n_gram_size = n_gram_size
        self.fast_analyzer = FastPatternAnalyzer()
        self.pattern_cache = {}

    def extract_patterns(self, words: List[str], pos_tags: List[str]) -> Tuple[Dict, Dict]:
        """Extract word and POS tag patterns using Cython implementation."""
        if not words or not pos_tags:
            return {}, {}

        word_patterns = self.fast_analyzer.extract_ngrams(words, self.n_gram_size)
        pos_patterns = self.fast_analyzer.extract_ngrams(pos_tags, self.n_gram_size)
        return word_patterns, pos_patterns

    def compute_pattern_similarity(self, patterns1: Dict, patterns2: Dict) -> float:
        """Compute pattern similarity using the FastPatternAnalyzer."""
        return self.fast_analyzer.compute_cosine_similarity(patterns1, patterns2)

    def process_chapter_pair(self, args) -> dict:
        """
        Process a single chapter pair – used for debugging or simple runs.
        
        For multiprocessing, use the global `process_single_pair` function.
        """
        return process_single_pair(args)


def compute_pattern_metrics(
    texts: Dict[str, List[str]], file_paths: List[str], n_gram_size: int = 3
) -> pd.DataFrame:
    """
    Compute pattern-based metrics for text pairs using parallel processing.

    Args:
        texts: A dictionary where each key is a file stem (name without extension) and
               the value is a list of chapter texts.
        file_paths: A list of file paths corresponding to the keys in `texts`.
        n_gram_size: The size of the n-grams to extract patterns.
    
    Returns:
        pd.DataFrame: A DataFrame containing similarity metrics for each pair of chapters.
    """
    tasks = []
    for i, file1 in enumerate(file_paths):
        for j, file2 in enumerate(file_paths[i + 1:], i + 1):
            file1_stem = Path(file1).stem
            file2_stem = Path(file2).stem
            chapters1 = texts[file1_stem]
            chapters2 = texts[file2_stem]

            for chapter_idx, (chapter1, chapter2) in enumerate(zip(chapters1, chapters2)):
                tasks.append((file1_stem, file2_stem, chapter_idx, chapter1, chapter2))

    # Process tasks in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_pair, tasks))

    return pd.DataFrame(results)


def visualize_pattern_analysis(pattern_df: pd.DataFrame, heatmaps_dir: Path, n_gram_size: int = 3) -> None:
    """
    Generate visualizations for pattern analysis results.

    Args:
        pattern_df: A DataFrame containing the results of pattern similarity analysis.
        heatmaps_dir: The directory path where visualizations will be saved.
        n_gram_size: The size of n-grams used in the analysis (default: 3).
    """
    metrics = {
        "POS Pattern Similarity": {
            "cmap": "YlOrRd",
            "description": "Similarity of POS tag patterns between texts (0-1).\n"
                           "Higher values indicate more similar grammatical structures.",
            "fmt": ".2f",
        },
        "Word Pattern Similarity": {
            "cmap": "YlGnBu",
            "description": "Similarity of word patterns between texts (0-1).\n"
                           "Higher values indicate more similar word usage patterns.",
            "fmt": ".2f",
        },
    }

    for metric, settings in metrics.items():
        pivot_data = pattern_df.pivot(
            index="Chapter",
            columns="Text Pair",
            values=metric
        ).fillna(0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=settings["fmt"],
            cmap=settings["cmap"],
            cbar_kws={"label": metric},
            annot_kws={"size": 10},
        )

        plt.title(f"Heatmap of {metric} by Chapter (n-gram size: {n_gram_size})", fontsize=16)
        plt.xlabel("Text Pair", fontsize=12)
        plt.ylabel("Chapter", fontsize=12)
        plt.text(0, -1.5, settings["description"], fontsize=10, color="black", ha="left")

        plt.tight_layout()
        output_path = heatmaps_dir / f"heatmap_pattern_{metric.lower().replace(' ', '_')}_n{n_gram_size}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
