"""Functions for analyzing text similarities."""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .metrics import (compute_normalized_lcs,
                      compute_normalized_syntactic_distance,
                      compute_weighted_jaccard)
from .text_processor import extract_words_and_pos


def compute_pairwise_analysis_pos(
    texts: Dict[str, List[str]], file_names: List[str]
) -> pd.DataFrame:
    """Compute pairwise analysis for POS-tagged texts.

    Args:
        texts: Dictionary mapping filenames to list of raw text for each chapter
        file_names: List of input file names

    Returns:
        DataFrame with pairwise analysis results
    """
    results = []
    pairs = list(combinations(range(len(file_names)), 2))

    # Create progress bar for text pairs
    pbar = tqdm(pairs, desc="Analyzing text pairs")
    for i, j in pbar:
        file1, file2 = file_names[i], file_names[j]
        file1_stem = Path(file1).stem
        file2_stem = Path(file2).stem
        pbar.set_description(f"Analyzing {file1_stem} vs {file2_stem}")

        chapters1, chapters2 = texts[file1_stem], texts[file2_stem]

        # Process each chapter pair
        for idx, (chapter1, chapter2) in enumerate(zip(chapters1, chapters2)):
            words1, pos1 = extract_words_and_pos(chapter1)
            words2, pos2 = extract_words_and_pos(chapter2)

            # Only compute and store normalized metrics (raw metrics are redundant)
            norm_syn_dist = compute_normalized_syntactic_distance(pos1, pos2)
            jaccard = compute_weighted_jaccard(words1, pos1, words2, pos2)
            norm_lcs = compute_normalized_lcs(words1, pos1, words2, pos2)

            results.append(
                {
                    "Text Pair": f"{file1_stem} vs {file2_stem}",
                    "Chapter": idx + 1,
                    "Normalized Syntactic Distance": norm_syn_dist,
                    "Weighted Jaccard Similarity (%)": jaccard * 100,
                    "Normalized LCS (%)": norm_lcs * 100,
                    "Chapter Length 1": len(words1),
                    "Chapter Length 2": len(words2),
                }
            )

    return pd.DataFrame(results)
