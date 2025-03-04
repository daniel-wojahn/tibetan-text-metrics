"""Functions for analyzing text similarities."""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

from .metrics import (compute_lcs,
                      compute_normalized_lcs,
                      compute_syntactic_distance,
                      compute_weighted_jaccard,
                      compute_wmd)
from .text_processor import extract_words_and_pos


def compute_pairwise_analysis_pos(
    texts: Dict[str, List[str]], model: KeyedVectors, file_names: List[str]
) -> pd.DataFrame:
    """Compute pairwise analysis for POS-tagged texts.

    Args:
        texts: Dictionary mapping filenames to list of raw text for each chapter
        model: Loaded word2vec model
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
        file1_name = Path(file1).name
        file2_name = Path(file2).name
        pbar.set_description(f"Analyzing {file1_name} vs {file2_name}")

        chapters1, chapters2 = texts[file1_name], texts[file2_name]

        # Process each chapter pair
        for idx, (chapter1, chapter2) in enumerate(zip(chapters1, chapters2)):
            words1, pos1 = extract_words_and_pos(chapter1)
            words2, pos2 = extract_words_and_pos(chapter2)

            syn_dist = compute_syntactic_distance(pos1, pos2)
            jaccard = compute_weighted_jaccard(words1, pos1, words2, pos2)
            lcs = compute_lcs(words1, pos1, words2, pos2)
            norm_lcs = compute_normalized_lcs(words1, pos1, words2, pos2)
            wmd = compute_wmd(words1, words2, model)

            results.append(
                {
                    "Text Pair": f"{file1_name} vs {file2_name}",
                    "Chapter": idx + 1,
                    "Syntactic Distance (POS Level)": syn_dist,
                    "Weighted Jaccard Similarity (%)": jaccard * 100,
                    "LCS Length": lcs,
                    "Normalized LCS (%)": norm_lcs * 100,
                    "Word Mover's Distance": wmd,
                    "Chapter Length 1": len(words1),
                    "Chapter Length 2": len(words2),
                }
            )

    return pd.DataFrame(results)
