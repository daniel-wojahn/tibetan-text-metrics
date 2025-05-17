import pandas as pd
from typing import Dict, List, Tuple
from .metrics import compute_all_metrics
from .semantic_embedding import get_sentence_transformer_model_and_device
from .tokenize import tokenize_texts
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


def process_texts(
    text_data: Dict[str, str], filenames: List[str], enable_semantic: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Processes uploaded texts, segments them by chapter marker, and computes metrics between chapters of different files.
    Args:
        text_data (Dict[str, str]): A dictionary mapping filenames to their content.
        filenames (List[str]): A list of filenames that were uploaded.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str]:
            - metrics_df: DataFrame with similarity metrics between corresponding chapters of file pairs.
            - word_counts_df: DataFrame with word counts for each segment (chapter) in each file.
            - warning: A string containing any warnings generated during processing (e.g., missing chapter markers).
    """
    st_model, st_device = None, None
    if enable_semantic:
        logger.info(
            "Semantic similarity enabled. Loading sentence transformer model..."
        )
        try:
            st_model, st_device = get_sentence_transformer_model_and_device()
            logger.info(
                f"Sentence transformer model loaded successfully on {st_device}."
            )
        except Exception as e:
            logger.error(
                f"Failed to load sentence transformer model: {e}. Semantic similarity will not be available."
            )
            # Optionally, add a warning to the UI if model loading fails
            # For now, keeping it as a logger.error. UI warning can be added later if desired.
            pass # Explicitly noting that we are not changing the warning handling for UI here.
    else:
        logger.info("Semantic similarity disabled. Skipping model loading.")

    # Detect chapter marker
    chapter_marker = "༈"
    fallback = False
    segment_texts = {}
    for fname in filenames:
        content = text_data[fname]
        if chapter_marker in content:
            segments = [
                seg.strip() for seg in content.split(chapter_marker) if seg.strip()
            ]
            for idx, seg in enumerate(segments):
                seg_id = f"{fname}|chapter {idx+1}"
                segment_texts[seg_id] = seg
        else:
            seg_id = f"{fname}|chapter 1"
            segment_texts[seg_id] = content.strip()
            fallback = True
    warning = ""
    if fallback:
        warning = (
            "No chapter marker found in one or more files. "
            "Each file will be treated as a single segment. "
            "For best results, add a unique marker (e.g., ༈) to separate chapters or sections."
        )
    # Group chapters by filename (preserving order)
    file_to_chapters = {}
    for seg_id in segment_texts:
        fname = seg_id.split("|")[0]
        file_to_chapters.setdefault(fname, []).append(seg_id)
    # For each pair of files, compare corresponding chapters (by index)
    results = []
    files = list(file_to_chapters.keys())
    for file1, file2 in combinations(files, 2):
        chaps1 = file_to_chapters[file1]
        chaps2 = file_to_chapters[file2]
        min_chaps = min(len(chaps1), len(chaps2))
        for idx in range(min_chaps):
            seg1 = chaps1[idx]
            seg2 = chaps2[idx]
            # Compute metrics for this chapter pair
            # Use compute_all_metrics on just these two segments
            pair_metrics = compute_all_metrics(
                {seg1: segment_texts[seg1], seg2: segment_texts[seg2]},
                model=st_model,
                device=st_device,
                enable_semantic=enable_semantic,
            )
            # Rename 'Text Pair' to show file stems and chapter number
            # Set Text Pair and Chapter columns
            pair_metrics.loc[:, "Text Pair"] = f"{file1} vs {file2}"
            pair_metrics.loc[:, "Chapter"] = idx + 1
            results.append(pair_metrics)
    if results:
        metrics_df = pd.concat(results, ignore_index=True)
    else:
        metrics_df = pd.DataFrame()

    # Calculate word counts
    word_counts_data = []
    for seg_id, text_content in segment_texts.items():
        fname, chapter_info = seg_id.split("|", 1)
        chapter_num = int(chapter_info.replace("chapter ", ""))
        # Use botok for accurate word count for raw Tibetan text
        tokenized_segments = tokenize_texts([text_content])  # Returns a list of lists
        if tokenized_segments and tokenized_segments[0]:
            word_count = len(tokenized_segments[0])
        else:
            word_count = 0
        word_counts_data.append(
            {
                "Filename": fname.replace(".txt", ""),
                "ChapterNumber": chapter_num,
                "SegmentID": seg_id,
                "WordCount": word_count,
            }
        )
    word_counts_df = pd.DataFrame(word_counts_data)
    if not word_counts_df.empty:
        word_counts_df = word_counts_df.sort_values(
            by=["Filename", "ChapterNumber"]
        ).reset_index(drop=True)

    return metrics_df, word_counts_df, warning
