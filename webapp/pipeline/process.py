import pandas as pd
from typing import Dict, List, Tuple
from .metrics import compute_all_metrics
from .semantic_embedding import get_model_and_device, train_fasttext_model, FASTTEXT_MODEL_ID
from .tokenize import tokenize_texts
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


def process_texts(
    text_data: Dict[str, str], 
    filenames: List[str], 
    enable_semantic: bool = True,
    model_name: str = "buddhist-nlp/buddhist-sentence-similarity",
    use_stopwords: bool = True,
    use_lite_stopwords: bool = False,
    progress_callback = None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Processes uploaded texts, segments them by chapter marker, and computes metrics between chapters of different files.
    
    Args:
        text_data (Dict[str, str]): A dictionary mapping filenames to their content.
        filenames (List[str]): A list of filenames that were uploaded.
        enable_semantic (bool, optional): Whether to compute semantic similarity metrics. 
            Requires loading a sentence transformer model, which can be time-consuming. Defaults to True.
        model_name (str, optional): The name of the sentence transformer model to use for semantic similarity.
            Must be a valid model identifier on Hugging Face. Defaults to "buddhist-nlp/buddhist-sentence-similarity".
        use_stopwords (bool, optional): Whether to use stopwords in the metrics calculation. Defaults to True.
        use_lite_stopwords (bool, optional): Whether to use the lite stopwords list (common particles only)
            instead of the comprehensive list. Only applies if use_stopwords is True. Defaults to False.
        progress_callback (callable, optional): A callback function for reporting progress updates.
            Should accept a float between 0 and 1 and a description string. Defaults to None.
            
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str]:
            - metrics_df: DataFrame with similarity metrics between corresponding chapters of file pairs.
                Contains columns: 'Text Pair', 'Chapter', 'Jaccard Similarity (%)', 'Normalized LCS',
                'Semantic Similarity' (if enable_semantic=True), and 'TF-IDF Cosine Sim'.
            - word_counts_df: DataFrame with word counts for each segment (chapter) in each file.
                Contains columns: 'Filename', 'ChapterNumber', 'SegmentID', 'WordCount'.
            - warning: A string containing any warnings generated during processing (e.g., missing chapter markers).
    
    Raises:
        RuntimeError: If the botok tokenizer fails to initialize.
        ValueError: If the input files cannot be processed or if metrics computation fails.
    """
    # Initialize model and device variables
    st_model, st_device = None, None
    model_warning = ""
    
    # Update progress if callback provided
    if progress_callback is not None:
        try:
            progress_callback(0.25, desc="Preparing for text analysis...")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")
            # Continue processing even if progress reporting fails
    
    # Load semantic model if enabled
    if enable_semantic:
        logger.info("Semantic similarity enabled. Loading embedding model...")
        try:
            logger.info("Using model: %s", model_name)
            
            # Check if this is a FastText model request
            if model_name == FASTTEXT_MODEL_ID:
                # Try to load the official Facebook FastText Tibetan model first
                if progress_callback is not None:
                    try:
                        progress_callback(0.25, desc="Loading official Facebook FastText Tibetan model...")
                    except Exception as e:
                        logger.warning("Progress callback error (non-critical): %s", str(e))
                
                st_model, st_device, model_type = get_model_and_device(model_id=model_name)
                
                # If model is None, we need to train a fallback model
                if st_model is None:
                    if progress_callback is not None:
                        try:
                            progress_callback(0.25, desc="Official model unavailable. Training fallback FastText model...")
                        except Exception as e:
                            logger.warning("Progress callback error (non-critical): %s", str(e))
                    
                    # Collect all text data for training
                    all_texts = list(text_data.values())
                    
                    # Train the model with standard parameters for stability
                    st_model = train_fasttext_model(all_texts, dim=100, epoch=5)
                    
                    if progress_callback is not None:
                        try:
                            progress_callback(0.3, desc="Fallback FastText model trained successfully")
                        except Exception as e:
                            logger.warning("Progress callback error (non-critical): %s", str(e))
                else:
                    if progress_callback is not None:
                        try:
                            progress_callback(0.3, desc="Official Facebook FastText Tibetan model loaded successfully")
                        except Exception as e:
                            logger.warning(f"Progress callback error (non-critical): {e}")
            else:
                # For sentence transformers
                st_model, st_device, model_type = get_model_and_device(model_id=model_name)
                logger.info(f"Model {model_name} loaded successfully on {st_device}.")
                
                if progress_callback is not None:
                    try:
                        progress_callback(0.3, desc="Model loaded successfully")
                    except Exception as e:
                        logger.warning(f"Progress callback error (non-critical): {e}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load sentence transformer model: {error_msg}. Semantic similarity will not be available.")
            
            # Create a user-friendly warning message
            if "is not a valid model identifier" in error_msg:
                model_warning = f"The model '{model_name}' could not be found on Hugging Face. Semantic similarity will not be available."
            elif "CUDA out of memory" in error_msg:
                model_warning = "Not enough GPU memory to load the semantic model. Try using a smaller model or disable semantic similarity."
            else:
                model_warning = f"Failed to load semantic model: {error_msg}. Semantic similarity will not be available."
                
            if progress_callback is not None:
                try:
                    progress_callback(0.3, desc="Continuing without semantic model")
                except Exception as e:
                    logger.warning(f"Progress callback error (non-critical): {e}")
    else:
        logger.info("Semantic similarity disabled. Skipping model loading.")
        if progress_callback is not None:
            try:
                progress_callback(0.3, desc="Processing text segments")
            except Exception as e:
                logger.warning(f"Progress callback error (non-critical): {e}")

    # Detect chapter marker and segment texts
    if progress_callback is not None:
        try:
            progress_callback(0.35, desc="Segmenting texts by chapters...")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")
        
    chapter_marker = "༈"
    fallback = False
    segment_texts = {}
    
    # Process each file
    for i, fname in enumerate(filenames):
        if progress_callback is not None and len(filenames) > 1:
            try:
                progress_callback(0.35 + (0.05 * (i / len(filenames))), 
                                desc=f"Segmenting file {i+1}/{len(filenames)}: {fname}")
            except Exception as e:
                logger.warning(f"Progress callback error (non-critical): {e}")
            
        content = text_data[fname]
        
        # Check if content is empty
        if not content.strip():
            logger.warning(f"File '{fname}' is empty or contains only whitespace.")
            continue
            
        # Split by chapter marker if present
        if chapter_marker in content:
            segments = [
                seg.strip() for seg in content.split(chapter_marker) if seg.strip()
            ]
            
            # Check if we have valid segments after splitting
            if not segments:
                logger.warning(f"File '{fname}' contains chapter markers but no valid text segments.")
                continue
                
            for idx, seg in enumerate(segments):
                seg_id = f"{fname}|chapter {idx+1}"
                segment_texts[seg_id] = seg
        else:
            # No chapter markers found, treat entire file as one segment
            seg_id = f"{fname}|chapter 1"
            segment_texts[seg_id] = content.strip()
            fallback = True
            
    # Generate warning if no chapter markers found
    warning = model_warning  # Include any model warnings
    if fallback:
        chapter_warning = (
            "No chapter marker found in one or more files. "
            "Each file will be treated as a single segment. "
            "For best results, add a unique marker (e.g., ༈) to separate chapters or sections."
        )
        warning = warning + " " + chapter_warning if warning else chapter_warning
        
    # Check if we have any valid segments
    if not segment_texts:
        logger.error("No valid text segments found in any of the uploaded files.")
        return pd.DataFrame(), pd.DataFrame(), "No valid text segments found in the uploaded files. Please check your files and try again."
    # Group chapters by filename (preserving order)
    if progress_callback is not None:
        try:
            progress_callback(0.4, desc="Organizing text segments...")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")
        
    file_to_chapters = {}
    for seg_id in segment_texts:
        fname = seg_id.split("|")[0]
        file_to_chapters.setdefault(fname, []).append(seg_id)
        
    # For each pair of files, compare corresponding chapters (by index)
    if progress_callback is not None:
        try:
            progress_callback(0.45, desc="Computing similarity metrics...")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")
        
    results = []
    files = list(file_to_chapters.keys())
    
    # Check if we have at least two files to compare
    if len(files) < 2:
        logger.warning("Need at least two files to compute similarity metrics.")
        return pd.DataFrame(), pd.DataFrame(), "Need at least two files to compute similarity metrics."
    
    # Track total number of comparisons for progress reporting
    total_comparisons = 0
    for file1, file2 in combinations(files, 2):
        chaps1 = file_to_chapters[file1]
        chaps2 = file_to_chapters[file2]
        total_comparisons += min(len(chaps1), len(chaps2))
    
    # Process each file pair
    comparison_count = 0
    for file1, file2 in combinations(files, 2):
        chaps1 = file_to_chapters[file1]
        chaps2 = file_to_chapters[file2]
        min_chaps = min(len(chaps1), len(chaps2))
        
        if progress_callback is not None:
            try:
                progress_callback(0.45, desc=f"Comparing {file1} with {file2}...")
            except Exception as e:
                logger.warning(f"Progress callback error (non-critical): {e}")
            
        for idx in range(min_chaps):
            seg1 = chaps1[idx]
            seg2 = chaps2[idx]
            
            # Update progress
            comparison_count += 1
            if progress_callback is not None and total_comparisons > 0:
                try:
                    progress_percentage = 0.45 + (0.25 * (comparison_count / total_comparisons))
                    progress_callback(progress_percentage, 
                                    desc=f"Computing metrics for chapter {idx+1} ({comparison_count}/{total_comparisons})")
                except Exception as e:
                    logger.warning(f"Progress callback error (non-critical): {e}")
            
            try:
                # Compute metrics for this chapter pair
                pair_metrics = compute_all_metrics(
                    {seg1: segment_texts[seg1], seg2: segment_texts[seg2]},
                    model=st_model,
                    device=st_device,
                    enable_semantic=enable_semantic,
                    model_type=model_type if 'model_type' in locals() else "sentence_transformer",
                    use_stopwords=use_stopwords,
                    use_lite_stopwords=use_lite_stopwords
                )
                
                # Rename 'Text Pair' to show file stems and chapter number
                pair_metrics.loc[:, "Text Pair"] = f"{file1} vs {file2}"
                pair_metrics.loc[:, "Chapter"] = idx + 1
                results.append(pair_metrics)
                
            except Exception as e:
                logger.error(f"Error computing metrics for {seg1} vs {seg2}: {e}")
                # Continue with other comparisons instead of failing completely
                continue
    
    # Create the metrics DataFrame
    if results:
        metrics_df = pd.concat(results, ignore_index=True)
    else:
        metrics_df = pd.DataFrame()
        warning += " No valid metrics could be computed. Please check your files and try again."

    # Calculate word counts
    if progress_callback is not None:
        try:
            progress_callback(0.75, desc="Calculating word counts...")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")
        
    word_counts_data = []
    
    # Process each segment
    for i, (seg_id, text_content) in enumerate(segment_texts.items()):
        # Update progress
        if progress_callback is not None and len(segment_texts) > 0:
            try:
                progress_percentage = 0.75 + (0.15 * (i / len(segment_texts)))
                progress_callback(progress_percentage, desc=f"Counting words in segment {i+1}/{len(segment_texts)}")
            except Exception as e:
                logger.warning(f"Progress callback error (non-critical): {e}")
            
        fname, chapter_info = seg_id.split("|", 1)
        chapter_num = int(chapter_info.replace("chapter ", ""))
        
        try:
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
        except Exception as e:
            logger.error(f"Error calculating word count for segment {seg_id}: {e}")
            # Add entry with 0 word count to maintain consistency
            word_counts_data.append(
                {
                    "Filename": fname.replace(".txt", ""),
                    "ChapterNumber": chapter_num,
                    "SegmentID": seg_id,
                    "WordCount": 0,
                }
            )
    
    # Create and sort the word counts DataFrame
    word_counts_df = pd.DataFrame(word_counts_data)
    if not word_counts_df.empty:
        word_counts_df = word_counts_df.sort_values(
            by=["Filename", "ChapterNumber"]
        ).reset_index(drop=True)
    
    if progress_callback is not None:
        try:
            progress_callback(0.95, desc="Analysis complete!")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")
        
    return metrics_df, word_counts_df, warning
