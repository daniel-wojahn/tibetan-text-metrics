import pandas as pd
from typing import Dict, List, Tuple
from .metrics import compute_all_metrics
from .hf_embedding import get_model as get_hf_model
from .tokenize import tokenize_texts
from .progressive_loader import MetricType
import logging
from itertools import combinations
import re

# FastText removed: always use Sentence Transformers


def get_botok_tokens_for_single_text(text: str, mode: str = "syllable") -> list[str]:
    """
    A wrapper around tokenize_texts to make it suitable for tokenize_fn 
    in generate_embeddings, which expects a function that tokenizes a single string.
    Accepts a 'mode' argument ('syllable' or 'word') to pass to tokenize_texts.
    """
    if not text.strip():
        return []
    # Pass the mode to tokenize_texts
    tokenized_list_of_lists = tokenize_texts([text], mode=mode)
    if tokenized_list_of_lists and tokenized_list_of_lists[0]:
        return tokenized_list_of_lists[0]
    return []

def clean_tibetan_text(text: str) -> str:
    """
    Applies light cleaning steps to Tibetan text:
    - Removes lnX/pX page/line markers.
    - Normalizes double tsheg to single tsheg.
    - Normalizes whitespace.
    """
    # Remove lnX/pX markers
    cleaned_text = re.sub(r"\s*(?:[lL][nN]|[pP])\d{1,3}[abAB]?\s*", " ", text)
    # Normalize double tsheg
    cleaned_text = re.sub(r"།\s*།", "།", cleaned_text)
    # Normalize spaces (multiple spaces to single, strip leading/trailing)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text



logger = logging.getLogger(__name__)


def process_texts(
    text_data: Dict[str, str], 
    filenames: List[str], 
    enable_semantic: bool = True,
    enable_fuzzy: bool = True,
    fuzzy_method: str = 'token_set',
    model_name: str = "sentence-transformers/LaBSE",
    use_stopwords: bool = True,
    use_lite_stopwords: bool = False,
    progress_callback = None,
    progressive_callback = None,
    batch_size: int = 32,
    show_progress_bar: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Processes uploaded texts, segments them by chapter marker, and computes metrics between chapters of different files.
    
    Args:
        text_data (Dict[str, str]): A dictionary mapping filenames to their content.
        filenames (List[str]): A list of filenames that were uploaded.
        enable_semantic (bool, optional): Whether to compute semantic similarity metrics. 
            Requires loading a sentence-transformer model, which can be time-consuming. Defaults to True.
        enable_fuzzy (bool, optional): Whether to compute fuzzy string similarity metrics.
            Uses TheFuzz library for approximate string matching. Defaults to True.
        fuzzy_method (str, optional): The fuzzy matching method to use. Options are:
            'token_set' - Order-independent token matching (default)
            'token_sort' - Order-normalized token matching
            'partial' - Best partial token matching
            'ratio' - Simple ratio matching
        model_name (str, optional): The Hugging Face sentence-transformer model to use for semantic similarity.
            Must be a valid model identifier on Hugging Face. Defaults to "sentence-transformers/LaBSE".
        use_stopwords (bool, optional): Whether to use stopwords in the metrics calculation. Defaults to True.
        use_lite_stopwords (bool, optional): Whether to use the lite stopwords list (common particles only)
            instead of the comprehensive list. Only applies if use_stopwords is True. Defaults to False.
        progress_callback (callable, optional): A callback function for reporting progress updates.
            Should accept a float between 0 and 1 and a description string. Defaults to None.
        progressive_callback (callable, optional): A callback function for sending incremental results.
            Used for progressive loading of metrics as they become available. Defaults to None.
            
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str]:
            - metrics_df: DataFrame with similarity metrics between corresponding chapters of file pairs.
                Contains columns: 'Text Pair', 'Chapter', 'Jaccard Similarity (%)', 'Normalized LCS',
                'Fuzzy Similarity' (if enable_fuzzy=True), 'Semantic Similarity' (if enable_semantic=True).
            - word_counts_df: DataFrame with word counts for each segment (chapter) in each file.
                Contains columns: 'Filename', 'ChapterNumber', 'SegmentID', 'WordCount'.
            - warning: A string containing any warnings generated during processing (e.g., missing chapter markers).
    
    Raises:
        RuntimeError: If the botok tokenizer fails to initialize.
        ValueError: If the input files cannot be processed or if metrics computation fails.
    """
    # Initialize model and model_type variables
    model, model_type = None, None # st_device removed
    warning = ""
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
            logger.info(f"Using model: {model_name}")
            # Always use Hugging Face sentence-transformers
            model, model_type = get_hf_model(model_id=model_name)

            if model:
                logger.info(f"Model '{model_name}' (type: {model_type}) loaded successfully.")
                if progress_callback is not None:
                    progress_callback(0.3, desc=f"Model '{model_name}' loaded.")
            else:
                model_warning = f"Model ('{model_name}') failed to load. Semantic similarity will be disabled."
                logger.warning(model_warning)
                warning = warning + f" {model_warning}" if 'warning' in locals() else model_warning
                enable_semantic = False
                if progress_callback is not None:
                    try:
                        progress_callback(0.3, desc="Unsupported model, continuing without semantic similarity.")
                    except Exception as e:
                        logger.warning(f"Progress callback error (non-critical): {e}")
        
        except Exception as e:  # General catch-all for unexpected errors during model loading attempts
            model_warning = f"An unexpected error occurred while attempting to load model '{model_name}': {e}. Semantic similarity will be disabled."
            logger.error(model_warning, exc_info=True)
            enable_semantic = False
            if progress_callback is not None:
                try:
                    progress_callback(0.3, desc="Error loading model, continuing without semantic similarity.")
                except Exception as e_cb:
                    logger.warning(f"Progress callback error (non-critical): {e_cb}")
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
                cleaned_seg = clean_tibetan_text(seg)
                segment_texts[seg_id] = cleaned_seg
        else:
            # No chapter markers found, treat entire file as one segment
            seg_id = f"{fname}|chapter 1"
            cleaned_content = clean_tibetan_text(content.strip())
            segment_texts[seg_id] = cleaned_content
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
    # Tokenize all segments at once for efficiency
    if progress_callback is not None:
        try:
            progress_callback(0.42, desc="Tokenizing all text segments...")
        except Exception as e:
            logger.warning(f"Progress callback error (non-critical): {e}")

    all_segment_ids = list(segment_texts.keys())
    all_segment_contents = list(segment_texts.values())
    tokenized_segments_list = tokenize_texts(all_segment_contents)

    segment_tokens = dict(zip(all_segment_ids, tokenized_segments_list))

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
    
    # Initialize results DataFrame for progressive updates
    results_columns = ['Text Pair', 'Chapter', 'Jaccard Similarity (%)', 'Normalized LCS']
    if enable_fuzzy:
        results_columns.append('Fuzzy Similarity')
    if enable_semantic:
        results_columns.append('Semantic Similarity')
    
    # Create empty DataFrame with the correct columns
    progressive_df = pd.DataFrame(columns=results_columns)
    
    # Track which metrics have been completed for progressive updates
    completed_metrics = []
    
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
                metrics_df = compute_all_metrics(
                    texts={seg1: segment_texts[seg1], seg2: segment_texts[seg2]},
                    token_lists={seg1: segment_tokens[seg1], seg2: segment_tokens[seg2]},
                    model=model,
                    enable_semantic=enable_semantic,
                    enable_fuzzy=enable_fuzzy,
                    fuzzy_method=fuzzy_method,
                    use_stopwords=use_stopwords,
                    use_lite_stopwords=use_lite_stopwords,
                )
                
                # Extract metrics from the DataFrame (should have only one row)
                if not metrics_df.empty:
                    pair_metrics = metrics_df.iloc[0].to_dict()
                else:
                    # Handle empty DataFrame case
                    logger.error(f"No metrics computed for {seg1} vs {seg2}")
                    pair_metrics = {
                        "Jaccard Similarity (%)": 0.0,
                        "Normalized LCS": 0.0,
                        "Fuzzy Similarity": 0.0 if enable_fuzzy else np.nan,
                        "Semantic Similarity": 0.0 if enable_semantic else np.nan
                    }
                
                # Format the results
                text_pair = f"{file1} vs {file2}"
                chapter_num = idx + 1
                
                result_row = {
                    "Text Pair": text_pair,
                    "Chapter": chapter_num,
                    "Jaccard Similarity (%)": pair_metrics["Jaccard Similarity (%)"],  # Already in percentage
                    "Normalized LCS": pair_metrics["Normalized LCS"],
                }
                
                # Add fuzzy similarity if enabled
                if enable_fuzzy:
                    result_row["Fuzzy Similarity"] = pair_metrics["Fuzzy Similarity"]
                    
                # Add semantic similarity if enabled and available
                if enable_semantic and "Semantic Similarity" in pair_metrics:
                    result_row["Semantic Similarity"] = pair_metrics["Semantic Similarity"]
                
                # Convert the dictionary to a DataFrame before appending
                result_df = pd.DataFrame([result_row])
                results.append(result_df)
                
                # Update progressive DataFrame and send update if callback is provided
                progressive_df = pd.concat(results, ignore_index=True)
                
                # Send progressive update if callback is provided
                if progressive_callback is not None:
                    # Determine which metrics are complete in this update
                    current_metrics = []
                    
                    # Always include these basic metrics
                    if "Jaccard Similarity (%)" in progressive_df.columns and MetricType.JACCARD not in completed_metrics:
                        current_metrics.append(MetricType.JACCARD)
                        completed_metrics.append(MetricType.JACCARD)
                        
                    if "Normalized LCS" in progressive_df.columns and MetricType.LCS not in completed_metrics:
                        current_metrics.append(MetricType.LCS)
                        completed_metrics.append(MetricType.LCS)
                    
                    # Add fuzzy if enabled and available
                    if enable_fuzzy and "Fuzzy Similarity" in progressive_df.columns and MetricType.FUZZY not in completed_metrics:
                        current_metrics.append(MetricType.FUZZY)
                        completed_metrics.append(MetricType.FUZZY)
                    
                    # Add semantic if enabled and available
                    if enable_semantic and "Semantic Similarity" in progressive_df.columns and MetricType.SEMANTIC not in completed_metrics:
                        current_metrics.append(MetricType.SEMANTIC)
                        completed_metrics.append(MetricType.SEMANTIC)
                    
                    # Create word counts DataFrame for progressive update
                    word_counts_data = []
                    for seg_id, tokens in segment_tokens.items():
                        filename, chapter_info = seg_id.split('|')
                        chapter_num = int(chapter_info.split()[1])
                        word_counts_data.append({
                            "Filename": filename,
                            "ChapterNumber": chapter_num,
                            "SegmentID": seg_id,
                            "WordCount": len(tokens)
                        })
                    word_counts_df_progressive = pd.DataFrame(word_counts_data)
                    
                    # Send the update
                    try:
                        progressive_callback(
                            progressive_df,
                            word_counts_df_progressive,
                            current_metrics,
                            warning,
                            False  # Not complete yet
                        )
                    except Exception as e:
                        logger.warning(f"Progressive callback error (non-critical): {e}")
                
            except Exception as e:
                logger.error(f"Error computing metrics for {seg1} vs {seg2}: {e}", exc_info=True)
                # Continue with other segmentsparisons instead of failing completely
                continue
    
    # Create the metrics DataFrame
    if results:
        # Results are already DataFrames, so we can concatenate them directly
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
        
    # Send final progressive update if callback is provided
    if progressive_callback is not None:
        try:
            # Send the complete results
            progressive_callback(
                metrics_df,
                word_counts_df,
                completed_metrics,
                warning,
                True  # Computation is complete
            )
        except Exception as e:
            logger.warning(f"Final progressive callback error (non-critical): {e}")
    
    # Return the results
    return metrics_df, word_counts_df, warning
