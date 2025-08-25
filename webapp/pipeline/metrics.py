import numpy as np
import pandas as pd
from typing import List, Dict, Union
from itertools import combinations

from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz
from .hf_embedding import generate_embeddings as generate_hf_embeddings
from .stopwords_bo import TIBETAN_STOPWORDS_SET
from .stopwords_lite_bo import TIBETAN_STOPWORDS_LITE_SET

import logging


# Attempt to import the Cython-compiled fast_lcs module
try:
    from .fast_lcs import compute_lcs_fast
    USE_CYTHON_LCS = True
except ImportError:
    # print("Cython fast_lcs not found, using Python LCS. For better performance, compile the Cython module.")
    USE_CYTHON_LCS = False

logger = logging.getLogger(__name__)




def compute_normalized_lcs(words1: List[str], words2: List[str]) -> float:
    # Calculate m and n (lengths) here, so they are available for normalization
    # regardless of which LCS implementation is used.
    m, n = len(words1), len(words2)

    if USE_CYTHON_LCS:
        # Use the Cython-compiled version if available
        lcs_length = compute_lcs_fast(words1, words2)
    else:
        # Fallback to pure Python implementation
        # m, n = len(words1), len(words2) # Moved to the beginning of the function
        # Using numpy array for dp table can be slightly faster than list of lists for large inputs
        # but the primary bottleneck is the Python loop itself compared to Cython.
        dp = np.zeros((m + 1, n + 1), dtype=np.int32)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
        lcs_length = int(dp[m, n])
    avg_length = (m + n) / 2
    return lcs_length / avg_length if avg_length > 0 else 0.0


def compute_fuzzy_similarity(words1: List[str], words2: List[str], method: str = 'token_set') -> float:
    """
    Computes fuzzy string similarity between two lists of words using TheFuzz.
    
    Args:
        words1: First list of tokens
        words2: Second list of tokens
        method: The fuzzy matching method to use:
                'token_set' - Order-independent token matching (default)
                'token_sort' - Order-normalized token matching
                'partial' - Best partial token matching
                'ratio' - Simple ratio matching
    
    Returns:
        float: Fuzzy similarity score between 0.0 and 1.0
    """
    if not words1 or not words2:
        return 0.0
        
    # Join tokens into strings for fuzzy matching
    text1 = " ".join(words1)
    text2 = " ".join(words2)
    
    # Apply the selected fuzzy matching method
    if method == 'token_set':
        # Best for texts with different word orders and partial overlaps
        score = fuzz.token_set_ratio(text1, text2)
    elif method == 'token_sort':
        # Good for texts with different word orders but similar content
        score = fuzz.token_sort_ratio(text1, text2)
    elif method == 'partial':
        # Best for finding shorter strings within longer ones
        score = fuzz.partial_ratio(text1, text2)
    else:  # 'ratio'
        # Simple Levenshtein distance ratio
        score = fuzz.ratio(text1, text2)
    
    # Convert score from 0-100 scale to 0-1 scale
    return score / 100.0



def compute_semantic_similarity(
    text1_segment: str,
    text2_segment: str,
    tokens1: List[str],
    tokens2: List[str],
    model,
    batch_size: int = 32,
    show_progress_bar: bool = False
) -> float:
    """Computes semantic similarity using a Sentence Transformer model only."""

    if model is None:
        logger.warning(
            "Embedding model not available for semantic similarity. Skipping calculation."
        )
        return np.nan

    if not text1_segment or not text2_segment:
        logger.info(
            "One or both texts are empty for semantic similarity. Returning 0.0."
        )
        return 0.0

    def _get_aggregated_embedding(
        raw_text_segment: str,
        _botok_tokens: List[str],
        model_obj,
        batch_size_param: int,
        show_progress_bar_param: bool
    ) -> Union[np.ndarray, None]:
        """Helper to get a single embedding for a text using Sentence Transformers."""
        if not raw_text_segment.strip():
            logger.info(
                f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
            )
            return None
            
        embedding = generate_hf_embeddings(
            texts=[raw_text_segment],
            model=model_obj,
            batch_size=batch_size_param,
            show_progress_bar=show_progress_bar_param
        )
        
        if embedding is None or embedding.size == 0: 
            logger.error(
                f"Failed to generate embedding for text: {raw_text_segment[:100]}..."
            )
            return None
        return embedding

    try:
        # Pass all relevant parameters to _get_aggregated_embedding
        emb1 = _get_aggregated_embedding(text1_segment, tokens1, model, batch_size, show_progress_bar)
        emb2 = _get_aggregated_embedding(text2_segment, tokens2, model, batch_size, show_progress_bar)

        if emb1 is None or emb2 is None or emb1.size == 0 or emb2.size == 0:
            logger.error(
                "Failed to obtain one or both embeddings for semantic similarity."
            )
            return np.nan

        # Ensure embeddings are numpy arrays (should be, but defensive)
        if not isinstance(emb1, np.ndarray):
            emb1 = np.array(emb1)
        if not isinstance(emb2, np.ndarray):
            emb2 = np.array(emb2)

        # Handle cases where embeddings are all zeros
        if np.all(emb1 == 0) and np.all(emb2 == 0):
            logger.info("Both embeddings are zero. Semantic similarity is 0.0.")
            return 0.0
        if np.all(emb1 == 0) or np.all(emb2 == 0):
            logger.info("One of the embeddings is zero. Semantic similarity is 0.0.")
            return 0.0
        
        # Handle NaN or Inf in embeddings
        if np.isnan(emb1).any() or np.isinf(emb1).any() or \
           np.isnan(emb2).any() or np.isinf(emb2).any():
            logger.warning("NaN or Inf found in embeddings. Semantic similarity set to 0.0.")
            return 0.0

        # Ensure embeddings are 2D for cosine_similarity: [1, dim]
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        similarity_score = cosine_similarity(emb1, emb2)[0][0]
        
        return max(0.0, float(similarity_score))

    except Exception as e:
        safe_text1 = str(text1_segment)[:100] if text1_segment is not None else "N/A"
        safe_text2 = str(text2_segment)[:100] if text2_segment is not None else "N/A"
        logger.error(
            f"Error during semantic similarity calculation:\nText1: {safe_text1}...\nText2: {safe_text2}...\nError: {e}"
        )
        logger.exception("Traceback for semantic similarity calculation error:")
        return np.nan


def compute_all_metrics(
    texts: Dict[str, str],
    token_lists: Dict[str, List[str]],
    model=None,
    enable_semantic: bool = True,
    enable_fuzzy: bool = True,
    fuzzy_method: str = 'token_set',
    use_stopwords: bool = True,
    use_lite_stopwords: bool = False,
    batch_size: int = 32,
    show_progress_bar: bool = False
) -> pd.DataFrame:
    """
    Computes all selected similarity metrics between pairs of texts.

    Args:
        texts (Dict[str, str]): A dictionary where keys are text identifiers (e.g., filenames or segment IDs)
                               and values are the text content strings.
        token_lists (Dict[str, List[str]]): Pre-tokenized text for each text identifier.
        model (SentenceTransformer, optional): The pre-loaded sentence transformer model.
                                              Defaults to None.
        enable_semantic (bool): Whether to compute semantic similarity. Defaults to True.
        enable_fuzzy (bool): Whether to compute fuzzy string similarity. Defaults to True.
        fuzzy_method (str): The fuzzy matching method to use ('token_set', 'token_sort', 'partial', 'ratio').
                           Defaults to 'token_set'.
        use_stopwords (bool): Whether to filter stopwords for Jaccard similarity. Defaults to True.
        use_lite_stopwords (bool): Whether to use the lite version of stopwords. Defaults to False.
        batch_size (int): Batch size for semantic similarity computation. Defaults to 32.
        show_progress_bar (bool): Whether to show progress bar for semantic similarity. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame where each row contains the metrics for a pair of texts,
                      including 'Text Pair', 'Jaccard Similarity (%)', 'Normalized LCS',
                      'Fuzzy Similarity', and 'Semantic Similarity'.
    """
    files = list(texts.keys())
    results = []
    corpus_for_sklearn_tfidf = []  # Kept for potential future use

    for fname, content in texts.items():
        # Use the pre-computed tokens from the token_lists dictionary
        current_tokens_for_file = token_lists.get(fname, [])
        corpus_for_sklearn_tfidf.append(" ".join(current_tokens_for_file) if current_tokens_for_file else "")

        
    for i, j in combinations(range(len(files)), 2):
        f1, f2 = files[i], files[j]
        words1_raw, words2_raw = token_lists[f1], token_lists[f2]

        # Select appropriate stopwords set based on user preference
        if use_stopwords:
            # Choose between regular and lite stopwords sets
            if use_lite_stopwords:
                stopwords_set_to_use = TIBETAN_STOPWORDS_LITE_SET
            else:
                stopwords_set_to_use = TIBETAN_STOPWORDS_SET
        else:
            # If stopwords are disabled, use an empty set
            stopwords_set_to_use = set()
            
        # Filter stopwords for Jaccard calculation
        words1_jaccard = [word for word in words1_raw if word not in stopwords_set_to_use]
        words2_jaccard = [word for word in words2_raw if word not in stopwords_set_to_use]

        jaccard = (
            len(set(words1_jaccard) & set(words2_jaccard)) / len(set(words1_jaccard) | set(words2_jaccard))
            if set(words1_jaccard) | set(words2_jaccard)  # Ensure denominator is not zero
            else 0.0
        )
        # LCS uses raw tokens (words1_raw, words2_raw) to provide a complementary metric.
        # Semantic similarity also uses raw text and its botok tokens for chunking decisions.
        jaccard_percent = jaccard * 100.0
        norm_lcs = compute_normalized_lcs(words1_raw, words2_raw)
        
        # Fuzzy Similarity Calculation
        if enable_fuzzy:
            fuzzy_sim = compute_fuzzy_similarity(words1_jaccard, words2_jaccard, method=fuzzy_method)
        else:
            fuzzy_sim = np.nan

        # Semantic Similarity Calculation
        if enable_semantic:
            # Pass raw texts and their pre-computed botok tokens
            semantic_sim = compute_semantic_similarity(
                texts[f1], texts[f2], words1_raw, words2_raw, model,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar
            )
        else:
            semantic_sim = np.nan
        results.append(
            {
                "Text Pair": f"{f1} vs {f2}",
                "Jaccard Similarity (%)": jaccard_percent,
                "Normalized LCS": norm_lcs,
                "Fuzzy Similarity": fuzzy_sim,
                "Semantic Similarity": semantic_sim
            }
        )
    return pd.DataFrame(results)
