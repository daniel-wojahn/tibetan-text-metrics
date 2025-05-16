import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import torch
from .semantic_embedding import generate_embeddings
from .tokenize import tokenize_texts
import logging
from numba import njit
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_CHUNK = 500  # Max tokens (words via botok) per chunk
CHUNK_OVERLAP = 50  # Number of tokens to overlap between chunks


def _chunk_text(
    original_text_content: str,
    tokens: List[str],
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Splits a list of tokens into chunks and reconstructs text segments from these token chunks.
    The reconstructed text segments are intended for embedding models.
    Args:
        original_text_content (str): The original raw text string. Used if no chunking is needed.
        tokens (List[str]): The list of botok tokens for the original_text_content.
        max_chunk_tokens (int): Maximum number of botok tokens per chunk.
        overlap_tokens (int): Number of botok tokens to overlap between chunks.

    Returns:
        List[str]: A list of text strings, where each string is a chunk.
    """
    if (
        not tokens
    ):  # Handles empty or whitespace-only original text that led to no tokens
        return [original_text_content] if original_text_content.strip() else []

    if len(tokens) <= max_chunk_tokens:
        # If not chunking, return the original text content directly, as per MEMORY[a777e6ad-11c4-4b90-8e6e-63a923a94432]
        # The memory states raw text segments are passed directly to the model.
        # Joining tokens here would alter spacing, etc.
        return [original_text_content]

    reconstructed_text_chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_chunk_tokens, len(tokens))
        current_chunk_botok_tokens = tokens[start_idx:end_idx]
        # Reconstruct the text chunk by joining the botok tokens. This is an approximation.
        # The semantic model's internal tokenizer will handle this string.
        reconstructed_text_chunks.append(" ".join(current_chunk_botok_tokens))

        if end_idx == len(tokens):
            break

        next_start_idx = start_idx + max_chunk_tokens - overlap_tokens
        if next_start_idx <= start_idx:
            next_start_idx = start_idx + 1
        start_idx = next_start_idx

    return reconstructed_text_chunks


@njit
def compute_normalized_lcs(words1: List[str], words2: List[str]) -> float:
    m, n = len(words1), len(words2)
    if m == 0 or n == 0:
        return 0.0
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


def compute_semantic_similarity(
    text1_segment: str,
    text2_segment: str,
    tokens1: List[str],
    tokens2: List[str],
    model,
    device,
) -> float:
    """Computes semantic similarity using a sentence transformer model, with chunking for long texts."""
    if model is None or device is None:
        logger.warning(
            "Semantic similarity model or device not available. Skipping calculation."
        )
        return np.nan  # Return NaN if model isn't loaded

    if not text1_segment or not text2_segment:
        logger.info(
            "One or both texts are empty for semantic similarity. Returning 0.0."
        )
        return 0.0  # Or np.nan, depending on desired behavior for empty inputs

    def _get_aggregated_embedding(
        raw_text_segment: str, botok_tokens: List[str], model_obj, device_str
    ) -> torch.Tensor | None:
        """Helper to get a single embedding for a text, chunking if necessary."""
        if (
            not botok_tokens and not raw_text_segment.strip()
        ):  # Check if effectively empty
            logger.info(
                f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
            )
            return None

        if len(botok_tokens) > MAX_TOKENS_PER_CHUNK:
            logger.info(
                f"Text segment with ~{len(botok_tokens)} tokens exceeds {MAX_TOKENS_PER_CHUNK}, chunking {raw_text_segment[:30]}..."
            )
            # Pass the original raw text and its pre-computed botok tokens to _chunk_text
            text_chunks = _chunk_text(
                raw_text_segment, botok_tokens, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP
            )
            if not text_chunks:
                logger.warning(
                    f"Chunking resulted in no chunks for segment: {raw_text_segment[:100]}..."
                )
                return None

            logger.info(
                f"Generated {len(text_chunks)} chunks for segment: {raw_text_segment[:30]}..."
            )
            chunk_embeddings = generate_embeddings(text_chunks, model_obj, device_str)

            if chunk_embeddings is None or chunk_embeddings.nelement() == 0:
                logger.error(
                    f"Failed to generate embeddings for chunks of text: {raw_text_segment[:100]}..."
                )
                return None
            # Mean pooling of chunk embeddings
            aggregated_embedding = torch.mean(chunk_embeddings, dim=0, keepdim=True)
            return aggregated_embedding
        else:
            # Text is short enough, embed raw text directly as per MEMORY[a777e6ad-11c4-4b90-8e6e-63a923a94432]
            if not raw_text_segment.strip():
                logger.info(
                    f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
                )
                return None

            embedding = generate_embeddings([raw_text_segment], model_obj, device_str)
            if embedding is None or embedding.nelement() == 0:
                logger.error(
                    f"Failed to generate embedding for text: {raw_text_segment[:100]}..."
                )
                return None
            return embedding  # Already [1, embed_dim]

    try:
        # Pass raw text and its pre-computed botok tokens
        embedding1 = _get_aggregated_embedding(text1_segment, tokens1, model, device)
        embedding2 = _get_aggregated_embedding(text2_segment, tokens2, model, device)

        if (
            embedding1 is None
            or embedding2 is None
            or embedding1.nelement() == 0
            or embedding2.nelement() == 0
        ):
            logger.error(
                "Failed to obtain one or both aggregated embeddings for semantic similarity."
            )
            return np.nan

        # Cosine similarity expects 2D arrays, embeddings are [1, embed_dim] and on CPU
        similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
        return float(similarity[0][0])
    except Exception as e:
        logger.error(
            f"Error computing semantic similarity with chunking:\nText1: '{text1_segment[:100]}...'\nText2: '{text2_segment[:100]}...'\nError: {e}",
            exc_info=True,
        )
        return np.nan


def compute_all_metrics(
    texts: Dict[str, str], model=None, device=None, enable_semantic: bool = True
) -> pd.DataFrame:
    """
    Computes all selected similarity metrics between pairs of texts.

    Args:
        texts (Dict[str, str]): A dictionary where keys are text identifiers (e.g., filenames or segment IDs)
                               and values are the text content strings.
        model (SentenceTransformer, optional): The pre-loaded sentence transformer model.
                                              Defaults to None.
        device (str, optional): The device the model is on ('cuda' or 'cpu').
                                Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame where each row contains the metrics for a pair of texts,
                      including 'Text Pair', 'Jaccard Similarity (%)', 'Normalized LCS',
                      and 'Semantic Similarity (BuddhistNLP)'.
    """
    files = list(texts.keys())
    results = []
    # Prepare token lists (always use tokenize_texts for raw Unicode)
    token_lists = {}
    corpus_for_tfidf = []  # For storing space-joined tokens for TF-IDF

    for fname, content in texts.items():
        tokenized_content = tokenize_texts([content])  # Returns a list of lists
        if tokenized_content and tokenized_content[0]:
            token_lists[fname] = tokenized_content[0]
        else:
            token_lists[fname] = []
        # Regardless of whether tokenized_content[0] exists, prepare entry for TF-IDF corpus
        # If tokens exist, join them; otherwise, use an empty string for that document
        corpus_for_tfidf.append(
            " ".join(token_lists[fname])
            if fname in token_lists and token_lists[fname]
            else ""
        )

    # TF-IDF Vectorization and Cosine Similarity Calculation
    if corpus_for_tfidf:
        # Using a dummy tokenizer and preprocessor as input is already tokenized (as space-separated strings)
        # and we don't want further case changes or token modifications for Tibetan.
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(), preprocessor=lambda x: x, token_pattern=None
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_for_tfidf)
        # Calculate pairwise cosine similarity on the TF-IDF matrix
        # This gives a square matrix where cosine_sim_matrix[i, j] is the similarity between doc i and doc j
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    else:
        # Handle case with no texts or all empty texts
        cosine_sim_matrix = np.array(
            [[]]
        )  # Or some other appropriate empty/default structure

    for i, j in combinations(range(len(files)), 2):
        f1, f2 = files[i], files[j]
        words1, words2 = token_lists[f1], token_lists[f2]
        jaccard = (
            len(set(words1) & set(words2)) / len(set(words1) | set(words2))
            if set(words1) | set(words2)
            else 0.0
        )
        jaccard_percent = jaccard * 100.0
        norm_lcs = compute_normalized_lcs(words1, words2)

        # Semantic Similarity Calculation
        if enable_semantic:
            # Pass raw texts and their pre-computed botok tokens
            semantic_sim = compute_semantic_similarity(
                texts[f1], texts[f2], words1, words2, model, device
            )
        else:
            semantic_sim = np.nan
        results.append(
            {
                "Text Pair": f"{f1} vs {f2}",
                "Jaccard Similarity (%)": jaccard_percent,
                "Normalized LCS": norm_lcs,
                # Pass tokens1 and tokens2 to compute_semantic_similarity
                "Semantic Similarity (BuddhistNLP)": semantic_sim,
                "TF-IDF Cosine Sim": (
                    cosine_sim_matrix[i, j]
                    if cosine_sim_matrix.size > 0
                    and i < cosine_sim_matrix.shape[0]
                    and j < cosine_sim_matrix.shape[1]
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(results)
