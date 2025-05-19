import numpy as np
import pandas as pd
from typing import List, Dict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import torch
from .semantic_embedding import generate_embeddings
from .tokenize import tokenize_texts
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from .stopwords_bo import TIBETAN_STOPWORDS, TIBETAN_STOPWORDS_SET
from .stopwords_lite_bo import TIBETAN_STOPWORDS_LITE, TIBETAN_STOPWORDS_LITE_SET

# Attempt to import the Cython-compiled fast_lcs module
try:
    from .fast_lcs import compute_lcs_fast
    USE_CYTHON_LCS = True
except ImportError:
    # print("Cython fast_lcs not found, using Python LCS. For better performance, compile the Cython module.")
    USE_CYTHON_LCS = False

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


def compute_semantic_similarity(
    text1_segment: str,
    text2_segment: str,
    tokens1: List[str],
    tokens2: List[str],
    model,
    device,
    model_type: str = "sentence_transformer",
    use_stopwords: bool = True,
    use_lite_stopwords: bool = False,
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
        raw_text_segment: str, botok_tokens: List[str], model_obj, device_str, model_type: str = "sentence_transformer", use_stopwords: bool = True, use_lite_stopwords: bool = False
    ) -> torch.Tensor | None:
        """Helper to get a single embedding for a text, chunking if necessary for transformer models."""
        if (
            not botok_tokens and not raw_text_segment.strip()
        ):  # Check if effectively empty
            logger.info(
                f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
            )
            return None
            
        # For FastText, we don't need chunking as it processes tokens directly
        if model_type == "fasttext":
            if not raw_text_segment.strip():
                logger.info(
                    f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
                )
                return None
                
            # Pass the raw text, pre-tokenized tokens, and stopword parameters
            embedding = generate_embeddings(
                [raw_text_segment], 
                model_obj, 
                device_str, 
                model_type, 
                tokenize_fn=botok_tokens, 
                use_stopwords=use_stopwords,
                use_lite_stopwords=use_lite_stopwords
            )
            
            if embedding is None or embedding.nelement() == 0:
                logger.error(
                    f"Failed to generate FastText embedding for text: {raw_text_segment[:100]}..."
                )
                return None
            return embedding  # Already [1, embed_dim]
        
        # For transformer models, check if all tokens are stopwords when filtering is enabled
        elif use_stopwords:
            # Filter stopwords to see if any content remains
            filtered_tokens = []
            if use_lite_stopwords:
                from .stopwords_lite_bo import TIBETAN_STOPWORDS_LITE_SET
                filtered_tokens = [token for token in botok_tokens if token not in TIBETAN_STOPWORDS_LITE_SET]
            else:
                from .stopwords_bo import TIBETAN_STOPWORDS_SET
                filtered_tokens = [token for token in botok_tokens if token not in TIBETAN_STOPWORDS_SET]
                
            # If all tokens were filtered out as stopwords, return zero embedding
            if not filtered_tokens:
                logger.info("All tokens in text are stopwords. Returning zero embedding.")
                # Create a zero tensor with the same dimension as the model's output
                # For transformer models, typically 384 or 768 dimensions
                embedding_dim = 384  # Default dimension for MiniLM models
                return torch.zeros(1, embedding_dim)
                
            # Continue with normal processing if content remains after filtering
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
                # Generate embeddings for each chunk using the model
                chunk_embeddings = generate_embeddings(text_chunks, model_obj, device_str, model_type)

                if chunk_embeddings is None or chunk_embeddings.nelement() == 0:
                    logger.error(
                        f"Failed to generate embeddings for chunks of text: {raw_text_segment[:100]}..."
                    )
                    return None
                # Mean pooling of chunk embeddings
                aggregated_embedding = torch.mean(chunk_embeddings, dim=0, keepdim=True)
                return aggregated_embedding
            else:
                # Text is short enough for transformer model, embed raw text directly
                if not raw_text_segment.strip():
                    logger.info(
                        f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
                    )
                    return None

                embedding = generate_embeddings([raw_text_segment], model_obj, device_str, model_type)
                if embedding is None or embedding.nelement() == 0:
                    logger.error(
                        f"Failed to generate embedding for text: {raw_text_segment[:100]}..."
                    )
                    return None
                return embedding  # Already [1, embed_dim]
        else:
            # No stopword filtering, proceed with normal processing
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
                # Generate embeddings for each chunk using the model
                chunk_embeddings = generate_embeddings(text_chunks, model_obj, device_str, model_type)

                if chunk_embeddings is None or chunk_embeddings.nelement() == 0:
                    logger.error(
                        f"Failed to generate embeddings for chunks of text: {raw_text_segment[:100]}..."
                    )
                    return None
                # Mean pooling of chunk embeddings
                aggregated_embedding = torch.mean(chunk_embeddings, dim=0, keepdim=True)
                return aggregated_embedding
            else:
                # Text is short enough for transformer model, embed raw text directly
                if not raw_text_segment.strip():
                    logger.info(
                        f"Text segment is empty or only whitespace: {raw_text_segment[:100]}... Returning None for embedding."
                    )
                    return None

                embedding = generate_embeddings([raw_text_segment], model_obj, device_str, model_type)
                if embedding is None or embedding.nelement() == 0:
                    logger.error(
                        f"Failed to generate embedding for text: {raw_text_segment[:100]}..."
                    )
                    return None
                return embedding  # Already [1, embed_dim]

    try:
        # Pass raw text and its pre-computed botok tokens with stopword preference
        embedding1 = _get_aggregated_embedding(text1_segment, tokens1, model, device, model_type, use_stopwords, use_lite_stopwords)
        embedding2 = _get_aggregated_embedding(text2_segment, tokens2, model, device, model_type, use_stopwords, use_lite_stopwords)

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

        # Check if both embeddings are zero vectors (which happens when all tokens are stopwords)
        if np.all(embedding1.numpy() == 0) and np.all(embedding2.numpy() == 0):
            # If both texts contain only stopwords, return 0 similarity
            return 0.0
            
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
    texts: Dict[str, str], model=None, device=None, enable_semantic: bool = True, 
    model_type: str = "sentence_transformer", use_stopwords: bool = True,
    use_lite_stopwords: bool = False
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
                      and 'Semantic Similarity'.
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
        try:
            # Using a dummy tokenizer and preprocessor as input is already tokenized (as space-separated strings)
            # and we don't want further case changes or token modifications for Tibetan.
            
            # Select appropriate stopwords list based on user preference
            if use_stopwords:
                # Choose between regular and lite stopwords list
                if use_lite_stopwords:
                    stopwords_to_use = TIBETAN_STOPWORDS_LITE
                else:
                    stopwords_to_use = TIBETAN_STOPWORDS
            else:
                # If stopwords are disabled, use an empty list
                stopwords_to_use = []
                
            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x.split(),
                preprocessor=lambda x: x,
                token_pattern=None,
                stop_words=stopwords_to_use
            )
            tfidf_matrix = vectorizer.fit_transform(corpus_for_tfidf)
            # Calculate pairwise cosine similarity on the TF-IDF matrix
            # This gives a square matrix where cosine_sim_matrix[i, j] is the similarity between doc i and doc j
            cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        except ValueError as e:
            if "empty vocabulary" in str(e):
                # If vocabulary is empty after stopword removal, create a zero matrix
                n = len(corpus_for_tfidf)
                cosine_sim_matrix = np.zeros((n, n))
            else:
                # Re-raise other ValueError
                raise
    else:
        # Handle case with no texts or all empty texts
        n = len(files) if files else 0
        cosine_sim_matrix = np.zeros((n, n))

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

        # Check if both texts only contain stopwords
        both_only_stopwords = len(words1_jaccard) == 0 and len(words2_jaccard) == 0

        jaccard = (
            len(set(words1_jaccard) & set(words2_jaccard)) / len(set(words1_jaccard) | set(words2_jaccard))
            if set(words1_jaccard) | set(words2_jaccard)  # Ensure denominator is not zero
            else 0.0
        )
        # LCS uses raw tokens (words1_raw, words2_raw) to provide a complementary metric.
        # Semantic similarity also uses raw text and its botok tokens for chunking decisions.
        jaccard_percent = jaccard * 100.0
        norm_lcs = compute_normalized_lcs(words1_raw, words2_raw)

        # Semantic Similarity Calculation
        if enable_semantic:
            # Pass raw texts and their pre-computed botok tokens
            semantic_sim = compute_semantic_similarity(
                texts[f1], texts[f2], words1_raw, words2_raw, model, device, model_type, use_stopwords, use_lite_stopwords
            )
        else:
            semantic_sim = np.nan
        results.append(
            {
                "Text Pair": f"{f1} vs {f2}",
                "Jaccard Similarity (%)": jaccard_percent,
                "Normalized LCS": norm_lcs,
                # Pass tokens1 and tokens2 to compute_semantic_similarity
                "Semantic Similarity": semantic_sim,
                "TF-IDF Cosine Sim": (
                    0.0 if both_only_stopwords else
                    cosine_sim_matrix[i, j]
                    if cosine_sim_matrix.size > 0
                    and i < cosine_sim_matrix.shape[0]
                    and j < cosine_sim_matrix.shape[1]
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(results)
