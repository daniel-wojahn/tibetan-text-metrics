"""
FastText embedding module for Tibetan text analysis.
This module provides functionality to train and use FastText models for Tibetan text embeddings.
"""

import os
import numpy as np
import fasttext
import logging
from typing import List, Optional
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "tibetan_fasttext.bin")
DEFAULT_CORPUS_PATH = os.path.join(DEFAULT_MODEL_DIR, "tibetan_corpus.txt")

# Official Facebook FastText Tibetan model
FACEBOOK_TIBETAN_MODEL_REPO = "facebook/fasttext-bo-vectors"
FACEBOOK_TIBETAN_MODEL_FILE = "model.bin"

# Default parameters for FastText - more conservative values for stability
DEFAULT_DIM = 100       # Standard dimension size
DEFAULT_EPOCH = 5       # Standard number of epochs
DEFAULT_MIN_COUNT = 5   # Standard minimum count
DEFAULT_WINDOW = 5      # Context window size
DEFAULT_MINN = 3        # Minimum length of char n-gram
DEFAULT_MAXN = 6        # Maximum length of char n-gram
DEFAULT_NEG = 5         # Number of negatives in negative sampling


def ensure_dir_exists(dir_path: str) -> None:
    """Ensure that the directory exists, creating it if necessary."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def tokenize_tibetan_text(text: str) -> List[str]:
    """
    Tokenize Tibetan text using simple whitespace tokenization for stability.
    
    Args:
        text: Tibetan text string to tokenize
        
    Returns:
        List of tokens
    """
    if not text or not text.strip():
        return []
    
    # Simple whitespace tokenization for stability
    tokens = text.split()
    return [t for t in tokens if t.strip()]


def prepare_corpus_file(texts: List[str], output_path: str = DEFAULT_CORPUS_PATH) -> str:
    """
    Prepare a corpus file from a list of texts for FastText training with Tibetan-specific preprocessing.
    
    Args:
        texts: List of Tibetan text strings
        output_path: Path where to save the corpus file
        
    Returns:
        Path to the created corpus file
    """
    ensure_dir_exists(os.path.dirname(output_path))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            if not text.strip():
                continue
                
            # Clean and preprocess text
            cleaned_text = text.replace('\n', ' ').strip()
            
            # Tokenize using Tibetan-specific tokenization
            tokens = tokenize_tibetan_text(cleaned_text)
            
            # Join tokens with spaces for FastText training
            # This preserves syllable boundaries which is important for Tibetan
            processed_text = ' '.join(tokens)
            
            if processed_text:
                f.write(processed_text + '\n')
    
    logger.info("Corpus file created at %s with Tibetan-specific preprocessing", output_path)
    return output_path


def train_fasttext_model(
    corpus_path: str = DEFAULT_CORPUS_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    dim: int = DEFAULT_DIM,
    epoch: int = DEFAULT_EPOCH,
    min_count: int = DEFAULT_MIN_COUNT,
    window: int = DEFAULT_WINDOW,
    minn: int = DEFAULT_MINN,
    maxn: int = DEFAULT_MAXN,
    neg: int = DEFAULT_NEG,
    model_type: str = "skipgram"
) -> fasttext.FastText._FastText:
    """
    Train a FastText model on Tibetan corpus using optimized parameters.
    
    Args:
        corpus_path: Path to the corpus file
        model_path: Path where to save the trained model
        dim: Embedding dimension (default: 300)
        epoch: Number of training epochs (default: 15)
        min_count: Minimum count of words (default: 3)
        window: Size of context window (default: 5)
        minn: Minimum length of char n-gram (default: 3)
        maxn: Maximum length of char n-gram (default: 6)
        neg: Number of negatives in negative sampling (default: 10)
        model_type: FastText model type ('skipgram' or 'cbow')
        
    Returns:
        Trained FastText model
    """
    ensure_dir_exists(os.path.dirname(model_path))
    
    logger.info("Training FastText model with %s, dim=%d, epoch=%d, window=%d, minn=%d, maxn=%d...", 
               model_type, dim, epoch, window, minn, maxn)
    
    # Preprocess corpus for Tibetan - segment by syllable points
    # This is based on research showing syllable segmentation works better for Tibetan
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Ensure syllable segmentation by adding spaces after Tibetan syllable markers (if not already present)
        # This improves model quality for Tibetan text according to research
        processed_content = content.replace('་', '་ ')
        
        # Write back the processed content
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        logger.info("Preprocessed corpus with syllable segmentation for Tibetan text")
    except Exception as e:
        logger.warning("Could not preprocess corpus for syllable segmentation: %s", str(e))
    
    # Train the model with optimized parameters
    model = fasttext.train_unsupervised(
        corpus_path,
        model=model_type,
        dim=dim,
        epoch=epoch,
        minCount=min_count,
        ws=window,        # Context window size
        minn=minn,        # Min length of char ngram
        maxn=maxn,        # Max length of char ngram
        neg=neg,          # Number of negatives sampled
        wordNgrams=2,     # Max length of word ngrams
        loss='ns',        # Use negative sampling
        bucket=2000000,   # Number of buckets
        thread=4,         # Number of threads
        t=1e-4,          # Sampling threshold
        lrUpdateRate=100  # Change the rate of updates for the learning rate
    )
    
    # Save the model
    model.save_model(model_path)
    logger.info("FastText model trained and saved to %s with optimized parameters for Tibetan text", model_path)
    
    return model


def load_fasttext_model(model_path: str = DEFAULT_MODEL_PATH, use_facebook_model: bool = True) -> Optional[fasttext.FastText._FastText]:
    """
    Load a pre-trained FastText model. By default, tries to load the official Facebook Tibetan model.
    If that fails or if use_facebook_model is False, falls back to the local model.
    
    Args:
        model_path: Path to the local FastText model (fallback)
        use_facebook_model: Whether to try loading the official Facebook Tibetan model first
        
    Returns:
        FastText model or None if loading fails
    """
    try:
        # First try to load the official Facebook Tibetan model if requested
        if use_facebook_model:
            try:
                logger.info("Attempting to download and load official Facebook FastText Tibetan model")
                facebook_model_path = hf_hub_download(
                    repo_id=FACEBOOK_TIBETAN_MODEL_REPO, 
                    filename=FACEBOOK_TIBETAN_MODEL_FILE,
                    cache_dir=DEFAULT_MODEL_DIR
                )
                logger.info("Loading official Facebook FastText Tibetan model from %s", facebook_model_path)
                return fasttext.load_model(facebook_model_path)
            except Exception as e:
                logger.warning("Could not load official Facebook FastText Tibetan model: %s", str(e))
                logger.info("Falling back to local model")
        
        # Fall back to local model
        if os.path.exists(model_path):
            logger.info("Loading local FastText model from %s", model_path)
            return fasttext.load_model(model_path)
        else:
            logger.warning("Model path %s does not exist", model_path)
            return None
    except Exception as e:
        logger.error("Error loading FastText model: %s", str(e))
        return None


def get_word_embedding(word: str, model: fasttext.FastText._FastText) -> np.ndarray:
    """
    Get embedding for a single word.
    
    Args:
        word: Input word
        model: FastText model
        
    Returns:
        Word embedding vector
    """
    return model.get_word_vector(word)


def get_text_embedding(
    text: str, 
    model: fasttext.FastText._FastText,
    tokenize_fn=None,
    use_stopwords: bool = True,
    stopwords_set=None,
    use_tfidf_weighting: bool = False,  # Disabled by default for stability
    corpus_token_freq=None
) -> np.ndarray:
    """
    Get embedding for a text by averaging word vectors with optional TF-IDF weighting.
    
    Args:
        text: Input text
        model: FastText model
        tokenize_fn: Optional tokenization function or pre-tokenized list
        use_stopwords: Whether to filter out stopwords before computing embeddings
        stopwords_set: Set of stopwords to filter out (if use_stopwords is True)
        use_tfidf_weighting: Whether to use TF-IDF weighting for averaging word vectors
        corpus_token_freq: Dictionary of token frequencies across corpus (required for TF-IDF)
        
    Returns:
        Text embedding vector
    """
    if not text.strip():
        return np.zeros(model.get_dimension())
    
    # Handle tokenization
    if tokenize_fn is None:
        # Simple whitespace tokenization as fallback
        tokens = text.split()
    elif isinstance(tokenize_fn, list):
        # If tokenize_fn is already a list of tokens, use it directly
        tokens = tokenize_fn
    else:
        # If tokenize_fn is a function, call it
        tokens = tokenize_fn(text)
    
    # Filter out stopwords if enabled and stopwords_set is provided
    if use_stopwords and stopwords_set is not None:
        tokens = [token for token in tokens if token not in stopwords_set]
    
    # If all tokens were filtered out as stopwords, return zero vector
    if not tokens:
        return np.zeros(model.get_dimension())
    
    # Filter out empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    if not tokens:
        return np.zeros(model.get_dimension())
        # Use simple averaging instead of TF-IDF weighting for more stable results
    if False and use_tfidf_weighting and corpus_token_freq is not None:  # Disabled for now
        # Calculate term frequencies in this document
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Total number of documents in corpus (estimated from corpus_token_freq)
        num_docs = max(corpus_token_freq.values()) if corpus_token_freq else 1
        
        # Calculate TF-IDF weights
        weights = []
        for token in tokens:
            # Term frequency in this document
            tf = token_counts[token] / len(tokens)
            
            # Inverse document frequency
            token_doc_freq = corpus_token_freq.get(token, 1)  # Default to 1 if not in corpus
            idf = np.log(num_docs / max(token_doc_freq, 1))  # Prevent division by zero
            
            # TF-IDF weight
            weights.append(tf * idf)
        
        # Normalize weights to sum to 1
        total_weight = sum(weights) if weights else 1
        if total_weight > 0:  # Prevent division by zero
            weights = [1.0 / len(tokens) for _ in tokens]  # Use equal weights for stability
        else:
            # If all weights are zero, use equal weights
            weights = [1.0 / len(tokens) for _ in tokens]
        
        # Get vectors for each token and apply weights
        vectors = [model.get_word_vector(token) for token in tokens]
        weighted_vectors = [w * v for w, v in zip(weights, vectors)]
        
        # Sum the weighted vectors
        return np.sum(weighted_vectors, axis=0)
    else:
        # Simple averaging if TF-IDF is not enabled or corpus frequencies not provided
        vectors = [model.get_word_vector(token) for token in tokens]
        return np.mean(vectors, axis=0)


def get_batch_embeddings(
    texts: List[str], 
    model: fasttext.FastText._FastText,
    tokenize_fn=None,
    use_stopwords: bool = True,
    stopwords_set=None,
    use_tfidf_weighting: bool = False,  # Disabled by default for stability
    corpus_token_freq=None
) -> np.ndarray:
    """
    Get embeddings for a batch of texts with optional TF-IDF weighting.
    
    Args:
        texts: List of input texts
        model: FastText model
        tokenize_fn: Optional tokenization function or pre-tokenized list of tokens
        use_stopwords: Whether to filter out stopwords before computing embeddings
        stopwords_set: Set of stopwords to filter out (if use_stopwords is True)
        use_tfidf_weighting: Whether to use TF-IDF weighting for averaging word vectors
        corpus_token_freq: Dictionary of token frequencies across corpus (required for TF-IDF)
        
    Returns:
        Array of text embedding vectors
    """
    # If corpus_token_freq is not provided but TF-IDF is requested, build it from the texts
    if False and use_tfidf_weighting and corpus_token_freq is None:  # Disabled for now
        logger.info("Building corpus token frequency dictionary for TF-IDF weighting")
        corpus_token_freq = {}
        
        # Tokenize all texts and count document frequencies
        for text in texts:
            if not text.strip():
                continue
                
            # Handle tokenization
            if tokenize_fn is None:
                tokens = text.split()
            elif isinstance(tokenize_fn, list):
                tokens = tokenize_fn
            else:
                tokens = tokenize_fn(text)
                
            # Filter stopwords if needed
            if use_stopwords and stopwords_set is not None:
                tokens = [token for token in tokens if token not in stopwords_set]
                
            # Count unique tokens in this document
            unique_tokens = set(token for token in tokens if token.strip())
            for token in unique_tokens:
                corpus_token_freq[token] = corpus_token_freq.get(token, 0) + 1
    
    # Get embeddings for each text
    embeddings = []
    for text in texts:
        embedding = get_text_embedding(
            text, 
            model, 
            tokenize_fn, 
            use_stopwords, 
            stopwords_set,
            use_tfidf_weighting,
            corpus_token_freq
        )
        embeddings.append(embedding)
    
    return np.array(embeddings)


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(embedding1, embedding2) / (norm1 * norm2)


def compute_pairwise_similarities(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings
        
    Returns:
        Matrix of pairwise similarities
    """
    # Normalize all vectors for faster computation
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Replace zero norms with 1 to avoid division by zero
    norms1 = np.where(norms1 == 0, 1, norms1)
    norms2 = np.where(norms2 == 0, 1, norms2)
    
    normalized1 = embeddings1 / norms1
    normalized2 = embeddings2 / norms2
    
    # Compute similarities
    return np.dot(normalized1, normalized2.T)
