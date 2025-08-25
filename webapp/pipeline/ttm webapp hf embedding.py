import logging
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Cache for loaded models with version information
_model_cache = {}

# Model version mapping
MODEL_VERSIONS = {
    "sentence-transformers/LaBSE": "v1.0",
    "intfloat/e5-base-v2": "v1.0",
}

def get_model(model_id: str) -> Tuple[Optional[SentenceTransformer], Optional[str]]:
    """
    Loads a SentenceTransformer model from the Hugging Face Hub with version tracking.

    Args:
        model_id (str): The identifier for the model to load (e.g., 'sentence-transformers/LaBSE').

    Returns:
        Tuple[Optional[SentenceTransformer], Optional[str]]: A tuple containing the loaded model and its type ('sentence-transformer'),
                                                              or (None, None) if loading fails.
    """
    # Include version information in cache key
    model_version = MODEL_VERSIONS.get(model_id, "unknown")
    cache_key = f"{model_id}@{model_version}"
    
    if cache_key in _model_cache:
        logger.info(f"Returning cached model: {model_id} (version: {model_version})")
        return _model_cache[cache_key], "sentence-transformer"

    logger.info(f"Loading SentenceTransformer model: {model_id} (version: {model_version})")
    try:
        model = SentenceTransformer(model_id)
        _model_cache[cache_key] = model
        logger.info(f"Model '{model_id}' (version: {model_version}) loaded successfully.")
        return model, "sentence-transformer"
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{model_id}': {e}", exc_info=True)
        return None, None

def generate_embeddings(
    texts: List[str], 
    model: SentenceTransformer, 
    batch_size: int = 32, 
    show_progress_bar: bool = False
) -> np.ndarray:
    """
    Generates embeddings for a list of texts using a SentenceTransformer model.

    Args:
        texts (list[str]): A list of texts to embed.
        model (SentenceTransformer): The loaded SentenceTransformer model.
        batch_size (int): The batch size for encoding.
        show_progress_bar (bool): Whether to display a progress bar.

    Returns:
        np.ndarray: A numpy array containing the embeddings. Returns an empty array of the correct shape on failure.
    """
    if not texts or not isinstance(model, SentenceTransformer):
        logger.warning("Invalid input for generating embeddings. Returning empty array.")
        # Return a correctly shaped empty array
        embedding_dim = model.get_sentence_embedding_dimension() if isinstance(model, SentenceTransformer) else 768 # Fallback
        return np.zeros((len(texts), embedding_dim))

    logger.info(f"Generating embeddings for {len(texts)} texts with {type(model).__name__}...")
    try:
        embeddings = model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True, 
            show_progress_bar=show_progress_bar
        )
        logger.info(f"Embeddings generated with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"An unexpected error occurred during embedding generation: {e}", exc_info=True)
        embedding_dim = model.get_sentence_embedding_dimension()
        return np.zeros((len(texts), embedding_dim))
