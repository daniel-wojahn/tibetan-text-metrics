import logging
import torch
from typing import List, Any
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define the model ID for the fine-tuned Tibetan MiniLM
DEFAULT_MODEL_NAME = "buddhist-nlp/buddhist-sentence-similarity"

# FastText model identifier
FASTTEXT_MODEL_ID = "fasttext-tibetan"


def get_model_and_device(
    model_id: str = DEFAULT_MODEL_NAME, device_preference: str = "auto"
):
    """
    Loads the Sentence Transformer model and determines the device.
    Priority: CUDA -> MPS (Apple Silicon) -> CPU.

    Args:
        model_id (str): The Hugging Face model ID.
        device_preference (str): Preferred device ("cuda", "mps", "cpu", "auto").

    Returns:
        tuple: (model, device_str)
               - model: The loaded SentenceTransformer model.
               - device_str: The device the model is loaded on ("cuda", "mps", or "cpu").
    """
    selected_device_str = ""

    if device_preference == "auto":
        if torch.cuda.is_available():
            selected_device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected_device_str = "mps"
        else:
            selected_device_str = "cpu"
    elif device_preference == "cuda" and torch.cuda.is_available():
        selected_device_str = "cuda"
    elif (
        device_preference == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        selected_device_str = "mps"
    else:  # Handles explicit "cpu" preference or fallback if preferred is unavailable
        selected_device_str = "cpu"

    logger.info("Attempting to use device: %s", selected_device_str)

    try:
        # Check if this is a FastText model request
        if model_id == FASTTEXT_MODEL_ID:
            try:
                # Import here to avoid dependency issues if FastText is not installed
                from .fasttext_embedding import load_fasttext_model
                model = load_fasttext_model()
                if model is None:
                    logger.warning("FastText model not found. You may need to train it first.")
                logger.info("FastText model loaded successfully.")
                # FastText always runs on CPU
                return model, "cpu", "fasttext"
            except ImportError:
                logger.error("FastText module not found. Please install it with 'pip install fasttext'.")
                raise
        else:
            logger.info(
                "Loading Sentence Transformer model: %s on device: %s",
                model_id, selected_device_str
            )
            # SentenceTransformer expects a string like 'cuda', 'mps', or 'cpu'
            model = SentenceTransformer(model_id, device=selected_device_str)
            logger.info("Model %s loaded successfully on %s.", model_id, selected_device_str)
            return model, selected_device_str, "sentence_transformer"
    except Exception as e:
        logger.error(
            "Error loading model %s on device %s: %s",
            model_id, selected_device_str, str(e)
        )
        # Fallback to CPU if the initially selected device (CUDA or MPS) failed
        if selected_device_str != "cpu":
            logger.warning(
                "Failed to load model on %s, attempting to load on CPU...",
                selected_device_str
            )
            fallback_device_str = "cpu"
            try:
                model = SentenceTransformer(model_id, device=fallback_device_str)
                logger.info(
                    "Model %s loaded successfully on CPU after fallback.",
                    model_id
                )
                return model, fallback_device_str, "sentence_transformer"
            except Exception as fallback_e:
                logger.error(
                    "Error loading model %s on CPU during fallback: %s",
                    model_id, str(fallback_e)
                )
                raise fallback_e  # Re-raise exception if CPU fallback also fails
        raise e  # Re-raise original exception if selected_device_str was already CPU or no fallback attempted


def generate_embeddings(texts: List[str], model: Any, device: str, model_type: str = "sentence_transformer", tokenize_fn=None, use_stopwords: bool = True, use_lite_stopwords: bool = False):
    """
    Generates embeddings for a list of texts using the provided model.

    Args:
        texts (list[str]): A list of texts to embed.
        model: The loaded model (SentenceTransformer or FastText).
        device (str): The device to use ("cuda", "mps", or "cpu").
        model_type (str): Type of model ("sentence_transformer" or "fasttext")
        tokenize_fn: Optional tokenization function or pre-tokenized list for FastText
        use_stopwords (bool): Whether to filter out stopwords for FastText embeddings

    Returns:
        torch.Tensor: A tensor containing the embeddings, moved to CPU.
    """
    if not texts:
        logger.warning(
            "No texts provided to generate_embeddings. Returning empty tensor."
        )
        return torch.empty(0)

    logger.info(f"Generating embeddings for {len(texts)} texts...")

    if model_type == "fasttext":
        try:
            # Import here to avoid dependency issues if FastText is not installed
            from .fasttext_embedding import get_batch_embeddings
            from .stopwords_bo import TIBETAN_STOPWORDS_SET
            
            # For FastText, get appropriate stopwords set if filtering is enabled
            stopwords_set = None
            if use_stopwords:
                # Choose between regular and lite stopwords sets
                if use_lite_stopwords:
                    from .stopwords_lite_bo import TIBETAN_STOPWORDS_LITE_SET
                    stopwords_set = TIBETAN_STOPWORDS_LITE_SET
                else:
                    from .stopwords_bo import TIBETAN_STOPWORDS_SET
                    stopwords_set = TIBETAN_STOPWORDS_SET
            
            # Pass tokenize_fn (pre-tokenized list) and stopwords parameters
            embeddings = get_batch_embeddings(
                texts, 
                model, 
                tokenize_fn, 
                use_stopwords=use_stopwords, 
                stopwords_set=stopwords_set
            )
            logger.info("FastText embeddings generated with shape: %s", str(embeddings.shape))
            # Convert numpy array to torch tensor for consistency
            return torch.tensor(embeddings)
        except ImportError:
            logger.error("FastText module not found. Please install it with 'pip install fasttext'.")
            raise
    else:  # sentence_transformer
        # The encode method of SentenceTransformer handles tokenization and pooling internally.
        # It also manages moving data to the model's device.
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        logger.info("Sentence Transformer embeddings generated with shape: %s", str(embeddings.shape))
        return (
            embeddings.cpu()
        )  # Ensure embeddings are on CPU for consistent further processing


def train_fasttext_model(corpus_texts: List[str], **kwargs):
    """
    Train a FastText model on the provided corpus texts.
    
    Args:
        corpus_texts: List of texts to use for training
        **kwargs: Additional parameters for training (dim, epoch, etc.)
        
    Returns:
        Trained model and path to the model file
    """
    try:
        from .fasttext_embedding import prepare_corpus_file, train_fasttext_model as train_ft
        
        # Prepare corpus file
        corpus_path = prepare_corpus_file(corpus_texts)
        
        # Train the model
        model = train_ft(corpus_path=corpus_path, **kwargs)
        
        return model
    except ImportError:
        logger.error("FastText module not found. Please install it with 'pip install fasttext'.")
        raise


if __name__ == "__main__":
    # Example usage:
    logger.info("Starting example usage of semantic_embedding module...")

    test_texts = [
        "བཀྲ་ཤིས་བདེ་ལེགས།",
        "hello world",  # Test with non-Tibetan to see behavior
        "དེ་རིང་གནམ་གཤིས་ཡག་པོ་འདུག",
    ]

    logger.info("Attempting to load model using default cache directory.")
    try:
        # Forcing CPU for this example to avoid potential CUDA issues in diverse environments
        # or if CUDA is not intended for this specific test.
        model, device, model_type = get_model_and_device(
            device_preference="cpu"  # Explicitly use CPU for this test run
        )

        if model:
            logger.info("Test model loaded on device: %s, type: %s", device, model_type)
            example_embeddings = generate_embeddings(test_texts, model, device, model_type)
            logger.info(
                "Generated example embeddings shape: %s",
                str(example_embeddings.shape)
            )
            if example_embeddings.nelement() > 0:  # Check if tensor is not empty
                logger.info(
                    "First embedding (first 10 dims): %s...",
                    str(example_embeddings[0][:10])
                )
            else:
                logger.info("Generated example embeddings tensor is empty.")
        else:
            logger.error("Failed to load model for example usage.")

    except Exception as e:
        logger.error("An error occurred during the example usage: %s", str(e))

    logger.info("Finished example usage.")
