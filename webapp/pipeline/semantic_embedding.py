import logging
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define the model ID for the fine-tuned Tibetan MiniLM
DEFAULT_MODEL_NAME = "buddhist-nlp/buddhist-sentence-similarity"


def get_sentence_transformer_model_and_device(
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

    logger.info(f"Attempting to use device: {selected_device_str}")

    try:
        logger.info(
            f"Loading Sentence Transformer model: {model_id} on device: {selected_device_str}"
        )
        # SentenceTransformer expects a string like 'cuda', 'mps', or 'cpu'
        model = SentenceTransformer(model_id, device=selected_device_str)
        logger.info(f"Model {model_id} loaded successfully on {selected_device_str}.")
        return model, selected_device_str
    except Exception as e:
        logger.error(
            f"Error loading model {model_id} on device {selected_device_str}: {e}"
        )
        # Fallback to CPU if the initially selected device (CUDA or MPS) failed
        if selected_device_str != "cpu":
            logger.warning(
                f"Failed to load model on {selected_device_str}, attempting to load on CPU..."
            )
            fallback_device_str = "cpu"
            try:
                model = SentenceTransformer(model_id, device=fallback_device_str)
                logger.info(
                    f"Model {model_id} loaded successfully on CPU after fallback."
                )
                return model, fallback_device_str
            except Exception as fallback_e:
                logger.error(
                    f"Error loading model {model_id} on CPU during fallback: {fallback_e}"
                )
                raise fallback_e  # Re-raise exception if CPU fallback also fails
        raise e  # Re-raise original exception if selected_device_str was already CPU or no fallback attempted


def generate_embeddings(texts: list[str], model, device: str):
    """
    Generates embeddings for a list of texts using the provided Sentence Transformer model.

    Args:
        texts (list[str]): A list of texts to embed.
        model: The loaded SentenceTransformer model.
        device (str): The device the model is on (primarily for logging, model.encode handles device).

    Returns:
        torch.Tensor: A tensor containing the embeddings, moved to CPU.
    """
    if not texts:
        logger.warning(
            "No texts provided to generate_embeddings. Returning empty tensor."
        )
        return torch.empty(0)

    logger.info(f"Generating embeddings for {len(texts)} texts...")

    # The encode method of SentenceTransformer handles tokenization and pooling internally.
    # It also manages moving data to the model's device.
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    logger.info(f"Embeddings generated with shape: {embeddings.shape}")
    return (
        embeddings.cpu()
    )  # Ensure embeddings are on CPU for consistent further processing


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
        st_model, st_device = get_sentence_transformer_model_and_device(
            device_preference="cpu"  # Explicitly use CPU for this test run
        )

        if st_model:
            logger.info(f"Test model loaded on device: {st_device}")
            example_embeddings = generate_embeddings(test_texts, st_model, st_device)
            logger.info(
                f"Generated example embeddings shape: {example_embeddings.shape}"
            )
            if example_embeddings.nelement() > 0:  # Check if tensor is not empty
                logger.info(
                    f"First embedding (first 10 dims): {example_embeddings[0][:10]}..."
                )
            else:
                logger.info("Generated example embeddings tensor is empty.")
        else:
            logger.error("Failed to load model for example usage.")

    except Exception as e:
        logger.error(f"An error occurred during the example usage: {e}")

    logger.info("Finished example usage.")
