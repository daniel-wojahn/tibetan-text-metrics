from typing import List

try:
    from botok import WordTokenizer

    # Initialize the tokenizer once at the module level
    BOTOK_TOKENIZER = WordTokenizer()
except ImportError:
    # Handle the case where botok might not be installed,
    # though it's a core dependency for this app.
    BOTOK_TOKENIZER = None
    print("ERROR: botok library not found. Tokenization will fail.")
    # Optionally, raise an error here if botok is absolutely critical for the app to even start
    # raise ImportError("botok is required for tokenization. Please install it.")


def tokenize_texts(texts: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of raw Tibetan texts using botok.
    Args:
        texts: List of raw text strings.
    Returns:
        List of tokenized texts (each as a list of tokens).
    """
    if BOTOK_TOKENIZER is None:
        # This case should ideally be handled more gracefully,
        # perhaps by preventing analysis if the tokenizer failed to load.
        raise RuntimeError(
            "Botok tokenizer failed to initialize. Cannot tokenize texts."
        )

    tokenized_texts_list = []
    for text_content in texts:
        tokens = [
            w.text for w in BOTOK_TOKENIZER.tokenize(text_content) if w.text.strip()
        ]
        tokenized_texts_list.append(tokens)
    return tokenized_texts_list
