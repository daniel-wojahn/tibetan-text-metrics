from typing import List, Dict
import hashlib
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize a cache for tokenization results
# Using a simple in-memory dictionary with text hash as key
_tokenization_cache: Dict[str, List[str]] = {}

# Maximum cache size (number of entries)
MAX_CACHE_SIZE = 1000

try:
    from botok import WordTokenizer

    # Initialize the tokenizer once at the module level
    BOTOK_TOKENIZER = WordTokenizer()
except ImportError:
    # Handle the case where botok might not be installed,
    # though it's a core dependency for this app.
    BOTOK_TOKENIZER = None
    logger.error("botok library not found. Tokenization will fail.")
    # Optionally, raise an error here if botok is absolutely critical for the app to even start
    # raise ImportError("botok is required for tokenization. Please install it.")


def _get_text_hash(text: str) -> str:
    """
    Generate a hash for the input text to use as a cache key.
    
    Args:
        text: The input text to hash
        
    Returns:
        A string representation of the MD5 hash of the input text
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def tokenize_texts(texts: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of raw Tibetan texts using botok, with caching for performance.
    
    This function maintains an in-memory cache of previously tokenized texts to avoid
    redundant processing of the same content. The cache uses MD5 hashes of the input
    texts as keys.
    
    Args:
        texts: List of raw text strings to tokenize.
        
    Returns:
        List of tokenized texts (each as a list of tokens).
        
    Raises:
        RuntimeError: If the botok tokenizer failed to initialize.
    """
    if BOTOK_TOKENIZER is None:
        # This case should ideally be handled more gracefully,
        # perhaps by preventing analysis if the tokenizer failed to load.
        raise RuntimeError(
            "Botok tokenizer failed to initialize. Cannot tokenize texts."
        )

    tokenized_texts_list = []
    
    # Process each text
    for text_content in texts:
        # Skip empty texts
        if not text_content.strip():
            tokenized_texts_list.append([])
            continue
            
        # Generate hash for cache lookup
        text_hash = _get_text_hash(text_content)
        
        # Check if we have this text in cache
        if text_hash in _tokenization_cache:
            # Cache hit - use cached tokens
            tokens = _tokenization_cache[text_hash]
            logger.debug(f"Cache hit for text hash {text_hash[:8]}...")
        else:
            # Cache miss - tokenize and store in cache
            try:
                tokens = [
                    w.text for w in BOTOK_TOKENIZER.tokenize(text_content) if w.text.strip()
                ]
                
                # Store in cache if not empty
                if tokens:
                    # If cache is full, remove a random entry (simple strategy)
                    if len(_tokenization_cache) >= MAX_CACHE_SIZE:
                        # Remove first key (oldest if ordered dict, random otherwise)
                        _tokenization_cache.pop(next(iter(_tokenization_cache)))
                    
                    _tokenization_cache[text_hash] = tokens
                    logger.debug(f"Added tokens to cache with hash {text_hash[:8]}...")
            except Exception as e:
                logger.error(f"Error tokenizing text: {e}")
                tokens = []
                
        tokenized_texts_list.append(tokens)
        
    return tokenized_texts_list
