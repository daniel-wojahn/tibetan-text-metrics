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


def tokenize_texts(texts: List[str], mode: str = "syllable") -> List[List[str]]:
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

    if mode not in ["word", "syllable"]:
        logger.warning(f"Invalid tokenization mode: '{mode}'. Defaulting to 'syllable'.")
        mode = "syllable"
    
    # Process each text
    for text_content in texts:
        # Skip empty texts
        if not text_content.strip():
            tokenized_texts_list.append([])
            continue
            
        # Generate hash for cache lookup
        cache_key_string = text_content + f"_{mode}" # Include mode in string for hashing
        text_hash = _get_text_hash(cache_key_string)
        
        # Check if we have this text in cache
        if text_hash in _tokenization_cache:
            # Cache hit - use cached tokens
            tokens = _tokenization_cache[text_hash]
            logger.debug(f"Cache hit for text hash {text_hash[:8]}... (mode: {mode})")
        else:
            # Cache miss - tokenize and store in cache
            try:
                current_tokens = []
                if BOTOK_TOKENIZER:
                    raw_botok_items = list(BOTOK_TOKENIZER.tokenize(text_content))
                    
                    if mode == "word":
                        for item_idx, w in enumerate(raw_botok_items):
                            if hasattr(w, 'text') and isinstance(w.text, str):
                                token_text = w.text.strip()
                                if token_text: # Ensure token is not empty or just whitespace
                                    current_tokens.append(token_text)
                            # Optionally log if w.text is not a string or missing, for debugging
                            # elif w.text is not None:
                            #     logger.debug(f"Token item {item_idx} has non-string text {type(w.text)} for hash {text_hash[:8]}. Skipping word.")
                            # else:
                            #     logger.debug(f"Token item {item_idx} missing text attribute for hash {text_hash[:8]}. Skipping word.")
                        logger.debug(
                            f"WORD TOKENS FORMED for hash {text_hash[:8]} (mode: {mode}, first 30): "
                            f"{current_tokens[:30]}"
                        )
                    elif mode == "syllable":
                        # This is the original syllable extraction logic
                        for item_idx, w in enumerate(raw_botok_items):
                            if hasattr(w, 'syls') and w.syls:
                                for syl_idx, syl_item in enumerate(w.syls):
                                    syllable_to_process = None
                                    if isinstance(syl_item, str):
                                        syllable_to_process = syl_item
                                    elif isinstance(syl_item, list):
                                        try:
                                            syllable_to_process = "".join(syl_item)
                                        except TypeError:
                                            logger.warning(
                                                f"Syllable item in w.syls was a list, but could not be joined (non-string elements?): {syl_item} "
                                                f"from word item {item_idx} (text: {getattr(w, 'text', 'N/A')}), syl_idx {syl_idx} "
                                                f"for hash {text_hash[:8]}. Skipping this syllable."
                                            )
                                            continue
                                    
                                    if syllable_to_process is not None:
                                        stripped_syl = syllable_to_process.strip()
                                        if stripped_syl:
                                            current_tokens.append(stripped_syl)
                                    elif syl_item is not None:
                                        logger.warning(
                                            f"Unexpected type for syllable item (neither str nor list): {type(syl_item)} ('{str(syl_item)[:100]}') "
                                            f"from word item {item_idx} (text: {getattr(w, 'text', 'N/A')}), syl_idx {syl_idx} "
                                            f"for hash {text_hash[:8]}. Skipping this syllable."
                                        )
                            elif hasattr(w, 'text') and w.text: # Fallback if no 'syls' but in syllable mode
                                if isinstance(w.text, str):
                                    token_text = w.text.strip()
                                    if token_text:
                                        current_tokens.append(token_text) # Treat as a single syllable/token
                                elif w.text is not None:
                                    logger.warning(
                                        f"Unexpected type for w.text (in syllable mode fallback): {type(w.text)} ('{str(w.text)[:100]}') "
                                        f"for item {item_idx} (POS: {getattr(w, 'pos', 'N/A')}) "
                                        f"for hash {text_hash[:8]}. Skipping this token."
                                    )
                        logger.debug(
                            f"SYLLABLE TOKENS FORMED for hash {text_hash[:8]} (mode: {mode}, first 30): "
                            f"{current_tokens[:30]}"
                        )
                    tokens = current_tokens
                else:
                    logger.error(f"BOTOK_TOKENIZER is None for text hash {text_hash[:8]}, cannot tokenize (mode: {mode}).")
                    tokens = []
                
                # Store in cache if not empty
                if tokens:
                    # If cache is full, remove a random entry (simple strategy)
                    if len(_tokenization_cache) >= MAX_CACHE_SIZE:
                        # Remove first key (oldest if ordered dict, random otherwise)
                        _tokenization_cache.pop(next(iter(_tokenization_cache)))
                    
                    _tokenization_cache[text_hash] = tokens
                    logger.debug(f"Added tokens to cache with hash {text_hash[:8]}... (mode: {mode})")
            except Exception as e:
                logger.error(f"Error tokenizing text (mode: {mode}): {e}")
                tokens = []
                
        tokenized_texts_list.append(tokens)
        
    return tokenized_texts_list
