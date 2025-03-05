"""Functions for reading and processing text files."""

import os
from pathlib import Path
from typing import Dict, List, Tuple


def read_text_files(file_paths: List[str]) -> Dict[str, List[str]]:
    """Read and process POS-tagged text files.

    Args:
        file_paths: List of paths to POS-tagged text files.

    Returns:
        Dictionary mapping filenames to list of raw text for each chapter.
    """
    texts = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            chapters = content.split("༈")
            # Use filename without extension as key
            texts[Path(file_path).stem] = [
                chapter.strip() for chapter in chapters if chapter.strip()
            ]
    return texts


def extract_words_and_pos(text: str) -> Tuple[List[str], List[str]]:
    """Extract words and POS tags from text.

    Args:
        text: Raw text with words and POS tags.

    Returns:
        Tuple of (words, pos_tags) lists.
    """
    tokens = text.split()
    words = []
    pos_tags = []
    for token in tokens:
        # Skip tokens without POS tags
        if "/" in token:
            # Split on last '/' to handle cases like "བསྒྲུབ་པ/n.v.fut"
            word, pos = token.rsplit("/", 1)
            if pos.strip():  # Only include if POS tag is not empty
                words.append(word.strip())
                pos_tags.append(pos.strip())
        else:
            # For testing: if strict mode is enabled, raise error
            if getattr(extract_words_and_pos, '_strict_mode', False):
                raise ValueError("All words must have POS tags in format 'word/tag'")
    return words, pos_tags
