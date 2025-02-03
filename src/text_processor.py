"""Functions for reading and processing text files."""

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
            # Use filename as key instead of full path
            texts[Path(file_path).name] = [
                chapter.strip() for chapter in chapters if chapter.strip()
            ]
    return texts


def extract_words_and_pos(text: str) -> Tuple[List[str], List[str]]:
    """Extract words and POS tags from text.

    Args:
        text: Text with words and POS tags in format "word1/tag1 word2/tag2".

    Returns:
        Tuple of (words, pos_tags) lists.

    Raises:
        ValueError: If any word is missing a POS tag or has an invalid tag format.
    """
    if not text.strip():
        return [], []

    words = []
    pos_tags = []

    for token in text.strip().split():
        if "/" not in token:
            raise ValueError(f"Missing POS tag in token: {token}")
        
        parts = token.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid POS tag format in token: {token}")
        
        word, pos = parts
        words.append(word)
        pos_tags.append(pos)

    return words, pos_tags
