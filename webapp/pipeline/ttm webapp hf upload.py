import os
from typing import List, Tuple


def handle_file_upload(files, input_type: str) -> Tuple[dict, List[str]]:
    """
    Reads uploaded files and returns their contents and filenames.
    Args:
        files: List of uploaded file objects from Gradio
        input_type: Type of input (raw, tokenized, pos-tagged)
    Returns:
        text_data: dict mapping filename to file content
        filenames: list of filenames
    """
    text_data = {}
    filenames = []
    for file in files:
        filename = os.path.basename(file.name)
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
        text_data[filename] = content
        filenames.append(filename)
    return text_data, filenames
