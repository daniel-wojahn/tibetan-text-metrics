# Tibetan Text Metrics Web App

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Project Status: Active – Web app version for accessible text analysis.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

A user-friendly web application for analyzing textual similarities and variations in Tibetan manuscripts. This tool provides a graphical interface to the core text comparison functionalities of the [Tibetan Text Metrics (TTM)](https://github.com/daniel-wojahn/tibetan-text-metrics) project, making it accessible to researchers without Python or command-line experience. Built with Python and Gradio.

## Background

The Tibetan Text Metrics project aims to provide quantitative methods for assessing textual similarities at the chapter or segment level, helping researchers understand patterns of textual evolution. This web application extends these capabilities by offering an intuitive interface, removing the need for manual script execution and environment setup for end-users.

## Key Features of the Web App

-   **Easy File Upload**: Upload one or more Tibetan `.txt` files directly through the browser.
-   **Automatic Segmentation**: Uses Tibetan section markers (e.g., `༈`) to automatically split texts into comparable chapters or sections.
-   **Core Metrics Computed**:
    -   **Jaccard Similarity (%)**: Measures vocabulary overlap between segments.
    -   **Normalized Longest Common Subsequence (LCS)**: Identifies the longest shared sequence of words, indicating direct textual parallels.
    -   **Semantic Similarity (BuddhistNLP)**: Uses the `buddhist-nlp/bodhi-sentence-cased-v1` model to compare the contextual meaning of segments.
    -   **TF-IDF Cosine Similarity**: Highlights texts that share important or characteristic terms by comparing their TF-IDF profiles.
-   **Handles Long Texts**: Implements automated chunking for semantic similarity to process texts exceeding the model's token limit.
-   **Interactive Visualizations**:
    -   Heatmaps for Jaccard, LCS, Semantic, and TF-IDF similarity metrics, providing a quick overview of inter-segment relationships.
    -   Bar chart displaying word counts per segment.
-   **Downloadable Results**: Export detailed metrics as a CSV file.
-   **Simplified Workflow**: No command-line interaction or Python scripting needed for analysis.

## Text Segmentation and Best Practices

**Why segment your texts?**

To obtain meaningful results, it is highly recommended to divide your Tibetan texts into logical chapters or sections before uploading. Comparing entire texts as a single unit often produces shallow or misleading results, especially for long or complex works. Chapters or sections allow the tool to detect stylistic, lexical, or structural differences that would otherwise be hidden.

**How to segment your texts:**

-   Use a clear marker (e.g., `༈` or another unique string) to separate chapters/sections in your `.txt` files.
-   Each segment should represent a coherent part of the text (e.g., a chapter, legal clause, or thematic section).
-   The tool will automatically split your file on this marker for analysis. If no marker is found, the entire file is treated as a single segment, and a warning will be issued.

**Best practices:**

-   Ensure your marker is unique and does not appear within a chapter.
-   Try to keep chapters/sections of similar length for more balanced comparisons.
-   For poetry or short texts, consider grouping several poems or stanzas as one segment.

## Implemented Metrics

The application computes and visualizes the following similarity metrics between corresponding chapters/segments of the uploaded texts:

1.  **Jaccard Similarity (%)**: This metric quantifies the lexical overlap between two text segments by comparing their sets of *unique* words. It essentially answers the question: 'Of all the distinct words found across these two segments, what proportion of them are present in both?' It is calculated as `(Number of common unique words) / (Total number of unique words in both texts combined) * 100`. Jaccard Similarity is insensitive to word order and word frequency; it only cares whether a unique word is present or absent. A higher percentage indicates a greater overlap in the vocabularies used in the two segments.
2.  **Normalized LCS (Longest Common Subsequence)**: This metric measures the length of the longest sequence of words that appears in *both* text segments, maintaining their original relative order. Importantly, these words do not need to be directly adjacent (contiguous) in either text. For example, if Text A is '<u>the</u> quick <u>brown</u> fox <u>jumps</u>' and Text B is '<u>the</u> lazy cat and <u>brown</u> dog <u>jumps</u> high', the LCS is 'the brown jumps'. The length of this common subsequence is then normalized (in this tool, by dividing by the length of the longer of the two segments) to provide a score, which is then presented as a percentage. A higher Normalized LCS score suggests more significant shared phrasing, direct textual borrowing, or strong structural parallelism, as it reflects similarities in how ideas are ordered and expressed sequentially.
    *   *Note on Interpretation*: It's possible for Normalized LCS to be higher than Jaccard Similarity. This often happens when texts share a substantial 'narrative backbone' or common ordered phrases (leading to a high LCS), even if they use varied surrounding vocabulary or introduce many unique words not part of these core sequences (which would lower the Jaccard score). LCS highlights this sequential, structural similarity, while Jaccard focuses on the overall shared vocabulary regardless of its arrangement.
3.  **Semantic Similarity (BuddhistNLP)**: Utilizes the `buddhist-nlp/bodhi-sentence-cased-v1` model to compute the cosine similarity between the semantic embeddings of text segments. This model is fine-tuned for Buddhist studies texts and captures nuances in meaning. For texts exceeding the model's 512-token input limit, an automated chunking strategy is employed: texts are divided into overlapping chunks, each chunk is embedded, and the resulting chunk embeddings are averaged (mean pooling) to produce a single representative vector for the entire segment before comparison.
4.  **TF-IDF Cosine Similarity**: This metric first calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word in each text segment. TF-IDF gives higher weight to words that are frequent within a particular segment but relatively rare across the entire collection of segments. This helps to identify terms that are characteristic or discriminative for a segment. Each segment is then represented as a vector of these TF-IDF scores. Finally, the cosine similarity is computed between these vectors. A score closer to 1 indicates that the two segments share more of these important, distinguishing terms, suggesting they cover similar specific topics or themes.

## Getting Started (Running Locally)

1.  Ensure you have Python 3.10 or newer.
2.  Navigate to the `webapp` directory:
    ```bash
    cd path/to/tibetan-text-metrics/webapp
    ```
3.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate    # On Windows
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Run the web application:
    ```bash
    python app.py
    ```
6.  Open your web browser and go to the local URL provided (usually `http://127.0.0.1:7860`).

## Usage

1.  **Upload Files**: Use the file upload interface to select one or more `.txt` files containing Tibetan Unicode text.
2.  **Run Analysis**: Click the "Run Analysis" button.
3.  **View Results**:
    -   A preview of the similarity metrics will be displayed.
    -   Download the full results as a CSV file.
    -   Interactive heatmaps for Jaccard Similarity, Normalized LCS, Semantic Similarity, and TF-IDF Cosine Similarity will be generated.
    -   A bar chart showing word counts per segment will also be available.
    -   Any warnings (e.g., regarding missing chapter markers) will be displayed.

## Structure

-   `app.py` — Gradio web app entry point and UI definition.
-   `pipeline/` — Modules for file handling, text processing, metrics calculation, and visualization.
    -   `process.py`: Core logic for segmenting texts and orchestrating metric computation.
    -   `metrics.py`: Implementation of Jaccard, LCS, and Semantic Similarity (including chunking).
    -   `semantic_embedding.py`: Handles loading and using the sentence transformer model.
    -   `tokenize.py`: Tibetan text tokenization using `botok`.
    -   `upload.py`: File upload handling (currently minimal).
    -   `visualize.py`: Generates heatmaps and word count plots.
-   `requirements.txt` — Python dependencies for the web application.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](../../LICENSE) file in the main project directory for details.

## Citation

If you use this web application or the underlying TTM tool in your research, please cite the main project:

```bibtex
@software{wojahn2025ttm,
  title = {TibetanTextMetrics (TTM): Computing Text Similarity Metrics on POS-tagged Tibetan Texts},
  author = {Daniel Wojahn},
  year = {2025},
  url = {https://github.com/daniel-wojahn/tibetan-text-metrics},
  version = {0.3.0}
}
```

---
For questions or issues specifically regarding the web application, please refer to the main project's [issue tracker](https://github.com/daniel-wojahn/tibetan-text-metrics/issues) or contact Daniel Wojahn.
