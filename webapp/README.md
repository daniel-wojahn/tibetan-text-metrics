---
title: Tibetan Text Metrics
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
python_version: 3.11
app_file: app.py
---

# Tibetan Text Metrics Web App

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Project Status: Active – Web app version for accessible text analysis.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

A user-friendly web application for analyzing textual similarities and variations in Tibetan manuscripts. This tool provides a graphical interface to the core text comparison functionalities of the [Tibetan Text Metrics (TTM)](https://github.com/daniel-wojahn/tibetan-text-metrics) project, making it accessible to researchers without Python or command-line experience. Built with Python, Cython, and Gradio.

## Background

The Tibetan Text Metrics project aims to provide quantitative methods for assessing textual similarities at the chapter or segment level, helping researchers understand patterns of textual evolution. This web application extends these capabilities by offering an intuitive interface, removing the need for manual script execution and environment setup for end-users.

## Key Features of the Web App

-   **Easy File Upload**: Upload one or more Tibetan `.txt` files directly through the browser.
-   **Automatic Segmentation**: Uses Tibetan section markers (e.g., `༈`) to automatically split texts into comparable chapters or sections.
-   **Core Metrics Computed**:
    -   **Jaccard Similarity (%)**: Measures vocabulary overlap between segments. *Common Tibetan stopwords can be filtered out to focus on meaningful lexical similarity.*
    -   **Normalized Longest Common Subsequence (LCS)**: Identifies the longest shared sequence of words, indicating direct textual parallels.
    -   **Fuzzy Similarity**: Uses fuzzy string matching to detect approximate matches between words, accommodating spelling variations and minor differences in Tibetan text.
    -   **Semantic Similarity**: Uses sentence-transformer embeddings (e.g., LaBSE) to compare the contextual meaning of segments. *Note: This metric works best when combined with other metrics for a more comprehensive analysis.*
-   **Handles Long Texts**: Implements automated handling for long segments when computing semantic embeddings.
-   **Model Selection**: Semantic similarity analysis uses Hugging Face sentence-transformer models (e.g., LaBSE).
-   **Stopword Filtering**: Three levels of filtering for Tibetan words:
    -   **None**: No filtering, includes all words
    -   **Standard**: Filters only common particles and punctuation
    -   **Aggressive**: Filters all function words including particles, pronouns, and auxiliaries
-   **Interactive Visualizations**:
    -   Heatmaps for Jaccard, LCS, Fuzzy, and Semantic similarity metrics, providing a quick overview of inter-segment relationships.
    -   Bar chart displaying word counts per segment.
-   **Advanced Interpretation**: Get scholarly insights about your results with a built-in analysis engine that:
    -   Examines your metrics and provides contextual interpretation of textual relationships
    -   Generates a dual-layer narrative analysis (scholarly and accessible)
    -   Identifies patterns across chapters and highlights notable textual relationships
    -   Connects findings to Tibetan textual studies concepts (transmission lineages, regional variants)
    -   Suggests questions for further investigation
-   **Downloadable Results**: Export detailed metrics as a CSV file and save heatmaps as PNG files.
-   **Simplified Workflow**: No command-line interaction or Python scripting needed for analysis.

## Advanced Features

### Using AI-Powered Analysis

The application includes an "Interpret Results" button that provides scholarly insights about your text similarity metrics. This feature:

1. Uses a selection of free OpenRouter models to analyze your results
2. Requires an OpenRouter API key (set via environment variable)
3. The AI will provide a comprehensive scholarly analysis including:
   - Introduction explaining the texts compared and general observations
   - Overall patterns across all chapters with visualized trends
   - Detailed examination of notable chapters (highest/lowest similarity)
   - Discussion of what different metrics reveal about textual relationships
   - Conclusions suggesting implications for Tibetan textual scholarship
   - Specific questions these findings raise for further investigation
   - Cautionary notes about interpreting perfect matches or zero similarity scores

### Data Processing

- **Automatic Filtering**: The system automatically filters out perfect matches (1.0 across all metrics) that may result from empty cells or identical text comparisons
- **Robust Analysis**: The system handles edge cases and provides meaningful metrics even with imperfect data

## Text Segmentation and Best Practices

**Why segment your texts?**

To obtain meaningful results, it is highly recommended to divide your Tibetan texts into logical chapters or sections before uploading. Comparing entire texts as a single unit often produces shallow or misleading results, especially for long or complex works. Chapters or sections allow the tool to detect stylistic, lexical, or structural differences that would otherwise be hidden.

**How to segment your texts:**

-   Use the Tibetan section marker (`༈` (sbrul shad)) to separate chapters/sections in your `.txt` files.
-   Each segment should represent a coherent part of the text (e.g., a chapter, legal clause, or thematic section).
-   The tool will automatically split your file on this marker for analysis. If no marker is found, the entire file is treated as a single segment, and a warning will be issued.

**Best practices:**

-   Ensure the marker is unique and does not appear within a chapter.
-   Try to keep chapters/sections of similar length for more balanced comparisons.
-   For poetry or short texts, consider grouping several poems or stanzas as one segment.

## Implemented Metrics

**Stopword Filtering:**
To enhance the accuracy and relevance of similarity scores, both the Jaccard Similarity and TF-IDF Cosine Similarity calculations incorporate a stopword filtering step. This process removes high-frequency, low-information Tibetan words (e.g., common particles, pronouns, and grammatical markers) before the metrics are computed. This ensures that the resulting scores are more reflective of meaningful lexical and thematic similarities between texts, rather than being skewed by the presence of ubiquitous common words.

The comprehensive list of Tibetan stopwords used is adapted and compiled from the following valuable resources:
- The **Divergent Discourses** (specifically, their Tibetan stopwords list available at [Zenodo Record 10148636](https://zenodo.org/records/10148636)).
- The **Tibetan Lucene Analyser** by the Buddhist Digital Archives (BUDA), available on [GitHub: buda-base/lucene-bo](https://github.com/buda-base/lucene-bo).

We extend our gratitude to the creators and maintainers of these projects for making their work available to the community.

Feel free to edit this list of stopwords to better suit your needs. The list is stored in the `pipeline/stopwords.py` file.

### The application computes and visualizes the following similarity metrics between corresponding chapters/segments of the uploaded texts:

1.  **Jaccard Similarity (%)**: This metric quantifies the lexical overlap between two text segments by comparing their sets of *unique* words, optionally **filtering out common Tibetan stopwords**. 
It essentially answers the question: 'Of all the distinct, meaningful words found across these two segments, what proportion of them are present in both?' 
It is calculated as `(Number of common unique meaningful words) / (Total number of unique meaningful words in both texts combined) * 100`. 
Jaccard Similarity is insensitive to word order and word frequency; it only cares whether a unique meaningful word is present or absent. 
A higher percentage indicates a greater overlap in the significant vocabularies used in the two segments.

**Stopword Filtering**: Three levels of filtering are available:
- **None**: No filtering, includes all words in the comparison
- **Standard**: Filters only common particles and punctuation
- **Aggressive**: Filters all function words including particles, pronouns, and auxiliaries

This helps focus on meaningful content words rather than grammatical elements.

2.  **Normalized LCS (Longest Common Subsequence)**: This metric measures the length of the longest sequence of words that appears in *both* text segments, maintaining their original relative order. Importantly, these words do not need to be directly adjacent (contiguous) in either text. For example, if Text A is '<u>the</u> quick <u>brown</u> fox <u>jumps</u>' and Text B is '<u>the</u> lazy cat and <u>brown</u> dog <u>jumps</u> high', the LCS is 'the brown jumps'. The length of this common subsequence is then normalized (in this tool, by dividing by the length of the longer of the two segments) to provide a score, which is then presented as a percentage. A higher Normalized LCS score suggests more significant shared phrasing, direct textual borrowing, or strong structural parallelism, as it reflects similarities in how ideas are ordered and expressed sequentially.
    *   *Note on Interpretation*: It's possible for Normalized LCS to be higher than Jaccard Similarity. This often happens when texts share a substantial 'narrative backbone' or common ordered phrases (leading to a high LCS), even if they use varied surrounding vocabulary or introduce many unique words not part of these core sequences (which would lower the Jaccard score). LCS highlights this sequential, structural similarity, while Jaccard focuses on the overall shared vocabulary regardless of its arrangement.
3.  **Fuzzy Similarity**: This metric uses fuzzy string matching algorithms to detect approximate matches between words, making it particularly valuable for Tibetan texts where spelling variations, dialectal differences, or scribal errors might be present. Unlike exact matching methods (such as Jaccard), fuzzy similarity can recognize when words are similar but not identical. The implementation offers multiple matching methods:
    - **Token Set Ratio** (default): Compares the sets of words regardless of order, finding the best alignment between them
    - **Token Sort Ratio**: Sorts the words alphabetically before comparing, useful for texts with similar vocabulary in different orders
    - **Partial Ratio**: Finds the best matching substring, helpful for detecting when one text contains parts of another
    - **Simple Ratio**: Performs character-by-character comparison, best for detecting minor spelling variations

    Scores range from 0 to 1, where 1 indicates perfect or near-perfect matches. This metric is particularly useful for identifying textual relationships that might be missed by exact matching methods, especially in manuscripts with orthographic variations.

**Stopword Filtering**: The same three levels of filtering used for Jaccard Similarity are applied to fuzzy matching:
- **None**: No filtering, includes all words in the comparison
- **Standard**: Filters only common particles and punctuation
- **Aggressive**: Filters all function words including particles, pronouns, and auxiliaries

4.  **Semantic Similarity**: Computes the cosine similarity between sentence-transformer embeddings (e.g., LaBSE) of text segments. Segments are embedded into high-dimensional vectors and compared via cosine similarity. Scores closer to 1 indicate a higher degree of semantic overlap.

**Stopword Filtering**: Three levels of filtering are available:
- **None**: No filtering, includes all words in the comparison
- **Standard**: Filters only common particles and punctuation
- **Aggressive**: Filters all function words including particles, pronouns, and auxiliaries

This helps focus on meaningful content words rather than grammatical elements.

## Getting Started (if run Locally)

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
5.  **Compile Cython Extension (Recommended for Performance)**:
    To speed up the Longest Common Subsequence (LCS) calculation, a Cython extension is provided. To compile it:
    ```bash
    # Ensure you are in the webapp directory
    python setup.py build_ext --inplace
    ```
    This step requires a C compiler. If you skip this, the application will use a slower, pure Python implementation for LCS.

6.  **Run the Web Application**:
    ```bash
    python app.py
    ```
7.  Open your web browser and go to the local URL provided (usually `http://127.0.0.1:7860`).

## Usage

1.  **Upload Files**: Use the file upload interface to select one or more `.txt` files containing Tibetan Unicode text.
2.  **Configure Options**: 
    - Choose whether to compute semantic similarity
    - Choose whether to compute fuzzy string similarity
    - Select a fuzzy matching method (Token Set, Token Sort, Partial, or Simple Ratio)
    - Select an embedding model for semantic analysis
    - Choose a stopword filtering level (None, Standard, or Aggressive)
3.  **Run Analysis**: Click the "Run Analysis" button.
3.  **View Results**:
    -   A preview of the similarity metrics will be displayed.
    -   Download the full results as a CSV file.
    -   Interactive heatmaps for Jaccard Similarity, Normalized LCS, Fuzzy Similarity, and Semantic Similarity will be generated. All heatmaps use a consistent color scheme where darker colors represent higher similarity.
    -   A bar chart showing word counts per segment will also be available.
    -   Any warnings (e.g., regarding missing chapter markers) will be displayed.

4.  **Get Interpretation** (Optional):
    -   After running the analysis, click the "Help Interpret Results" button.
    -   No API key or internet connection required! The system uses a built-in rule-based analysis engine.
    -   The system will analyze your metrics and provide insights about patterns, relationships, and notable findings in your data.
    -   This feature helps researchers understand the significance of the metrics and identify interesting textual relationships between chapters.

## Embedding Model

Semantic similarity uses Hugging Face sentence-transformer models (default: `sentence-transformers/LaBSE`). These models provide context-aware, segment-level embeddings suitable for comparing Tibetan text passages.

## Structure

-   `app.py` — Gradio web app entry point and UI definition.
-   `pipeline/` — Modules for file handling, text processing, metrics calculation, and visualization.
    -   `process.py`: Core logic for segmenting texts and orchestrating metric computation.
    -   `metrics.py`: Implementation of Jaccard, LCS, and Semantic Similarity.
    -   `hf_embedding.py`: Handles loading and using sentence-transformer models.
    -   `tokenize.py`: Tibetan text tokenization using `botok`.
    -   `upload.py`: File upload handling (currently minimal).
    -   `visualize.py`: Generates heatmaps and word count plots.
-   `requirements.txt` — Python dependencies for the web application.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](../../LICENSE) file in the main project directory for details.

## Research and Acknowledgements

We acknowledge the broader Tibetan NLP community for tokenization and stopword resources leveraged in this project, including the Divergent Discourses stopword list and BUDA's lucene-bo analyzer.

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
