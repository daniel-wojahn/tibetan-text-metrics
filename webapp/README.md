---
title: Tibetan Text Metrics
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
python_version: 3.10
app_file: app.py
models:
  - buddhist-nlp/buddhist-sentence-similarity
  - fasttext-tibetan
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
    -   **Semantic Similarity**: Uses embedding models to compare the contextual meaning of segments. Users can select between a transformer-based model (buddhist-nlp/buddhist-sentence-similarity) specialized for Buddhist texts or a FastText model optimized for Tibetan language. *Note: This metric works best when combined with other metrics for a more comprehensive analysis.*
    -   **TF-IDF Cosine Similarity**: Highlights texts that share important or characteristic terms by comparing their TF-IDF profiles. *Common Tibetan stopwords can be excluded to ensure TF-IDF weights highlight genuinely characteristic terms.*
-   **Handles Long Texts**: Implements automated chunking for semantic similarity to process texts exceeding the model's token limit.
-   **Model Selection**: Choose from specialized embedding models for semantic similarity analysis:
    -   **Buddhist-NLP Transformer**: Pre-trained model specialized for Buddhist texts
    -   **FastText**: Train a custom model directly on your Tibetan corpus with optimizations specifically for Tibetan language
-   **Stopword Filtering**: Three levels of filtering for Tibetan words:
    -   **None**: No filtering, includes all words
    -   **Standard**: Filters only common particles and punctuation
    -   **Aggressive**: Filters all function words including particles, pronouns, and auxiliaries
-   **Interactive Visualizations**:
    -   Heatmaps for Jaccard, LCS, Semantic, and TF-IDF similarity metrics, providing a quick overview of inter-segment relationships.
    -   Bar chart displaying word counts per segment.
-   **Rule-Based Interpretation**: Get help understanding your results with a built-in analysis engine that examines your metrics and provides insights about textual relationships.
-   **Downloadable Results**: Export detailed metrics as a CSV file and save heatmaps as PNG files.
-   **Simplified Workflow**: No command-line interaction or Python scripting needed for analysis.

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
3.  **Semantic Similarity (BuddhistNLP)**: Utilizes the `buddhist-nlp/bodhi-sentence-cased-v1` model to compute the cosine similarity between the semantic embeddings of text segments. This model is fine-tuned for Buddhist studies texts and captures nuances in meaning. For texts exceeding the model's 512-token input limit, an automated chunking strategy is employed: texts are divided into overlapping chunks, each chunk is embedded, and the resulting chunk embeddings are averaged (mean pooling) to produce a single representative vector for the entire segment before comparison.
4.  **TF-IDF Cosine Similarity**: This metric first calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word in each text segment, optionally **filtering out common Tibetan stopwords**. 
TF-IDF gives higher weight to words that are frequent within a particular segment but relatively rare across the entire collection of segments. 
This helps to identify terms that are characteristic or discriminative for a segment. When stopword filtering is enabled, the TF-IDF scores better reflect genuinely significant terms.
Each segment is then represented as a vector of these TF-IDF scores. 
Finally, the cosine similarity is computed between these vectors. 
A score closer to 1 indicates that the two segments share more of these important, distinguishing terms, suggesting they cover similar specific topics or themes.

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
    - Select an embedding model for semantic analysis
    - Choose a stopword filtering level (None, Standard, or Aggressive)
3.  **Run Analysis**: Click the "Run Analysis" button.
3.  **View Results**:
    -   A preview of the similarity metrics will be displayed.
    -   Download the full results as a CSV file.
    -   Interactive heatmaps for Jaccard Similarity, Normalized LCS, Semantic Similarity, and TF-IDF Cosine Similarity will be generated. All heatmaps use a consistent color scheme where darker colors represent higher similarity.
    -   A bar chart showing word counts per segment will also be available.
    -   Any warnings (e.g., regarding missing chapter markers) will be displayed.

4.  **Get Interpretation** (Optional):
    -   After running the analysis, click the "Help Interpret Results" button.
    -   No API key or internet connection required! The system uses a built-in rule-based analysis engine.
    -   The system will analyze your metrics and provide insights about patterns, relationships, and notable findings in your data.
    -   This feature helps researchers understand the significance of the metrics and identify interesting textual relationships between chapters.

## Embedding Models

The application offers two specialized approaches for calculating semantic similarity in Tibetan texts:

1. **Buddhist-NLP Transformer** (Default option):
   - A specialized model fine-tuned for Buddhist text similarity
   - Provides excellent results for Tibetan Buddhist texts
   - Pre-trained and ready to use, no training required
   - Best for general Buddhist terminology and concepts

2. **FastText Model**:
   - Uses the official Facebook FastText Tibetan model (facebook/fasttext-bo-vectors)
   - Pre-trained on a large corpus of Tibetan text from Wikipedia and other sources
   - Falls back to training a custom model on your texts if the official model cannot be loaded
   - Respects your stopword filtering settings when creating embeddings
   - Uses simple word vector averaging for stable embeddings

**When to choose FastText**:
- When you want high-quality word embeddings specifically trained for Tibetan language
- When you need a model that can handle out-of-vocabulary words through character n-grams
- When you want to benefit from Facebook's large-scale pre-training on Tibetan text
- When you need more control over how stopwords affect semantic analysis

## Structure

-   `app.py` — Gradio web app entry point and UI definition.
-   `pipeline/` — Modules for file handling, text processing, metrics calculation, and visualization.
    -   `process.py`: Core logic for segmenting texts and orchestrating metric computation.
    -   `metrics.py`: Implementation of Jaccard, LCS, and Semantic Similarity (including chunking).
    -   `semantic_embedding.py`: Handles loading and using the selected embedding models.
    -   `fasttext_embedding.py`: Provides functionality for training and using FastText models.
    -   `tokenize.py`: Tibetan text tokenization using `botok`.
    -   `upload.py`: File upload handling (currently minimal).
    -   `visualize.py`: Generates heatmaps and word count plots.
-   `requirements.txt` — Python dependencies for the web application.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](../../LICENSE) file in the main project directory for details.

## Research and Acknowledgements

The FastText implementation for Tibetan text has been optimized based on research findings from several studies on Tibetan natural language processing:

1. Di, R., Tashi, N., & Lin, J. (2019). Improving Tibetan Word Segmentation Based on Multi-Features Fusion. *IEEE Access*, 7, 178057-178069.
   - Informed our syllable-based tokenization approach and the importance of preserving Tibetan syllable markers

2. Tashi, N., Rabgay, T., & Wangchuk, K. (2020). Tibetan Word Segmentation using Syllable-based Maximum Matching with Potential Syllable Merging. *Engineering Applications of Artificial Intelligence*, 93, 103716.
   - Provided insights on syllable segmentation for Tibetan text processing

3. Tashi, N., Rai, A. K., Mittal, P., & Sharma, A. K. (2018). A Novel Approach to Feature Extraction for Tibetan Text Classification. *Journal of Information Processing Systems*, 14(1), 211-224.
   - Guided our parameter optimization for FastText, including embedding dimensions and n-gram settings

4. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the Association for Computational Linguistics*, 5, 135-146.
   - The original FastText paper that introduced the subword-enriched word embeddings we use

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
