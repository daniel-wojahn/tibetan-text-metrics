# Tibetan Text Metrics (TTM)

[![codecov](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics/branch/main/graph/badge.svg)](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14992358.svg)](https://doi.org/10.5281/zenodo.14992358)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/daniel-wojahn/tibetan-text-metrics/graphs/commit-activity)

A Python tool designed to analyze textual similarities and variations in Tibetan manuscripts using multiple computational approaches.

## Background & Motivation

TibetanTextMetrics (TTM) grew out of the challenge of analysing multiple editions of the 17th-century Tibetan legal text "The Pronouncements in Sixteen Chapters" (*zhal lce bcu drug*) as part of the [Law in Historic Tibet](https://www.law.ox.ac.uk/law-historic-tibet) project at the Centre for Socio-Legal Studies at the University of Oxford. My original approach stemmed from an understanding within the Tibetan scholarly tradition that all *zhal lce bcu drug* editions are essentially identical. Thus the plan was for a critical edition using all available editions. However, a preliminary attempt using [CollateX](https://collatex.net/) revealed substantial differences between editions, particularly in certain chapters, resulting in a convoluted apparatus that was very hard to navigate. While CollateX is ideal for texts with minor variations, the large variations between these editions required a different analytical approach. Simple comparison methods such as difflib or online plagiarism checkers offered limited insights. In order to perform a more in-depth analysis, including semantic, structural and content-based comparisons (as far as possible for the Tibetan language), I developed TTM. This tool provides the quantitative metrics necessary to effectively assess textual similarities at the chapter level, helping researchers to understand broader patterns of textual evolution.

### Key features and use cases:

- **Chapter/Section-based Analysis**: Uses Tibetan section markers (*sbrul shad*, ༈) to automatically split texts into comparable units, enabling targeted analysis of specific sections
- **Flexible Text Segmentation**: Adaptable for various historical texts and genres (a corpus like the many Sakya Genealogies (*sa skya gdung rabs*) or different editions of biographical literature (*rnam thar*) come to mind)
- **Text Evolution Analysis**: Helps trace how texts evolved over time by identifying where successive authors and editors incorporated additional material
- **Data-Driven Insights**: Provides quantitative metrics to complement qualitative textual analysis

## Features

- **Syntactic Distance**: Counts the number of operations needed to transform one POS tag sequence into another. Also provides a normalized version (0-1) scaled by text length, enabling fair comparison between chapters of different sizes.

- **Weighted Jaccard Similarity**: Measures vocabulary overlap with POS-based weighting (customizable in `metrics.py`), allowing you to emphasize content words like nouns and verbs over function words.

- **Longest Common Subsequence (LCS)**: Identifies shared sequences of words (Cython-optimized, ~196x faster). Both raw counts and normalized percentages (relative to average text length) are provided.

- **Principal Component Analysis (PCA)**: Provides multi-dimensional visualization of textual relationships, offering a holistic approach to identifying manuscript traditions by combining multiple metrics into an intuitive visual representation.

- **Visualizations**: Generate heatmaps for individual metrics and PCA plots that help identify clusters of similar texts and chapters.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/daniel-wojahn/tibetan-text-metrics.git
cd tibetan-text-metrics
```
2. Set up your environment:

   **For Windows:**
   - Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
     - During installation, select "Desktop development with C++"
     - This is required for compiling Cython extensions
   - Use a virtual environment (recommended):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **For macOS/Linux:**
   - Use Python 3.10 or later (Python 3.13 has known issues with gensim dependencies)
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```

   For development, install additional tools:
   ```bash
   pip install -e ".[dev]"
   ```

   Development dependencies include:
   - Testing: pytest with coverage reporting
   - Code Quality: black (formatting), isort (import sorting), flake8 (linting)
   - Type Checking: mypy with pandas-stubs
   - Security: bandit
   - Performance: memory-profiler

   **Note for Windows users:** If you encounter any issues with Cython compilation, ensure that:
   - Visual Studio Build Tools are properly installed
   - You're using Python 3.10+ (Python 3.13 has known issues with gensim dependencies)
   - Your environment variables are correctly set (usually handled by VS Build Tools installer)

## Usage

> **⚠️ Important**: This tool requires at least one Tibetan section marker (*sbrul shad*, ༈) at the beginning of each input text. These markers are essential for the text segmentation functionality and preprocessing steps.

1. Prepare your input files:
   - Place your POS-tagged Tibetan text files in the `input_files/` directory
   - Files should use the format: `word1/POS1 word2/POS2 word3/POS3`
   - For POS-tagging, you can use [ACTib](https://github.com/lothelanor/actib), which combines [Botok](https://github.com/OpenPecha/botok) for tokenization with Memory-Based Tagger for POS tagging. Note that POS tagging for Classical Tibetan is still an active area of research, and manual validation of the results is recommended.

2. Run the analysis:

   ```cmd
   python -m tibetan_text_metrics.main
   ```
   Or if you're running from the repo root:
   ```bash
   python -m src.tibetan_text_metrics.main
   ```
   The tool will automatically process all text files in the `input_files` directory. On Windows, this directory will be at `input_files\` relative to your project root.

3. View results:
   - CSV file with metrics: `output/metrics/pos_tagged_analysis.csv`
   - Heatmap visualizations: `output/heatmaps/`
   - PCA visualizations: `output/pca/interactive_pca_visualization.html`

## Output

The tool generates:
- **CSV files**:
  - `output/metrics/pos_tagged_analysis.csv`: Complete pairwise analysis with all metrics (both raw and normalized)
  - `output/pca/pca_data.csv`: PCA coordinates with normalized metrics for further analysis
  - `output/pca/pca_loadings.csv`: Shows how each metric contributes to the principal components

- **Visualizations**:
  - `output/heatmaps/`: Heatmaps for each metric (normalized and raw versions)
  - `output/pca/interactive_pca_visualization.html`: Interactive PCA plots showing chapter relationships by text pair and by chapter with explanations
  - `output/pca/`: Feature vector projections showing the contribution of each metric to the principal components

For the Weighted Jaccard Similarity metric, you can customize POS tag weights in `metrics.py` to control how different parts of speech affect the similarity score. This allows you to give more weight to content words (nouns, verbs) versus function words, for example.

### Understanding Normalized Metrics

Normalization is crucial when comparing chapters of different lengths. For example:
- Raw syntactic distance between two 1000-word chapters might be 200 operations
- Raw syntactic distance between two 100-word chapters might be 20 operations
- Despite the 10x difference in raw values, both represent the same proportional difference (20%)

Normalized metrics address this by scaling values relative to text length, providing fair comparisons that aren't biased by chapter size.

### PCA Visualization

The tool includes an enhanced Principal Component Analysis (PCA) visualization that helps interpret the relationships between different text chapters based on multiple metrics simultaneously:

- **Interactive HTML plot** with hover information for each data point
- **Clear metric labels** positioned in the corners of the visualization
- **Built-in explanation** of how to interpret the PCA results
- **Visual clustering** to identify outliers and pattern groups

The PCA visualization can be found in `output/pca/interactive_pca_visualization.html` and provides:

1. A scatterplot where each point represents a chapter comparison
2. Points are colored by text pair to identify patterns within comparisons
3. Main and secondary cluster regions with clear visual boundaries

This visualization is particularly useful for identifying which chapters have unusual similarity patterns compared to others.

The PCA analysis is based on the metrics calculated during the pairwise comparison process, providing a holistic view that combines multiple similarity measures into a single visualization.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details or visit the [Creative Commons](https://creativecommons.org/licenses/by/4.0/) website.

## Contributing

Contributions to this project are welcome and appreciated! This is my first open-source repository, and I'm excited to collaborate with others.

Here are some ways you can contribute:

- **Bug reports**: If you find a bug, please open an issue with a clear description and steps to reproduce it
- **Feature requests**: Have an idea for a new feature? Feel free to suggest it by opening an issue
- **Code contributions**: Pull requests for bug fixes, features, or documentation improvements are all welcome
- **Documentation**: Improvements to the documentation, examples, or tutorials are very valuable

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{wojahn2025ttm,
  title = {TibetanTextMetrics (TTM): Computing Text Similarity Metrics on POS-tagged Tibetan Texts},
  author = {Daniel Wojahn},
  year = {2025},
  url = {https://github.com/daniel-wojahn/tibetan-text-metrics},
  version = {0.1.0}
}
```

MLA:
```text
Wojahn, Daniel. "TibetanTextMetrics (TTM): Computing Text Similarity Metrics on POS-tagged Tibetan Texts." Version 0.1.0, 2025, github.com/daniel-wojahn/tibetan-text-metrics.
