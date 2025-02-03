# Tibetan Text Metrics (TTM)

[![codecov](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics/branch/main/graph/badge.svg)](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A Python tool designed to analyze textual similarities and variations in Tibetan manuscripts using multiple computational approaches.

## Background & Motivation

TibetanTextMetrics (TTM) grew out of the challenge of analysing multiple editions of a 17th-century Tibetan legal text, "The Sixteen Legal Pronouncements" (*zhal lce bu drug*), as part of the [Law in Historic Tibet](https://www.law.ox.ac.uk/law-historic-tibet) project based at the Centre for Socio-Legal Studies, University of Oxford. While traditional collation tools like CollateX excel at creating critical editions, they can generate overwhelmingly detailed apparatuses. When dealing with substantially different editions rather than mere variant readings, a different approach becomes necessary. TTM addresses this by offering quantitative metrics to efficiently assess textual similarities at a chapter level, enabling researchers to identify and understand broader patterns of textual development.

Key features and use cases:

- **Chapter/Section-based Analysis**: Uses Tibetan section markers (*sbrul shad*, ༈) to automatically split texts into comparable units, enabling targeted analysis of specific sections
- **Flexible Text Segmentation**: Adaptable for various historical texts and genres (a corpus like the many Sakya Genealogies (*sa skya gdung rabs*) or different editions of biographical literature (*rnam thar*) come to mind)
- **Text Evolution Analysis**: Helps trace how texts evolved over time by identifying where successive authors and editors incorporated additional material
- **Data-Driven Insights**: Provides quantitative metrics to complement qualitative textual analysis

## Features

- Syntactic Distance (POS Level): Counts the number of operations needed to transform one POS tag sequence into another
- Weighted Jaccard Similarity: Measures vocabulary overlap with POS-based weighting (customizable in `metrics.py`)
- Longest Common Subsequence (LCS): Identifies shared sequences of words (Cython-optimized, ~196x faster)
- Word Mover's Distance (WMD): Measures semantic similarity using word embeddings
- Generate visualizations of similarity metrics
- Support for POS-tagged Tibetan text analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/daniel-wojahn/tibetan-text-metrics.git
cd tibetan-text-metrics
```

2. Download the required word2vec files:
   - Download the MiLMo Word2Vec model (syllable-level) from [Hugging Face](https://huggingface.co/CMLI-NLP/MiLMo/tree/main)
   - Place the model files in `src/word2vec/藏文-音节/`
   - Required files: `word2vec_zang_yinjie.vec`

3. Set up your environment:

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

   **For Windows:**
   ```cmd
   python -m tibetan_text_metrics.main
   ```

   **For macOS/Linux:**
   ```bash
   python -m tibetan_text_metrics.main
   ```

   The tool will automatically process all text files in the `input_files` directory. On Windows, this directory will be at `input_files\` relative to your project root.

3. View results:
   - CSV file with metrics: `output/pos_tagged_analysis.csv`
   - Heatmap visualizations: `output/heatmap_*.png`

## Output

The tool generates:
- CSV file with pairwise similarity metrics
- Heatmap visualizations for each metric:
  - Syntactic distance (POS level)
  - Weighted Jaccard similarity
  - LCS length
  - Word Mover's Distance

For the Weighted Jaccard Similarity metric, you can customize POS tag weights in `metrics.py` to control how different parts of speech affect the similarity score. This allows you to give more weight to content words (nouns, verbs) versus function words, for example.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details or visit the [Creative Commons](https://creativecommons.org/licenses/by/4.0/) website.

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