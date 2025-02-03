# Tibetan Text Metrics (TTM)

[![CI](https://github.com/daniel-wojahn/tibetan-text-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/daniel-wojahn/tibetan-text-metrics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics/branch/main/graph/badge.svg)](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for computing various text similarity metrics on POS-tagged Tibetan texts. This tool is designed to analyze textual similarities and variations in Tibetan manuscripts using multiple computational approaches.

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

3. Install the package and its dependencies:
```bash
pip install -e .
```

This will install all required dependencies as specified in `pyproject.toml`. For development, you can install additional tools with:
```bash
pip install -e ".[dev]"
```

This includes:
- pytest and pytest-cov for testing and coverage
- black, isort, and flake8 for code formatting
- mypy for type checking
- bandit for security checks

## Usage

1. Prepare your input files:
   - Place your POS-tagged Tibetan text files in the `input_files/` directory
   - Files should use the format: `word1/POS1 word2/POS2 word3/POS3`
   - For POS-tagging, we recommend using [ACTib](https://github.com/lothelanor/actib), a reliable tool for Tibetan text annotation

2. Run the analysis:
```bash
python -m tibetan_text_metrics.main
```
The tool will automatically process all text files in the `input_files` directory.

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

For the Weighted Jaccard Similarity metric, you can customize POS tag weights in `metrics.py` to control how different parts of speech affect the similarity score - play around with different weights to see how they affect the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{wojahn2025ttm,
  title = {TibetanTextMetrics (TTM): Computing Text Similarity Metrics on POS-tagged Tibetan Texts},
  author = {Daniel Wojahn},
  year = {2025},
  url = {https://github.com/daniel-wojahn/tibetan-text-metrics}
}