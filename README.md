# Tibetan Text Metrics (TTM)

[![CI](https://github.com/daniel-wojahn/tibetan-text-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/daniel-wojahn/tibetan-text-metrics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics/branch/main/graph/badge.svg)](https://codecov.io/gh/daniel-wojahn/tibetan-text-metrics)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python tool for computing various text similarity metrics on POS-tagged Tibetan texts. This tool is designed to analyze textual similarities and variations in Tibetan manuscripts using multiple computational approaches.

## Features

- Computes multiple similarity metrics between Tibetan text witnesses:
  - Syntactic Distance (POS Level): Counts the number of operations (insertions, deletions, substitutions) needed to transform one POS tag sequence into another
  - Weighted Jaccard Similarity: Measures vocabulary overlap with POS-based weighting
  - Longest Common Subsequence (LCS) Length: Identifies shared sequences of words
  - Word Mover's Distance (WMD): Measures semantic similarity using word embeddings
- Generates heatmap visualizations for each metric
- Supports POS-tagged text input
- Progress bar for long-running comparisons
- Model caching for improved performance

## Prerequisites

- Python 3.10 (recommended, required for gensim compatibility)
- MiLMo Word2Vec model (syllable-level) (download from [Hugging Face](https://huggingface.co/CMLI-NLP/MiLMo/tree/main))
- POS-tagged input files using [ACTib](https://github.com/lothelanor/actib)

## Project Structure

```
TTM/
├── .github/          # CI/CD workflows
├── input_files/      # POS-tagged input texts
├── output/          # Analysis results and visualizations
├── src/             # Main package source code
│   ├── tibetan_text_metrics/
│   │   ├── analyzer.py      # Core analysis functions
│   │   ├── metrics.py       # Similarity metrics implementation
│   │   ├── text_processor.py # Text processing utilities
│   │   ├── visualizer.py    # Visualization functions
│   │   └── word2vec/        # Word embeddings model
│   ├── main.py            # Main script
│   └── fast_lcs.pyx       # Cython implementation of LCS
├── tests/           # Test files
├── pyproject.toml   # Project configuration
└── setup.py         # Build configuration
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/daniel-wojahn/tibetan-text-metrics.git
cd tibetan-text-metrics
```

2. Download the required word2vec files:
   - Create a directory: `src/tibetan_text_metrics/word2vec/`
   - Download the following files and place them in this directory:
     - `藏文-词级别/word2vec_zang_tool.vec`
     - `藏文-音节/word2vec_zang_yinjie.vec`
   - Note: These files are not included in the repository due to their size. Please contact the maintainers for access to these files.

3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies and build Cython extensions:
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

The tool uses Cython for optimized performance, particularly for the LCS (Longest Common Subsequence) computation. If the Cython extension fails to build, the tool will automatically fall back to a pure Python implementation.

## Input Format

- Input files should be plain text files (`.txt`) with POS-tagged words
- Each word should be followed by its POS tag, separated by a forward slash (/)
- Example: `ཆོས/[NOUN] ཀྱི/[PART] དབྱིངས/[NOUN] སུ/[PART]`
- Files should be placed in the `input_files` directory

## Usage

1. Place your POS-tagged text files in the `input_files` directory.

2. Edit `main.py` to specify the order of text files for comparison:
```python
file_paths = [
    "input_files/Text1.txt",
    "input_files/Text2.txt",
    "input_files/Text3.txt",
    # Add more files as needed
]
```

Note: The order of files is important, especially when comparing three or more texts, as it determines how the texts are paired and displayed in the heatmaps.

3. Run the analysis:
```bash
python -m tibetan_text_metrics.main
```

## Output

The tool generates:
1. A CSV file (`output/pos_tagged_analysis.csv`) containing all computed metrics:
   - Syntactic Distance: Raw number of edit operations needed (higher = more different)
   - Weighted Jaccard Similarity: Percentage of weighted overlap (0-100%)
   - LCS Length: Number of matching words in the longest common subsequence
   - Word Mover's Distance: Semantic distance based on word embeddings (lower = more similar)
2. Heatmap visualizations for each metric in the `output` directory:
   - `heatmap_syntactic_distance.png`
   - `heatmap_weighted_jaccard.png`
   - `heatmap_lcs.png`
   - `heatmap_wmd.png`

## Performance

The tool is optimized for performance using Cython for computationally intensive operations:
- LCS computation is ~196x faster with Cython
- Memory usage is optimized and stable (peak usage ~500MB including Word2Vec model)

## Performance Profiling

The project includes profiling tools:
- `profile_script.py`: Script for performance analysis
- `profile.stats`: Generated profiling statistics

Run profiling with:
```bash
python profile_script.py
```

## Development

### Setting up the development environment

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

Run the test suite with coverage:
```bash
pytest --cov=tibetan_text_metrics tests/
```

### Code Quality Tools

The project uses several tools to maintain code quality:
- `black` for code formatting
- `isort` for import sorting
- `flake8` for style guide enforcement
- `mypy` for static type checking
- `bandit` for security checks

These checks are automatically run through pre-commit hooks and GitHub Actions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{tibetan_text_metrics,
  title = {TibetanTextMetrics (TTM): Computing Text Similarity Metrics on POS-tagged Tibetan Texts},
  author = {Daniel Wojahn},
  year = {2025},
  url = {https://github.com/daniel-wojahn/tibetan-text-metrics}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.