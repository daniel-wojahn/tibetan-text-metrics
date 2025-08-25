# Tibetan Text Metrics (TTM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14992358.svg)](https://doi.org/10.5281/zenodo.14992358)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/daniel-wojahn/tibetan-text-metrics/graphs/commit-activity)

A Python tool designed to analyze textual similarities and variations in Tibetan manuscripts using multiple computational approaches.

## Background & Motivation

TibetanTextMetrics (TTM) grew out of the challenge of analysing multiple editions of the 17th-century Tibetan legal text "The Pronouncements in Sixteen Chapters" (*zhal lce bcu drug*) as part of the [Law in Historic Tibet](https://www.law.ox.ac.uk/law-historic-tibet) project at the Centre for Socio-Legal Studies at the University of Oxford. My original approach stemmed from an understanding within the Tibetan scholarly tradition that all *zhal lce bcu drug* editions are essentially identical. Thus the plan was for a critical edition using all available editions. However, a preliminary attempt using [CollateX](https://collatex.net/) revealed substantial differences between editions, particularly in certain chapters, resulting in a convoluted apparatus that was very hard to navigate. While CollateX is ideal for texts with minor variations, the large variations between these editions required a different analytical approach. Simple comparison methods such as difflib or online plagiarism checkers offered limited insights. In order to perform a more in-depth analysis, including semantic, structural and content-based comparisons (as far as possible for the Tibetan language), I developed TTM. This tool provides the quantitative metrics necessary to effectively assess textual similarities at the chapter level, helping researchers to understand broader patterns of textual evolution.

## Key features and use cases:

- **Chapter/Section-based Analysis**: Uses Tibetan section markers (*sbrul shad*, ༈) to automatically split texts into comparable units, enabling targeted analysis of specific sections
- **Flexible Text Segmentation**: Adaptable for various historical texts and genres (a corpus like the many Sakya Genealogies (*sa skya gdung rabs*) or different editions of biographical literature (*rnam thar*) come to mind)
- **Text Evolution Analysis**: Helps trace how texts evolved over time by identifying where successive authors and editors incorporated additional material
- **Data-Driven Insights**: Provides quantitative metrics to complement qualitative textual analysis
- **N-gram Pattern Analysis**: Compares how often different word or POS tag sequences (n-grams) appear in each text, using a pattern-based approach that detects similarities in _style_ and _structure_, not just exact matches.
 
## Web App (Gradio) — run locally

The project includes a user-friendly web interface located in `ttm-webapp-hf/` that exposes the core TTM comparison workflow (upload Tibetan `.txt` files, segment by `༈`, compute Jaccard, Normalized LCS, Fuzzy, and Semantic similarity, and visualize results via heatmaps and bar charts).

Quick start for the web app:

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

cd ttm-webapp-hf
pip install -r requirements.txt

# Optional but recommended (faster LCS)
python setup.py build_ext --inplace

# Run the app
python app.py
```

Then open the provided local URL (usually http://127.0.0.1:7860). For full details (features, stopword levels, embedding models, interpretation helper), see `ttm-webapp-hf/README.md`.

## Approach & Metrics

This tool uses several complementary approaches to compare Tibetan texts:

- **Syntactic Distance (Normalized):** Measures how similarly two texts are structured grammatically. A low score means the sentence structures are alike; a high score means they are quite different. This highlights differences in style or editing.

- **Weighted Jaccard Similarity:** Calculates how much important vocabulary (especially nouns and verbs) is shared between texts. A higher score means the texts use similar vocabulary; a lower score means they focus on different things.

- **Normalized LCS (Longest Common Subsequence):** Finds the longest sequence of words that appears in both texts, even if there are gaps. A longer shared sequence suggests similar or copied passages; a shorter one means more differences.

- **Pattern Recognition (N-gram Analysis):** Looks for repeating patterns of words or grammatical tags (like pairs or triplets). If two texts use similar patterns often, they may be stylistically or structurally related, even if the exact words differ. This helps identify stylistic or structural resemblance, not just direct copying.

- **Principal Component Analysis (PCA):** Combines the above metrics into a multi-dimensional visualization, helping you spot clusters, similarities, and outliers among texts and chapters.

- **Visualizations:** Generates heatmaps for each metric to help you quickly see which texts or chapters are most similar or different.

### Why These Metrics?
- **Content-level analysis** (e.g., word2vec, BERT, etc.) is not yet reliable for Tibetan due to limited resources and training data.
- **Structural metrics** (POS-based, LCS, Jaccard) are language-agnostic and robust even with limited semantic resources.
- **Weighted approaches** allow you to emphasize linguistically important features (e.g., nouns, verbs).

### Interpreting the Results
- **Normalized Syntactic Distance**: 0 means identical structure, 1 means completely different.
- **Weighted Jaccard**: Higher is more overlap (max 100%).
- **Normalized LCS**: Higher is more sequential similarity (max 100%).

### Limitations
- **Semantic depth**: Without reliable word embeddings or semantic models for Tibetan, the analysis can’t fully capture meaning-level similarity.
- **POS tagging accuracy**: All structural metrics depend on the quality of POS tagging.


## Output

- **CSV file**: `output/metrics/pos_tagged_analysis.csv` with normalized metrics only.
- **Visualizations**: Heatmaps for each normalized metric in `output/heatmaps/`.
- **PCA**: Principal Component Analysis based on normalized metrics.

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
   - Use Python 3.10 or later
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```


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
   The tool will prompt you to select an n-gram size for pattern analysis:
   - **n=2 (Bigrams)**: Best for finding common word pairs and basic phrases
   - **n=3 (Trigrams)**: Default, good balance of specificity and coverage
   - **n=4 (4-grams)**: Better for identifying recurring expressions
   - **n=5 (5-grams)**: Best for finding exact repeated passages
   
   Choose based on your analysis needs - larger n-grams are better for finding specific textual parallels, while smaller n-grams help identify general structural similarities.
   
   The tool will then process all text files in the `input_files` directory. On Windows, this directory will be at `input_files\` relative to your project root.

3. View results:
   - CSV file with metrics: `output/metrics/pos_tagged_analysis.csv`
   - Heatmap visualizations: `output/heatmaps/`
   - PCA visualizations: `output/pca/interactive_pca_visualization.html`

## Future Directions: Phrase Detection

Phrase detection aims to go beyond fixed-length n-grams by identifying meaningful, recurring expressions in Tibetan texts—such as set formulas, idioms, or common syntactic constructions. This can be done using:

- **Rule-based methods** (e.g., splitting by Tibetan punctuation or known grammatical patterns)
- **Statistical methods** (e.g., finding word combinations that occur together more than by chance)
- **Machine learning** (e.g., algorithms that learn to spot phrase boundaries from data)

In Tibetan, this could mean detecting multi-syllable expressions, formulaic legal or religious phrases, or idioms. Integrating phrase detection would make the tool even more linguistically aware and valuable for genres where set expressions matter. This is a planned area for future development—suggestions and collaborations are welcome!

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details or visit the [Creative Commons](https://creativecommons.org/licenses/by/4.0/) website.

## Further Reading

TTM's development draws upon established computational linguistics approaches:

Graham, S., Milligan, I., & Weingart, S. 2016. _Exploring Big Historical Data: The Historian’s Macroscope_. Imperial College Press.
 
Jurafsky, D., & Martin, J. H. 2014. _Speech and Language Processing_. (Second edition, Pearson new international edition). Pearson.
 
Li, Y., Li, X., Wang, Y., Lv, H., Li, F., & Duo, L. 2022. "Character-based Joint Word Segmentation and Part-of-Speech Tagging for Tibetan Based on Deep Learning". In _ACM Transactions on Asian and Low-Resource Language Information Processing_, 21(5). [https://doi.org/10.1145/3511600](https://doi.org/10.1145/3511600).
 
Veidlinger, D. (2019). "Computational Linguistics and the Buddhist Corpus". In D. Veidlinger (Ed.), _Digital Humanities and Buddhism: An Introduction_, 43–58. Berlin, Boston: De Gruyter. [https://doi.org/10.1515/9783110519082-003](https://doi.org/10.1515/9783110519082-003).
 
Wang, J., & Dong, Y. (2020). "Measurement of Text Similarity: A Survey". _Information (Basel), 11(9), 421_. [https://doi.org/10.3390/info11090421](https://doi.org/10.3390/info11090421).

Check out also Paul Vierthaler’s [“Hacking the Humanities” workshop](https://www.youtube.com/playlist?list=PL6kqrM2i6BPIpEF5yHPNkYhjHm-FYWh17)

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
  version = {0.3.0}
}
```

MLA:
```text
Wojahn, Daniel. "TibetanTextMetrics (TTM): Computing Text Similarity Metrics on POS-tagged Tibetan Texts." Version 0.3.0, 2025, github.com/daniel-wojahn/tibetan-text-metrics.
