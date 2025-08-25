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
 
## Web App (Gradio) — run locally

The project includes a user-friendly web interface located in `webapp/` that exposes the core TTM comparison workflow (upload Tibetan `.txt` files, segment by `༈`, compute Jaccard, Normalized LCS, Fuzzy, and Semantic similarity, and visualize results via heatmaps and bar charts).

Quick start for the web app:

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

cd webapp
pip install -r requirements.txt

# Optional but recommended (faster LCS)
python setup.py build_ext --inplace

# Run the app
python app.py
```

Then open the provided local URL (usually http://127.0.0.1:7860). For full details (features, stopword levels, embedding models, interpretation helper), see `webapp/README.md`.

## Approach & Metrics

This tool uses complementary approaches to compare Tibetan texts at the chapter/segment level:

- **Jaccard Similarity (%)**: Lexical overlap over unique tokens. Supports stopword filtering (None, Standard, Aggressive) using curated Tibetan lists in `webapp/pipeline/stopwords_bo.py` and `webapp/pipeline/stopwords_lite_bo.py`.
- **Normalized LCS (Longest Common Subsequence)**: Sequential/structural similarity over token sequences. Normalized by the average length of the two segments (range 0–1). Uses an optional Cython acceleration if compiled (`webapp/fast_lcs.pyx`).
- **Fuzzy Similarity**: Approximate string similarity via TheFuzz with selectable methods (token_set, token_sort, partial, ratio). Honors the selected stopword filtering.
- **Semantic Similarity**: Cosine similarity of sentence-transformer embeddings (default: `sentence-transformers/LaBSE`).

### Interpreting the Results
- **Jaccard Similarity**: 0–100%. Higher = more shared unique words (after optional stopword filtering).
- **Normalized LCS**: 0–1. Higher = longer ordered sequences in common.
- **Fuzzy Similarity**: 0–1. Higher = more approximate matches (robust to spelling/order variations).
- **Semantic Similarity**: 0–1. Higher = more similar meanings (embedding-based).

### Limitations
- **Semantic depth**: Because embedding-based similarity for Tibetan is still developing, it should be combined with structural and lexical metrics for comprehensive insights.
- **Segmentation quality**: Results depend on sensible chapter/section segmentation using `༈`.


## Output

- **CSV file**: A `results.csv` file is written to the current working directory when running the web app (`webapp/`).
- **Visualizations**: Interactive Plotly heatmaps for Jaccard, LCS, Fuzzy, and Semantic similarity, plus a word count bar chart — displayed in the app UI. Heatmaps are not saved automatically; export via screenshots or Plotly utilities if needed.

### Optional: AI Interpretation (OpenRouter)
- **Send results to an LLM**: Click “Help Interpret Results” to send `results.csv` to an LLM via OpenRouter for a scholarly interpretation of the metrics. Set `OPENROUTER_API_KEY` in your environment. If unavailable, the app falls back to a rule‑based analysis.
- **Prompt summary**: The app converts the results DataFrame into a Markdown table and asks for a concise, scholarly interpretation focusing on implications and relationships between texts (not restating values). It includes the model name and uses a system prompt framing the assistant as a senior scholar of Tibetan Buddhist textual criticism.
- **Available models (auto‑fallback order)**: `qwen/qwen3-235b-a22b-07-25:free`, `deepseek/deepseek-r1-0528:free`, `google/gemma-2-9b-it:free`, `moonshotai/kimi-k2:free` (see `webapp/pipeline/llm_service.py`).

## Installation

1. Clone this repository:
```bash
git clone https://github.com/daniel-wojahn/tibetan-text-metrics.git
cd tibetan-text-metrics
```
2. Create and activate a virtual environment (Python 3.10+):

   **Windows**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the web app dependencies:
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```

4. (Optional) Compile the Cython LCS extension for speed:
   ```bash
   python setup.py build_ext --inplace
   ```

4a. (Optional) Enable AI interpretation via OpenRouter:

   Set your OpenRouter API key so the “Help Interpret Results” feature can call an LLM.

   **macOS/Linux (bash/zsh)**
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

   **Windows (PowerShell)**
   ```powershell
   setx OPENROUTER_API_KEY "your_api_key_here"
   # Restart terminal to apply
   ```

   **Windows (cmd, current session only)**
   ```cmd
   set OPENROUTER_API_KEY=your_api_key_here
   ```

5. Run the web app:
   ```bash
   python app.py
   ```


## Usage

- Upload two or more `.txt` files containing Tibetan Unicode text.
- Use the Tibetan section marker `༈` to separate chapters/sections; without it, each file is treated as one segment and a warning is shown.
- Choose whether to compute Semantic and Fuzzy similarity and select a stopword filtering level.
- After running, download `results.csv` and explore the heatmaps and word count chart.

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
