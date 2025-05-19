# Tibetan Text Metrics Web Application User Guide

## Introduction

Welcome to the Tibetan Text Metrics Web Application! This user-friendly tool allows you to analyze textual similarities and variations in Tibetan manuscripts using multiple computational approaches. The application provides a graphical interface to the core functionalities of the Tibetan Text Metrics (TTM) project.

## Getting Started

### System Requirements

- Modern web browser (Chrome, Firefox, Safari, or Edge)
- For local installation: Python 3.10 or newer
- Sufficient RAM for processing large texts (4GB minimum, 8GB recommended)

### Installation and Setup

#### Online Demo

The easiest way to try the application is through our Hugging Face Spaces demo:
[daniel-wojahn/ttm-webapp-hf](https://huggingface.co/spaces/daniel-wojahn/ttm-webapp-hf)

Note: The free tier of Hugging Face Spaces may have performance limitations compared to running locally.

#### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/daniel-wojahn/tibetan-text-metrics.git
   cd tibetan-text-metrics/webapp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:7860
   ```

## Using the Application

### Step 1: Upload Your Tibetan Text Files

1. Click the "Upload Tibetan .txt files" button to select one or more `.txt` files containing Tibetan text.
2. Files should be in UTF-8 or UTF-16 encoding.
3. Maximum file size: 10MB per file (for optimal performance, use files under 1MB).
4. For best results, your texts should be segmented into chapters/sections using the Tibetan marker '༈' (*sbrul shad*).

### Step 2: Configure Analysis Options

1. **Semantic Similarity**: Choose whether to compute semantic similarity metrics.
   - "Yes" (default): Includes semantic similarity in the analysis (slower but more comprehensive).
   - "No": Skips semantic similarity calculation for faster processing.

2. **Embedding Model**: Select the model to use for semantic similarity analysis.
   - **sentence-transformers/all-MiniLM-L6-v2** (default): General purpose sentence embedding model (fastest option).
   - **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**: Multilingual model with good performance for many languages.
   - **buddhist-nlp/buddhist-sentence-similarity**: Optimized for Buddhist text similarity.
   - **xlm-roberta-base**: Multilingual model that includes Tibetan.

3. Click the "Run Analysis" button to start processing.

### Step 3: View and Interpret Results

After processing, the application displays several visualizations and metrics:

#### Word Count Chart

Shows the number of words in each chapter/segment of each file, allowing you to compare the relative lengths of different texts.

#### Similarity Metrics

The application computes four different similarity metrics between corresponding chapters of different files:

1. **Jaccard Similarity (%)**: Measures vocabulary overlap between segments after filtering out common Tibetan stopwords. A higher percentage indicates a greater overlap in the significant vocabularies used in the two segments.

2. **Normalized LCS (Longest Common Subsequence)**: Measures the length of the longest sequence of words that appears in both text segments, maintaining their original relative order. A higher score suggests more significant shared phrasing, direct textual borrowing, or strong structural parallelism.

3. **Semantic Similarity**: Uses a transformer-based model to compute the cosine similarity between the semantic embeddings of text segments. This captures similarities in meaning even when different vocabulary is used.

4. **TF-IDF Cosine Similarity**: Compares texts based on their important, characteristic terms by giving higher weight to words that are frequent within a particular segment but relatively rare across the entire collection.

#### Heatmap Visualizations

Each metric has a corresponding heatmap visualization where:
- Rows represent chapters/segments
- Columns represent text pairs being compared
- Color intensity indicates similarity (brighter = more similar)

### Tips for Effective Analysis

1. **Text Segmentation**: For meaningful chapter-level comparisons, ensure your texts are segmented using the Tibetan marker '༈' (*sbrul shad*).

2. **File Naming**: Use descriptive filenames to make the comparison results easier to interpret.

3. **Model Selection**: 
   - For faster processing, use the default model or disable semantic similarity.
   - For Buddhist texts, the buddhist-nlp/buddhist-sentence-similarity model may provide better results.

4. **File Size**: 
   - Keep individual files under 1MB for optimal performance.
   - Very large files (>10MB) are not supported and will trigger an error.

5. **Comparing Multiple Texts**: The application requires at least two text files to compute similarity metrics.

## Understanding the Metrics

### Jaccard Similarity (%)

This metric quantifies the lexical overlap between two text segments by comparing their sets of unique words, after filtering out common Tibetan stopwords. It essentially answers the question: 'Of all the distinct, meaningful words found across these two segments, what proportion of them are present in both?'

It is calculated as:
```
(Number of common unique meaningful words) / (Total number of unique meaningful words in both texts combined) * 100
```

Jaccard Similarity is insensitive to word order and word frequency; it only cares whether a unique meaningful word is present or absent. A higher percentage indicates a greater overlap in the significant vocabularies used in the two segments.

### Normalized LCS (Longest Common Subsequence)

This metric measures the length of the longest sequence of words that appears in both text segments, maintaining their original relative order. Importantly, these words do not need to be directly adjacent (contiguous) in either text.

For example, if Text A is 'the quick brown fox jumps' and Text B is 'the lazy cat and brown dog jumps high', the LCS is 'the brown jumps'.

The length of this common subsequence is then normalized to provide a score. A higher Normalized LCS score suggests more significant shared phrasing, direct textual borrowing, or strong structural parallelism, as it reflects similarities in how ideas are ordered and expressed sequentially.

Unlike other metrics, LCS does not filter out stopwords, allowing it to capture structural similarities and the flow of language, including the use of particles and common words that contribute to sentence construction.

### Semantic Similarity

This metric utilizes transformer-based models to compute the cosine similarity between the semantic embeddings of text segments. The model converts each text segment into a high-dimensional vector that captures its semantic meaning.

For texts exceeding the model's token limit, an automated chunking strategy is employed: texts are divided into overlapping chunks, each chunk is embedded, and the resulting chunk embeddings are averaged to produce a single representative vector for the entire segment before comparison.

A higher score indicates that the texts express similar concepts or ideas, even if they use different vocabulary or phrasing.

### TF-IDF Cosine Similarity

This metric first calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word in each text segment, after filtering out common Tibetan stopwords. TF-IDF gives higher weight to words that are frequent within a particular segment but relatively rare across the entire collection of segments.

Each segment is then represented as a vector of these TF-IDF scores, and the cosine similarity is computed between these vectors. A score closer to 1 indicates that the two segments share more of these important, distinguishing terms, suggesting they cover similar specific topics or themes.

## Troubleshooting

### Common Issues and Solutions

1. **"Empty vocabulary" error**:
   - This can occur if a text contains only stopwords or if tokenization fails.
   - Solution: Check your input text to ensure it contains valid Tibetan content.

2. **Model loading errors**:
   - If a model fails to load, the application will continue without semantic similarity.
   - Solution: Try a different model or disable semantic similarity.

3. **Performance issues with large files**:
   - Solution: Split large files into smaller ones or use fewer files at once.

4. **No results displayed**:
   - Solution: Ensure you have uploaded at least two valid text files and that they contain comparable content.

5. **Encoding issues**:
   - If your text appears garbled, it may have encoding problems.
   - Solution: Ensure your files are saved in UTF-8 or UTF-16 encoding.

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the [GitHub repository](https://github.com/daniel-wojahn/tibetan-text-metrics) for updates or known issues.
2. Submit an issue on GitHub with details about your problem.

## Acknowledgments

The Tibetan Text Metrics project was developed as part of the [Law in Historic Tibet](https://www.law.ox.ac.uk/law-historic-tibet) project at the Centre for Socio-Legal Studies at the University of Oxford.

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
