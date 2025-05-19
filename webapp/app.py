import gradio as gr
from pathlib import Path
from pipeline.process import process_texts
from pipeline.visualize import generate_visualizations, generate_word_count_chart
from pipeline.llm_interpreter import get_interpretation
import logging
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from theme import tibetan_theme

logger = logging.getLogger(__name__)


# Main interface logic
def main_interface():
    with gr.Blocks(
        theme=tibetan_theme,
        title="Tibetan Text Metrics Web App",
        css=tibetan_theme.get_css_string(),
    ) as demo:
        gr.Markdown(
            """# Tibetan Text Metrics Web App
<span style='font-size:18px;'>A user-friendly web application for analyzing textual similarities and variations in Tibetan manuscripts, providing a graphical interface to the core functionalities of the [Tibetan Text Metrics (TTM)](https://github.com/daniel-wojahn/tibetan-text-metrics) project.</span>
        """,
            elem_classes="gr-markdown",
        )

        with gr.Row(elem_id="steps-row"):
            with gr.Column(scale=1, elem_classes="step-column"):
                with gr.Group():
                    gr.Markdown(
                        """
                    ## Step 1: Upload Your Tibetan Text Files
                    <span style='font-size:16px;'>Upload one or more `.txt` files. Each file should contain Unicode Tibetan text, segmented into chapters/sections if possible using the marker '༈' (<i>sbrul shad</i>).</span>
                    """,
                        elem_classes="gr-markdown",
                    )
                    file_input = gr.File(
                        label="Upload Tibetan .txt files",
                        file_types=[".txt"],
                        file_count="multiple",
                    )
                    gr.Markdown(
                        "<small>Note: Maximum file size: 10MB per file. For optimal performance, use files under 1MB.</small>",
                        elem_classes="gr-markdown"
                    )
            with gr.Column(scale=1, elem_classes="step-column"):
                with gr.Group():
                    gr.Markdown(
                        """## Step 2: Configure and run the analysis
<span style='font-size:16px;'>Choose your analysis options and click the button below to compute metrics and view results. For meaningful analysis, ensure your texts are segmented by chapter or section using the marker '༈' (<i>sbrul shad</i>). The tool will split files based on this marker.</span>
                    """,
                        elem_classes="gr-markdown",
                    )
                    semantic_toggle_radio = gr.Radio(
                        label="Compute semantic similarity?",
                        choices=["Yes", "No"],
                        value="Yes",
                        info="Semantic similarity will be time-consuming. Choose 'No' to speed up analysis if these metrics are not required.",
                        elem_id="semantic-radio-group",
                    )
                    
                    model_dropdown = gr.Dropdown(
                        label="Embedding Model",
                        choices=[
                            "buddhist-nlp/buddhist-sentence-similarity",
                            "fasttext-tibetan"
                        ],
                        value="buddhist-nlp/buddhist-sentence-similarity",
                        info="Select the embedding model for semantic similarity.<br><br>"
                             "<b>Model information:</b><br>"
                             "• <a href='https://huggingface.co/buddhist-nlp/buddhist-sentence-similarity' target='_blank'>buddhist-nlp/buddhist-sentence-similarity</a>: Specialized model fine-tuned for Buddhist text similarity. Provides the best results for Tibetan Buddhist texts.<br>"
                             "• <b>fasttext-tibetan</b>: Uses the official Facebook FastText Tibetan model pre-trained on a large corpus. If the official model cannot be loaded, it will fall back to training a custom model on your uploaded texts.",
                        visible=True,
                        interactive=True
                    )
                    
                    stopwords_dropdown = gr.Dropdown(
                        label="Stopword Filtering",
                        choices=[
                            "None (No filtering)", 
                            "Standard (Common particles only)", 
                            "Aggressive (All function words)"
                        ],
                        value="Aggressive (All function words)",  # Default to aggressive filtering
                        info="Choose how aggressively to filter out common Tibetan particles and function words when calculating similarity. This helps focus on meaningful content words."
                    )

                    process_btn = gr.Button(
                        "Run Analysis", elem_id="run-btn", variant="primary"
                    )

        gr.Markdown(
            """## Results
        """,
            elem_classes="gr-markdown",
        )
        # The heatmap_titles and metric_tooltips dictionaries are defined here
        # heatmap_titles = { ... }
        # metric_tooltips = { ... }
        csv_output = gr.File(label="Download CSV Results")
        metrics_preview = gr.Dataframe(
            label="Similarity Metrics Preview", interactive=False, visible=True
        )
        
        # LLM Interpretation components
        with gr.Row():
            with gr.Column():
                interpret_btn = gr.Button("Help Interpret Results", variant="primary")
                gr.Markdown(
                    "<small>Using free Claude 3.5 Sonnet via <a href='https://developer.puter.com/tutorials/free-unlimited-claude-35-sonnet-api/' target='_blank'>Puter API</a>. No API key required!</small>",
                    elem_classes="gr-markdown"
                )
                interpretation_output = gr.Markdown(
                    value="Click 'Help Interpret Results' to get an AI-powered interpretation of your similarity metrics."
                )
        
        word_count_plot = gr.Plot(label="Word Counts per Segment")
        # Heatmap tabs for each metric
        heatmap_titles = {
            "Jaccard Similarity (%)": "Jaccard Similarity (%): Higher scores (darker) mean more shared unique words.",
            "Normalized LCS": "Normalized LCS: Higher scores (darker) mean longer shared sequences of words.",
            "Semantic Similarity": "Semantic Similarity (using word embeddings/experimental): Higher scores (darker) mean more similar meanings.",
            "TF-IDF Cosine Sim": "TF-IDF Cosine Similarity: Higher scores (darker) mean texts share more important, distinctive vocabulary.",
        }

        metric_tooltips = {
            "Jaccard Similarity (%)": """
### Jaccard Similarity (%)
This metric quantifies the lexical overlap between two text segments by comparing their sets of *unique* words, optionally filtering out common Tibetan stopwords. 

It essentially answers the question: 'Of all the distinct words found across these two segments, what proportion of them are present in both?' It is calculated as `(Number of common unique words) / (Total number of unique words in both texts combined) * 100`. 

Jaccard Similarity is insensitive to word order and word frequency; it only cares whether a unique word is present or absent. A higher percentage indicates a greater overlap in the vocabularies used in the two segments.

**Stopword Filtering**: When enabled (via the "Filter Stopwords" checkbox), common Tibetan particles and function words are filtered out before comparison. This helps focus on meaningful content words rather than grammatical elements.
""",
            "Normalized LCS": """
### Normalized LCS (Longest Common Subsequence)
This metric measures the length of the longest sequence of words that appears in *both* text segments, maintaining their original relative order. 
Importantly, these words do not need to be directly adjacent (contiguous) in either text. 
For example, if Text A is '<u>the</u> quick <u>brown</u> fox <u>jumps</u>' and Text B is '<u>the</u> lazy cat and <u>brown</u> dog <u>jumps</u> high', the LCS is 'the brown jumps'. 
The length of this common subsequence is then normalized (in this tool, by dividing by the length of the longer of the two segments) to provide a score, which is then presented as a percentage. 
A higher Normalized LCS score suggests more significant shared phrasing, direct textual borrowing, or strong structural parallelism, as it reflects similarities in how ideas are ordered and expressed sequentially.

**No Stopword Filtering.** Unlike metrics such as Jaccard Similarity or TF-IDF Cosine Similarity (which typically filter out common stopwords to focus on content-bearing words), the LCS calculation in this tool intentionally uses the raw, unfiltered sequence of tokens from your texts. This design choice allows LCS to capture structural similarities and the flow of language, including the use of particles and common words that contribute to sentence construction and narrative sequence. By not removing stopwords, LCS can reveal similarities in phrasing and textual structure that might otherwise be obscured, making it a valuable complement to metrics that focus purely on lexical overlap of keywords.

**Note on Interpretation**: It is possible for Normalized LCS to be higher than Jaccard Similarity. This often happens when texts share a substantial 'narrative backbone' or common ordered phrases (leading to a high LCS), even if they use varied surrounding vocabulary or introduce many unique words not part of these core sequences (which would lower the Jaccard score). LCS highlights this sequential, structural similarity, while Jaccard focuses on the overall shared vocabulary regardless of its arrangement.
""",
            "Semantic Similarity": """
### Semantic Similarity
Computes the cosine similarity between semantic embeddings of text segments using one of two approaches:

**1. Transformer-based Model**: Pre-trained model that understand contextual relationships between words.
   - `buddhist-nlp/buddhist-sentence-similarity`: Specialized for Buddhist texts

**2. FastText Model**: Uses the official Facebook FastText Tibetan model (facebook/fasttext-bo-vectors) pre-trained on a large corpus of Tibetan text. Falls back to a custom model only if the official model cannot be loaded.
   - Creates embeddings specifically tailored to your corpus vocabulary
   - Better for specialized Tibetan texts with domain-specific terminology
   - Trained when first selected and saved for future use
   - Optimized for Tibetan language with:
     - Syllable-based tokenization preserving Tibetan syllable markers
     - TF-IDF weighted averaging for word vectors (distinct from the TF-IDF Cosine Similarity metric)
     - Enhanced parameters based on Tibetan NLP research

**Chunking for Long Texts**: For texts exceeding the model's token limit, an automated chunking strategy is employed: texts are divided into overlapping chunks, each chunk is embedded, and the resulting embeddings are averaged to produce a single vector for the entire segment.

**Stopword Filtering**: When enabled (via the "Filter Stopwords" checkbox), common Tibetan particles and function words are filtered out before computing embeddings. This helps focus on meaningful content words. Transformer models process the full text regardless of stopword filtering setting.

**Note**: This metric works best when combined with other metrics for a more comprehensive analysis.
""",
            "TF-IDF Cosine Sim": """
### TF-IDF Cosine Similarity
This metric calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word in each text segment, optionally filtering out common Tibetan stopwords. 

TF-IDF gives higher weight to words that are frequent within a particular segment but relatively rare across the entire collection of segments. This helps identify terms that are characteristic or discriminative for a segment. When stopword filtering is enabled, the TF-IDF scores better reflect genuinely significant terms by excluding common particles and function words.

Each segment is represented as a vector of these TF-IDF scores, and the cosine similarity is computed between these vectors. A score closer to 1 indicates that the two segments share more important, distinguishing terms, suggesting they cover similar specific topics or themes.

**Stopword Filtering**: When enabled (via the "Filter Stopwords" checkbox), common Tibetan particles and function words are filtered out. This can be toggled on/off to compare results with and without stopwords.
""",
        }
        heatmap_tabs = {}
        gr.Markdown("## Detailed Metric Analysis", elem_classes="gr-markdown")
        with gr.Tabs(elem_id="heatmap-tab-group"):
            for metric_key, descriptive_title in heatmap_titles.items():
                with gr.Tab(metric_key):
                    if metric_key in metric_tooltips:
                        gr.Markdown(value=metric_tooltips[metric_key])
                    else:
                        gr.Markdown(
                            value=f"### {metric_key}\nDescription not found."
                        )  # Fallback
                    heatmap_tabs[metric_key] = gr.Plot(
                        label=f"Heatmap: {metric_key}", show_label=False
                    )

        # The outputs in process_btn.click should use the short metric names as keys for heatmap_tabs
        # e.g., heatmap_tabs["Jaccard Similarity (%)"]
        # Ensure the plot is part of the layout. This assumes plots are displayed sequentially
        # within the current gr.Tab("Results"). If they are in specific TabItems, this needs adjustment.
        # For now, this modification focuses on creating the plot object and making it an output.
        # The visual placement depends on how Gradio renders children of gr.Tab or if there's another container.

        warning_box = gr.Markdown(visible=False)

        def run_pipeline(files, enable_semantic, model_name, stopwords_option="Aggressive (All function words)", progress=gr.Progress()):
            """Run the text analysis pipeline on the uploaded files.

            Args:
                files: List of uploaded files
                enable_semantic: Whether to compute semantic similarity
                model_name: Name of the embedding model to use
                stopwords_option: Stopword filtering level (None, Standard, or Aggressive)
                progress: Gradio progress indicator

            Returns:
                Tuple of (metrics_df, heatmap_jaccard, heatmap_lcs, heatmap_semantic, heatmap_tfidf, word_count_fig)
            """
            # Initialize progress tracking
            try:
                progress_tracker = gr.Progress()
            except Exception as e:
                logger.warning(f"Could not initialize progress tracker: {e}")
                progress_tracker = None
            # Initialize all return values to ensure defined paths for all outputs
            csv_path_res = None
            metrics_preview_df_res = None # Can be a DataFrame or a string message
            word_count_fig_res = None
            jaccard_heatmap_res = None
            lcs_heatmap_res = None
            semantic_heatmap_res = None
            tfidf_heatmap_res = None
            warning_update_res = gr.update(value="", visible=False) # Default: no warning

            """
            Processes uploaded files, computes metrics, generates visualizations, and prepares outputs for the UI.

            Args:
                files (List[FileStorage]): A list of file objects uploaded by the user.

            Returns:
                tuple: A tuple containing the following elements in order:
                    - csv_path (str | None): Path to the generated CSV results file, or None on error.
                    - metrics_preview_df (pd.DataFrame | str | None): DataFrame for metrics preview, error string, or None.
                    - word_count_fig (matplotlib.figure.Figure | None): Plot of word counts, or None on error.
                    - jaccard_heatmap (matplotlib.figure.Figure | None): Jaccard similarity heatmap, or None.
                    - lcs_heatmap (matplotlib.figure.Figure | None): LCS heatmap, or None.
                    - semantic_heatmap (matplotlib.figure.Figure | None): Semantic similarity heatmap, or None.
                    - warning_update (gr.update): Gradio update for the warning box.
            """
            # Check if files are provided
            if not files:
                return (
                    None,
                    "Please upload files to analyze.",
                    None,  # word_count_plot
                    None,  # jaccard_heatmap
                    None,  # lcs_heatmap
                    None,  # semantic_heatmap
                    None,  # tfidf_heatmap
                    gr.update(value="Please upload files.", visible=True),
                )
                
            # Check file size limits (10MB per file)
            for file in files:
                file_size_mb = Path(file.name).stat().st_size / (1024 * 1024)
                if file_size_mb > 10:
                    return (
                        None,
                        f"File '{Path(file.name).name}' exceeds the 10MB size limit (size: {file_size_mb:.2f}MB).",
                        None, None, None, None, None,
                        gr.update(value=f"Error: File '{Path(file.name).name}' exceeds the 10MB size limit.", visible=True),
                    )

            try:
                if progress_tracker is not None:
                    try:
                        progress_tracker(0.1, desc="Preparing files...")
                    except Exception as e:
                        logger.warning(f"Progress update error (non-critical): {e}")
                
                # Get filenames and read file contents
                filenames = [
                    Path(file.name).name for file in files
                ]  # Use Path().name to get just the filename
                text_data = {}
                
                # Read files with progress updates
                for i, file in enumerate(files):
                    file_path = Path(file.name)
                    filename = file_path.name
                    if progress_tracker is not None:
                        try:
                            progress_tracker(0.1 + (0.1 * (i / len(files))), desc=f"Reading file: {filename}")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    
                    try:
                        text_data[filename] = file_path.read_text(encoding="utf-8-sig")
                    except UnicodeDecodeError:
                        # Try with different encodings if UTF-8 fails
                        try:
                            text_data[filename] = file_path.read_text(encoding="utf-16")
                        except UnicodeDecodeError:
                            return (
                                None,
                                f"Error: Could not decode file '{filename}'. Please ensure it contains valid Tibetan text in UTF-8 or UTF-16 encoding.",
                                None, None, None, None, None,
                                gr.update(value=f"Error: Could not decode file '{filename}'.", visible=True),
                            )

                # Configure semantic similarity
                enable_semantic_bool = enable_semantic == "Yes"
                
                if progress_tracker is not None:
                    try:
                        progress_tracker(0.2, desc="Loading model..." if enable_semantic_bool else "Processing text...")
                    except Exception as e:
                        logger.warning(f"Progress update error (non-critical): {e}")
                
                # Process texts with selected model
                # Convert stopword option to appropriate parameters
                use_stopwords = stopwords_option != "None (No filtering)"
                use_lite_stopwords = stopwords_option == "Standard (Common particles only)"
                
                df_results, word_counts_df_data, warning_raw = process_texts(
                    text_data, filenames, 
                    enable_semantic=enable_semantic_bool, 
                    model_name=model_name,
                    use_stopwords=use_stopwords,
                    use_lite_stopwords=use_lite_stopwords,
                    progress_callback=progress_tracker
                )

                if df_results.empty:
                    warning_md = f"**⚠️ Warning:** {warning_raw}" if warning_raw else ""
                    warning_message = (
                        "No common chapters found or results are empty. " + warning_md
                    )
                    metrics_preview_df_res = warning_message
                    warning_update_res = gr.update(value=warning_message, visible=True)
                    # Results for this case are set, then return
                else:
                    # Generate visualizations
                    if progress_tracker is not None:
                        try:
                            progress_tracker(0.8, desc="Generating visualizations...")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    
                    # heatmap_titles is already defined in the outer scope of main_interface
                    heatmaps_data = generate_visualizations(
                        df_results, descriptive_titles=heatmap_titles
                    )
                    
                    # Generate word count chart
                    if progress_tracker is not None:
                        try:
                            progress_tracker(0.9, desc="Creating word count chart...")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    word_count_fig_res = generate_word_count_chart(word_counts_df_data)
                    
                    # Save results to CSV
                    if progress_tracker is not None:
                        try:
                            progress_tracker(0.95, desc="Saving results...")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    csv_path_res = "results.csv"
                    df_results.to_csv(csv_path_res, index=False)
                    
                    # Prepare final output
                    warning_md = f"**⚠️ Warning:** {warning_raw}" if warning_raw else ""
                    metrics_preview_df_res = df_results.head(10)

                    jaccard_heatmap_res = heatmaps_data.get("Jaccard Similarity (%)")
                    lcs_heatmap_res = heatmaps_data.get("Normalized LCS")
                    semantic_heatmap_res = heatmaps_data.get(
                        "Semantic Similarity"
                    )
                    tfidf_heatmap_res = heatmaps_data.get("TF-IDF Cosine Sim")
                    warning_update_res = gr.update(
                        visible=bool(warning_raw), value=warning_md
                    )

            except Exception as e:
                logger.error(f"Error in run_pipeline: {e}", exc_info=True)
                # metrics_preview_df_res and warning_update_res are set here.
                # Other plot/file path variables will retain their initial 'None' values set at function start.
                metrics_preview_df_res = f"Error: {str(e)}" 
                warning_update_res = gr.update(value=f"Error: {str(e)}", visible=True)

            return (
                csv_path_res,
                metrics_preview_df_res,
                word_count_fig_res,
                jaccard_heatmap_res,
                lcs_heatmap_res,
                semantic_heatmap_res,
                tfidf_heatmap_res,
                warning_update_res
            )

        # Function to interpret results using LLM
        def interpret_results(csv_path):
            try:
                if not csv_path or not Path(csv_path).exists():
                    return "Please run the analysis first to generate results."
                
                # Read the CSV file
                df_results = pd.read_csv(csv_path)
                
                # Get interpretation from LLM (using Puter API - free Claude 3.5 Sonnet)
                interpretation = get_interpretation(df_results)
                
                return interpretation
            except Exception as e:
                logger.error(f"Error in interpret_results: {e}", exc_info=True)
                return f"Error interpreting results: {str(e)}"
        
        process_btn.click(
            fn=run_pipeline,
            inputs=[file_input, semantic_toggle_radio, model_dropdown, stopwords_dropdown],
            outputs=[
                csv_output,
                metrics_preview,
                word_count_plot,
                heatmap_tabs["Jaccard Similarity (%)"],
                heatmap_tabs["Normalized LCS"],
                heatmap_tabs["Semantic Similarity"],
                heatmap_tabs["TF-IDF Cosine Sim"],
                warning_box,
            ]
        )
        
        # Connect the interpret button
        interpret_btn.click(
            fn=interpret_results,
            inputs=[csv_output],
            outputs=interpretation_output
        )
        
    return demo


if __name__ == "__main__":
    demo = main_interface()
    demo.launch()