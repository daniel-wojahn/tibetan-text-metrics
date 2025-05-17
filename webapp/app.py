import gradio as gr
from pathlib import Path
from pipeline.process import process_texts
from pipeline.visualize import generate_visualizations, generate_word_count_chart
import logging

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
        word_count_plot = gr.Plot(label="Word Counts per Segment")
        # Heatmap tabs for each metric
        heatmap_titles = {
            "Jaccard Similarity (%)": "Jaccard Similarity (%): Higher scores (brighter) mean more shared unique words.",
            "Normalized LCS": "Normalized LCS: Higher scores (brighter) mean longer shared sequences of words.",
            "Semantic Similarity (BuddhistNLP)": "Semantic Similarity (BuddhistNLP - using word embeddings/experimental): Higher scores (brighter) mean more similar meanings.",
            "TF-IDF Cosine Sim": "TF-IDF Cosine Similarity: Higher scores mean texts share more important, distinctive vocabulary.",
        }

        metric_tooltips = {
            "Jaccard Similarity (%)": """
### Jaccard Similarity (%)
This metric quantifies the lexical overlap between two text segments by comparing their sets of *unique* words, after **filtering out common Tibetan stopwords**. 
It essentially answers the question: 'Of all the distinct, meaningful words found across these two segments, what proportion of them are present in both?' 
It is calculated as `(Number of common unique meaningful words) / (Total number of unique meaningful words in both texts combined) * 100`. 
Jaccard Similarity is insensitive to word order and word frequency; it only cares whether a unique meaningful word is present or absent. 
A higher percentage indicates a greater overlap in the significant vocabularies used in the two segments.

**Stopword Filtering**: uses a range of stopwords to filter out common Tibetan words that do not contribute to the semantic content of the text.
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
### Semantic Similarity (Experimental)
Utilizes the `<a href="https://huggingface.co/buddhist-nlp/buddhist-sentence-similarity">buddhist-nlp/buddhist-sentence-similarity</a>` model to compute the cosine similarity between the semantic embeddings of text segments. 
This model is fine-tuned for Buddhist studies texts and captures nuances in meaning. 
For texts exceeding the model's 512-token input limit, an automated chunking strategy is employed: texts are divided into overlapping chunks, each chunk is embedded, and the resulting chunk embeddings are averaged (mean pooling) to produce a single representative vector for the entire segment before comparison.

**Note**: This metric is experimental and may not perform well for all texts. It is recommended to use it in combination with other metrics for a more comprehensive analysis.
""",
            "TF-IDF Cosine Sim": """
### TF-IDF Cosine Similarity
This metric first calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores for each word in each text segment, **after filtering out common Tibetan stopwords**. 
TF-IDF gives higher weight to words that are frequent within a particular segment but relatively rare across the entire collection of segments. 
This helps to identify terms that are characteristic or discriminative for a segment. By excluding stopwords, the TF-IDF scores better reflect genuinely significant terms.
Each segment is then represented as a vector of these TF-IDF scores. 
Finally, the cosine similarity is computed between these vectors. 
A score closer to 1 indicates that the two segments share more of these important, distinguishing terms, suggesting they cover similar specific topics or themes.

**Stopword Filtering**: uses a range of stopwords to filter out common Tibetan words that do not contribute to the semantic content of the text.
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

        def run_pipeline(files, enable_semantic_str):
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

            try:
                filenames = [
                    Path(file.name).name for file in files
                ]  # Use Path().name to get just the filename
                text_data = {
                    Path(file.name)
                    .name: Path(file.name)
                    .read_text(encoding="utf-8-sig")
                    for file in files
                }

                enable_semantic_bool = enable_semantic_str == "Yes"
                df_results, word_counts_df_data, warning_raw = process_texts(
                    text_data, filenames, enable_semantic=enable_semantic_bool
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
                    # heatmap_titles is already defined in the outer scope of main_interface
                    heatmaps_data = generate_visualizations(
                        df_results, descriptive_titles=heatmap_titles
                    )
                    word_count_fig_res = generate_word_count_chart(word_counts_df_data)
                    csv_path_res = "results.csv"
                    df_results.to_csv(csv_path_res, index=False)
                    warning_md = f"**⚠️ Warning:** {warning_raw}" if warning_raw else ""
                    metrics_preview_df_res = df_results.head(10)

                    jaccard_heatmap_res = heatmaps_data.get("Jaccard Similarity (%)")
                    lcs_heatmap_res = heatmaps_data.get("Normalized LCS")
                    semantic_heatmap_res = heatmaps_data.get(
                        "Semantic Similarity (BuddhistNLP)"
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
                warning_update_res,
            )

        process_btn.click(
            run_pipeline,
            inputs=[file_input, semantic_toggle_radio],
            outputs=[
                csv_output,
                metrics_preview,
                word_count_plot,
                heatmap_tabs["Jaccard Similarity (%)"],
                heatmap_tabs["Normalized LCS"],
                heatmap_tabs["Semantic Similarity (BuddhistNLP)"],
                heatmap_tabs["TF-IDF Cosine Sim"],
                warning_box,
            ],
        )
    return demo


if __name__ == "__main__":
    demo = main_interface()
    demo.launch()
