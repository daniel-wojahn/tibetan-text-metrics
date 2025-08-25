import gradio as gr
from pathlib import Path
from pipeline.process import process_texts
from pipeline.visualize import generate_visualizations, generate_word_count_chart
from pipeline.llm_service import LLMService
from pipeline.progressive_ui import ProgressiveUI, create_progressive_callback
import logging
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
from theme import tibetan_theme

load_dotenv()

logger = logging.getLogger(__name__)
def main_interface():
    with gr.Blocks(
        theme=tibetan_theme,
        title="Tibetan Text Metrics Web App",
        css=tibetan_theme.get_css_string() + ".metric-description, .step-box { padding: 1.5rem !important; }"
    ) as demo:
        gr.Markdown(
            """# Tibetan Text Metrics Web App
<span style='font-size:18px;'>A user-friendly web application for analyzing textual similarities and variations in Tibetan manuscripts, providing a graphical interface to the core functionalities of the [Tibetan Text Metrics (TTM)](https://github.com/daniel-wojahn/tibetan-text-metrics) project. Powered by advanced language models via OpenRouter for in-depth text analysis.</span>
        """,

            elem_classes="gr-markdown",
        )

        with gr.Row(elem_id="steps-row"):
            with gr.Column(scale=1, elem_classes="step-column"):
                with gr.Group(elem_classes="step-box"):
                    gr.Markdown(
                        """
                    ## Step 1: Upload Your Tibetan Text Files
                    <span style='font-size:16px;'>Upload two or more `.txt` files. Each file should contain Unicode Tibetan text, segmented into chapters/sections if possible using the marker '༈' (<i>sbrul shad</i>).</span>
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
                with gr.Group(elem_classes="step-box"):
                    gr.Markdown(
                        """## Step 2: Configure and run the analysis
<span style='font-size:16px;'>Choose your analysis options and click the button below to compute metrics and view results. For meaningful analysis, ensure your texts are segmented by chapter or section using the marker '༈' (<i>sbrul shad</i>). The tool will split files based on this marker.</span>
                    """,
                        elem_classes="gr-markdown",
                    )
                    semantic_toggle_radio = gr.Radio(
                        label="Compute semantic similarity? (Experimental)",
                        choices=["Yes", "No"],
                        value="No",
                        info="Semantic similarity will be time-consuming. Choose 'No' to speed up analysis if these metrics are not required.",
                        elem_id="semantic-radio-group",
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "sentence-transformers/LaBSE"
                        ],
                        label="Select Embedding Model",
                        value="sentence-transformers/LaBSE",
                        info="Select the embedding model to use for semantic similarity analysis. Only Hugging Face sentence-transformers are supported."
                    )
                    
                    with gr.Accordion("Advanced Options", open=False):
                        batch_size_slider = gr.Slider(
                            minimum=1,
                            maximum=64,
                            value=8,
                            step=1,
                            label="Batch Size (for Hugging Face models)",
                            info="Adjust based on your hardware (VRAM). Lower this if you encounter memory issues."
                        )
                        progress_bar_checkbox = gr.Checkbox(
                            label="Show Embedding Progress Bar",
                            value=False,
                            info="Display a progress bar during embedding generation. Useful for large datasets."
                        )

                    stopwords_dropdown = gr.Dropdown(
                        label="Stopword Filtering",
                        choices=[
                            "None (No filtering)", 
                            "Standard (Common particles only)", 
                            "Aggressive (All function words)"
                        ],
                        value="Standard (Common particles only)",  # Default
                        info="Choose how aggressively to filter out common Tibetan particles and function words when calculating similarity. This helps focus on meaningful content words."
                    )
                    
                    fuzzy_toggle_radio = gr.Radio(
                        label="Enable Fuzzy String Matching",
                        choices=["Yes", "No"],
                        value="Yes",
                        info="Fuzzy matching helps detect similar but not identical text segments. Useful for identifying variations and modifications."
                    )
                    
                    fuzzy_method_dropdown = gr.Dropdown(
                        label="Fuzzy Matching Method",
                        choices=[
                            "token_set - Order-independent matching",
                            "token_sort - Order-normalized matching",
                            "partial - Best partial matching",
                            "ratio - Simple ratio matching"
                        ],
                        value="token_set - Order-independent matching",
                        info="Select the fuzzy matching algorithm to use:\n\n• token_set: Best for texts with different word orders and partial overlaps. Compares unique words regardless of their order (recommended for Tibetan texts).\n\n• token_sort: Good for texts with different word orders but similar content. Sorts words alphabetically before comparing.\n\n• partial: Best for finding shorter strings within longer ones. Useful when one text is a fragment of another.\n\n• ratio: Simple Levenshtein distance ratio. Best for detecting small edits and typos in otherwise identical texts."
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
        # States for data persistence
        state_text_data = gr.State()
        state_df_results = gr.State()
        
        # LLM Interpretation components
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "## AI Analysis\n*The AI will analyze your text similarities and provide insights into patterns and relationships.*",
                    elem_classes="gr-markdown"
                )
                
                # Add the interpret button
                with gr.Row():
                    interpret_btn = gr.Button(
                        "Help Interpret Results",
                        variant="primary",
                        elem_id="interpret-btn"
                    )
                # Create a placeholder message with proper formatting and structure
                initial_message = """
## Analysis of Tibetan Text Similarity Metrics

<small>*Click the 'Help Interpret Results' button above to generate an AI-powered analysis of your similarity metrics.*</small>
"""
                interpretation_output = gr.Markdown(
                    value=initial_message,
                    elem_id="llm-analysis"
                )
        
        # Heatmap tabs for each metric
        heatmap_titles = {
            "Jaccard Similarity (%)": "Higher scores mean more shared unique words.",
            "Normalized LCS": "Higher scores mean longer shared sequences of words.",
            "Fuzzy Similarity": "Higher scores mean more similar text with fuzzy matching tolerance for variations.",
            "Semantic Similarity": "Higher scores mean more similar meanings.",
            "Word Counts": "Word Counts: Bar chart showing the number of words in each segment after tokenization.",
        }

        metric_tooltips = {
            "Jaccard Similarity (%)": """
### Jaccard Similarity (%)
This metric quantifies the lexical overlap between two text segments by comparing their sets of *unique* words, optionally filtering out common Tibetan stopwords. 

It essentially answers the question: 'Of all the distinct words found across these two segments, what proportion of them are present in both?' It is calculated as `(Number of common unique words) / (Total number of unique words in both texts combined) * 100`. 

Jaccard Similarity is insensitive to word order and word frequency; it only cares whether a unique word is present or absent. A higher percentage indicates a greater overlap in the vocabularies used in the two segments.

**Stopword Filtering**: When enabled (via the "Filter Stopwords" checkbox), common Tibetan particles and function words are filtered out before comparison. This helps focus on meaningful content words rather than grammatical elements.
""",
            "Fuzzy Similarity": """
### Fuzzy Similarity
This metric measures the approximate string similarity between text segments using fuzzy matching algorithms from TheFuzz library. Unlike exact matching metrics, fuzzy similarity can detect similarities even when texts contain variations, misspellings, or different word orders.

Fuzzy similarity is particularly useful for Tibetan texts that may have orthographic variations, scribal differences, or regional spelling conventions. It provides a score between 0 and 1, where higher values indicate greater similarity.

**Available Methods**:
- **Token Set Ratio**: Compares the unique words in each text regardless of order (best for texts with different word arrangements)
- **Token Sort Ratio**: Normalizes word order before comparison (good for texts with similar content but different ordering)
- **Partial Ratio**: Finds the best matching substring (useful for texts where one is contained within the other)
- **Simple Ratio**: Direct character-by-character comparison (best for detecting minor variations)

**Stopword Filtering**: When enabled, common Tibetan particles and function words are filtered out before comparison, focusing on meaningful content words.
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
This metric measures similarity in meaning between text segments using sentence-transformer models from Hugging Face (e.g., LaBSE). Text segments are embedded into high-dimensional vectors and compared via cosine similarity. Scores closer to 1 indicate higher semantic overlap.

Key points:
- Context-aware embeddings capture nuanced meanings and relationships.
- Designed for sentence/segment-level representations, not just words.
- Works well alongside Jaccard and LCS for a holistic view.
- Stopword filtering: When enabled, common Tibetan particles and function words are filtered before embedding to focus on content-bearing terms.
""",
            "Word Counts": """
### Word Counts per Segment
This chart displays the number of words in each segment of your texts after tokenization.

The word count is calculated after applying the selected tokenization and stopword filtering options. This visualization helps you understand the relative sizes of different text segments and can reveal patterns in text structure across your documents.

**Key points**:
- Longer bars indicate segments with more words
- Segments are grouped by source document
- Useful for identifying structural patterns and content distribution
- Can help explain similarity metric variations (longer texts may show different patterns)
""",
            "Structural Analysis": """
### Structural Analysis
This advanced analysis examines the structural relationships between text segments across your documents. It identifies patterns of similarity and difference that may indicate textual dependencies, common sources, or editorial modifications.

The structural analysis combines multiple similarity metrics to create a comprehensive view of how text segments relate to each other, highlighting potential stemmatic relationships and textual transmission patterns.

**Key points**:
- Identifies potential source-target relationships between texts
- Visualizes text reuse patterns across segments
- Helps reconstruct possible stemmatic relationships
- Provides insights into textual transmission and editorial history

**Note**: This analysis is computationally intensive and only available after the initial metrics calculation is complete.
"""

        }
        heatmap_tabs = {}
        gr.Markdown("## Detailed Metric Analysis", elem_classes="gr-markdown")
        
        with gr.Tabs(elem_id="heatmap-tab-group"):
            # Process all metrics
            metrics_to_display = heatmap_titles
            
            for metric_key, descriptive_title in metrics_to_display.items():
                with gr.Tab(metric_key):
                    # Set CSS class based on metric type
                    if metric_key == "Jaccard Similarity (%)":
                        css_class = "metric-info-accordion jaccard-info"
                        accordion_title = "Understanding Vocabulary Overlap"
                    elif metric_key == "Normalized LCS":
                        css_class = "metric-info-accordion lcs-info"
                        accordion_title = "Understanding Sequence Patterns"
                    elif metric_key == "Fuzzy Similarity":
                        css_class = "metric-info-accordion fuzzy-info"
                        accordion_title = "Understanding Fuzzy Matching"
                    elif metric_key == "Semantic Similarity":
                        css_class = "metric-info-accordion semantic-info"
                        accordion_title = "Understanding Meaning Similarity"
                    elif metric_key == "Word Counts":
                        css_class = "metric-info-accordion wordcount-info"
                        accordion_title = "Understanding Text Length"
                    else:
                        css_class = "metric-info-accordion"
                        accordion_title = f"About {metric_key}"
                    
                    # Create the accordion with appropriate content
                    with gr.Accordion(accordion_title, open=False, elem_classes=css_class):
                        if metric_key == "Word Counts":
                            gr.Markdown("""
                            ### Word Counts per Segment
                            This chart displays the number of words in each segment of your texts after tokenization.
                            """)
                        elif metric_key in metric_tooltips:
                            gr.Markdown(value=metric_tooltips[metric_key], elem_classes="metric-description")
                        else:
                            gr.Markdown(value=f"### {metric_key}\nDescription not found.")
                    
                    # Add the appropriate plot
                    if metric_key == "Word Counts":
                        word_count_plot = gr.Plot(label="Word Counts per Segment", show_label=False, scale=1, elem_classes="metric-description")
                    else:
                        heatmap_tabs[metric_key] = gr.Plot(label=f"Heatmap: {metric_key}", show_label=False, elem_classes="metric-heatmap")

            # Structural Analysis Tab
            # Structural analysis tab removed - see dedicated collation app
        # For now, this modification focuses on creating the plot object and making it an output.
        # The visual placement depends on how Gradio renders children of gr.Tab or if there's another container.

        warning_box = gr.Markdown(visible=False)
        
        # Create a container for metric progress indicators
        with gr.Row(visible=False) as progress_container:
            # Progress indicators will be created dynamically by ProgressiveUI
            gr.Markdown("Metric progress will appear here during analysis")

        def run_pipeline(files, enable_semantic, enable_fuzzy, fuzzy_method, model_name, stopwords_option, batch_size, show_progress, progress=gr.Progress()):
            """Processes uploaded files, computes metrics, generates visualizations, and prepares outputs for the UI.
            
            Args:
                files: A list of file objects uploaded by the user.
                enable_semantic: Whether to compute semantic similarity.
                enable_fuzzy: Whether to compute fuzzy string similarity.
                fuzzy_method: The fuzzy matching method to use.
                model_name: Name of the embedding model to use.
                stopwords_option: Stopword filtering level (None, Standard, or Aggressive).
                batch_size: Batch size for embedding generation.
                show_progress: Whether to show progress bars during embedding.
                progress: Gradio progress indicator.
                
            Returns:
                tuple: Results for UI components including metrics, visualizations, and state.
            """
            # Initialize return values with defaults
            csv_path_res = None
            metrics_preview_df_res = pd.DataFrame()
            word_count_fig_res = None
            jaccard_heatmap_res = None
            lcs_heatmap_res = None
            fuzzy_heatmap_res = None
            semantic_heatmap_res = None
            warning_update_res = gr.update(visible=False)
            state_text_data_res = None
            state_df_results_res = None
            
            # Create a ProgressiveUI instance for handling progressive updates
            progressive_ui = ProgressiveUI(
                metrics_preview=metrics_preview,
                word_count_plot=word_count_plot,
                jaccard_heatmap=heatmap_tabs["Jaccard Similarity (%)"],
                lcs_heatmap=heatmap_tabs["Normalized LCS"],
                fuzzy_heatmap=heatmap_tabs["Fuzzy Similarity"],
                semantic_heatmap=heatmap_tabs["Semantic Similarity"],
                warning_box=warning_box,
                progress_container=progress_container,
                heatmap_titles=heatmap_titles
            )
            
            # Make progress container visible during analysis
            progress_container.update(visible=True)
            
            # Create a progressive callback function
            progressive_callback = create_progressive_callback(progressive_ui)
            # Check if files are provided
            if not files:
                return (
                    None,
                    pd.DataFrame({"Message": ["Please upload files to analyze."]}),
                    None,  # word_count_plot
                    None,  # jaccard_heatmap
                    None,  # lcs_heatmap
                    None,  # fuzzy_heatmap
                    None,  # semantic_heatmap
                    None,  # warning update
                    None,  # state_text_data
                    None  # state_df_results
                )
                
            # Check file size limits (10MB per file)
            for file in files:
                file_size_mb = Path(file.name).stat().st_size / (1024 * 1024)
                if file_size_mb > 10:
                    return (
                        None,
                        pd.DataFrame({"Error": [f"File '{Path(file.name).name}' exceeds the 10MB size limit (size: {file_size_mb:.2f}MB)."]}),
                        None,  # word_count_plot
                        None,  # jaccard_heatmap
                        None,  # lcs_heatmap
                        None,  # fuzzy_heatmap
                        None,  # semantic_heatmap
                        gr.update(value=f"Error: File '{Path(file.name).name}' exceeds the 10MB size limit.", visible=True),
                        None,  # state_text_data
                        None  # state_df_results
                    )

            try:
                if progress is not None:
                    try:
                        progress(0.1, desc="Preparing files...")
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
                    if progress is not None:
                        try:
                            progress(0.1 + (0.1 * (i / len(files))), desc=f"Reading file: {filename}")
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
                                pd.DataFrame({"Error": [f"Could not decode file '{filename}'. Please ensure it contains valid Tibetan text in UTF-8 or UTF-16 encoding."]}),
                                None,  # word_count_plot
                                None,  # jaccard_heatmap
                                None,  # lcs_heatmap
                                None,  # fuzzy_heatmap
                                None,  # semantic_heatmap
                                gr.update(value=f"Error: Could not decode file '{filename}'.", visible=True),
                                None,  # state_text_data
                                None  # state_df_results
                            )

                # Configure semantic similarity and fuzzy matching
                enable_semantic_bool = enable_semantic == "Yes"
                enable_fuzzy_bool = enable_fuzzy == "Yes"
                
                # Extract the fuzzy method from the dropdown value
                fuzzy_method_value = fuzzy_method.split(' - ')[0] if fuzzy_method else 'token_set'
                
                if progress is not None:
                    try:
                        progress(0.2, desc="Loading model..." if enable_semantic_bool else "Processing text...")
                    except Exception as e:
                        logger.warning(f"Progress update error (non-critical): {e}")
                
                # Process texts with selected model
                # Convert stopword option to appropriate parameters
                use_stopwords = stopwords_option != "None (No filtering)"
                use_lite_stopwords = stopwords_option == "Standard (Common particles only)"
                
                # For Hugging Face models, the UI value is the correct model ID
                internal_model_id = model_name

                df_results, word_counts_df_data, warning_raw = process_texts(
                    text_data=text_data,
                    filenames=filenames,
                    enable_semantic=enable_semantic_bool,
                    enable_fuzzy=enable_fuzzy_bool,
                    fuzzy_method=fuzzy_method_value,
                    model_name=internal_model_id,
                    use_stopwords=use_stopwords,
                    use_lite_stopwords=use_lite_stopwords,
                    progress_callback=progress,
                    progressive_callback=progressive_callback,
                    batch_size=batch_size,
                    show_progress_bar=show_progress
                )

                if df_results.empty:
                    warning_md = f"**⚠️ Warning:** {warning_raw}" if warning_raw else ""
                    warning_message = "No common chapters found or results are empty. " + (warning_raw or "")
                    metrics_preview_df_res = pd.DataFrame({"Message": [warning_message]})
                    warning_update_res = gr.update(value=warning_md or warning_message, visible=True)
                    # No structural analysis in this app
                else:
                    # Generate visualizations
                    if progress is not None:
                        try:
                            progress(0.8, desc="Generating visualizations...")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    
                    # heatmap_titles is already defined in the outer scope of main_interface
                    heatmaps_data = generate_visualizations(
                        df_results, descriptive_titles=heatmap_titles
                    )
                    
                    # Generate word count chart
                    if progress is not None:
                        try:
                            progress(0.9, desc="Creating word count chart...")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    word_count_fig_res = generate_word_count_chart(word_counts_df_data)
                    
                    # Store state data for potential future use
                    state_text_data_res = text_data
                    state_df_results_res = df_results
                    logger.info("Analysis complete, storing state data")
                    
                    # Save results to CSV
                    if progress is not None:
                        try:
                            progress(0.95, desc="Saving results...")
                        except Exception as e:
                            logger.warning(f"Progress update error (non-critical): {e}")
                    csv_path_res = "results.csv"
                    df_results.to_csv(csv_path_res, index=False)
                    
                    # Prepare final output
                    warning_md = f"**⚠️ Warning:** {warning_raw}" if warning_raw else ""
                    metrics_preview_df_res = df_results.head(10)

                    jaccard_heatmap_res = heatmaps_data.get("Jaccard Similarity (%)")
                    lcs_heatmap_res = heatmaps_data.get("Normalized LCS")
                    fuzzy_heatmap_res = heatmaps_data.get("Fuzzy Similarity")
                    semantic_heatmap_res = heatmaps_data.get(
                        "Semantic Similarity"
                    )
                    # TF-IDF has been completely removed
                    warning_update_res = gr.update(
                        visible=bool(warning_raw), value=warning_md
                    )

            except Exception as e:
                logger.error(f"Error in run_pipeline: {e}", exc_info=True)
                # Ensure DataFrame for metrics preview on error
                metrics_preview_df_res = pd.DataFrame({"Error": [str(e)]})
                warning_update_res = gr.update(value=f"Error: {str(e)}", visible=True)

            return (
                csv_path_res,
                metrics_preview_df_res,
                word_count_fig_res,
                jaccard_heatmap_res,
                lcs_heatmap_res,
                fuzzy_heatmap_res,
                semantic_heatmap_res,
                warning_update_res,
                state_text_data_res,
                state_df_results_res,
            )

        # Function to interpret results using LLM
        def interpret_results(csv_path, progress=gr.Progress()):
            try:
                if not csv_path or not Path(csv_path).exists():
                    return "Please run the analysis first to generate results."
                
                # Read the CSV file
                df_results = pd.read_csv(csv_path)
                
                # Show detailed progress messages with percentages
                progress(0, desc="Preparing data for analysis...")
                progress(0.1, desc="Analyzing similarity patterns...")
                progress(0.2, desc="Connecting to Mistral 7B via OpenRouter...")
                
                # Get interpretation from LLM (using OpenRouter API)
                progress(0.3, desc="Generating scholarly interpretation (this may take 20-40 seconds)...")
                llm_service = LLMService()
                interpretation = llm_service.analyze_similarity(df_results)
                
                # Simulate completion steps
                progress(0.9, desc="Formatting results...")
                progress(0.95, desc="Applying scholarly formatting...")
                
                # Completed
                progress(1.0, desc="Analysis complete!")
                
                # Add a timestamp to the interpretation
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                interpretation = f"{interpretation}\n\n<small>Analysis generated on {timestamp}</small>"
                return interpretation
            except Exception as e:
                logger.error(f"Error in interpret_results: {e}", exc_info=True)
                return f"Error interpreting results: {str(e)}"
        
        process_btn.click(
            fn=run_pipeline,
            inputs=[file_input, semantic_toggle_radio, fuzzy_toggle_radio, fuzzy_method_dropdown, model_dropdown, stopwords_dropdown, batch_size_slider, progress_bar_checkbox],
            outputs=[
                csv_output,
                metrics_preview,
                word_count_plot,
                heatmap_tabs["Jaccard Similarity (%)"],
                heatmap_tabs["Normalized LCS"],
                heatmap_tabs["Fuzzy Similarity"],
                heatmap_tabs["Semantic Similarity"],
                warning_box,
                state_text_data,
                state_df_results,
            ]
        )

        # Structural analysis functionality removed - see dedicated collation app
        
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