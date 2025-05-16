import plotly.graph_objects as go
import pandas as pd
import plotly.express as px  # For color palettes
import numpy as np  # Ensure numpy is imported, in case pivot_table uses it for aggfunc


def generate_visualizations(metrics_df: pd.DataFrame, descriptive_titles: dict = None):
    """
    Generate heatmap visualizations for all metrics.
    Args:
        metrics_df: DataFrame with similarity metrics (segment-level)
    Returns:
        heatmaps: dict of {metric_name: plotly Figure} for each metric
    """

    # Identify all numeric metric columns (exclude 'Text Pair' and 'Chapter')
    metric_cols = [
        col
        for col in metrics_df.columns
        if col not in ["Text Pair", "Chapter"] and metrics_df[col].dtype != object
    ]
    for col in metrics_df.columns:
        if "Pattern Similarity" in col and col not in metric_cols:
            metric_cols.append(col)

    # --- Heatmaps for each metric ---
    heatmaps = {}
    # Using 'Reds' colormap as requested for a red/white gradient.
    # Chapter 1 will be at the top of the Y-axis due to sort_index(ascending=False).
    for metric in metric_cols:
        # Check if all values for this metric are NaN
        if metrics_df[metric].isnull().all():
            heatmaps[metric] = None
            continue  # Move to the next metric

        pivot = metrics_df.pivot(index="Chapter", columns="Text Pair", values=metric)
        pivot = pivot.sort_index(ascending=False)  # Invert Y-axis: Chapter 1 at the top
        # Additional check: if pivot is empty or all NaNs after pivoting (e.g., due to single chapter comparisons)
        if pivot.empty or pivot.isnull().all().all():
            heatmaps[metric] = None
            continue

        cleaned_columns = [col.replace(".txt", "") for col in pivot.columns]
        cmap = "Reds"  # Apply 'Reds' colormap to all heatmaps
        text = [
            [f"{val:.2f}" if pd.notnull(val) else "" for val in row]
            for row in pivot.values
        ]
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=cleaned_columns,
                y=pivot.index,
                colorscale=cmap,
                zmin=float(np.nanmin(pivot.values)),
                zmax=float(np.nanmax(pivot.values)),
                text=text,
                texttemplate="%{text}",
                hovertemplate="Chapter %{y}<br>Text Pair: %{x}<br>Value: %{z:.2f}<extra></extra>",
                colorbar=dict(title=metric, thickness=20, tickfont=dict(size=14)),
            )
        )
        plot_title = (
            descriptive_titles.get(metric, metric) if descriptive_titles else metric
        )
        fig.update_layout(
            title=plot_title,
            xaxis_title="Text Pair",
            yaxis_title="Chapter",
            autosize=False,
            width=1350,
            height=1200,
            font=dict(size=16),
            margin=dict(l=140, b=80, t=60),
        )
        fig.update_xaxes(tickangle=30, tickfont=dict(size=16))
        fig.update_yaxes(tickfont=dict(size=16), autorange="reversed")
        # Ensure all integer chapter numbers are shown if the axis is numeric and reversed
        if pd.api.types.is_numeric_dtype(pivot.index):
            fig.update_yaxes(
                tickmode="array",
                tickvals=pivot.index,
                ticktext=[str(i) for i in pivot.index],
            )
        heatmaps[metric] = fig

    # Use all features including pattern similarities if present
    if not metrics_df.empty:
        # Remove '.txt' from Text Pair labels
        metrics_df = metrics_df.copy()
        metrics_df["Text Pair"] = metrics_df["Text Pair"].str.replace(
            ".txt", "", regex=False
        )
    return heatmaps


def generate_word_count_chart(word_counts_df: pd.DataFrame):
    """
    Generates a bar chart for word counts per segment (file/chapter).
    Args:
        word_counts_df: DataFrame with 'Filename', 'ChapterNumber', 'SegmentID', 'WordCount'.
    Returns:
        plotly Figure for the bar chart, or None if input is empty.
    """
    if word_counts_df.empty:
        return None

    fig = go.Figure()

    # Assign colors based on Filename
    unique_files = sorted(word_counts_df["Filename"].unique())
    colors = px.colors.qualitative.Plotly  # Get a default Plotly color sequence

    for i, filename in enumerate(unique_files):
        file_df = word_counts_df[word_counts_df["Filename"] == filename].sort_values(
            "ChapterNumber"
        )
        fig.add_trace(
            go.Bar(
                x=file_df["ChapterNumber"],
                y=file_df["WordCount"],
                name=filename,
                marker_color=colors[i % len(colors)],
                text=file_df["WordCount"],
                textposition="auto",
                customdata=file_df[["Filename"]],  # Pass Filename for hovertemplate
                hovertemplate="<b>File</b>: %{customdata[0]}<br>"
                + "<b>Chapter</b>: %{x}<br>"
                + "<b>Word Count</b>: %{y}<extra></extra>",
            )
        )

    fig.update_layout(
        title_text="Word Counts per Chapter (Grouped by File)",
        xaxis_title="Chapter Number",
        yaxis_title="Word Count",
        barmode="group",
        font=dict(size=14),
        legend_title_text="Filename",
        xaxis=dict(
            type="category"
        ),  # Treat chapter numbers as categories for distinct grouping
        autosize=True,
        margin=dict(l=80, r=50, b=100, t=50, pad=4),
    )
    # Ensure x-axis ticks are shown for all chapter numbers present
    all_chapter_numbers = sorted(word_counts_df["ChapterNumber"].unique())
    fig.update_xaxes(
        tickmode="array",
        tickvals=all_chapter_numbers,
        ticktext=[str(ch) for ch in all_chapter_numbers],
    )

    return fig
