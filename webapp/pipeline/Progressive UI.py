"""
Progressive UI module for the Tibetan Text Metrics app.

This module provides UI components and helper functions for progressive loading
and updating of metrics as they are computed.
"""

import gradio as gr
import pandas as pd
from typing import Dict, List, Any, Callable
from .progressive_loader import ProgressiveResult, MetricType
from .visualize import generate_visualizations, generate_word_count_chart
import logging

logger = logging.getLogger(__name__)

class ProgressiveUI:
    """
    Manages progressive UI updates for the Tibetan Text Metrics app.
    
    This class handles the incremental updates of UI components as metrics
    are computed, allowing for a more responsive user experience.
    """
    
    def __init__(self, 
                 metrics_preview: gr.Dataframe,
                 word_count_plot: gr.Plot,
                 jaccard_heatmap: gr.Plot,
                 lcs_heatmap: gr.Plot,
                 fuzzy_heatmap: gr.Plot,
                 semantic_heatmap: gr.Plot,
                 warning_box: gr.Markdown,
                 progress_container: gr.Row,
                 heatmap_titles: Dict[str, str],
                 structural_btn=None):
        """
        Initialize the ProgressiveUI.
        
        Args:
            metrics_preview: Gradio Dataframe component for metrics preview
            word_count_plot: Gradio Plot component for word count visualization
            jaccard_heatmap: Gradio Plot component for Jaccard similarity heatmap
            lcs_heatmap: Gradio Plot component for LCS similarity heatmap
            fuzzy_heatmap: Gradio Plot component for fuzzy similarity heatmap
            semantic_heatmap: Gradio Plot component for semantic similarity heatmap
            warning_box: Gradio Markdown component for warnings
            progress_container: Gradio Row component to hold progress indicators
            heatmap_titles: Dictionary mapping metric names to descriptive titles
        """
        self.metrics_preview = metrics_preview
        self.word_count_plot = word_count_plot
        self.jaccard_heatmap = jaccard_heatmap
        self.lcs_heatmap = lcs_heatmap
        self.fuzzy_heatmap = fuzzy_heatmap
        self.semantic_heatmap = semantic_heatmap
        self.warning_box = warning_box
        self.progress_container = progress_container
        self.heatmap_titles = heatmap_titles
        self.structural_btn = structural_btn
        
        # Create progress indicators for each metric
        with self.progress_container:
            self.jaccard_progress = gr.Markdown("🔄 **Jaccard Similarity:** Waiting...", elem_id="jaccard_progress")
            self.lcs_progress = gr.Markdown("🔄 **Normalized LCS:** Waiting...", elem_id="lcs_progress")
            self.fuzzy_progress = gr.Markdown("🔄 **Fuzzy Similarity:** Waiting...", elem_id="fuzzy_progress")
            self.semantic_progress = gr.Markdown("🔄 **Semantic Similarity:** Waiting...", elem_id="semantic_progress")
            self.word_count_progress = gr.Markdown("🔄 **Word Counts:** Waiting...", elem_id="word_count_progress")
        
        # Track which components have been updated
        self.updated_components = set()
        
    def update(self, result: ProgressiveResult) -> Dict[gr.components.Component, Any]:
        """
        Update UI components based on progressive results.
        
        Args:
            result: ProgressiveResult object containing the current state of computation
            
        Returns:
            Dictionary mapping Gradio components to their updated values
        """
        updates = {}
        
        # Always update metrics preview if we have data
        if not result.metrics_df.empty:
            updates[self.metrics_preview] = result.metrics_df.head(10)
        
        # Update warning if present
        if result.warning:
            warning_md = f"**⚠️ Warning:** {result.warning}" if result.warning else ""
            updates[self.warning_box] = gr.update(value=warning_md, visible=True)
        
        # Generate visualizations for completed metrics
        if not result.metrics_df.empty:
            # Generate heatmaps for available metrics
            heatmaps_data = generate_visualizations(
                result.metrics_df, descriptive_titles=self.heatmap_titles
            )
            
            # Update heatmaps and progress indicators for completed metrics
            for metric_type in result.completed_metrics:
                if metric_type == MetricType.JACCARD:
                    # Update progress indicator
                    updates[self.jaccard_progress] = "✅ **Jaccard Similarity:** Complete"
                    
                    # Update heatmap if not already updated
                    if self.jaccard_heatmap not in self.updated_components:
                        if "Jaccard Similarity (%)" in heatmaps_data:
                            updates[self.jaccard_heatmap] = heatmaps_data["Jaccard Similarity (%)"]
                            self.updated_components.add(self.jaccard_heatmap)
                        
                elif metric_type == MetricType.LCS:
                    # Update progress indicator
                    updates[self.lcs_progress] = "✅ **Normalized LCS:** Complete"
                    
                    # Update heatmap if not already updated
                    if self.lcs_heatmap not in self.updated_components:
                        if "Normalized LCS" in heatmaps_data:
                            updates[self.lcs_heatmap] = heatmaps_data["Normalized LCS"]
                            self.updated_components.add(self.lcs_heatmap)
                        
                elif metric_type == MetricType.FUZZY:
                    # Update progress indicator
                    updates[self.fuzzy_progress] = "✅ **Fuzzy Similarity:** Complete"
                    
                    # Update heatmap if not already updated
                    if self.fuzzy_heatmap not in self.updated_components:
                        if "Fuzzy Similarity" in heatmaps_data:
                            updates[self.fuzzy_heatmap] = heatmaps_data["Fuzzy Similarity"]
                            self.updated_components.add(self.fuzzy_heatmap)
                        
                elif metric_type == MetricType.SEMANTIC:
                    # Update progress indicator
                    updates[self.semantic_progress] = "✅ **Semantic Similarity:** Complete"
                    
                    # Update heatmap if not already updated
                    if self.semantic_heatmap not in self.updated_components:
                        if "Semantic Similarity" in heatmaps_data:
                            updates[self.semantic_heatmap] = heatmaps_data["Semantic Similarity"]
                            self.updated_components.add(self.semantic_heatmap)
        
        # Generate word count chart if we have data
        if not result.word_counts_df.empty:
            # Update progress indicator
            updates[self.word_count_progress] = "✅ **Word Counts:** Complete"
            
            # Update chart if not already updated
            if self.word_count_plot not in self.updated_components:
                updates[self.word_count_plot] = generate_word_count_chart(result.word_counts_df)
                self.updated_components.add(self.word_count_plot)
        
        # Update progress indicators for metrics in progress
        if not result.is_complete:
            # Update progress indicators for metrics that are still in progress
            if MetricType.JACCARD not in result.completed_metrics:
                updates[self.jaccard_progress] = "⏳ **Jaccard Similarity:** In progress..."
            if MetricType.LCS not in result.completed_metrics:
                updates[self.lcs_progress] = "⏳ **Normalized LCS:** In progress..."
            if MetricType.FUZZY not in result.completed_metrics:
                updates[self.fuzzy_progress] = "⏳ **Fuzzy Similarity:** In progress..."
            if MetricType.SEMANTIC not in result.completed_metrics:
                updates[self.semantic_progress] = "⏳ **Semantic Similarity:** In progress..."
            if self.word_count_plot not in self.updated_components:
                updates[self.word_count_progress] = "⏳ **Word Counts:** In progress..."
        else:
            # If computation is complete, enable structural button if available
            if self.structural_btn is not None:
                updates[self.structural_btn] = gr.update(interactive=True)
                logger.info("Enabling structural analysis button via progressive UI")
            
        return updates


def create_progressive_callback(progressive_ui: ProgressiveUI) -> Callable:
    """
    Create a callback function for progressive updates.
    
    Args:
        progressive_ui: ProgressiveUI instance to handle updates
        
    Returns:
        Callback function that can be passed to process_texts
    """
    def callback(metrics_df: pd.DataFrame, 
                word_counts_df: pd.DataFrame,
                completed_metrics: List[MetricType],
                warning: str,
                is_complete: bool) -> None:
        """
        Callback function for progressive updates.
        
        Args:
            metrics_df: DataFrame with current metrics
            word_counts_df: DataFrame with word counts
            completed_metrics: List of completed metric types
            warning: Warning message
            is_complete: Whether computation is complete
        """
        result = ProgressiveResult(
            metrics_df=metrics_df,
            word_counts_df=word_counts_df,
            completed_metrics=completed_metrics,
            warning=warning,
            is_complete=is_complete
        )
        
        # Get updates for UI components
        updates = progressive_ui.update(result)
        
        # Apply updates to UI components
        for component, value in updates.items():
            try:
                # Handle different component types appropriately
                if isinstance(component, gr.Markdown):
                    # For Markdown components, directly set the value
                    component.value = value
                elif isinstance(value, gr.update):
                    # For gr.update objects (like button state updates)
                    component.update(**value.kwargs)
                elif isinstance(component, (gr.Plot, gr.Dataframe, gr.HTML, gr.File)):
                    # For Plot, Dataframe, HTML, and File components, use the update method with the value
                    if value is not None:  # Only update if we have a value
                        component.update(value=value)
                elif hasattr(component, 'update'):
                    # For other components with update method
                    component.update(value=value)
                else:
                    logger.warning(f"Cannot update component of type {type(component)}")
            except Exception as e:
                logger.warning(f"Error updating component: {e}")
            
    return callback
