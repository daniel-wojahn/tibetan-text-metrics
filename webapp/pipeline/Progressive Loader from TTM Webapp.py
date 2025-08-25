"""
Progressive loading module for the Tibetan Text Metrics app.

This module provides functionality for progressive loading and updating of metrics
as they are computed, allowing for a more responsive user experience.
"""

import pandas as pd
from typing import List, Callable, Optional
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Enum for different types of metrics that can be progressively loaded."""
    JACCARD = "Jaccard Similarity (%)"
    LCS = "Normalized LCS"
    FUZZY = "Fuzzy Similarity"
    SEMANTIC = "Semantic Similarity"
    WORD_COUNT = "Word Count"


@dataclass
class ProgressiveResult:
    """Class to store progressive computation results."""
    metrics_df: pd.DataFrame
    word_counts_df: pd.DataFrame
    completed_metrics: List[MetricType]
    warning: str
    is_complete: bool


class ProgressiveLoader:
    """
    Manages progressive loading of metrics computation results.
    
    This class handles the incremental updates of metrics as they are computed,
    allowing the UI to display partial results before the entire computation is complete.
    """
    
    def __init__(self, update_callback: Optional[Callable[[ProgressiveResult], None]] = None):
        """
        Initialize the ProgressiveLoader.
        
        Args:
            update_callback: Function to call when new results are available.
                            Should accept a ProgressiveResult object.
        """
        self.update_callback = update_callback
        self.metrics_df = pd.DataFrame()
        self.word_counts_df = pd.DataFrame()
        self.completed_metrics = []
        self.warning = ""
        self.is_complete = False
        self.last_update_time = 0
        self.update_interval = 0.5  # Minimum seconds between updates to avoid UI thrashing
    
    def update(self, 
               metrics_df: Optional[pd.DataFrame] = None,
               word_counts_df: Optional[pd.DataFrame] = None, 
               completed_metric: Optional[MetricType] = None,
               warning: Optional[str] = None,
               is_complete: bool = False) -> None:
        """
        Update the progressive results and trigger the callback if enough time has passed.
        
        Args:
            metrics_df: Updated metrics DataFrame
            word_counts_df: Updated word counts DataFrame
            completed_metric: Newly completed metric type
            warning: Warning message to display
            is_complete: Whether the computation is complete
        """
        current_time = time.time()
        
        # Update internal state
        if metrics_df is not None:
            self.metrics_df = metrics_df
        
        if word_counts_df is not None:
            self.word_counts_df = word_counts_df
            
        if completed_metric is not None and completed_metric not in self.completed_metrics:
            self.completed_metrics.append(completed_metric)
            
        if warning:
            self.warning = warning
            
        self.is_complete = is_complete
        
        # Only trigger update if enough time has passed or if this is the final update
        if (current_time - self.last_update_time >= self.update_interval) or is_complete:
            self._trigger_update()
            self.last_update_time = current_time
    
    def _trigger_update(self) -> None:
        """Trigger the update callback with the current state."""
        if self.update_callback:
            try:
                result = ProgressiveResult(
                    metrics_df=self.metrics_df.copy(),
                    word_counts_df=self.word_counts_df.copy(),
                    completed_metrics=self.completed_metrics.copy(),
                    warning=self.warning,
                    is_complete=self.is_complete
                )
                self.update_callback(result)
            except Exception as e:
                logger.error(f"Error in progressive loader update callback: {e}", exc_info=True)
