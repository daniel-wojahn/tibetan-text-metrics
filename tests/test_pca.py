import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tibetan_text_metrics.pca_core import (
    prepare_data_for_pca,
    transform_data_with_pca,
    identify_clusters_and_outliers
)


@pytest.fixture
def sample_results_df():
    """Create a sample DataFrame for testing PCA functions."""
    return pd.DataFrame({
        'Text Pair': ['A vs B'] * 3 + ['A vs C'] * 3 + ['B vs C'] * 3,
        'Chapter': ['1', '2', '3'] * 3,
        'Chapter Length 1': [100, 150, 200] * 3,
        'Chapter Length 2': [120, 140, 180] * 3,
        'Normalized Syntactic Distance': [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5, 0.4, 0.3],
        'Weighted Jaccard Similarity (%)': [80, 70, 60, 40, 30, 20, 50, 60, 70],
        'Normalized LCS (%)': [75, 65, 55, 35, 25, 15, 45, 55, 65]
    })


def test_prepare_data_for_pca(sample_results_df):
    """Test the function that prepares data for PCA."""
    features, metadata = prepare_data_for_pca(sample_results_df)
    
    # Check the correct columns were selected for features
    assert list(features.columns) == [
        'Normalized Syntactic Distance',
        'Weighted Jaccard Similarity (%)',
        'Normalized LCS (%)'
    ]
    
    # Check the correct columns were included in metadata
    assert list(metadata.columns) == [
        'Text Pair', 'Chapter', 'Chapter Length 1', 'Chapter Length 2'
    ]
    
    # Check that the correct number of rows are present
    assert len(features) == 9
    assert len(metadata) == 9


def test_transform_data_with_pca(sample_results_df):
    """Test the PCA transformation function."""
    features, _ = prepare_data_for_pca(sample_results_df)
    transformed_data, loadings, components, explained_variance = transform_data_with_pca(features)
    
    # Check the shapes of the output
    assert transformed_data.shape == (9, 2)
    assert loadings.shape == (3, 2)
    assert components.shape == (2, 3)
    assert len(explained_variance) == 2
    
    # Check that explained variance sums to less than or equal to 1
    assert sum(explained_variance) <= 1.0


def test_identify_clusters_and_outliers(sample_results_df):
    """Test the cluster and outlier identification."""
    features, metadata = prepare_data_for_pca(sample_results_df)
    transformed_data, _, _, _ = transform_data_with_pca(features)
    
    # Create a DataFrame with the PCA results and metadata
    pca_df = pd.DataFrame(
        transformed_data,
        columns=['Principal Component 1', 'Principal Component 2']
    )
    
    # Add metadata back to the PCA results
    pca_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)
    
    # Test the function
    labeled_df, outliers = identify_clusters_and_outliers(pca_df, distance_threshold=1.0)
    
    # Check that the labeled dataframe has the expected columns
    assert 'Distance_From_Center' in labeled_df.columns
    assert 'Is_Outlier' in labeled_df.columns
    assert 'Text Pair Category' in labeled_df.columns
    
    # Check that we have the correct number of rows
    assert len(labeled_df) == 9
