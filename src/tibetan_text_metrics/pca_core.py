from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prepare_data_for_pca(results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for PCA analysis.
    
    Args:
        results_df: DataFrame containing pairwise analysis results
        
    Returns:
        Tuple of (feature_df, metadata_df) for PCA analysis
    """
    # Create a new dataframe with chapter and text pair as indices
    metadata = results_df[['Text Pair', 'Chapter', 'Chapter Length 1', 'Chapter Length 2']]
    
    # Select numeric features for PCA
    # Check if the pattern metrics exist in the DataFrame
    pattern_metrics = ['POS Pattern Similarity', 'Word Pattern Similarity']
    available_metrics = [m for m in pattern_metrics if m in results_df.columns]

    # Base metrics always included
    base_metrics = [
        'Normalized Syntactic Distance',
        'Weighted Jaccard Similarity (%)',
        # 'Normalized LCS (%)'  # DROPPED: highly correlated with Jaccard
    ]

    # Only keep POS Pattern Similarity (drop Word Pattern Similarity if both present)
    selected_pattern_metrics = []
    if 'POS Pattern Similarity' in available_metrics:
        selected_pattern_metrics.append('POS Pattern Similarity')
    elif 'Word Pattern Similarity' in available_metrics:
        selected_pattern_metrics.append('Word Pattern Similarity')

    # Combine base metrics with selected pattern metric
    selected_metrics = base_metrics + selected_pattern_metrics

    # Select features for PCA
    features = results_df[selected_metrics]
    
    return features, metadata


def perform_direct_text_pca(text_data: Dict[str, List[str]], file_paths: List[str], pca_dir: pd.DataFrame) -> None:
    """Perform PCA analysis directly on text features rather than metrics.
    
    This alternative approach bypasses calculated metrics and works directly with
    the text content using a TF-IDF representation of word frequencies.
    
    Args:
        text_data: Dictionary with file names as keys and lists of words as values
        file_paths: List of file paths to process
        pca_dir: Directory to save PCA outputs
    """
    # Skip direct text PCA as requested by the user
    print("Skipping direct text-based PCA analysis as requested.")
    return


def transform_data_with_pca(features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Transform data using PCA.
    
    Args:
        features: DataFrame containing features for PCA analysis
        
    Returns:
        Tuple of (transformed_data, loadings, components, explained_variances)
    """
    # Standardize the features (important for PCA)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA with 2 components
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(scaled_features)
    
    # Get the feature loadings (how each feature contributes to each PC)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance: PC1 {explained_variance[0]:.2f}, PC2 {explained_variance[1]:.2f}")
    print(f"Total explained variance: {sum(explained_variance):.2f}\n")
    
    return transformed_data, loadings, pca.components_, list(explained_variance)


def identify_clusters_and_outliers(
    pca_df: pd.DataFrame, 
    distance_threshold: float = 1.5,  # More selective global threshold
    local_threshold: float = 2.0     # More selective local threshold
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify clusters and outliers in PCA data using both global and local approaches.
    
    Args:
        pca_df: DataFrame with PCA results
        distance_threshold: Global threshold for outlier detection (default: 1.5)
        local_threshold: Local threshold for cluster-based outlier detection (default: 2.0)
        
    Returns:
        Tuple of (pca_df with cluster labels, outliers DataFrame)
    """
    # Calculate global distances from center
    pca_df['Distance_From_Center'] = np.sqrt(
        pca_df['Principal Component 1']**2 + 
        pca_df['Principal Component 2']**2
    )
    
    # Calculate local distances within text pair groups
    pca_df['Local_Distance'] = 0.0
    for pair in pca_df['Text Pair'].unique():
        mask = pca_df['Text Pair'] == pair
        if mask.sum() > 1:  # Only if we have multiple points
            group = pca_df[mask]
            centroid = group[['Principal Component 1', 'Principal Component 2']].mean()
            distances = np.sqrt(
                (group['Principal Component 1'] - centroid['Principal Component 1'])**2 +
                (group['Principal Component 2'] - centroid['Principal Component 2'])**2
            )
            pca_df.loc[mask, 'Local_Distance'] = distances
    
    # Global outlier detection using MAD
    median_distance = pca_df['Distance_From_Center'].median()
    mad = np.median(np.abs(pca_df['Distance_From_Center'] - median_distance))
    global_outliers = pca_df['Distance_From_Center'] > median_distance + distance_threshold * mad
    
    # Only use global outlier detection for simplicity and robustness
    pca_df['Is_Outlier'] = global_outliers
    
    # Extract outliers for focused analysis
    outliers = pca_df[pca_df['Is_Outlier']].copy()
    
    # Categorize the text pairs for coloring
    # Create a derived category for coloring (just the pair of texts, not the specific chapter)
    pca_df['Text Pair Category'] = pca_df['Text Pair'].apply(
        lambda x: ' vs '.join(sorted(x.split(' vs ')))
    )
    
    return pca_df, outliers


def prepare_feature_vectors(
    features: pd.DataFrame, 
    loadings: np.ndarray
) -> pd.DataFrame:
    """Prepare feature vectors data for visualization.
    
    Args:
        features: DataFrame of original features
        loadings: Array of feature loadings from PCA
        
    Returns:
        DataFrame with feature vectors information
    """
    # Create a DataFrame to show feature loadings
    feature_loadings = pd.DataFrame(
        loadings,
        index=features.columns,
        columns=['PC1', 'PC2']
    )
    
    # Display the loadings
    print("\nFeature loadings (contribution of each feature to the principal components):")
    print(feature_loadings)
    
    return feature_loadings
