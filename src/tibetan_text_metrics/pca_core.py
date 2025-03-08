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
    features = results_df[[
        'Normalized Syntactic Distance',
        'Weighted Jaccard Similarity (%)',
        'Normalized LCS (%)'
    ]]
    
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
    distance_threshold: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify clusters and outliers in PCA data.
    
    Args:
        pca_df: DataFrame with PCA results
        distance_threshold: Threshold for outlier detection (default: 1.5)
        
    Returns:
        Tuple of (pca_df with cluster labels, outliers DataFrame)
    """
    # Calculate the Euclidean distance from each point to the center (origin)
    pca_df['Distance_From_Center'] = np.sqrt(
        pca_df['Principal Component 1']**2 + 
        pca_df['Principal Component 2']**2
    )
    
    # Calculate the median and MAD (Median Absolute Deviation) for robust outlier detection
    median_distance = pca_df['Distance_From_Center'].median()
    mad = np.median(np.abs(pca_df['Distance_From_Center'] - median_distance))
    
    # Define points that are far from the center as outliers (using MAD)
    pca_df['Is_Outlier'] = pca_df['Distance_From_Center'] > median_distance + distance_threshold * mad
    
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
