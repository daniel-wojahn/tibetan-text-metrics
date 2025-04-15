from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .pca_core import (
    prepare_data_for_pca,
    transform_data_with_pca,
    identify_clusters_and_outliers,
    prepare_feature_vectors
)


def perform_pca_analysis(results_df: pd.DataFrame, pca_dir: Path) -> None:
    """Perform PCA analysis on text metrics and visualize results.
    
    This function creates responsive, interactive visualizations of PCA results.
    It automatically handles nested directory paths by creating parent directories
    as needed.
    
    Args:
        results_df: DataFrame containing pairwise analysis results
        pca_dir: Directory to save PCA outputs (can be a nested directory path)
    """
    # Create directory and any parent directories if they don't exist
    pca_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    features, metadata = prepare_data_for_pca(results_df)
    
    # Transform data using PCA
    transformed_data, loadings, components, explained_variance = transform_data_with_pca(features)
    
    # Create a DataFrame with the PCA results and metadata
    pca_df = pd.DataFrame(
        transformed_data,
        columns=['Principal Component 1', 'Principal Component 2']
    )
    
    # Add metadata back to the PCA results
    pca_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)
    
    # Identify clusters and outliers
    pca_df, _ = identify_clusters_and_outliers(pca_df)
    
    # Prepare feature vectors data
    feature_loadings = prepare_feature_vectors(features, loadings)
    
    # Create the interactive visualization
    create_interactive_visualization(
        pca_df=pca_df,
        feature_loadings=feature_loadings,
        features=features,
        explained_variance=explained_variance,
        pca_dir=pca_dir
    )


def create_interactive_visualization(
    pca_df: pd.DataFrame,
    feature_loadings: pd.DataFrame,
    features: pd.DataFrame,
    explained_variance: List[float],
    pca_dir: Path
) -> None:
    """Create an interactive PCA visualization.
    
    Args:
        pca_df: DataFrame with PCA results and metadata
        feature_loadings: DataFrame with feature loadings
        features: Original feature DataFrame
        explained_variance: List of explained variance ratios
        pca_dir: Directory to save visualization
    """
    # Define the metrics columns for hover data
    metrics_columns = list(features.columns)
    
    # Create a distinct, colorblind-friendly palette
    import plotly.express as px
    
    # Get unique text pairs for coloring
    text_pair_categories = pca_df['Text Pair Category'].unique()
    
    # Create a custom color scale
    colors = px.colors.qualitative.Dark24[:len(text_pair_categories)]
    
    # Create a plotly figure for interactive visualization
    fig = make_subplots(rows=1, cols=1)
    
    # Add traces for each text pair category
    for i, text_pair in enumerate(text_pair_categories):
        df_subset = pca_df[pca_df['Text Pair Category'] == text_pair]
        
        # Create hover text with detailed information
        hover_text = [f"Text Pair: {row['Text Pair']}<br>"
                     f"Chapter: {row['Chapter']}<br>"
                     f"PC1: {row['Principal Component 1']:.4f}<br>"
                     f"PC2: {row['Principal Component 2']:.4f}"
                     for _, row in df_subset.iterrows()]
        
        # Add scatter trace for this text pair
        fig.add_trace(
            go.Scatter(
                x=df_subset['Principal Component 1'],
                y=df_subset['Principal Component 2'],
                mode='markers+text',
                marker=dict(size=12, color=colors[i]),  # Slightly larger markers
                text=df_subset['Chapter'],
                textposition="top center",
                hoverinfo="text",
                hovertext=hover_text,
                name=text_pair
            )
        )
    
    # Add feature vectors as arrows with improved styling and clarity
    # Define distinct colors for each feature to make them more distinguishable
    feature_colors = {
        'Normalized Syntactic Distance': '#d62728',    # Red
        'Weighted Jaccard Similarity (%)': '#2ca02c',  # Green
        'Normalized LCS (%)': '#1f77b4',              # Blue
        'POS Pattern Similarity': '#9467bd',           # Purple
        'Word Pattern Similarity': '#ff7f0e'           # Orange
    }
    
    # Embed feature field labels in the corners of the plot as part of the raster grid
    # Using neutral colors and minimal styling
    for i, feature in enumerate(feature_loadings.index):
        # Get the loadings for this feature
        pc1_loading = feature_loadings.iloc[i, 0]  # PC1 loading
        pc2_loading = feature_loadings.iloc[i, 1]  # PC2 loading
        
        # Position labels in the corners based on their typical positions in the PCA space
        # For Normalized Syntactic Distance (usually negative PC1)
        if feature == 'Normalized Syntactic Distance':
            label_x = -3.5  # Far left corner
            label_y = -2.5  # Bottom left corner
        # For Weighted Jaccard (usually positive PC1)
        elif feature == 'Weighted Jaccard Similarity (%)':
            label_x = 3.5   # Far right corner
            label_y = 2.5   # Top right corner
        # For Normalized LCS
        elif feature == 'Normalized LCS (%)':
            label_x = 3.5   # Far right corner
            label_y = -2.5  # Bottom right corner
        # For POS Pattern Similarity
        elif feature == 'POS Pattern Similarity':
            label_x = -3.5  # Far left corner
            label_y = 2.5   # Top left corner
        # For Word Pattern Similarity
        elif feature == 'Word Pattern Similarity':
            label_x = -1.0  # Center
            label_y = 3.0   # Top center
        else:
            # Fallback positioning
            label_x = 0.0
            label_y = 0.0  # Center
        
        # Simplify the feature name for display
        if feature == 'Normalized Syntactic Distance':
            display_name = 'Normalized Syntactic Distance'
        elif feature == 'Weighted Jaccard Similarity (%)':
            display_name = 'Weighted Jaccard Similarity'
        elif feature == 'Normalized LCS (%)':
            display_name = 'Normalized LCS'
        elif feature == 'POS Pattern Similarity':
            display_name = 'POS Pattern Similarity'
        elif feature == 'Word Pattern Similarity':
            display_name = 'Word Pattern Similarity'
        else:
            display_name = feature.split('(')[0]
        
        # Create label as part of the grid with neutral styling
        fig.add_annotation(
            x=label_x,
            y=label_y,
            text=f"<b>{display_name}</b>",  # Simplified name
            showarrow=False,  # No arrow
            font=dict(
                color='#555555',  # Neutral gray color
                size=12,          # Smaller size
                family="Arial, sans-serif"
            ),
            bgcolor="rgba(245, 245, 245, 0.5)",  # Very light gray, mostly transparent
            bordercolor="#cccccc",             # Light gray border
            borderwidth=1,                    # Thinner border
            borderpad=3,                      # Less padding
            opacity=0.8,                      # More transparent overall
            align="center"
        )
    
    # Create clusters to visually group data
    # Extract outliers from pca_df
    outliers = pca_df[pca_df['Is_Outlier']].copy()
    
    # Group outliers into regions instead of individual highlights
    if len(outliers) > 0:
        # Use simple clustering approach to group outliers into maximum 4 regions
        from sklearn.cluster import KMeans
        
        # For small number of outliers, use fewer clusters
        n_clusters = min(2, len(outliers)) if len(outliers) < 4 else min(3, len(outliers))
        
        # Only use KMeans if we have enough points
        if len(outliers) >= n_clusters:
            # Extract PCA coordinates for clustering
            outlier_coords = outliers[['Principal Component 1', 'Principal Component 2']].values
            
            # Fit KMeans to outlier points
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(outlier_coords)
            
            # Group outliers by cluster
            for cluster_id in range(n_clusters):
                cluster_points = outliers[cluster_labels == cluster_id]
                
                if len(cluster_points) > 0:
                    # Calculate the center and radius of the cluster
                    pc1_values = cluster_points['Principal Component 1'].values
                    pc2_values = cluster_points['Principal Component 2'].values
                    
                    # Get center coordinates
                    center_x = pc1_values.mean()
                    center_y = pc2_values.mean()
                    
                    # Calculate a reasonable radius to encompass the points (with padding)
                    radius_x = max(0.6, pc1_values.std() * 2 + 0.3)
                    radius_y = max(0.6, pc2_values.std() * 2 + 0.3)
                    
                    # Create oval shape around the cluster
                    fig.add_shape(
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=center_x - radius_x,
                        y0=center_y - radius_y,
                        x1=center_x + radius_x,
                        y1=center_y + radius_y,
                        fillcolor="rgba(0, 128, 128, 0.2)",  # Teal color with transparency
                        line=dict(color="teal", width=1, dash="dash"),
                        name=f"Outlier Group {cluster_id+1}"
                    )
                    
                    # Add a label to the cluster
                    fig.add_annotation(
                        x=center_x,
                        y=center_y + radius_y + 0.1,
                        text=f"<b>Outlier Group {cluster_id+1}</b>",
                        showarrow=False,
                        font=dict(color="teal", size=12),
                        bgcolor="white",
                        bordercolor="teal",
                        borderwidth=1,
                        borderpad=3,
                        opacity=0.9
                    )
        else:
            # For very few outliers, just create one region
            pc1_values = outliers['Principal Component 1'].values
            pc2_values = outliers['Principal Component 2'].values
            
            center_x = pc1_values.mean()
            center_y = pc2_values.mean()
            
            # Create a single region for all outliers
            radius_x = max(0.6, pc1_values.std() * 2 + 0.3) if len(pc1_values) > 1 else 0.6
            radius_y = max(0.6, pc2_values.std() * 2 + 0.3) if len(pc2_values) > 1 else 0.6
            
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=center_x - radius_x,
                y0=center_y - radius_y,
                x1=center_x + radius_x,
                y1=center_y + radius_y,
                fillcolor="rgba(0, 128, 128, 0.2)",  # Teal color with transparency
                line=dict(color="teal", width=1, dash="dash"),
                name="Outlier Region"
            )
            
            # Add a label
            fig.add_annotation(
                x=center_x,
                y=center_y + radius_y + 0.1,
                text=f"<b>Outliers</b>",
                showarrow=False,
                font=dict(color="teal", size=12),
                bgcolor="white",
                bordercolor="teal",
                borderwidth=1,
                borderpad=3,
                opacity=0.9
            )
    
    # Analyze data distribution to determine if we need cluster regions
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Get all points excluding outliers
    non_outliers = pca_df[~pca_df['Is_Outlier']]
    coords = non_outliers[['Principal Component 1', 'Principal Component 2']].values
    
    # Only try clustering if we have enough points
    if len(coords) >= 4:
        # Try clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(coords)
        
        # Calculate silhouette score to determine if clusters are well-separated
        sil_score = silhouette_score(coords, cluster_labels) if len(set(cluster_labels)) > 1 else 0
        
        # Only show clusters if they are well-separated (silhouette score > 0.3)
        if sil_score > 0.3:
            # Get points for each cluster
            cluster_points = [non_outliers[cluster_labels == i] for i in range(2)]
            
            # Sort clusters by size (main cluster should be larger)
            cluster_points.sort(key=len, reverse=True)
            
            # Add regions for both clusters
            for i, points in enumerate(cluster_points):
                # Calculate region bounds with padding
                x_min, x_max = points['Principal Component 1'].min(), points['Principal Component 1'].max()
                y_min, y_max = points['Principal Component 2'].min(), points['Principal Component 2'].max()
                
                # Add padding proportional to the cluster size
                padding = 0.2 + (0.1 * len(points) / len(non_outliers))
                
                x0 = x_min - padding
                y0 = y_min - padding
                x1 = x_max + padding
                y1 = y_max + padding
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                
                # Style based on cluster type
                if i == 0:  # Main cluster
                    color = "orange"
                    fill_color = "rgba(255, 165, 0, 0.1)"
                    label = "Main Cluster"
                else:  # Secondary cluster
                    color = "lightgreen"
                    fill_color = "rgba(144, 238, 144, 0.1)"
                    label = "Secondary Cluster"
                
                # Add cluster region
                fig.add_shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    fillcolor=fill_color,
                    line=dict(color=color, width=1, dash="dash"),
                    name=f"{label} Region"
                )
                
                # Add label
                fig.add_annotation(
                    x=center_x,
                    y=y1 + 0.2,
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    font=dict(color=color, size=12),
                    bgcolor="white",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=3,
                    opacity=0.9
                )
        else:
            # If clusters aren't well-separated, just show one main region
            padding = 0.3
            x0 = non_outliers['Principal Component 1'].min() - padding
            y0 = non_outliers['Principal Component 2'].min() - padding
            x1 = non_outliers['Principal Component 1'].max() + padding
            y1 = non_outliers['Principal Component 2'].max() + padding
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor="rgba(255, 165, 0, 0.1)",
                line=dict(color="orange", width=1, dash="dash"),
                name="Main Region"
            )
            
            fig.add_annotation(
                x=center_x,
                y=y1 + 0.2,
                text="<b>Main Cluster</b>",
                showarrow=False,
                font=dict(color="#ff8c00", size=12),
                bgcolor="white",
                bordercolor="orange",
                borderwidth=1,
                borderpad=3,
                opacity=0.9
            )
    
    # Add a legend interaction tip annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=1.05,  # Adjusted position to be more visible
        text="<i>Tip: Click on colored dots in the legend to show/hide specific text pairs</i>",
        showarrow=False,
        font=dict(size=12, color="#555555"),
        bgcolor="rgba(240,240,240,0.8)",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=4,
        opacity=0.95
    )
    
    # Update layout with improved styling
    
    # Update layout with improved styling and better axis proportions
    fig.update_layout(
        title=None,  # Remove the plot title since we already have it in the HTML
        xaxis_title={
            'text': f'Principal Component 1 ({explained_variance[0]:.2%} variance explained)',
            'font': dict(family="Arial, sans-serif", size=14, color="#333333"),
            'standoff': 15
        },
        yaxis_title={
            'text': f'Principal Component 2 ({explained_variance[1]:.2%} variance explained)',
            'font': dict(family="Arial, sans-serif", size=14, color="#333333"),
            'standoff': 15
        },
        xaxis=dict(
            range=[-4, 4],  # Slightly wider range
            constrain='domain'
        ), 
        yaxis=dict(
            range=[-3, 3],  # Slightly wider range
            scaleanchor="x",
            scaleratio=1
        ),
        hovermode='closest',
        showlegend=True,
        legend_title_text='<b>Text Pairs</b>',
        # Make width and height responsive by using 100% values
        # Default values will be used for non-responsive environments
        width=1800,
        height=1200,
        autosize=True,
        # Do NOT use aspect ratio constraints
        margin=dict(l=60, r=60, t=150, b=100),  # Increased top margin for the tip annotation
        legend=dict(y=-0.15, orientation='h', font=dict(size=12)),  # Adjusted legend position
        plot_bgcolor='rgba(240,240,240,0.9)',  # Slightly gray background for better contrast
    )
    
    # Create  explanation for the PCA
    enhanced_explanation = f"""
    <h2>Principal Component Analysis (PCA) Explained</h2>
    <h3>What is PCA?</h3>
    <p>A technique that combines our text similarity metrics into two dimensions that we can visualize.</p>
    <h3>Metrics Included:</h3>
    <ul>
        <li><strong>Distance metric:</strong> Normalized Syntactic Distance</li>
        <li><strong>Lexical similarity:</strong> Weighted Jaccard Similarity</li>
        <li><strong>Structural similarity:</strong> Normalized LCS, POS Pattern Similarity, Word Pattern Similarity</li>
    </ul>

    <h3>How PCA Actually Finds Patterns:</h3>
    <ol>
        <li><strong>Standardizes</strong> all metrics to the same scale</li>
        <li><strong>Calculates correlations</strong> between all metrics</li>
        <li><strong>Identifies directions</strong> of maximum variance</li>
        <li><strong>Projects</strong> the data onto these new directions</li>
    </ol>
    <h3>What "Variance Explained" Means:</h3>
    <ul>
        <li><strong>PC1: {explained_variance[0]:.1%}</strong> - This first dimension captures {explained_variance[0]:.1%} of the total variance in the data</li>
        <li><strong>PC2: {explained_variance[1]:.1%}</strong> - This second dimension captures {explained_variance[1]:.1%} of the remaining patterns</li>
        <li><strong>Total: {sum(explained_variance):.1%}</strong> - Together, these two dimensions preserve {sum(explained_variance):.1%} of the information from all metrics</li>
    </ul>
    <p><em>For our text metrics:</em></p>
    <ul>
        <li>When texts are similar, similarity metrics are high while distance metrics are low</li>
        <li>PC1 captures the primary relationship between these metrics</li>
        <li>PC2 captures secondary patterns where metrics disagree</li>
        <li>Pattern metrics add new dimensions of analysis based on n-gram sequences</li>
    </ul>
    <h3>How to Read This Plot:</h3>
    <ul>
        <li><strong>Right side:</strong> Higher similarity (Jaccard, LCS, Pattern Similarity)</li>
        <li><strong>Left side:</strong> Higher distance (Syntactic Distance)</li>
        <li><strong>Vertical position:</strong> Captures subtler differences between metrics</li>
        <li><strong>Legend interaction:</strong> Click on colored dots in the legend to show/hide specific text pairs</li>
    </ul>
    <h3>The Colored Circles:</h3>
    <ul>
        <li><strong>Orange circle:</strong> The main cluster where most text comparisons fall (normal similarity patterns)</li>
        <li><strong>Teal circle:</strong> Outlier texts where metrics show unusual or contradictory patterns</li>
    </ul>
    <p><em>Hover over any point for details</em></p>
    """
    
    # Ensure pca_dir exists - handle any nested subdirectories in the path
    pca_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate just the plotly figure without full HTML, set responsive to True
    plot_div = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})

    # --- PATCH: Improved responsive CSS for PCA visualization ---
    responsive_css = '''
            body, html { margin: 0; padding: 0; font-family: Arial, sans-serif; }
            .main-container { display: flex; flex-direction: column; width: 100%; margin: 0 auto; }
            .container { display: flex; flex-direction: row; flex-wrap: wrap; width: 100%; }
            .explanation { flex: 0 0 330px; padding: 20px; font-size: 13px; min-width: 220px; max-width: 350px; box-sizing: border-box; }
            .plot-container { flex: 1; min-width: 300px; width: 100%; height: 80vh; box-sizing: border-box; }
            h1 { font-size: 22px; text-align: center; margin: 15px 0; }
            h2 { font-size: 18px; }
            h3 { font-size: 15px; }
            .js-plotly-plot { width: 100% !important; height: 100% !important; }
            /* Responsive adaptations */
            @media (max-width: 1100px) {
              .container { flex-direction: column; }
              .explanation { width: 100%; max-width: 100%; min-width: 0; }
              .plot-container { width: 100%; min-height: 60vh; }
            }
            @media (max-width: 768px) {
              .plot-container { height: 60vh; }
              body { font-size: 14px; }
              h1, h2, h3 { margin: 10px 0; }
            }
    '''
    # --- END PATCH ---

    custom_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tibetan Text Similarities: PCA Visualization</title>
        <style>
        {responsive_css}
        </style>
        <!-- Additional script to ensure Plotly is fully responsive -->
        <script>
            window.addEventListener('resize', function() {{
            // Trigger a Plotly resize event when window size changes
            if (window.Plotly) {{
                var plotContainers = document.querySelectorAll('.js-plotly-plot');
                plotContainers.forEach(function(container) {{
                    Plotly.Plots.resize(container);
                }});
            }}
        }}); 
        </script>
    </head>
    <body>
        <div class="main-container">
            <h1>Tibetan Text Similarities: PCA Visualization</h1>
            <div class="container">
                <div class="explanation">
                    {enhanced_explanation}
                </div>
                <div class="plot-container">
                    {plot_div}
                    <!-- PATCH: Autoscale axes on load -->
                    <script>
                    window.addEventListener('DOMContentLoaded', function() {{
                        var plot = document.querySelector('.js-plotly-plot');
                        if (window.Plotly && plot) {{
                            Plotly.relayout(plot, {{
                                'xaxis.autorange': true,
                                'yaxis.autorange': true
                            }});
                        }}
                    }});
                    </script>
                    <!-- END PATCH -->
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

    # Write the custom HTML to file
    with open(str(pca_dir / 'interactive_pca_visualization.html'), 'w') as f:
        f.write(custom_html)

    print(f"Saved interactive PCA visualization to {pca_dir / 'interactive_pca_visualization.html'}")
