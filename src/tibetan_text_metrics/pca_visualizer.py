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
    pca_df, outliers = identify_clusters_and_outliers(pca_df)
    
    # Prepare feature vectors data
    feature_loadings = prepare_feature_vectors(features, loadings)
    
    # Create the interactive visualization
    create_interactive_visualization(
        pca_df=pca_df,
        feature_loadings=feature_loadings,
        features=features,
        explained_variance=explained_variance,
        pca_dir=pca_dir,
        outliers=outliers
    )


def create_interactive_visualization(
    pca_df: pd.DataFrame,
    feature_loadings: pd.DataFrame,
    features: pd.DataFrame,
    explained_variance: List[float],
    pca_dir: Path,
    outliers: Optional[pd.DataFrame] = None
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
        'Normalized LCS (%)': '#1f77b4'               # Blue
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
        else:
            # Fallback positioning
            label_x = -3.5
            label_y = 2.5  # Top left corner
        
        # Simplify the feature name for display
        if feature == 'Normalized Syntactic Distance':
            display_name = 'Normalized Syntactic Distance'
        elif feature == 'Weighted Jaccard Similarity (%)':
            display_name = 'Weighted Jaccard Similarity'
        elif feature == 'Normalized LCS (%)':
            display_name = 'Normalized LCS'
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
    # Group outliers into regions instead of individual highlights
    if outliers is not None and len(outliers) > 0:
        # Use simple clustering approach to group outliers into maximum 4 regions
        from sklearn.cluster import KMeans
        
        # If we have very few outliers, just group them into 1 or 2 clusters
        n_clusters = min(4, len(outliers))
        
        # Only use KMeans if we have enough points
        if len(outliers) >= n_clusters:
            # Extract PCA coordinates for clustering
            outlier_coords = outliers[['Principal Component 1', 'Principal Component 2']].values
            
            # Cluster the outliers
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
    
    # Add large ellipse to indicate the main cluster
    main_cluster_x0, main_cluster_y0 = -1, -1
    main_cluster_x1, main_cluster_y1 = 2, 1.5
    main_cluster_center_x = (main_cluster_x0 + main_cluster_x1) / 2
    main_cluster_center_y = (main_cluster_y0 + main_cluster_y1) / 2
    
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=main_cluster_x0,
        y0=main_cluster_y0,
        x1=main_cluster_x1,
        y1=main_cluster_y1,
        fillcolor="rgba(255, 165, 0, 0.1)",  # Orange with transparency
        line=dict(color="orange", width=1, dash="dash"),
        name="Main Cluster Region"
    )
    
    # Add label for main cluster
    fig.add_annotation(
        x=main_cluster_center_x,
        y=main_cluster_y1 + 0.2,
        text="<b>Main Cluster</b>",
        showarrow=False,
        font=dict(color="#ff8c00", size=12),  # Dark orange
        bgcolor="white",
        bordercolor="orange",
        borderwidth=1,
        borderpad=3,
        opacity=0.9
    )
    
    # Add light green region for the secondary cluster if any patterns emerge
    secondary_cluster_x0, secondary_cluster_y0 = -2.5, -1.5
    secondary_cluster_x1, secondary_cluster_y1 = -0.5, 1.5
    secondary_cluster_center_x = (secondary_cluster_x0 + secondary_cluster_x1) / 2
    secondary_cluster_center_y = (secondary_cluster_y0 + secondary_cluster_y1) / 2
    
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=secondary_cluster_x0,
        y0=secondary_cluster_y0,
        x1=secondary_cluster_x1,
        y1=secondary_cluster_y1,
        fillcolor="rgba(144, 238, 144, 0.1)",  # Light green with transparency
        line=dict(color="lightgreen", width=1, dash="dash"),
        name="Secondary Cluster Region"
    )
    
    # Add label for secondary cluster
    fig.add_annotation(
        x=secondary_cluster_center_x,
        y=secondary_cluster_y1 + 0.2,
        text="<b>Secondary Cluster</b>",
        showarrow=False,
        font=dict(color="green", size=12),
        bgcolor="white",
        bordercolor="lightgreen",
        borderwidth=1,
        borderpad=3,
        opacity=0.9
    )
    
    # Update layout with improved styling and better axis proportions
    fig.update_layout(
        title=None,  # Remove the plot title since we already have it in the HTML
        xaxis_title={
            'text': f'Principal Component 1 ({explained_variance[0]:.2%} variance explained)',
            'font': dict(family="Arial, sans-serif", size=16, color="#333333"),
            'standoff': 15
        },
        yaxis_title={
            'text': f'Principal Component 2 ({explained_variance[1]:.2%} variance explained)',
            'font': dict(family="Arial, sans-serif", size=16, color="#333333"),
            'standoff': 15
        },
        # Set dynamic axis ranges that can be overridden by responsive sizing
        xaxis=dict(
            range=[-3, 3],
            constrain='domain'
        ), 
        yaxis=dict(
            range=[-2, 2],
            scaleanchor="x",
            scaleratio=1
        ),  
        hovermode='closest',
        legend_title_text='<b>Text Pairs</b>',
        # Make width and height responsive by using 100% values
        # Default values will be used for non-responsive environments
        width=1600,
        height=1200,
        autosize=True,
        # Do NOT use aspect ratio constraints
        margin=dict(l=60, r=60, t=120, b=100),  # More top margin for title
        legend=dict(y=-0.15, orientation='h', font=dict(size=12)),  # Adjusted legend position
        plot_bgcolor='rgba(240,240,240,0.9)',  # Slightly gray background for better contrast
    )
    
    # Create enhanced explanation for the PCA
    enhanced_explanation = f"""
    <h2>Principal Component Analysis (PCA) Explained</h2>
    <h3>What is PCA?</h3>
    <p>A technique that combines our three text similarity metrics into two dimensions that we can visualize.</p>
    
    <h3>How PCA Actually Finds Patterns:</h3>
    <ol>
        <li><strong>Standardizes</strong> all metrics to the same scale</li>
        <li><strong>Calculates correlations</strong> between all metrics</li>
        <li><strong>Identifies directions</strong> of maximum variance</li>
        <li><strong>Projects</strong> the data onto these new directions</li>
    </ol>
    
    <p><em>For our text metrics:</em></p>
    <ul>
        <li>When texts are similar, Jaccard and LCS are high while Syntactic Distance is low</li>
        <li>PC1 ({explained_variance[0]:.1%} of variance) captures this primary relationship</li>
        <li>PC2 ({explained_variance[1]:.1%} of variance) captures secondary patterns where metrics disagree</li>
    </ul>
    
    <h3>How to Read This Plot:</h3>
    <ul>
        <li><strong>Right side</strong>: Jaccard and LCS are high, Syntactic Distance is low (similar texts)</li>
        <li><strong>Left side</strong>: Jaccard and LCS are low, Syntactic Distance is high (different texts)</li>
        <li><strong>Vertical position</strong>: Captures subtler differences between metrics</li>
    </ul>
    
    <h3>The Colored Circles:</h3>
    <ul>
        <li><strong>Orange circle</strong>: The main cluster where most text comparisons fall (normal similarity patterns)</li>
        <li><strong>Teal circle</strong>: Outlier texts where metrics show unusual or contradictory patterns</li>
    </ul>
    
    <p><em>Hover over any point for details</em></p>
    """
    
    # Ensure pca_dir exists - handle any nested subdirectories in the path
    pca_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a complete custom HTML layout with flexbox
    html_file = pca_dir / 'interactive_pca_visualization.html'
    
    # Generate just the plotly figure without full HTML, set responsive to True
    plot_div = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
    
    # Create a fully custom HTML with responsive design
    custom_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Tibetan Text Similarities: PCA Visualization</title>
        <style>
            body, html {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
            .main-container {{ display: flex; flex-direction: column; width: 100%; margin: 0 auto; }}
            .container {{ display: flex; flex-direction: row; flex-wrap: wrap; width: 100%; }}
            .explanation {{ flex: 0 0 300px; padding: 20px; }}
            .plot-container {{ flex: 1; min-width: 300px; width: 100%; height: 80vh; }}
            h1 {{ font-size: 22px; text-align: center; margin: 15px 0; }}
            .js-plotly-plot {{ width: 100% !important; height: 100% !important; }}
            
            /* Responsive adaptations */
            @media (max-width: 1400px) {{ 
                .container {{ flex-direction: column; }} 
                .explanation {{ flex: none; width: auto; }} 
                .plot-container {{ width: 100%; min-height: 80vh; }} 
            }}
            @media (max-width: 768px) {{ 
                .plot-container {{ height: 90vh; }} 
                body {{ font-size: 14px; }}
                h1, h2, h3 {{ margin: 10px 0; }}
            }}
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
                </div>
            </div>
        </div>
    </body>
    </html>
    '''
    
    # Write the custom HTML to file
    with open(str(html_file), 'w') as f:
        f.write(custom_html)
    
    print(f"Saved interactive PCA visualization to {html_file}")
