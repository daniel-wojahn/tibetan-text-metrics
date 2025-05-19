"""
LLM-based interpreter for Tibetan Text Metrics results.
This module provides functionality to analyze similarity metrics using an LLM
and generate human-readable interpretations.
"""

import pandas as pd
import logging
from .puter_api import call_puter_claude
from typing import Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

class LLMInterpreter:
    """
    Uses the Puter API to access Claude 3.5 Sonnet (free, no API key required) for interpreting similarity metrics.
    """
    
    def __init__(self):
        """
        Initialize the LLM interpreter.
        No API key required for Puter API.
        """
        pass
    
    def interpret_results(self, 
                          results_df: pd.DataFrame, 
                          max_tokens: int = 1500) -> str:
        """
        Interpret the similarity metrics using Claude 3.5 Sonnet via Puter API.
        
        Args:
            results_df: DataFrame containing similarity metrics
            max_tokens: Maximum tokens for the response
            
        Returns:
            str: Human-readable interpretation of the results
        """
        try:
            # Prepare the data for the LLM
            df_summary = self._prepare_data_summary(results_df)
            
            # Create the prompt
            prompt = self._create_interpretation_prompt(df_summary, results_df)
            
            # Comprehensive system message for Claude
            system_message = """
            You are an expert assistant specializing in interpreting Tibetan text similarity metrics for scholarly research. 
            
            ABOUT THE METRICS:
            - Jaccard Similarity (%): Measures vocabulary overlap between segments. Higher percentages indicate more shared unique words.
            - Normalized LCS: Measures the longest common subsequence of words that appear in both texts in the same order. Higher values indicate longer shared sequences.
            - Semantic Similarity: Uses embedding models to compare contextual meaning. Higher values indicate more similar meanings.
            - TF-IDF Cosine Sim: Highlights texts that share important or characteristic terms. Higher values indicate more shared important terms.
            
            ABOUT TIBETAN TEXTS:
            - Tibetan texts often have complex structures with many particles and function words.
            - Texts may be segmented into chapters or sections using the Tibetan section marker (༈).
            - Different versions of the same text may have variations in spelling, word choice, or phrasing.
            - Similarity between texts can indicate textual relationships, transmission history, or shared sources.
            
            YOUR TASK:
            - Analyze the similarity metrics to identify patterns and relationships between text segments.
            - Focus on the most significant findings that would be valuable to Tibetan text researchers.
            - Explain what the metrics suggest about textual relationships in clear, scholarly language.
            - Highlight any interesting outliers or notable segment pairs.
            - Suggest potential next steps for further analysis based on the patterns observed.
            
            Keep your response concise, focused, and scholarly. Avoid technical jargon about the metrics themselves and focus on what the results mean for understanding the texts.
            """
            
            # Use Puter API to access Claude 3.5 Sonnet (free, no API key required)
            response_text = call_puter_claude(
                prompt=prompt,
                system_message=system_message,
                max_tokens=max_tokens
            )
            return response_text
            
        except Exception as e:
            logger.error("Error with Puter API: %s", str(e))
            return "Error interpreting results: " + str(e)
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare a summary of the data for the LLM.
        
        Args:
            df: DataFrame containing similarity metrics
            
        Returns:
            Dict containing summary statistics
        """
        summary = {
            "num_segments": len(df),
            "metrics": [],
            "segment_pairs": []
        }
        
        # Get text columns and metric columns
        text_cols = ['Text A', 'Text B', 'Text A ID', 'Text B ID']
        text_cols = [col for col in text_cols if col in df.columns]
        metric_cols = [col for col in df.columns if col not in text_cols]
        
        summary["metrics"] = metric_cols
        
        # Ensure metrics are numeric before calculating statistics
        numeric_df = df.copy()
        for col in metric_cols:
            # Try to convert to numeric, set non-convertible values to NaN
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        # Add summary statistics for each metric
        for metric in metric_cols:
            try:
                summary["{}_avg".format(metric)] = numeric_df[metric].mean()
                summary["{}_min".format(metric)] = numeric_df[metric].min()
                summary["{}_max".format(metric)] = numeric_df[metric].max()
            except Exception as e:
                logger.warning("Could not calculate statistics for %s: %s", metric, str(e))
                summary["{}_avg".format(metric)] = "N/A"
                summary["{}_min".format(metric)] = "N/A"
                summary["{}_max".format(metric)] = "N/A"
        
        # Get the top 5 most similar pairs for each metric
        for metric in metric_cols:
            try:
                # Try to sort by the metric
                top_pairs = numeric_df.sort_values(by=metric, ascending=False).head(5)
                
                # Get the text columns that exist
                text_a_col = 'Text A' if 'Text A' in df.columns else None
                text_b_col = 'Text B' if 'Text B' in df.columns else None
                
                # Create the top pairs list
                if text_a_col and text_b_col:
                    # Use the original df to get the text values
                    pairs_list = []
                    for _, row in top_pairs.iterrows():
                        # Get the index to find the corresponding row in the original df
                        idx = row.name
                        if idx in df.index:
                            # Get the original text values and the metric value
                            text_a = df.loc[idx, text_a_col]
                            text_b = df.loc[idx, text_b_col]
                            metric_val = numeric_df.loc[idx, metric]
                            pairs_list.append([text_a, text_b, float(metric_val)])
                    
                    summary["{}_top_pairs".format(metric)] = pairs_list
                else:
                    # If text columns don't exist, just use the index as identifiers
                    summary["{}_top_pairs".format(metric)] = [["Item {}".format(row.name), "Item {}".format(row.name), float(row[metric])] 
                                                   for _, row in top_pairs.iterrows()]
            except Exception as e:
                logger.warning("Could not get top pairs for %s: %s", metric, str(e))
                summary["{}_top_pairs".format(metric)] = []
        
        return summary
    
    def _create_interpretation_prompt(self, df_summary: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Create a prompt for the LLM to interpret the similarity metrics.
        
        Args:
            df_summary: Summary statistics from the metrics DataFrame
            df: The full metrics DataFrame
            
        Returns:
            str: Prompt for the LLM
        """
        # Create a detailed prompt with the metrics data
        prompt = """
        I need help interpreting similarity metrics between Tibetan text segments. Here's a summary of the data:
        
        Number of text files analyzed: {}
        Number of segments/chapters: {}
        
        Metrics Summary:
        - Jaccard Similarity: Mean = {:.2f}%, Min = {:.2f}%, Max = {:.2f}%
        - Normalized LCS: Mean = {:.4f}, Min = {:.4f}, Max = {:.4f}
        - Semantic Similarity: Mean = {:.4f}, Min = {:.4f}, Max = {:.4f}
        - TF-IDF Cosine Similarity: Mean = {:.4f}, Min = {:.4f}, Max = {:.4f}
        
        Notable observations:
        - Segments with highest Jaccard similarity: {}
        - Segments with highest Normalized LCS: {}
        - Segments with highest Semantic similarity: {}
        - Segments with highest TF-IDF similarity: {}
        
        Here's the full metrics data (first 20 rows):
        {}
        
        Please provide a detailed chapter-level analysis of these metrics. Focus on:
        1. Specific chapter pairs that show interesting relationships
        2. Patterns across different chapters and texts
        3. What these metrics suggest about textual relationships and transmission history
        4. Notable outliers or unexpected similarities/differences
        
        Analyze each metric type (Jaccard, LCS, Semantic, TF-IDF) separately and then provide an integrated analysis.
        """.format(
            df_summary.get('num_files', 'N/A'),
            df_summary.get('num_segments', 'N/A'),
            df_summary.get('jaccard_mean', 0),
            df_summary.get('jaccard_min', 0),
            df_summary.get('jaccard_max', 0),
            df_summary.get('lcs_mean', 0),
            df_summary.get('lcs_min', 0),
            df_summary.get('lcs_max', 0),
            df_summary.get('semantic_mean', 0),
            df_summary.get('semantic_min', 0),
            df_summary.get('semantic_max', 0),
            df_summary.get('tfidf_mean', 0),
            df_summary.get('tfidf_min', 0),
            df_summary.get('tfidf_max', 0),
            df_summary.get('highest_jaccard_pairs', 'N/A'),
            df_summary.get('highest_lcs_pairs', 'N/A'),
            df_summary.get('highest_semantic_pairs', 'N/A'),
            df_summary.get('highest_tfidf_pairs', 'N/A'),
            df.head(20).to_string()
        )
        return prompt

def get_interpretation(results_df: pd.DataFrame) -> str:
    """
    Get an interpretation of the similarity metrics using Claude 3.5 Sonnet via Puter API.
    No API key required - completely free.
    
    Args:
        results_df: DataFrame containing similarity metrics
        
    Returns:
        str: Human-readable interpretation of the results
    """
    interpreter = LLMInterpreter()
    return interpreter.interpret_results(results_df)
