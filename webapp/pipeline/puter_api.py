"""
Text metrics interpreter module that provides rule-based analysis of similarity metrics.
"""

import logging

# Set up logging
logger = logging.getLogger(__name__)

def call_puter_claude(prompt: str, system_message: str = "", max_tokens: int = 1500) -> str:
    """
    Provides a rule-based analysis of similarity metrics.
    
    Args:
        prompt: The prompt containing metrics data
        system_message: Not used, kept for API compatibility
        max_tokens: Not used, kept for API compatibility
        
    Returns:
        str: A detailed analysis of the metrics
    """
    try:
        # Extract metrics data from the prompt
        return analyze_metrics_from_prompt(prompt)
    except Exception as e:
        logger.error("Error analyzing metrics: %s", str(e))
        return generate_general_guidance()


def analyze_metrics_from_prompt(prompt: str) -> str:
    """
    Analyzes metrics data from the prompt and provides insights.
    
    Args:
        prompt: The prompt containing metrics data
        
    Returns:
        str: A detailed analysis of the metrics
    """
    # Extract CSV data from the prompt if possible
    try:
        # Look for the CSV data section in the prompt
        import re
        import pandas as pd
        from io import StringIO
        
        # Try to extract the CSV data table from the prompt
        csv_pattern = r"Here's the full metrics data.*?\n(.*?)\n\n"  # Match the CSV data section
        csv_match = re.search(csv_pattern, prompt, re.DOTALL)
        
        if csv_match:
            # Extract the CSV-like data and parse it
            csv_text = csv_match.group(1)
            
            # Try to identify chapter numbers and text pairs
            chapter_pattern = r"(Ngari|Bhutan|Dolpo|Mustang)\s+(\d+)\s+vs\s+(Ngari|Bhutan|Dolpo|Mustang)\s+(\d+)"
            chapter_matches = re.findall(chapter_pattern, csv_text)
            
            # Extract metrics data using regex
            metrics_pattern = r"(\d+\.\d+)"
            metrics_matches = re.findall(metrics_pattern, csv_text)
            
            # Determine if we have high similarity values
            high_jaccard = any(float(val) > 30 for val in metrics_matches[::4] if val.replace('.', '', 1).isdigit())
            high_lcs = any(float(val) > 0.4 for val in metrics_matches[1::4] if val.replace('.', '', 1).isdigit())
            high_semantic = any(float(val) > 0.7 for val in metrics_matches[2::4] if val.replace('.', '', 1).isdigit())
            high_tfidf = any(float(val) > 0.5 for val in metrics_matches[3::4] if val.replace('.', '', 1).isdigit())
            
            # Identify specific chapter pairs with high similarity
            high_similarity_pairs = []
            for i, match in enumerate(chapter_matches[:5]):
                if i < len(metrics_matches) // 4:
                    jaccard = float(metrics_matches[i*4]) if i*4 < len(metrics_matches) else 0
                    lcs = float(metrics_matches[i*4+1]) if i*4+1 < len(metrics_matches) else 0
                    semantic = float(metrics_matches[i*4+2]) if i*4+2 < len(metrics_matches) else 0
                    tfidf = float(metrics_matches[i*4+3]) if i*4+3 < len(metrics_matches) else 0
                    
                    if jaccard > 30 or lcs > 0.4 or semantic > 0.7 or tfidf > 0.5:
                        high_similarity_pairs.append((match[0], match[1], match[2], match[3], jaccard, lcs, semantic, tfidf))
        else:
            # If we can't extract the CSV data, fall back to simple pattern matching
            high_jaccard = "high Jaccard" in prompt.lower() or "jaccard similarity above" in prompt.lower()
            high_lcs = "high normalized LCS" in prompt.lower() or "LCS values above" in prompt.lower()
            high_semantic = "high semantic similarity" in prompt.lower() or "semantic similarity above" in prompt.lower()
            high_tfidf = "high TF-IDF" in prompt.lower() or "TF-IDF above" in prompt.lower()
            high_similarity_pairs = []
        
        # Generate a more specific analysis based on what we found
        analysis = """## Analysis of Tibetan Text Similarity Metrics

Based on the metrics data provided, here's an interpretation of the textual relationships at the chapter level:

### Key Findings:"""
        
        # Add insights based on detected patterns and specific chapter pairs
        if high_similarity_pairs:
            # Add specific chapter-level analysis
            analysis += """

#### Notable Chapter Relationships:"""
            
            # Add details about specific high-similarity pairs
            for pair in high_similarity_pairs[:3]:  # Show up to 3 high similarity pairs
                text_a, chapter_a, text_b, chapter_b, jaccard, lcs, semantic, tfidf = pair
                # Avoid f-string with conditional expressions inside by preparing them first
                vocab_strength = "strong" if jaccard > 30 else "moderate"
                struct_strength = "significant" if lcs > 0.4 else "some"
                sem_strength = "high" if semantic > 0.7 else "moderate"
                topic_desc = "likely cover the same topics" if semantic > 0.7 else "may address related concepts"
                
                analysis += f"""

- **{text_a} Chapter {chapter_a} & {text_b} Chapter {chapter_b}**:
  - Jaccard Similarity: {jaccard:.2f}%
  - Normalized LCS: {lcs:.4f}
  - Semantic Similarity: {semantic:.4f}
  - TF-IDF Similarity: {tfidf:.4f}
  
  This chapter pair shows {vocab_strength} vocabulary overlap and {struct_strength} structural similarity. The {sem_strength} semantic similarity suggests they {topic_desc}."""
        
        # Add general patterns based on metrics
        analysis += """

#### General Patterns:"""
        
        if high_jaccard:
            analysis += """
- **Strong Vocabulary Overlap**: The high Jaccard similarity indicates substantial shared vocabulary between segments. This suggests these texts likely share content or derive from a common source."""
        else:
            analysis += """
- **Vocabulary Differences**: The Jaccard similarity values suggest distinct vocabulary usage between segments, which may indicate different authorship, time periods, or subject matter."""
            
        if high_lcs:
            analysis += """
- **Sequential Similarity**: The high Normalized LCS values show that significant portions of text appear in the same sequence across segments. This points to direct textual borrowing or a shared textual tradition."""
        else:
            analysis += """
- **Structural Variation**: The Normalized LCS values indicate considerable differences in word sequence and structure, suggesting independent composition or significant reworking of content."""
            
        if high_semantic:
            analysis += """
- **Conceptual Alignment**: The semantic similarity scores reveal that these texts express similar concepts and ideas, even if using different specific terminology."""
        else:
            analysis += """
- **Conceptual Divergence**: The semantic similarity scores suggest these texts address different concepts or topics, despite any surface-level similarities."""
            
        if high_tfidf:
            analysis += """
- **Shared Key Terminology**: The TF-IDF similarity indicates these texts share important characteristic terms that define their subject matter, suggesting topical alignment."""
        else:
            analysis += """
- **Distinct Terminology Focus**: The TF-IDF similarity suggests these texts emphasize different key terms, which may indicate different focuses within a broader subject area."""
        
        # Add chapter-specific interpretation guidance
        analysis += """

### Chapter-Level Interpretation Guidance:

1. **Textual Relationships**: The metrics suggest varying degrees of relationship between different chapters. Pay special attention to chapters with high similarity across multiple metrics, as these likely represent the strongest textual connections.

2. **Transmission Patterns**: The pattern of similarities across chapters may reveal how these texts were transmitted or evolved. Look for sequences of chapters with similar patterns that might indicate systematic borrowing or adaptation.

3. **Metric Divergence**: When chapters show high similarity in one metric but low in others (e.g., high Jaccard but low LCS), this often indicates:
   - Shared vocabulary but different structure (reworked content)
   - Similar structure but different vocabulary (formulaic patterns with term substitution)
   - Similar meaning expressed differently (paraphrasing)

4. **Next Steps**: For deeper analysis, examine specific chapter pairs with interesting patterns, particularly those showing unexpected similarity or difference. Consider comparing the actual text content of these chapters to verify the computational findings."""
        
        # Add suggestions for further research
        analysis += """

### Suggestions for Further Research:

1. **Focused Textual Comparison**: Conduct close readings of the specific chapter pairs identified with high similarity to verify the computational findings.

2. **Historical Context**: Investigate the historical relationship between these text traditions, particularly focusing on potential transmission routes or shared sources.

3. **Sequential Analysis**: Examine whether similarities follow a sequential pattern across chapters, which might indicate systematic borrowing or adaptation.

4. **Content Analysis**: For chapters with high semantic similarity but lower lexical similarity, analyze how similar concepts are expressed with different vocabulary."""

        
        return analysis
        
    except Exception as e:
        logger.error("Error extracting metrics from prompt: %s", str(e))
        return generate_general_guidance()


def generate_general_guidance() -> str:
    """
    Provides general guidance on interpreting similarity metrics.
    
    Returns:
        str: General guidance on interpreting metrics
    """
    return """## Guide to Interpreting Tibetan Text Similarity Metrics

When analyzing your similarity metrics, consider these key points:

### Understanding Each Metric:

1. **Jaccard Similarity (%)**: 
   - Measures shared vocabulary between texts
   - Higher values (>50%) indicate substantial shared vocabulary
   - Useful for identifying texts with similar terminology
   - Less sensitive to word order or structure

2. **Normalized LCS**: 
   - Measures the longest sequence of words appearing in the same order
   - Higher values (>0.5) indicate significant structural similarity
   - Helps identify direct textual borrowing or shared passages
   - Sensitive to text structure and sequence

3. **Semantic Similarity**: 
   - Measures similarity in meaning using embedding models
   - Higher values suggest conceptually related content
   - Can detect similar ideas expressed with different vocabulary
   - Particularly useful for identifying conceptual relationships

4. **TF-IDF Cosine Similarity**: 
   - Highlights texts sharing important characteristic terms
   - Emphasizes distinctive terminology rather than common words
   - Higher values indicate shared subject-specific vocabulary
   - Useful for identifying topical relationships

### Analytical Approaches:

- **Cross-Metric Analysis**: Compare results across different metrics to get a more complete picture
- **Pattern Identification**: Look for clusters of high similarity or unusual divergences
- **Outlier Analysis**: Pay special attention to segments with unexpectedly high or low similarity
- **Segment Length Consideration**: Factor in the length of segments when interpreting results

For the most insightful analysis, combine these quantitative metrics with your qualitative understanding of the texts and their historical context."""
