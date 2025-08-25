"""
LLM Service for Tibetan Text Metrics

This module provides a unified interface for analyzing text similarity metrics
using both LLM-based and rule-based approaches.
"""

import os
import json
import logging
import requests
import pandas as pd
import re

# Set up logging
logger = logging.getLogger(__name__)

# Try to load environment variables
ENV_LOADED = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_LOADED = True
except ImportError:
    logger.warning("python-dotenv not installed. Using system environment variables.")

# Constants
DEFAULT_MAX_TOKENS = 4000
PREFERRED_MODELS = [
    "qwen/qwen3-235b-a22b-07-25:free",
    "deepseek/deepseek-r1-0528:free",
    "google/gemma-2-9b-it:free",
    "moonshotai/kimi-k2:free"
]
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9

class LLMService:
    """
    Service for analyzing text similarity metrics using LLMs and rule-based methods.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the LLM service.
        
        Args:
            api_key: Optional API key for OpenRouter. If not provided, will try to load from environment.
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.models = PREFERRED_MODELS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
    
    def analyze_similarity(
        self, 
        results_df: pd.DataFrame, 
        use_llm: bool = True,
    ) -> str:
        """
        Analyze similarity metrics using either LLM or rule-based approach.
        
        Args:
            results_df: DataFrame containing similarity metrics
            use_llm: Whether to use LLM for analysis (falls back to rule-based if False or on error)
            
        Returns:
            str: Analysis of the metrics in markdown format with appropriate fallback messages
        """
        # If LLM is disabled, use rule-based analysis
        if not use_llm:
            logger.info("LLM analysis disabled. Using rule-based analysis.")
            return self._analyze_with_rules(results_df)
            
        # Try LLM analysis if enabled
        try:
            if not self.api_key:
                raise ValueError("No OpenRouter API key provided. Please set the OPENROUTER_API_KEY environment variable.")
                
            logger.info("Attempting LLM-based analysis...")
            return self._analyze_with_llm(results_df, max_tokens=DEFAULT_MAX_TOKENS)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in LLM analysis: {error_msg}")
            
            # Create a user-friendly error message
            if "no openrouter api key" in error_msg.lower():
                error_note = "OpenRouter API key not found. Please set the `OPENROUTER_API_KEY` environment variable to use this feature."
            elif "payment" in error_msg.lower() or "402" in error_msg:
                error_note = "OpenRouter API payment required. Falling back to rule-based analysis."
            elif "invalid" in error_msg.lower() or "401" in error_msg:
                error_note = "Invalid OpenRouter API key. Falling back to rule-based analysis."
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                error_note = "API rate limit exceeded. Falling back to rule-based analysis."
            else:
                error_note = f"LLM analysis failed: {error_msg[:200]}. Falling back to rule-based analysis."
            
            # Get rule-based analysis
            rule_based_analysis = self._analyze_with_rules(results_df)
            
            # Combine the error message with the rule-based analysis
            return f"## Analysis of Tibetan Text Similarity Metrics\n\n*Note: {error_note}*\n\n{rule_based_analysis}"
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the DataFrame for analysis.
        
        Args:
            df: Input DataFrame with similarity metrics
            
        Returns:
            pd.DataFrame: Cleaned and prepared DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Clean text columns
        text_cols = ['Text A', 'Text B']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
                df[col] = df[col].str.replace('.txt$', '', regex=True)
        
        # Filter out perfect matches (likely empty cells)
        metrics_cols = ['Jaccard Similarity (%)', 'Normalized LCS']
        if all(col in df.columns for col in metrics_cols):
            mask = ~((df['Jaccard Similarity (%)'] == 100.0) & 
                    (df['Normalized LCS'] == 1.0))
            df = df[mask].copy()
        
        return df
    
    def _analyze_with_llm(self, df: pd.DataFrame, max_tokens: int) -> str:
        """
        Analyze metrics using an LLM via OpenRouter API, with fallback models.

        Args:
            df: Prepared DataFrame with metrics
            max_tokens: Maximum tokens for the response

        Returns:
            str: LLM analysis in markdown format
        """
        last_error = None

        for model in self.models:
            try:
                # Create the prompt inside the loop to include the current model name
                prompt = self._create_llm_prompt(df, model)
                logger.info(f"Attempting analysis with model: {model}")
                response = self._call_openrouter_api(
                    model=model,
                    prompt=prompt,
                    system_message=self._get_system_prompt(),
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                logger.info(f"Successfully received response from {model}.")
                return self._format_llm_response(response, df, model)

            except requests.exceptions.HTTPError as e:
                last_error = e
                # Check for server-side errors (5xx) to decide whether to retry
                if 500 <= e.response.status_code < 600:
                    logger.warning(f"Model {model} failed with server error {e.response.status_code}. Trying next model.")
                    continue  # Try the next model
                else:
                    # For client-side errors (4xx), fail immediately
                    logger.error(f"LLM analysis failed with client error: {e}")
                    raise e
            except Exception as e:
                last_error = e
                logger.error(f"An unexpected error occurred with model {model}: {e}")
                continue # Try next model on other errors too

        # If all models failed, raise the last recorded error
        logger.error("All LLM models failed.")
        if last_error:
            raise last_error
        else:
            raise Exception("LLM analysis failed for all available models.")
    
    def _analyze_with_rules(self, df: pd.DataFrame) -> str:
        """
        Analyze metrics using rule-based approach.
        
        Args:
            df: Prepared DataFrame with metrics
            
        Returns:
            str: Rule-based analysis in markdown format
        """
        analysis = ["## Tibetan Text Similarity Analysis (Rule-Based)"]
        
        # Basic stats
        text_a_col = 'Text A' if 'Text A' in df.columns else None
        text_b_col = 'Text B' if 'Text B' in df.columns else None
        
        if text_a_col and text_b_col:
            unique_texts = set(df[text_a_col].unique()) | set(df[text_b_col].unique())
            analysis.append(f"- **Texts analyzed:** {', '.join(sorted(unique_texts))}")
        
        # Analyze each metric
        metric_analyses = []
        
        if 'Jaccard Similarity (%)' in df.columns:
            jaccard_analysis = self._analyze_jaccard(df)
            metric_analyses.append(jaccard_analysis)
            
        if 'Normalized LCS' in df.columns:
            lcs_analysis = self._analyze_lcs(df)
            metric_analyses.append(lcs_analysis)
            
        # TF-IDF analysis removed
        
        # Add all metric analyses
        if metric_analyses:
            analysis.extend(metric_analyses)
        
        # Add overall interpretation
        analysis.append("\n## Overall Interpretation")
        analysis.append(self._generate_overall_interpretation(df))
        
        return "\n\n".join(analysis)
    
    def _analyze_jaccard(self, df: pd.DataFrame) -> str:
        """Analyze Jaccard similarity scores."""
        jaccard = df['Jaccard Similarity (%)'].dropna()
        if jaccard.empty:
            return ""
            
        mean_jaccard = jaccard.mean()
        max_jaccard = jaccard.max()
        min_jaccard = jaccard.min()
        
        analysis = [
            "### Jaccard Similarity Analysis",
            f"- **Range:** {min_jaccard:.1f}% to {max_jaccard:.1f}% (mean: {mean_jaccard:.1f}%)"
        ]
        
        # Interpret the scores
        if mean_jaccard > 60:
            analysis.append("- **High vocabulary overlap** suggests texts share significant content or are from the same tradition.")
        elif mean_jaccard > 30:
            analysis.append("- **Moderate vocabulary overlap** indicates some shared content or themes.")
        else:
            analysis.append("- **Low vocabulary overlap** suggests texts are on different topics or from different traditions.")
        
        # Add top pairs
        top_pairs = df.nlargest(3, 'Jaccard Similarity (%)')
        if not top_pairs.empty:
            analysis.append("\n**Most similar pairs:**")
            for _, row in top_pairs.iterrows():
                text_a = row.get('Text A', 'Text 1')
                text_b = row.get('Text B', 'Text 2')
                score = row['Jaccard Similarity (%)']
                analysis.append(f"- {text_a} ↔ {text_b}: {score:.1f}%")
        
        return "\n".join(analysis)
    
    def _analyze_lcs(self, df: pd.DataFrame) -> str:
        """Analyze Longest Common Subsequence scores."""
        lcs = df['Normalized LCS'].dropna()
        if lcs.empty:
            return ""
            
        mean_lcs = lcs.mean()
        max_lcs = lcs.max()
        min_lcs = lcs.min()
        
        analysis = [
            "### Structural Similarity (LCS) Analysis",
            f"- **Range:** {min_lcs:.2f} to {max_lcs:.2f} (mean: {mean_lcs:.2f})"
        ]
        
        # Interpret the scores
        if mean_lcs > 0.7:
            analysis.append("- **High structural similarity** suggests texts follow similar organizational patterns.")
        elif mean_lcs > 0.4:
            analysis.append("- **Moderate structural similarity** indicates some shared organizational elements.")
        else:
            analysis.append("- **Low structural similarity** suggests different organizational approaches.")
        
        # Add top pairs
        top_pairs = df.nlargest(3, 'Normalized LCS')
        if not top_pairs.empty:
            analysis.append("\n**Most structurally similar pairs:**")
            for _, row in top_pairs.iterrows():
                text_a = row.get('Text A', 'Text 1')
                text_b = row.get('Text B', 'Text 2')
                score = row['Normalized LCS']
                analysis.append(f"- {text_a} ↔ {text_b}: {score:.2f}")
        
        return "\n".join(analysis)
    
    # TF-IDF analysis method removed
    
    def _generate_overall_interpretation(self, df: pd.DataFrame) -> str:
        """Generate an overall interpretation of the metrics."""
        interpretations = []
        
        # Get metrics if they exist
        has_jaccard = 'Jaccard Similarity (%)' in df.columns
        has_lcs = 'Normalized LCS' in df.columns
        
        # Calculate means for available metrics
        metrics = {}
        if has_jaccard:
            metrics['jaccard'] = df['Jaccard Similarity (%)'].mean()
        if has_lcs:
            metrics['lcs'] = df['Normalized LCS'].mean()
        # TF-IDF metrics removed
        
        # Generate interpretation based on metrics
        if metrics:
            interpretations.append("Based on the analysis of similarity metrics:")
            
            if has_jaccard and metrics['jaccard'] > 60:
                interpretations.append("- The high Jaccard similarity indicates significant vocabulary overlap between texts, "
                                     "suggesting they may share common sources or be part of the same textual tradition.")
            
            if has_lcs and metrics['lcs'] > 0.7:
                interpretations.append("- The high LCS score indicates strong structural similarity, "
                                     "suggesting the texts may follow similar organizational patterns or share common structural elements.")
            
            # TF-IDF interpretation removed
            
            # Add cross-metric interpretations
            if has_jaccard and has_lcs and metrics['jaccard'] > 60 and metrics['lcs'] > 0.7:
                interpretations.append("\nThe combination of high Jaccard and LCS similarities strongly suggests "
                                     "that these texts are closely related, possibly being different versions or "
                                     "transmissions of the same work or sharing a common source.")
            
            # TF-IDF cross-metric interpretation removed
        
        # Add general guidance if no specific patterns found
        if not interpretations:
            interpretations.append("The analysis did not reveal strong patterns in the similarity metrics. "
                                 "This could indicate that the texts are either very similar or very different "
                                 "across all measured dimensions.")
        
        return "\n\n".join(interpretations)
    
    def _create_llm_prompt(self, df: pd.DataFrame, model_name: str) -> str:
        """
        Create a prompt for the LLM based on the DataFrame.
        
        Args:
            df: Prepared DataFrame with metrics
            model_name: Name of the model being used
            
        Returns:
            str: Formatted prompt for the LLM
        """
        # Convert DataFrame to markdown for the prompt
        md_table = df.to_markdown(index=False)
        
        # Create the prompt
        prompt = f"""
# Tibetan Text Similarity Analysis

## Introduction

You will be provided with a table of text similarity scores in Markdown format. Your task is to provide a scholarly interpretation of these results for an academic article on Tibetan textual analysis. Do not simply restate the data. Instead, focus on the *implications* of the scores. What do they suggest about the relationships between the texts? Consider potential reasons for both high and low similarity across different metrics (e.g., shared vocabulary vs. structural differences).

**Data:**
{md_table}

## Instructions

Your analysis will be performed using the `{model_name}` model. Provide a concise, scholarly analysis in well-structured markdown.
"""
        

        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are a senior scholar of Tibetan Buddhist texts, specializing in textual criticism. Your task is to analyze the provided similarity metrics and provide expert insights into the relationships between these texts. Ground your analysis in the data, be precise, and focus on what the metrics reveal about the texts' transmission and history."""
    
    def _call_openrouter_api(self, model: str, prompt: str, system_message: str = None, max_tokens: int = None, temperature: float = None, top_p: float = None) -> str:
        """
        Call the OpenRouter API.
        
        Args:
            model: Model to use for the API call
            prompt: The user prompt
            system_message: Optional system message
            max_tokens: Maximum tokens for the response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            str: The API response
            
        Raises:
            ValueError: If API key is missing or invalid
            requests.exceptions.RequestException: For network-related errors
            Exception: For other API-related errors
        """
        if not self.api_key:
            error_msg = "OpenRouter API key not provided. Please set the OPENROUTER_API_KEY environment variable."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/daniel-wojahn/tibetan-text-metrics",
            "X-Title": "Tibetan Text Metrics"
        }
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": model,  # Use the model parameter here
            "messages": messages,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
        }
        
        try:
            logger.info(f"Calling OpenRouter API with model: {model}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            # Handle different HTTP status codes
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content'].strip()
                else:
                    error_msg = "Unexpected response format from OpenRouter API"
                    logger.error(f"{error_msg}: {result}")
                    raise ValueError(error_msg)
                    
            elif response.status_code == 401:
                error_msg = "Invalid OpenRouter API key. Please check your API key and try again."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            elif response.status_code == 402:
                error_msg = "OpenRouter API payment required. Please check your OpenRouter account balance or billing status."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            elif response.status_code == 429:
                error_msg = "API rate limit exceeded. Please try again later or check your OpenRouter rate limits."
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            else:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to OpenRouter API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse OpenRouter API response: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
    
    def _format_llm_response(self, response: str, df: pd.DataFrame, model_name: str) -> str:
        """
        Format the LLM response for display.
        
        Args:
            response: Raw LLM response
            df: Original DataFrame for reference
            model_name: Name of the model used
            
        Returns:
            str: Formatted response with fallback if needed
        """
        # Basic validation
        if not response or len(response) < 100:
            raise ValueError("Response too short or empty")
        
        # Check for garbled output (random numbers, nonsensical patterns)
        # This is a simple heuristic - look for long sequences of numbers or strange patterns
        suspicious_patterns = [
            r'\d{8,}',  # Long number sequences
            r'[0-9,.]{20,}',  # Long sequences of digits, commas and periods
            r'[\W]{20,}',  # Long sequences of non-word characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, response):
                logger.warning(f"Detected potentially garbled output matching pattern: {pattern}")
                # Don't immediately raise - we'll do a more comprehensive check
        
        # Check for content quality - ensure it has expected sections
        expected_content = [
            "introduction", "analysis", "similarity", "patterns", "conclusion", "question"
        ]
        
        # Count how many expected content markers we find
        content_matches = sum(1 for term in expected_content if term.lower() in response.lower())
        
        # If we find fewer than 3 expected content markers, log a warning
        if content_matches < 3:
            logger.warning(f"LLM response missing expected content sections (found {content_matches}/6)")
        
        # Check for text names from the dataset
        # Extract text names from the Text Pair column
        text_names = set()
        if "Text Pair" in df.columns:
            for pair in df["Text Pair"]:
                if isinstance(pair, str) and " vs " in pair:
                    texts = pair.split(" vs ")
                    text_names.update(texts)
        
        # Check if at least some text names appear in the response
        text_name_matches = sum(1 for name in text_names if name in response)
        if text_names and text_name_matches == 0:
            logger.warning("LLM response does not mention any of the text names from the dataset. The analysis may be generic.")
        
        # Ensure basic markdown structure
        if '##' not in response:
            response = f"## Analysis of Tibetan Text Similarity\n\n{response}"
        
        # Add styling to make the output more readable
        response = f"<div class='llm-analysis'>\n{response}\n</div>"
        
        # Format the response into a markdown block
        formatted_response = f"""## AI-Powered Analysis (Model: {model_name})\n\n{response}"""
    
        return formatted_response
    

