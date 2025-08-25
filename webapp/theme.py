import gradio as gr
from gradio.themes.utils import colors, sizes, fonts


class TibetanAppTheme(gr.themes.Soft):
    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,  # Primary interactive elements (e.g., #2563eb)
            secondary_hue=colors.orange,  # For accents if needed, or default buttons
            neutral_hue=colors.slate,  # For backgrounds, borders, and text
            font=[
                fonts.GoogleFont("Inter"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
            radius_size=sizes.radius_md,  # General radius, can be overridden (16px was for cards)
            text_size=sizes.text_md,  # Base font size (16px)
        )
        self.theme_vars_for_set = {
            # Global & Body Styles
            "body_background_fill": "#f0f2f5",
            "body_text_color": "#333333",
            # Card Styles (.gr-group)
            "block_background_fill": "#ffffff",
            "block_radius": "16px",  # May need to be removed if not a valid settable CSS var
            "block_shadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
            "block_padding": "24px",
            "block_border_width": "0px",
            # Markdown Styles
            "body_text_color_subdued": "#4b5563",
            # Button Styles
            "button_secondary_background_fill": "#ffffff",
            "button_secondary_text_color": "#374151",
            "button_secondary_border_color": "#d1d5db",
            "button_secondary_border_color_hover": "#adb5bd",
            "button_secondary_background_fill_hover": "#f9fafb",
            # Primary Button
            "button_primary_background_fill": "#2563eb",
            "button_primary_text_color": "#ffffff",
            "button_primary_border_color": "transparent",
            "button_primary_background_fill_hover": "#1d4ed8",
            # HR style
            "border_color_accent_subdued": "#e5e7eb",
        }
        super().set(**self.theme_vars_for_set)

        # Store CSS overrides; these will be converted to a string and applied via gr.Blocks(css=...)
        self.css_overrides = {
            ".gradio-container, .gr-block, .gr-markdown, label, input, .gr-slider, .gr-radio, .gr-button": {
                "font-family": ", ".join(self.font),
                "font-size": "16px !important",
                "line-height": "1.6 !important",
                "color": "#333333 !important",
            },
            ".gr-group": {"margin-bottom": "24px !important"},  # min-height removed
            ".gr-markdown": {
                "background": "transparent !important",
                "font-size": "1em !important",
                "margin-bottom": "16px !important",
            },
            ".gr-markdown h1": {
                "font-size": "28px !important",
                "font-weight": "600 !important",
                "margin-bottom": "8px !important",
                "color": "#111827 !important",
            },
            ".gr-markdown h2": {
                "font-size": "26px !important",
                "font-weight": "600 !important",
                "color": "var(--primary-600, #2563eb) !important",
                "margin-top": "32px !important",
                "margin-bottom": "16px !important",
            },
            ".gr-markdown h3": {
                "font-size": "22px !important",
                "font-weight": "600 !important",
                "color": "#1f2937 !important",
                "margin-top": "24px !important",
                "margin-bottom": "12px !important",
            },
            ".gr-markdown p, .gr-markdown span": {
                "font-size": "16px !important",
                "color": "#4b5563 !important",
            },
            ".gr-button button": {
                "border-radius": "8px !important",
                "padding": "10px 20px !important",
                "font-weight": "500 !important",
                "box-shadow": "0 1px 2px 0 rgba(0, 0, 0, 0.05) !important",
                "border": "1px solid #d1d5db !important",
                "background-color": "#ffffff !important",
                "color": "#374151 !important",
            },
            "#run-btn": {
                "background": "var(--button-primary-background-fill) !important",
                "color": "var(--button-primary-text-color) !important",
                "font-weight": "bold !important",
                "font-size": "24px !important",
                "border": "none !important",
                "box-shadow": "var(--button-primary-shadow) !important",
            },
            "#run-btn:hover": {  # Changed selector
                "background": "var(--button-primary-background-fill-hover) !important",
                "box-shadow": "0px 4px 12px rgba(0, 0, 0, 0.15) !important",
                "transform": "translateY(-1px) !important",
            },
            ".gr-button button:hover": {
                "background-color": "#f9fafb !important",
                "border-color": "#adb5bd !important",
            },
            "hr": {
                "margin": "32px 0 !important",
                "border": "none !important",
                "border-top": "1px solid var(--border-color-accent-subdued) !important",
            },
            ".gr-slider, .gr-radio, .gr-file": {"margin-bottom": "20px !important"},
            ".gr-radio .gr-form button": {
                "background-color": "#f3f4f6 !important",
                "color": "#374151 !important",
                "border": "1px solid #d1d5db !important",
                "border-radius": "6px !important",
                "padding": "8px 16px !important",
                "font-weight": "500 !important",
            },
            ".gr-radio .gr-form button:hover": {
                "background-color": "#e5e7eb !important",
                "border-color": "#9ca3af !important",
            },
            ".gr-radio .gr-form button.selected": {
                "background-color": "var(--primary-500, #3b82f6) !important",
                "color": "#ffffff !important",
                "border-color": "var(--primary-500, #3b82f6) !important",
            },
            ".gr-radio .gr-form button.selected:hover": {
                "background-color": "var(--primary-600, #2563eb) !important",
                "border-color": "var(--primary-600, #2563eb) !important",
            },
            "#semantic-radio-group span": {  # General selector, refined size
                "font-size": "17px !important",
                "font-weight": "500 !important",
            },
            "#semantic-radio-group div": {  # General selector, refined size
                "font-size": "14px !important"
            },
            # Row and Column flex styles for equal height
            "#steps-row": {
                "display": "flex !important",
                "align-items": "stretch !important",
            },
            ".step-column": {
                "display": "flex !important",
                "flex-direction": "column !important",
            },
            ".step-column > .gr-group": {
                "flex-grow": "1 !important",
                "display": "flex !important",
                "flex-direction": "column !important",
            },
            ".tabs > .tab-nav": {"border-bottom": "1px solid #d1d5db !important"},
            ".tabs > .tab-nav > button.selected": {
                "border-bottom": "2px solid var(--primary-500) !important",
                "color": "var(--primary-500) !important",
                "background-color": "transparent !important",
            },
            ".tabs > .tab-nav > button": {
                "color": "#6b7280 !important",
                "background-color": "transparent !important",
                "padding": "10px 15px !important",
                "border-bottom": "2px solid transparent !important",
            },
            
            # Custom styling for metric accordions
            ".metric-info-accordion": {
                "border-left": "4px solid #3B82F6 !important",
                "margin-bottom": "1rem !important",
                "background-color": "#F8FAFC !important",
                "border-radius": "6px !important",
                "overflow": "hidden !important",
            },
            ".jaccard-info": {
                "border-left-color": "#3B82F6 !important",  # Blue
            },
            ".lcs-info": {
                "border-left-color": "#10B981 !important",  # Green
            },
            ".semantic-info": {
                "border-left-color": "#8B5CF6 !important",  # Purple
            },
            ".wordcount-info": {
                "border-left-color": "#EC4899 !important",  # Pink
            },
            
            # Accordion header styling
            ".metric-info-accordion > .label-wrap": {
                "font-weight": "600 !important",
                "padding": "12px 16px !important",
                "background-color": "#F1F5F9 !important",
                "border-bottom": "1px solid #E2E8F0 !important",
            },
            
            # Accordion content styling
            ".metric-info-accordion > .wrap": {
                "padding": "16px !important",
            },
            
            # Word count plot styling - full width
            ".tabs > .tab-content > div[data-testid='tabitem'] > .plot": {
                "width": "100% !important",
            },
            
            # Heatmap plot styling - responsive sizing
            ".tabs > .tab-content > div[data-testid='tabitem'] > .plotly": {
                "width": "100% !important",
                "height": "auto !important",
            },
            
            # Specific heatmap container styling
            ".metric-heatmap": {
                "max-width": "100% !important",
                "overflow-x": "auto !important",
            },
            
            # LLM Analysis styling
            ".llm-analysis": {
                "background-color": "#f8f9fa !important",
                "border-left": "4px solid #3B82F6 !important",
                "border-radius": "8px !important",
                "padding": "20px 24px !important",
                "margin": "16px 0 !important",
                "box-shadow": "0 2px 8px rgba(0, 0, 0, 0.05) !important",
            },
            ".llm-analysis h2": {
                "color": "#1e40af !important",
                "font-size": "24px !important",
                "margin-bottom": "16px !important",
                "border-bottom": "1px solid #e5e7eb !important",
                "padding-bottom": "8px !important",
            },
            ".llm-analysis h3, .llm-analysis h4": {
                "color": "#1e3a8a !important",
                "margin-top": "20px !important",
                "margin-bottom": "12px !important",
            },
            ".llm-analysis p": {
                "line-height": "1.7 !important",
                "margin-bottom": "12px !important",
            },
            ".llm-analysis ul, .llm-analysis ol": {
                "margin-left": "24px !important",
                "margin-bottom": "16px !important",
            },
            ".llm-analysis li": {
                "margin-bottom": "6px !important",
            },
            ".llm-analysis strong, .llm-analysis b": {
                "color": "#1f2937 !important",
                "font-weight": "600 !important",
            },
        }

    def get_css_string(self) -> str:
        """Converts the self.css_overrides dictionary into a CSS string."""
        css_parts = []
        for selector, properties in self.css_overrides.items():
            props_str = "\n".join(
                [f"    {prop}: {value};" for prop, value in properties.items()]
            )
            css_parts.append(f"{selector} {{\n{props_str}\n}}")
        return "\n\n".join(css_parts)


# Instantiate the theme for easy import
tibetan_theme = TibetanAppTheme()
