import re
from typing import Optional


def strip_think_tags(text: Optional[str]) -> str:
    """
    Remove <think>...</think> tags and their content from text.

    Args:
        text: The text to process, which may contain think tags

    Returns:
        The text with all think tags and their content removed
    """
    if not text:
        return ""

    # Use regex to remove <think>...</think> tags and everything between them
    # The (?s) flag makes . match newlines as well
    # The *? makes it non-greedy to handle multiple think tags correctly
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up any double spaces or extra newlines that might be left
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)

    return cleaned_text.strip()
