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


class StreamingThinkTagFilter:
    """
    A stateful filter for removing think tags from streaming text.
    Handles cases where tags span multiple chunks.
    """

    def __init__(self):
        self.buffer = ""
        self.in_think_tag = False
        self.pending_output = ""

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk and return displayable text.

        Args:
            chunk: The new text chunk from the stream

        Returns:
            Text that can be safely displayed (with think tags removed)
        """
        # Add chunk to buffer
        self.buffer += chunk

        # Process buffer to extract displayable content
        output = ""
        i = 0

        while i < len(self.buffer):
            if self.in_think_tag:
                # Look for closing tag
                close_index = self.buffer.find('</think>', i)
                if close_index != -1:
                    # Found closing tag, skip to after it
                    i = close_index + 8  # len('</think>')
                    self.in_think_tag = False
                else:
                    # No closing tag yet, wait for more chunks
                    break
            else:
                # Look for opening tag
                open_index = self.buffer.find('<think>', i)
                if open_index != -1:
                    # Output text before the tag
                    output += self.buffer[i:open_index]
                    i = open_index + 7  # len('<think>')
                    self.in_think_tag = True
                else:
                    # Check if we might have a partial tag at the end
                    partial_tag_start = max(i, len(self.buffer) - 7)
                    for j in range(partial_tag_start, len(self.buffer)):
                        if (
                            self.buffer[j:].startswith('<')
                            or self.buffer[j:].startswith('<t')
                            or self.buffer[j:].startswith('<th')
                            or self.buffer[j:].startswith('<thi')
                            or self.buffer[j:].startswith('<thin')
                            or self.buffer[j:].startswith('<think')
                        ):
                            # Might be start of tag, output up to this point
                            output += self.buffer[i:j]
                            # Keep the potential tag start in buffer
                            self.buffer = self.buffer[j:]
                            return output

                    # No partial tags, output the rest
                    output += self.buffer[i:]
                    self.buffer = ""
                    break

        # Update buffer to remove processed content
        if i < len(self.buffer):
            self.buffer = self.buffer[i:]
        else:
            self.buffer = ""

        return output

    def flush(self) -> str:
        """
        Get any remaining buffered content (called at end of stream).

        Returns:
            Any remaining displayable text
        """
        # If we're still in a think tag at the end, it wasn't closed properly
        # In this case, we just return empty string
        if self.in_think_tag:
            return ""

        # Return any remaining buffer content
        output = self.buffer
        self.buffer = ""
        return output
