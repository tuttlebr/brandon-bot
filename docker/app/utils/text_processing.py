import logging
import re
import unicodedata
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def clean_content(content: str) -> str:
    """
    Clean content text by removing formatting artifacts and ensuring
    plain text display

    Args:
        content: Raw content string from search results

    Returns:
        str: Cleaned content suitable for markdown display
    """
    if not content:
        return ""

    # # Remove common markdown formatting artifacts

    # # Remove markdown headers (# ## ###)
    # content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)

    # # Remove markdown bold/italic formatting
    # # (**text**, *text*, __text__, _text_)
    # content = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", content)
    # content = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", content)

    # # Remove markdown links but keep the text [text](url) -> text
    # content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

    # # Remove HTML tags
    # content = re.sub(r"<[^>]+>", "", content)

    # # Remove excessive whitespace and normalize line breaks
    # content = re.sub(r"\s+", " ", content)

    # # Remove leading/trailing quotes that might be artifacts
    # content = content.strip("\"'")

    content = content.strip()
    return content


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
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # # Clean up any double spaces or extra newlines that might be left
    # cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    # cleaned_text = re.sub(r'  +', ' ', cleaned_text)

    return cleaned_text.strip()


def escape_markdown_dollars(text: Optional[str]) -> str:
    """
    Escape dollar signs in text for Streamlit markdown rendering.

    Streamlit's markdown renderer interprets dollar signs as LaTeX math
    delimiters. This function escapes standalone dollar signs while
    preserving intended LaTeX math expressions that use the ${...} syntax.

    Args:
        text: The text to process

    Returns:
        Text with dollar signs properly escaped for Streamlit markdown
    """
    if not text:
        return ""

    # Escape all dollar signs first
    escaped_text = text.replace("$", "\\$")

    # But preserve LaTeX math expressions that use ${...} syntax
    escaped_text = escaped_text.replace("\\${", "${")

    return escaped_text


def sanitize_markdown_for_streamlit(text: Optional[str]) -> str:
    """
    Sanitize markdown text for safe display in Streamlit.

    This function prevents markdown rendering issues that can cause
    corruption with certain characters, especially during streaming updates.

    Args:
        text: The markdown text to sanitize

    Returns:
        Sanitized markdown text safe for Streamlit display
    """
    if not text:
        return ""

    # First escape dollar signs
    text = escape_markdown_dollars(text)

    # Convert escaped newline sequences ("\\n") into actual newlines. This
    # handles cases where upstream tools double-escape newlines, causing the
    # literal characters "\n" to appear in the rendered Markdown instead of
    # real line breaks. We replace *after* dollar-sign escaping to preserve any
    # potential LaTeX constructs.
    # text = text.replace("\\n", "\n")

    # # Remove any null bytes or non-printable characters that might cause issues
    # text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')

    # # ------------------------------------------------------------------
    # # Fix malformed quote / block-quote artifacts sometimes produced by
    # # the LLM when JSON strings containing markdown are parsed.  These
    # # show up as literal sequences like ">"">" which render as
    # # distracting characters in Streamlit.  We remove stray quote marks
    # # that directly bracket block-quote symbols (>) while preserving
    # # legitimate quotes elsewhere.
    # # ------------------------------------------------------------------
    # # Pattern 1: quote(s) + > + optional quote(s)  -> keep a single '>'
    # text = re.sub(r'"+\s*>\s*"*', '> ', text)

    # # Pattern 2: > followed by quote(s) (e.g., '>"') -> '>'
    # text = re.sub(r'>\s*"+', '>', text)

    # # Collapse any lingering repeated '> ' sequences
    # text = re.sub(r'(>\s*){2,}', '> ', text)

    # # Remove any null bytes or non-printable characters that might cause issues
    # text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')

    # # Fix potential issues with repeated quote markers
    # # Replace multiple consecutive > with a single >
    # text = re.sub(r'>{2,}', '>', text)

    # # Ensure quote blocks have proper spacing
    # text = re.sub(r'^>', '> ', text, flags=re.MULTILINE)

    return text


def sanitize_python_input(text: str) -> str:
    """
    Sanitize user input to make it safer for Python processing.

    Args:
        text: The text to sanitize

    Returns:
        Sanitized text safe for Python processing
    """
    if not text:
        return ""

    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8")
    )


def romanize_text(text: str) -> str:
    """
    Romanize/transliterate all non-ASCII characters to their closest
    ASCII equivalent.

    This function uses Unicode normalization to convert accented
    characters and other diacritical marks to their base ASCII forms.
    For more complex transliteration (e.g., Chinese, Japanese, Arabic),
    consider using the 'unidecode' library.

    Args:
        text: The input string to romanize

    Returns:
        A string with non-ASCII characters converted to their closest
        ASCII equivalent. If an error occurs, returns the original
        input string.
    """
    if not text:
        return text

    try:
        # First, normalize the text to NFD (decomposed) form
        # This separates base characters from combining characters
        normalized = unicodedata.normalize("NFD", text)

        # Build the result character by character
        result = []
        for char in normalized:
            # Get the Unicode category
            category = unicodedata.category(char)

            # Skip combining marks (Mn = Mark, Nonspacing)
            if category == "Mn":
                continue

            # Try to get the ASCII representation
            try:
                # Try to encode as ASCII
                ascii_char = char.encode("ascii").decode("ascii")
                result.append(ascii_char)
            except UnicodeEncodeError:
                # For characters that can't be converted to ASCII,
                # try to get their Unicode name and extract a
                # reasonable replacement
                try:
                    name = unicodedata.name(char, None)
                    if name:
                        # Handle some common cases
                        if "LATIN" in name and "LETTER" in name:
                            # Extract the base letter from names like
                            # "LATIN SMALL LETTER A WITH ACUTE"
                            parts = name.split()
                            if len(parts) >= 4:
                                letter = parts[3]
                                if len(letter) == 1 and letter.isalpha():
                                    result.append(
                                        letter.lower()
                                        if "SMALL" in name
                                        else letter
                                    )
                                    continue

                        # Handle quotation marks and apostrophes
                        if "QUOTATION MARK" in name:
                            result.append('"')
                            continue
                        elif (
                            "APOSTROPHE" in name
                            or name == "RIGHT SINGLE QUOTATION MARK"
                        ):
                            result.append("'")
                            continue

                        # Handle dashes and hyphens
                        if "DASH" in name or "HYPHEN" in name:
                            result.append("-")
                            continue

                        # Handle spaces
                        if "SPACE" in name:
                            result.append(" ")
                            continue

                    # If we can't handle it, skip the character
                    # (This is where unidecode would provide better
                    # coverage)

                except ValueError:
                    # Character has no Unicode name, skip it
                    pass

        return "".join(result)

    except Exception as e:  # pylint: disable=broad-except
        # We catch all exceptions to ensure we always return a string
        logger.warning(
            "Error in romanize_text: %s. Returning original input.", str(e)
        )
        return text


class TextProcessor:
    """
    Context-aware text processor that applies different sanitization
    strategies based on the use case.
    """

    @staticmethod
    def for_filename(text: str, max_length: int = 255) -> str:
        """
        Process text to create safe filenames.

        Args:
            text: Input text to process
            max_length: Maximum length for filename (default 255)

        Returns:
            Safe filename string
        """
        # Romanize to ASCII for maximum compatibility
        safe = romanize_text(text)
        # Replace unsafe filename characters
        safe = re.sub(r'[<>:"/\\|?*]', "_", safe)
        # Replace multiple spaces/underscores with single underscore
        safe = re.sub(r"[\s_]+", "_", safe)
        # Remove leading/trailing underscores and dots
        safe = safe.strip("_.").strip()
        # Ensure non-empty
        if not safe:
            safe = "unnamed"
        # Truncate if needed
        if len(safe) > max_length:
            safe = safe[:max_length].rstrip("_")
        return safe

    @staticmethod
    def for_search(text: str) -> str:
        """
        Normalize text for search operations while preserving original.
        Returns lowercase, normalized text for better matching.

        Args:
            text: Input text to normalize

        Returns:
            Normalized search string
        """
        if not text:
            return ""
        # Normalize Unicode (NFKC for compatibility)
        normalized = unicodedata.normalize("NFKC", text)
        # Convert to lowercase for case-insensitive search
        return normalized.lower().strip()

    @staticmethod
    def for_display(text: str, preserve_unicode: bool = True) -> str:
        """
        Clean text for display while preserving Unicode by default.

        Args:
            text: Input text to clean
            preserve_unicode: Whether to preserve Unicode characters

        Returns:
            Cleaned display text
        """
        if not text:
            return ""

        # Don't normalize whitespace aggressively - preserve markdown
        # formatting. Only clean up excessive blank lines (more than 2)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Clean up trailing whitespace on each line without breaking markdown
        lines = text.split("\n")
        cleaned_lines = [line.rstrip() for line in lines]
        text = "\n".join(cleaned_lines)

        if not preserve_unicode:
            # Only romanize if explicitly requested
            text = romanize_text(text)

        return text.strip()

    @staticmethod
    def for_code_identifier(text: str) -> str:
        """
        Convert text to valid Python/programming identifier.

        Args:
            text: Input text to convert

        Returns:
            Valid identifier string
        """
        # First romanize to get ASCII
        ascii_text = romanize_text(text)
        # Replace non-alphanumeric with underscores
        identifier = re.sub(r"[^a-zA-Z0-9]", "_", ascii_text)
        # Remove leading numbers
        identifier = re.sub(r"^[0-9]+", "", identifier)
        # Replace multiple underscores
        identifier = re.sub(r"_+", "_", identifier)
        # Remove leading/trailing underscores
        identifier = identifier.strip("_")
        # Ensure non-empty
        if not identifier:
            identifier = "var"
        return identifier

    @staticmethod
    def for_api_key(text: str) -> str:
        """
        Sanitize text that might contain API keys or secrets.
        Preserves exact formatting but strips dangerous characters.

        Args:
            text: Input text to sanitize

        Returns:
            Sanitized string safe for API usage
        """
        if not text:
            return ""
        # Remove control characters but preserve all printable chars
        return "".join(char for char in text if char.isprintable())

    @staticmethod
    def for_database_query(text: str) -> str:
        """
        Prepare text for database queries (basic SQL injection prevention).
        Note: Always use parameterized queries in production!

        Args:
            text: Input text to sanitize

        Returns:
            Escaped text for database usage
        """
        if not text:
            return ""
        # Basic escaping - replace single quotes
        # This is NOT sufficient for production - use parameterized queries!
        return text.replace("'", "''")

    @staticmethod
    def get_processing_info(text: str) -> Dict[str, Any]:
        """
        Analyze text and return information about what processing
        might be needed.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with processing recommendations
        """
        info = {
            "has_unicode": any(ord(char) > 127 for char in text),
            "has_special_chars": bool(re.search(r"[^a-zA-Z0-9\s]", text)),
            "has_control_chars": any(not char.isprintable() for char in text),
            "script_types": set(),
            "needs_normalization": False,
        }

        # Detect script types
        for char in text:
            if ord(char) > 127:
                name = unicodedata.name(char, "")
                if "CJK" in name:
                    info["script_types"].add("CJK")
                elif "ARABIC" in name:
                    info["script_types"].add("Arabic")
                elif "HEBREW" in name:
                    info["script_types"].add("Hebrew")
                elif "CYRILLIC" in name:
                    info["script_types"].add("Cyrillic")
                elif "DEVANAGARI" in name:
                    info["script_types"].add("Devanagari")

        # Check if normalization would change the text
        normalized = unicodedata.normalize("NFC", text)
        if normalized != text:
            info["needs_normalization"] = True

        return info


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
                close_index = self.buffer.find("</think>", i)
                if close_index != -1:
                    # Found closing tag, skip to after it
                    i = close_index + 8  # len('</think>')
                    self.in_think_tag = False
                else:
                    # No closing tag yet, wait for more chunks
                    break
            else:
                # Look for opening tag
                open_index = self.buffer.find("<think>", i)
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
                            self.buffer[j:].startswith("<")
                            or self.buffer[j:].startswith("<t")
                            or self.buffer[j:].startswith("<th")
                            or self.buffer[j:].startswith("<thi")
                            or self.buffer[j:].startswith("<thin")
                            or self.buffer[j:].startswith("<think")
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


# Usage Examples and Best Practices
# ==================================
#
# 1. File Operations:
#    filename = TextProcessor.for_filename("My Résumé 2024.pdf")
#    # Result: "My_Resume_2024.pdf"
#
# 2. Search Operations:
#    # Store original, search on normalized
#    message = {
#        'content': user_input,
#        'search_content': TextProcessor.for_search(user_input)
#    }
#
# 3. Display Operations:
#    # Preserve Unicode by default
#    display_text = TextProcessor.for_display(raw_text)
#    # Force ASCII only when needed
#    ascii_only = TextProcessor.for_display(raw_text, preserve_unicode=False)
#
# 4. Code Generation:
#    var_name = TextProcessor.for_code_identifier("User's Name")
#    # Result: "Users_Name"
#
# 5. API/Security:
#    clean_key = TextProcessor.for_api_key(api_input)
#    # Removes control characters but preserves the key
#
# 6. Text Analysis:
#    info = TextProcessor.get_processing_info(text)
#    if info['has_unicode'] and 'CJK' in info['script_types']:
#        # Handle CJK text differently
#
# 7. Legacy Functions:
#    # Use sparingly - only when you need forced ASCII conversion
#    ascii_only = sanitize_python_input(text)
#    # For more control over romanization
#    romanized = romanize_text(text)
#
# Best Practices:
# - Always preserve Unicode when possible
# - Use context-specific processors
# - Store both original and processed versions when needed
# - Don't apply blanket sanitization to all text
# - Consider user locale and preferences
