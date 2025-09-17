import re
import unicodedata
from typing import Any, Dict, List, Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


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


def strip_analysis_blocks(text: Optional[str]) -> str:
    """
    Remove analysis...assistantfinal blocks from text.

    This handles the specific format from gpt-oss-120b model where reasoning
    content is prefixed with 'analysis' and suffixed with 'assistantfinal'.

    Args:
        text: The text to process, which may contain analysis blocks

    Returns:
        The text with analysis blocks removed, keeping only content after
    """
    if not text:
        return ""

    # Look for pattern: "analysis" at start, followed by content, ending with "assistantfinal"
    # We want to keep everything after this block
    pattern = r"^analysis.*?assistantfinal"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return cleaned_text.strip()


def strip_all_thinking_formats(text: Optional[str]) -> str:
    """
    Remove all known thinking/reasoning formats from text.

    This includes:
    - <think>...</think> tags
    - analysis...assistantfinal blocks

    Args:
        text: The text to process

    Returns:
        The text with all thinking formats removed
    """
    if not text:
        return ""

    # First strip think tags
    text = strip_think_tags(text)

    # Then strip analysis blocks
    text = strip_analysis_blocks(text)

    return text.strip()


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
    Now supports dynamic thinking tag configuration per model.
    """

    def __init__(
        self,
        model_name: str = None,
        thinking_start: str = None,
        thinking_stop: str = None,
    ):
        """
        Initialize the filter with optional model-specific tags

        Args:
            model_name: Name of the model (for schema lookup)
            thinking_start: Custom thinking start tag (overrides model schema)
            thinking_stop: Custom thinking stop tag (overrides model schema)
        """
        self.buffer = ""
        self.in_think_tag = False
        self.pending_output = ""
        self.found_stop_tag = False  # Track if we've found the stop tag (for stop-only filtering)

        # Get thinking tags from schema manager or use provided ones
        if thinking_start is not None and thinking_stop is not None:
            self.thinking_start = thinking_start
            self.thinking_stop = thinking_stop
        elif model_name:
            from utils.llm_schema_manager import schema_manager

            tags = schema_manager.get_thinking_tags(model_name)
            self.thinking_start, self.thinking_stop = tags
        else:
            # Use default tags
            from utils.config import config

            self.thinking_start = config.llm.DEFAULT_THINKING_START
            self.thinking_stop = config.llm.DEFAULT_THINKING_STOP

        # Cache tag lengths for efficiency and check if filtering is enabled
        self.start_tag_len = (
            len(self.thinking_start) if self.thinking_start else 0
        )
        self.stop_tag_len = (
            len(self.thinking_stop) if self.thinking_stop else 0
        )
        # Enable filtering if we have at least one tag (start OR stop)
        self.filtering_enabled = bool(
            self.thinking_start or self.thinking_stop
        )
        self.has_start_tag = bool(self.thinking_start)
        self.has_stop_tag = bool(self.thinking_stop)

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk and return displayable text.

        Args:
            chunk: The new text chunk from the stream

        Returns:
            Text that can be safely displayed (with think tags removed)
        """
        # If filtering is disabled (empty tags), pass through
        if not self.filtering_enabled:
            return chunk

        # Add chunk to buffer
        self.buffer += chunk

        # Process buffer to extract displayable content
        output = ""
        i = 0

        while i < len(self.buffer):
            if self.in_think_tag:
                # Look for closing tag (if we have one)
                if self.has_stop_tag:
                    close_index = self.buffer.find(self.thinking_stop, i)
                    if close_index != -1:
                        # Found closing tag, skip to after it
                        i = close_index + self.stop_tag_len
                        self.in_think_tag = False
                    else:
                        # No closing tag yet, wait for more chunks
                        break
                else:
                    # No stop tag configured, consume everything until end
                    self.buffer = ""
                    break
            else:
                # Handle different scenarios based on available tags
                if self.has_start_tag and self.has_stop_tag:
                    # Standard case: both start and stop tags
                    open_index = self.buffer.find(self.thinking_start, i)
                    if open_index != -1:
                        # Output text before the tag
                        output += self.buffer[i:open_index]
                        i = open_index + self.start_tag_len
                        self.in_think_tag = True
                    else:
                        # Check for partial start tag
                        partial_tag_start = max(
                            i, len(self.buffer) - self.start_tag_len
                        )
                        for j in range(partial_tag_start, len(self.buffer)):
                            remaining_buffer = self.buffer[j:]
                            if self.thinking_start.startswith(
                                remaining_buffer
                            ):
                                output += self.buffer[i:j]
                                self.buffer = self.buffer[j:]
                                return output
                        # No partial tags, output the rest
                        output += self.buffer[i:]
                        self.buffer = ""
                        break
                elif self.has_stop_tag and not self.has_start_tag:
                    # Only stop tag - this means everything before the stop tag is thinking content
                    # We need to buffer everything until we find the stop tag
                    stop_index = self.buffer.find(self.thinking_stop, i)
                    if stop_index != -1:
                        # Found stop tag - discard everything before it and the tag itself
                        # Only output content after the stop tag
                        remaining_content = self.buffer[
                            stop_index + self.stop_tag_len :
                        ]
                        output += remaining_content
                        self.buffer = ""
                        self.found_stop_tag = True
                        break
                    else:
                        # No stop tag found yet - check for partial stop tag
                        partial_found = False
                        for j in range(
                            max(i, len(self.buffer) - self.stop_tag_len),
                            len(self.buffer),
                        ):
                            remaining_buffer = self.buffer[j:]
                            if self.thinking_stop.startswith(
                                remaining_buffer
                            ) and len(remaining_buffer) < len(
                                self.thinking_stop
                            ):
                                # Found partial stop tag, keep it in buffer and don't output anything
                                self.buffer = self.buffer[j:]
                                partial_found = True
                                break

                        if not partial_found:
                            # No stop tag found yet - if we already found it in previous chunks,
                            # output everything. Otherwise, don't output anything (it's thinking content)
                            if self.found_stop_tag:
                                output += self.buffer[i:]
                                self.buffer = ""
                            else:
                                # Buffer everything - it might all be thinking content
                                self.buffer = self.buffer  # Keep entire buffer
                        break
                elif self.has_start_tag and not self.has_stop_tag:
                    # Only start tag - filter everything after start tag
                    open_index = self.buffer.find(self.thinking_start, i)
                    if open_index != -1:
                        # Output text before the tag, then consume everything after
                        output += self.buffer[i:open_index]
                        self.buffer = ""
                        break
                    else:
                        # Check for partial start tag
                        partial_tag_start = max(
                            i, len(self.buffer) - self.start_tag_len
                        )
                        for j in range(partial_tag_start, len(self.buffer)):
                            remaining_buffer = self.buffer[j:]
                            if self.thinking_start.startswith(
                                remaining_buffer
                            ):
                                output += self.buffer[i:j]
                                self.buffer = self.buffer[j:]
                                return output
                        # No partial tags, output the rest
                        output += self.buffer[i:]
                        self.buffer = ""
                        break
                else:
                    # No tags configured, pass through
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


class StreamingAnalysisBlockFilter:
    """
    A stateful filter for removing analysis...assistantfinal blocks from streaming text.
    Handles cases where the block spans multiple chunks.
    Now supports dynamic analysis tag configuration per model.
    """

    def __init__(
        self,
        model_name: str = None,
        analysis_start: str = None,
        analysis_stop: str = None,
    ):
        """
        Initialize the filter with optional model-specific tags

        Args:
            model_name: Name of the model (for schema lookup)
            analysis_start: Custom analysis start tag (overrides model schema)
            analysis_stop: Custom analysis stop tag (overrides model schema)
        """
        self.buffer = ""
        self.found_analysis_start = False
        self.found_assistant_final = False
        self.analysis_block_content = ""

        # Get analysis tags from schema manager or use provided ones
        if analysis_start and analysis_stop:
            self.analysis_start = analysis_start
            self.analysis_stop = analysis_stop
        elif model_name:
            from utils.llm_schema_manager import schema_manager

            self.analysis_start, self.analysis_stop = (
                schema_manager.get_analysis_tags(model_name)
            )
        else:
            # Use default tags
            self.analysis_start = "analysis"
            self.analysis_stop = "assistantfinal"

        # Handle None values (when analysis blocks aren't used)
        self.use_analysis_filtering = (
            self.analysis_start is not None and self.analysis_stop is not None
        )

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk and return displayable text.

        Args:
            chunk: The new text chunk from the stream

        Returns:
            Text that can be safely displayed (with analysis blocks removed)
        """
        # Skip processing if analysis filtering is disabled
        if not self.use_analysis_filtering:
            return chunk

        # Add chunk to buffer
        self.buffer += chunk

        # If we haven't found the analysis start yet
        if not self.found_analysis_start:
            # Check if buffer starts with analysis start tag
            if self.buffer.startswith(self.analysis_start):
                self.found_analysis_start = True
                self.analysis_block_content = self.buffer
                # Don't output anything yet
                return ""
            elif len(self.buffer) < len(self.analysis_start):
                # Not enough data to determine, wait for more
                return ""
            else:
                # Buffer doesn't start with analysis start tag, output and clear
                output = self.buffer
                self.buffer = ""
                return output

        # We're inside an analysis block
        if self.found_analysis_start and not self.found_assistant_final:
            self.analysis_block_content += chunk

            # Check if we've found the end marker
            if self.analysis_stop in self.analysis_block_content:
                self.found_assistant_final = True

                # Find the position after the analysis stop tag
                end_index = self.analysis_block_content.find(
                    self.analysis_stop
                ) + len(self.analysis_stop)

                # Extract content after the block
                if end_index < len(self.analysis_block_content):
                    remaining_content = self.analysis_block_content[end_index:]
                    self.buffer = ""
                    self.found_analysis_start = False
                    self.found_assistant_final = False
                    self.analysis_block_content = ""
                    return remaining_content
                else:
                    # No content after the block yet
                    self.buffer = ""
                    return ""
            else:
                # Still collecting the analysis block
                return ""

        # Normal processing after analysis block
        output = self.buffer
        self.buffer = ""
        return output

    def flush(self) -> str:
        """
        Get any remaining buffered content (called at end of stream).

        Returns:
            Any remaining displayable text
        """
        # If we're still in an analysis block at the end, discard it
        if self.found_analysis_start and not self.found_assistant_final:
            return ""

        # Return any remaining buffer content
        output = self.buffer
        self.buffer = ""
        return output


class StreamingCombinedThinkingFilter:
    """
    A combined filter that handles both think tags and analysis blocks in streaming text.
    Now supports dynamic schema configuration per model.
    """

    def __init__(
        self,
        model_name: str = None,
        thinking_start: str = None,
        thinking_stop: str = None,
        analysis_start: str = None,
        analysis_stop: str = None,
    ):
        """
        Initialize the combined filter with optional model-specific tags

        Args:
            model_name: Name of the model (for schema lookup)
            thinking_start: Custom thinking start tag
            thinking_stop: Custom thinking stop tag
            analysis_start: Custom analysis start tag
            analysis_stop: Custom analysis stop tag
        """
        self.think_filter = StreamingThinkTagFilter(
            model_name=model_name,
            thinking_start=thinking_start,
            thinking_stop=thinking_stop,
        )
        self.analysis_filter = StreamingAnalysisBlockFilter(
            model_name=model_name,
            analysis_start=analysis_start,
            analysis_stop=analysis_stop,
        )

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk through both filters.

        Args:
            chunk: The new text chunk from the stream

        Returns:
            Text that can be safely displayed
        """
        # First process through analysis filter
        intermediate = self.analysis_filter.process_chunk(chunk)

        # Then process through think tag filter
        if intermediate:
            return self.think_filter.process_chunk(intermediate)
        return ""

    def flush(self) -> str:
        """
        Flush both filters.

        Returns:
            Any remaining displayable text
        """
        analysis_output = self.analysis_filter.flush()
        if analysis_output:
            self.think_filter.buffer += analysis_output

        return self.think_filter.flush()


class StreamingToolCallFilter:
    """
    A stateful filter for extracting tool calls from streaming text and removing them from display.
    Handles cases where tool calls span multiple chunks and supports dynamic tool call formats.
    """

    def __init__(
        self,
        model_name: str = None,
        tool_start: str = None,
        tool_stop: str = None,
    ):
        """
        Initialize the filter with optional model-specific tags

        Args:
            model_name: Name of the model (for schema lookup)
            tool_start: Custom tool call start tag (overrides model schema)
            tool_stop: Custom tool call stop tag (overrides model schema)
        """
        self.buffer = ""
        self.in_tool_call = False
        self.tool_call_content = ""
        self.extracted_tool_calls = []

        # Get tool call tags from schema manager or use provided ones
        if tool_start is not None and tool_stop is not None:
            self.tool_start = tool_start
            self.tool_stop = tool_stop
        elif model_name:
            from utils.llm_schema_manager import schema_manager

            self.tool_start, self.tool_stop = schema_manager.get_tool_tags(
                model_name
            )
        else:
            # Use default tags
            from utils.config import config

            self.tool_start = config.llm.DEFAULT_TOOL_START
            self.tool_stop = config.llm.DEFAULT_TOOL_STOP

        # Cache tag lengths for efficiency and check if filtering is enabled
        self.start_tag_len = len(self.tool_start) if self.tool_start else 0
        self.stop_tag_len = len(self.tool_stop) if self.tool_stop else 0
        # Only enable filtering if BOTH start and stop tags are present and non-empty
        # If both are null/empty, it means use OpenAI format only (no content extraction)
        self.filtering_enabled = bool(
            self.tool_start
            and self.tool_stop
            and self.tool_start.strip()
            and self.tool_stop.strip()
        )

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk, extract tool calls, and return displayable text.

        Args:
            chunk: The new text chunk from the stream

        Returns:
            Text that can be safely displayed (with tool calls removed)
        """
        # If filtering is disabled (empty tags), pass through
        if not self.filtering_enabled:
            return chunk

        # Add chunk to buffer
        self.buffer += chunk

        # Process buffer to extract displayable content and tool calls
        output = ""
        i = 0

        while i < len(self.buffer):
            if self.in_tool_call:
                # Look for closing tag
                close_index = self.buffer.find(self.tool_stop, i)
                if close_index != -1:
                    # Found closing tag, extract the tool call content
                    self.tool_call_content += self.buffer[i:close_index]

                    # Debug logging
                    logger.debug(
                        "Extracted tool call content: %s",
                        repr(self.tool_call_content),
                    )

                    # Try to parse the tool call
                    self._extract_tool_call(self.tool_call_content)

                    # Reset state and skip to after the closing tag
                    i = close_index + self.stop_tag_len
                    self.in_tool_call = False
                    self.tool_call_content = ""
                else:
                    # No closing tag yet, accumulate content and wait
                    self.tool_call_content += self.buffer[i:]
                    self.buffer = ""  # Clear buffer since we processed it all
                    return output  # Return what we have so far
            else:
                # Look for opening tag
                open_index = self.buffer.find(self.tool_start, i)
                if open_index != -1:
                    # Output text before the tag
                    output += self.buffer[i:open_index]
                    i = open_index + self.start_tag_len
                    self.in_tool_call = True
                else:
                    # Check if we might have a partial tag at the end
                    partial_tag_start = max(
                        i, len(self.buffer) - self.start_tag_len
                    )
                    for j in range(partial_tag_start, len(self.buffer)):
                        # Check for partial start tag
                        remaining_buffer = self.buffer[j:]
                        if self.tool_start.startswith(remaining_buffer):
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

    def _extract_tool_call(self, tool_call_content: str):
        """
        Extract and parse a tool call from the content

        Args:
            tool_call_content: The content between tool call tags
        """
        try:
            import json

            # Clean and normalize the content
            cleaned_content = tool_call_content.strip()

            # Remove any leading/trailing whitespace and newlines more aggressively
            cleaned_content = re.sub(r"^\s+", "", cleaned_content)
            cleaned_content = re.sub(r"\s+$", "", cleaned_content)

            # Try to extract JSON from the content if it's mixed with other text
            # Look for JSON-like patterns
            json_match = re.search(r"\{.*\}", cleaned_content, re.DOTALL)
            if json_match:
                cleaned_content = json_match.group(0)

            # Try to parse as JSON
            tool_call_data = json.loads(cleaned_content)

            # Validate tool call structure
            if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                self.extracted_tool_calls.append(
                    {
                        "name": tool_call_data["name"],
                        "arguments": tool_call_data.get("arguments", {}),
                        "source": "dynamic_schema",
                    }
                )
                logger.debug("Extracted tool call: %s", tool_call_data["name"])
            elif isinstance(tool_call_data, list):
                # Handle array of tool calls
                for call in tool_call_data:
                    if isinstance(call, dict) and "name" in call:
                        self.extracted_tool_calls.append(
                            {
                                "name": call["name"],
                                "arguments": call.get("arguments", {}),
                                "source": "dynamic_schema",
                            }
                        )
                        logger.debug("Extracted tool call: %s", call["name"])

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse tool call content as JSON: %s. Content: %s",
                e,
                repr(tool_call_content[:100]),
            )
        except Exception as e:
            logger.error("Error extracting tool call: %s", e)

    def get_extracted_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get all extracted tool calls

        Returns:
            List of extracted tool call dictionaries
        """
        return self.extracted_tool_calls.copy()

    def clear_extracted_tool_calls(self):
        """Clear the list of extracted tool calls"""
        self.extracted_tool_calls.clear()

    def flush(self) -> str:
        """
        Get any remaining buffered content (called at end of stream).

        Returns:
            Any remaining displayable text
        """
        # If we're still in a tool call at the end, it wasn't closed properly
        # Log this as a warning but don't include it in output
        if self.in_tool_call:
            logger.warning("Unclosed tool call detected at end of stream")
            return ""

        # Return any remaining buffer content
        output = self.buffer
        self.buffer = ""
        return output


class StreamingCompleteFilter:
    """
    A comprehensive filter that handles thinking tags, analysis blocks, and tool calls
    with dynamic schema support for different LLM models.
    """

    def __init__(
        self,
        model_name: str = None,
        thinking_start: str = None,
        thinking_stop: str = None,
        analysis_start: str = None,
        analysis_stop: str = None,
        tool_start: str = None,
        tool_stop: str = None,
    ):
        """
        Initialize the complete filter with optional model-specific tags

        Args:
            model_name: Name of the model (for schema lookup)
            thinking_start: Custom thinking start tag
            thinking_stop: Custom thinking stop tag
            analysis_start: Custom analysis start tag
            analysis_stop: Custom analysis stop tag
            tool_start: Custom tool call start tag
            tool_stop: Custom tool call stop tag
        """
        self.think_filter = StreamingThinkTagFilter(
            model_name=model_name,
            thinking_start=thinking_start,
            thinking_stop=thinking_stop,
        )
        self.analysis_filter = StreamingAnalysisBlockFilter(
            model_name=model_name,
            analysis_start=analysis_start,
            analysis_stop=analysis_stop,
        )
        self.tool_filter = StreamingToolCallFilter(
            model_name=model_name, tool_start=tool_start, tool_stop=tool_stop
        )

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk through all filters.

        Args:
            chunk: The new text chunk from the stream

        Returns:
            Text that can be safely displayed
        """
        # First process through tool call filter to extract tool calls
        intermediate = self.tool_filter.process_chunk(chunk)

        # Then process through analysis filter
        if intermediate:
            intermediate = self.analysis_filter.process_chunk(intermediate)

        # Finally process through think tag filter
        if intermediate:
            return self.think_filter.process_chunk(intermediate)
        return ""

    def get_extracted_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all extracted tool calls from the tool filter"""
        return self.tool_filter.get_extracted_tool_calls()

    def clear_extracted_tool_calls(self):
        """Clear extracted tool calls"""
        self.tool_filter.clear_extracted_tool_calls()

    def flush(self) -> str:
        """
        Flush all filters.

        Returns:
            Any remaining displayable text
        """
        # Flush tool filter first
        tool_output = self.tool_filter.flush()
        if tool_output:
            self.analysis_filter.buffer += tool_output

        # Flush analysis filter
        analysis_output = self.analysis_filter.flush()
        if analysis_output:
            self.think_filter.buffer += analysis_output

        return self.think_filter.flush()


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
