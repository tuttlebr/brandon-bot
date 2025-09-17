"""
Simple Tool Call Filter

A simplified, more robust implementation of tool call filtering
that handles streaming content correctly.
"""

import json
import re
from typing import Any, Dict, List

from utils.logging_config import get_logger

logger = get_logger(__name__)


class SimpleStreamingToolFilter:
    """
    A simplified streaming filter for extracting tool calls
    """

    def __init__(
        self,
        model_name: str = None,
        tool_start: str = None,
        tool_stop: str = None,
    ):
        """
        Initialize the filter

        Args:
            model_name: Name of the model (for schema lookup)
            tool_start: Tool call start tag
            tool_stop: Tool call stop tag
        """
        self.accumulated_content = ""
        self.extracted_tool_calls = []

        # Get tool call tags
        if tool_start is not None and tool_stop is not None:
            self.tool_start = tool_start
            self.tool_stop = tool_stop
        elif model_name:
            from utils.llm_schema_manager import schema_manager

            self.tool_start, self.tool_stop = schema_manager.get_tool_tags(
                model_name
            )
        else:
            from utils.config import config

            self.tool_start = config.llm.DEFAULT_TOOL_START
            self.tool_stop = config.llm.DEFAULT_TOOL_STOP

        # Only enable filtering if BOTH tags are present and non-empty
        # Empty strings or None mean no filtering (use OpenAI format)
        self.filtering_enabled = bool(
            self.tool_start
            and self.tool_stop
            and self.tool_start.strip()
            and self.tool_stop.strip()
        )

    def process_chunk(self, chunk: str) -> str:
        """
        Process a chunk and return displayable content

        Args:
            chunk: The new chunk from the stream

        Returns:
            Content safe to display (with tool calls removed)
        """
        if not self.filtering_enabled:
            return chunk

        # Accumulate all content
        self.accumulated_content += chunk

        # Extract and remove complete tool calls
        while True:
            start_idx = self.accumulated_content.find(self.tool_start)
            if start_idx == -1:
                break

            end_idx = self.accumulated_content.find(self.tool_stop, start_idx)
            if end_idx == -1:
                break  # Incomplete tool call, wait for more content

            # Extract tool call content
            tool_content_start = start_idx + len(self.tool_start)
            tool_content = self.accumulated_content[tool_content_start:end_idx]

            # Parse and store the tool call
            self._extract_tool_call(tool_content)

            # Remove the tool call from accumulated content
            self.accumulated_content = (
                self.accumulated_content[:start_idx]
                + self.accumulated_content[end_idx + len(self.tool_stop) :]
            )

        # Return the current accumulated content (this is what should be displayed)
        # In a real streaming scenario, you'd return only new displayable content
        # For now, return everything since we're processing in chunks
        return self.accumulated_content

    def _extract_tool_call(self, tool_call_content: str):
        """
        Extract and parse a tool call

        Args:
            tool_call_content: Content between tool call tags
        """
        try:
            # Clean the content
            cleaned_content = tool_call_content.strip()

            # Try to extract JSON if mixed with other content
            json_match = re.search(r"\{.*\}", cleaned_content, re.DOTALL)
            if json_match:
                cleaned_content = json_match.group(0)

            # Parse JSON
            tool_call_data = json.loads(cleaned_content)

            if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                self.extracted_tool_calls.append(
                    {
                        "name": tool_call_data["name"],
                        "arguments": tool_call_data.get("arguments", {}),
                        "source": "dynamic_schema",
                    }
                )
                logger.debug("Extracted tool call: %s", tool_call_data["name"])

        except Exception as e:
            logger.warning(
                "Failed to extract tool call: %s. Content: %s",
                e,
                repr(tool_call_content[:100]),
            )

    def get_extracted_tool_calls(self) -> List[Dict[str, Any]]:
        """Get all extracted tool calls"""
        return self.extracted_tool_calls.copy()

    def clear_extracted_tool_calls(self):
        """Clear extracted tool calls"""
        self.extracted_tool_calls.clear()

    def get_display_content(self) -> str:
        """Get the current content safe for display"""
        return self.accumulated_content

    def flush(self) -> str:
        """Flush and return final content"""
        return self.accumulated_content
