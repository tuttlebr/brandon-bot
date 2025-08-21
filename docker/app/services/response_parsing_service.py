"""
Response Parsing Service

This service handles parsing of LLM responses, including extraction
of tool calls from both standard OpenAI format and custom formats.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from utils.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class ResponseParsingService:
    """Service for parsing LLM responses and extracting tool calls"""

    def parse_response(self, response: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse LLM response to extract content and tool calls

        Args:
            response: LLM response object

        Returns:
            Tuple of (content, tool_calls)
        """
        try:
            message = response.choices[0].message
            content = message.content or ""

            # Extract standard OpenAI tool calls
            openai_tool_calls = self._extract_openai_tool_calls(message)

            # Extract custom tool calls from content
            custom_tool_calls = self._extract_custom_tool_calls(content)

            # Normalize all tool calls
            all_tool_calls = self._normalize_tool_calls(
                openai_tool_calls, custom_tool_calls
            )

            # Clean content if custom tool calls were found
            if custom_tool_calls:
                content = self._clean_tool_instructions(content)

            if len(all_tool_calls) == 0:
                all_tool_calls = None

            return content, all_tool_calls

        except Exception as e:
            logger.error("Error parsing response: %s", e)
            raise LLMServiceError(f"Failed to parse LLM response: {e}")

    def _extract_openai_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        """Extract standard OpenAI format tool calls"""
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return []

        tool_calls = []
        for tool_call in message.tool_calls:
            try:
                tool_calls.append(
                    {
                        'name': tool_call.function.name,
                        'arguments': json.loads(tool_call.function.arguments),
                        'id': getattr(tool_call, 'id', None),
                    }
                )
            except Exception as e:
                logger.error("Error parsing OpenAI tool call: %s", e)

        return tool_calls

    def _extract_custom_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract custom format tool calls from content"""
        if not content:
            return []

        # Pattern for custom tool calls: <TOOLCALL-[...]</TOOLCALL>
        pattern = r'<TOOLCALL[^>]*?\[(.*?)\]</TOOLCALL>'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

        tool_calls = []
        for match in matches:
            try:
                # Parse JSON array
                parsed = json.loads(
                    f"[{match}]" if not match.startswith("[") else match
                )
                if isinstance(parsed, list):
                    for item in parsed:
                        if (
                            isinstance(item, dict)
                            and 'name' in item
                            and 'arguments' in item
                        ):
                            tool_calls.append(item)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse custom tool call: %s", e)

        return tool_calls

    def _normalize_tool_calls(
        self, openai_calls: List[Dict[str, Any]], custom_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize tool calls to a consistent format"""
        normalized = []

        # Add OpenAI calls
        for call in openai_calls:
            normalized.append(
                {
                    'name': call['name'],
                    'arguments': call['arguments'],
                    'source': 'openai',
                    'id': call.get('id'),
                }
            )

        # Add custom calls
        for call in custom_calls:
            normalized.append(
                {
                    'name': call['name'],
                    'arguments': call.get('arguments', {}),
                    'source': 'custom',
                    'id': None,
                }
            )

        return normalized

    def _clean_tool_instructions(self, content: str) -> str:
        """Remove tool call instructions from content"""
        if not content:
            return content

        # Remove tool call patterns
        patterns = [
            r'<TOOLCALL[^>]*?\[.*?\]</TOOLCALL>',
            r'<toolcall[^>]*?\[.*?\]</toolcall>',
        ]

        cleaned = content
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        return cleaned.strip()
