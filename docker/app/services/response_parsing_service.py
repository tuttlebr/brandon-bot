"""
Response Parsing Service

This service handles parsing of LLM responses, including extraction
of tool calls from both standard OpenAI format and custom formats.
"""

import json
import re
from typing import Any, Dict, List, Tuple

from utils.exceptions import LLMServiceError
from utils.logging_config import get_logger
from utils.text_processing import strip_all_thinking_formats

logger = get_logger(__name__)


class ResponseParsingService:
    """Service for parsing LLM responses and extracting tool calls"""

    def __init__(self, model_name: str = None):
        """
        Initialize the response parsing service

        Args:
            model_name: Name of the model for schema lookup
        """
        self.model_name = model_name

    def parse_response(
        self, response: Any, model_name: str = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse LLM response to extract content and tool calls

        Args:
            response: LLM response object
            model_name: Optional model name for schema lookup
                       (overrides instance model_name)

        Returns:
            Tuple of (content, tool_calls)
        """
        try:
            # Use provided model name or fall back to instance model name
            effective_model_name = model_name or self.model_name

            message = response.choices[0].message
            content = message.content or ""

            # Extract standard OpenAI tool calls
            openai_tool_calls = self._extract_openai_tool_calls(message)

            # Extract custom tool calls from content using dynamic schema
            custom_tool_calls = self._extract_custom_tool_calls(
                content, effective_model_name
            )

            # Extract tool calls from dynamic schema tags
            schema_tool_calls = self._extract_schema_tool_calls(
                content, effective_model_name
            )

            # Fallback: If no structured tool calls were found but we have
            # content, try additional content-based extraction methods for
            # models that may embed tool calls in content instead of using
            # structured format
            if (
                not openai_tool_calls
                and not custom_tool_calls
                and not schema_tool_calls
                and content
            ):
                logger.debug(
                    "No structured tool calls found, attempting "
                    "content-based fallback extraction"
                )
                fallback_tool_calls = self._extract_fallback_tool_calls(
                    content, effective_model_name
                )
                if fallback_tool_calls:
                    logger.info(
                        "Found %d tool calls using fallback content "
                        "extraction",
                        len(fallback_tool_calls),
                    )
                    schema_tool_calls.extend(fallback_tool_calls)

            # Normalize all tool calls
            all_tool_calls = self._normalize_tool_calls(
                openai_tool_calls, custom_tool_calls, schema_tool_calls
            )

            # Clean content if any tool calls were found
            if custom_tool_calls or schema_tool_calls:
                content = self._clean_tool_instructions(
                    content, effective_model_name
                )

            # Strip all thinking/reasoning formats using dynamic schema
            content = self._strip_thinking_formats(
                content, effective_model_name
            )

            if len(all_tool_calls) == 0:
                all_tool_calls = None

            return content, all_tool_calls

        except Exception as e:
            logger.error("Error parsing response: %s", e)
            raise LLMServiceError(f"Failed to parse LLM response: {e}") from e

    def _extract_openai_tool_calls(self, message: Any) -> List[Dict[str, Any]]:
        """Extract standard OpenAI format tool calls"""
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []

        tool_calls = []
        for tool_call in message.tool_calls:
            try:
                # Log the raw arguments for debugging
                raw_args = tool_call.function.arguments
                logger.debug(
                    "Parsing tool call '%s' with arguments: %s",
                    tool_call.function.name,
                    repr(raw_args),
                )

                # Handle empty or invalid arguments gracefully
                if not raw_args or not raw_args.strip():
                    logger.warning(
                        "Empty or whitespace-only arguments for tool call "
                        "'%s', using empty dict",
                        tool_call.function.name,
                    )
                    parsed_args = {}
                else:
                    # Parse the arguments
                    logger.info(f"Raw arguments: {raw_args}")
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        logger.error(
                            "JSON parsing error for tool call '%s': %s. "
                            "Raw arguments: %s. Using empty dict as "
                            "fallback.",
                            tool_call.function.name,
                            e,
                            repr(raw_args),
                        )
                        # Fallback to empty dict instead of failing
                        parsed_args = {}

                tool_calls.append(
                    {
                        "name": tool_call.function.name,
                        "arguments": parsed_args,
                        "id": getattr(tool_call, "id", None),
                    }
                )
            except Exception as e:
                logger.error(
                    "Error parsing OpenAI tool call '%s': %s. "
                    "Skipping this tool call.",
                    tool_call.function.name,
                    e,
                )

        return tool_calls

    def _extract_custom_tool_calls(
        self, content: str, model_name: str = None
    ) -> List[Dict[str, Any]]:
        """Extract custom format tool calls from content
        (legacy TOOLCALL format)"""
        if not content:
            return []

        # Pattern for custom tool calls: <TOOLCALL-[...]</TOOLCALL>
        pattern = r"<TOOLCALL[^>]*?\[(.*?)\]</TOOLCALL>"
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
                            and "name" in item
                            and "arguments" in item
                        ):
                            tool_calls.append(item)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse custom tool call: %s", e)

        return tool_calls

    def _extract_schema_tool_calls(
        self, content: str, model_name: str = None
    ) -> List[Dict[str, Any]]:
        """Extract tool calls using dynamic schema tags"""
        if not content or not model_name:
            return []

        try:
            from utils.llm_schema_manager import schema_manager

            tool_start, tool_stop = schema_manager.get_tool_tags(model_name)

            # If both tool tags are empty/null, don't extract from content
            # This means the model uses standard OpenAI tool_calls format only
            if not tool_start or not tool_stop:
                logger.debug(
                    "Model '%s' has null tool tags - using OpenAI format only",
                    model_name,
                )
                return []

            # Create pattern for the dynamic tool call tags
            # Escape special regex characters in the tags
            escaped_start = re.escape(tool_start)
            escaped_stop = re.escape(tool_stop)
            pattern = f"{escaped_start}(.*?){escaped_stop}"

            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

            tool_calls = []
            for match in matches:
                try:
                    # Try to parse as JSON
                    tool_call_data = json.loads(match.strip())

                    # Handle single tool call (dict)
                    if (
                        isinstance(tool_call_data, dict)
                        and "name" in tool_call_data
                    ):
                        tool_calls.append(
                            {
                                "name": tool_call_data["name"],
                                "arguments": tool_call_data.get(
                                    "arguments", {}
                                ),
                                "source": "dynamic_schema",
                            }
                        )
                    # Handle multiple tool calls (list)
                    elif isinstance(tool_call_data, list):
                        for call in tool_call_data:
                            if isinstance(call, dict) and "name" in call:
                                tool_calls.append(
                                    {
                                        "name": call["name"],
                                        "arguments": call.get("arguments", {}),
                                        "source": "dynamic_schema",
                                    }
                                )

                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse schema tool call as JSON: %s", e
                    )
                except Exception as e:
                    logger.error("Error parsing schema tool call: %s", e)

            if tool_calls:
                logger.info(
                    "Extracted %d tool calls using dynamic schema for"
                    " model %s",
                    len(tool_calls),
                    model_name,
                )

            return tool_calls

        except Exception as e:
            logger.error("Error extracting schema tool calls: %s", e)
            return []

    def _extract_fallback_tool_calls(
        self, content: str, model_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback method to extract tool calls from content using various
        patterns. This is used when structured tool calls are empty but
        content might contain tool calls in different formats.
        """
        if not content:
            return []

        tool_calls = []

        # Get valid tool names for validation
        valid_tool_names = self._get_valid_tool_names()

        # Pattern 1: Look for JSON-like function calls in content
        # Matches patterns like: function_name({"arg": "value"})
        json_function_pattern = r"(\w+)\s*\(\s*(\{[^}]*\})\s*\)"
        json_matches = re.findall(json_function_pattern, content, re.DOTALL)

        for func_name, args_str in json_matches:
            # Validate tool name before processing
            if func_name not in valid_tool_names:
                logger.debug(
                    f"Ignoring invalid tool name '{func_name}' from "
                    "fallback extraction"
                )
                continue
            try:
                args = json.loads(args_str)
                tool_calls.append(
                    {
                        "name": func_name,
                        "arguments": args,
                        "source": "fallback_json_function",
                    }
                )
                logger.debug(
                    f"Extracted fallback tool call: {func_name} "
                    f"with args: {args}"
                )
            except json.JSONDecodeError:
                continue

        # Pattern 2: Look for explicit function call declarations
        # Matches patterns like: "I'll use the search_web function with
        # query: 'example'"
        function_intent_pattern = (
            r"(?:I'll use|using|call|invoke)\s+(?:the\s+)?(\w+)\s+"
            r"(?:function|tool)(?:\s+with\s+(\w+):\s*['\"]([^'\"]+)"
            r"['\"])?|(\w+)\(['\"]([^'\"]*)['\"](?:,\s*['\"]([^'\"]*)"
            r"['\"])*\)"
        )
        intent_matches = re.findall(
            function_intent_pattern, content, re.IGNORECASE
        )

        for match in intent_matches:
            func_name = match[0] or match[3]
            if func_name:
                # Validate tool name before processing
                if func_name not in valid_tool_names:
                    logger.debug(
                        f"Ignoring invalid tool name '{func_name}' from "
                        "fallback intent extraction"
                    )
                    continue
                # Try to extract arguments from the context
                args = {}
                if match[1] and match[2]:  # parameter name and value
                    args[match[1]] = match[2]
                elif match[4]:  # positional argument found
                    args["query"] = match[4]  # Common parameter name

                tool_calls.append(
                    {
                        "name": func_name,
                        "arguments": args,
                        "source": "fallback_intent",
                    }
                )
                logger.debug(
                    f"Extracted fallback intent tool call: {func_name} "
                    f"with args: {args}"
                )

        # Pattern 3: Look for markdown-style code blocks with function calls
        # Matches patterns like: ```function_name\n{"arg": "value"}\n```
        code_block_pattern = r"```(\w+)\s*\n(\{.*?\})\s*\n```"
        code_matches = re.findall(code_block_pattern, content, re.DOTALL)

        for func_name, args_str in code_matches:
            # Validate tool name before processing
            if func_name not in valid_tool_names:
                logger.debug(
                    f"Ignoring invalid tool name '{func_name}' from "
                    "fallback code block extraction"
                )
                continue
            try:
                args = json.loads(args_str)
                tool_calls.append(
                    {
                        "name": func_name,
                        "arguments": args,
                        "source": "fallback_code_block",
                    }
                )
                logger.debug(
                    "Extracted fallback code block tool call: "
                    f"{func_name} with args: {args}"
                )
            except json.JSONDecodeError:
                continue

        if tool_calls:
            logger.info(
                "Fallback extraction found %d potential tool calls in content",
                len(tool_calls),
            )

        return tool_calls

    def _get_valid_tool_names(self) -> set:
        """Get set of valid tool names from the registry for validation"""
        try:
            from tools.registry import ToolRegistry

            registry = ToolRegistry.get_instance()
            # Get both registered instances and lazy-loaded tool names
            all_tool_names = set(registry._tools.keys()) | set(
                registry._factory.get_registered_tools()
            )
            return all_tool_names
        except Exception as e:
            logger.error(f"Error getting valid tool names: {e}")
            # Return empty set to be safe - will reject all fallback
            # extractions
            return set()

    def _normalize_tool_calls(
        self,
        openai_calls: List[Dict[str, Any]],
        custom_calls: List[Dict[str, Any]],
        schema_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize tool calls to a consistent format"""
        normalized = []

        # Add OpenAI calls
        for call in openai_calls:
            normalized.append(
                {
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "source": "openai",
                    "id": call.get("id"),
                }
            )

        # Add custom calls (legacy format)
        for call in custom_calls:
            normalized.append(
                {
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                    "source": "custom",
                    "id": None,
                }
            )

        # Add schema calls (dynamic format)
        for call in schema_calls:
            normalized.append(
                {
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                    "source": call.get("source", "dynamic_schema"),
                    "id": None,
                }
            )

        return normalized

    def _clean_tool_instructions(
        self, content: str, model_name: str = None
    ) -> str:
        """Remove tool call instructions from content using dynamic schema"""
        if not content:
            return content

        cleaned = content

        # Remove legacy tool call patterns first
        legacy_patterns = [
            r"<TOOLCALL[^>]*?\[.*?\]</TOOLCALL>",
            r"<toolcall[^>]*?\[.*?\]</toolcall>",
        ]

        for pattern in legacy_patterns:
            cleaned = re.sub(
                pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE
            )

        # Remove dynamic schema tool call patterns if model is specified
        if model_name:
            try:
                from utils.llm_schema_manager import schema_manager

                tool_start, tool_stop = schema_manager.get_tool_tags(
                    model_name
                )

                # Only clean if both tags are present
                if tool_start and tool_stop:
                    # Create pattern for the dynamic tool call tags
                    escaped_start = re.escape(tool_start)
                    escaped_stop = re.escape(tool_stop)
                    pattern = f"{escaped_start}.*?{escaped_stop}"

                    cleaned = re.sub(
                        pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE
                    )

            except Exception as e:
                logger.error("Error cleaning dynamic schema tool calls: %s", e)

        return cleaned.strip()

    def _strip_thinking_formats(
        self, content: str, model_name: str = None
    ) -> str:
        """Strip thinking formats using dynamic schema"""
        if not content:
            return content

        # If no model name provided, use the default stripping function
        if not model_name:
            return strip_all_thinking_formats(content)

        try:
            from utils.llm_schema_manager import schema_manager

            # Get schema for the model
            schema = schema_manager.get_schema(model_name)

            # Strip thinking tags - handle asymmetric cases
            if schema.thinking_start and schema.thinking_stop:
                # Standard case: both start and stop tags
                escaped_start = re.escape(schema.thinking_start)
                escaped_stop = re.escape(schema.thinking_stop)
                pattern = f"{escaped_start}.*?{escaped_stop}"
                content = re.sub(
                    pattern, "", content, flags=re.DOTALL | re.IGNORECASE
                )
            elif schema.thinking_stop and not schema.thinking_start:
                # Only stop tag - remove everything before and including
                # the stop tag
                stop_index = content.find(schema.thinking_stop)
                if stop_index != -1:
                    # Keep only content after the stop tag
                    content = content[stop_index + len(schema.thinking_stop) :]
                    logger.info(
                        "Applied stop-only filtering - removed %d chars before"
                        " stop tag",
                        stop_index,
                    )
            elif schema.thinking_start and not schema.thinking_stop:
                # Only start tag - remove everything after the start tag
                escaped_start = re.escape(schema.thinking_start)
                pattern = f"{escaped_start}.*$"
                content = re.sub(
                    pattern, "", content, flags=re.DOTALL | re.IGNORECASE
                )

            # Strip analysis blocks if configured
            if schema.analysis_start and schema.analysis_stop:
                escaped_start = re.escape(schema.analysis_start)
                escaped_stop = re.escape(schema.analysis_stop)
                pattern = f"{escaped_start}.*?{escaped_stop}"
                content = re.sub(
                    pattern, "", content, flags=re.DOTALL | re.IGNORECASE
                )

            return content.strip()

        except Exception as e:
            logger.error(
                "Error stripping thinking formats with dynamic schema: %s", e
            )
            # Fall back to default stripping
            return strip_all_thinking_formats(content)
