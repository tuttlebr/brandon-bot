import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List

from models.chat_config import ChatConfig
from openai import OpenAI
from tools import (
    execute_news_with_dict,
    execute_retrieval_with_dict,
    execute_tavily_with_dict,
    execute_weather_with_dict,
    get_news_tool_definition,
    get_retrieval_tool_definition,
    get_tavily_tool_definition,
    get_weather_tool_definition,
)
from utils.system_prompt import TOOL_PROMPT

tavily_tool_def = get_tavily_tool_definition()
weather_tool_def = get_weather_tool_definition()
retrieval_tool_def = get_retrieval_tool_definition()
news_tool_def = get_news_tool_definition()
MAX_TURNS = 9
ALL_TOOLS = [tavily_tool_def, weather_tool_def, retrieval_tool_def, news_tool_def]
tools = {
    "tavily_internet_search": execute_tavily_with_dict,
    "get_weather": execute_weather_with_dict,
    "retrieval_search": execute_retrieval_with_dict,
    "tavily_news_search": execute_news_with_dict,
}


class LLMService:
    """Service for handling Large Language Model interactions"""

    def __init__(self, config: ChatConfig):
        """
        Initialize the LLM service

        Args:
            config: Configuration for the LLM service
        """
        self.config = config
        self.client = self._initialize_client()
        self.last_tool_responses = []  # Store tool responses for context extraction

    def _initialize_client(self) -> OpenAI:
        """Initialize the OpenAI client"""
        try:
            return OpenAI(api_key=self.config.api_key, base_url=self.config.llm_endpoint)
        except Exception as e:
            logging.error(f"Failed to initialize LLM client: {e}")
            raise

    def _parse_custom_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse custom tool call format with flexible pattern matching.
        Handles variations like:
        - <TOOLCALL-[...]</TOOLCALL>
        - <TOOLCALL[...]</TOOLCALL>
        - <TOOLCALL"[...]</TOOLCALL>
        - <toolcall-[...]</toolcall>

        Args:
            content: The response content that may contain custom tool calls

        Returns:
            List of parsed tool call dictionaries, empty list if none found
        """
        if not content:
            return []

        # Multiple flexible regex patterns to handle different variations
        patterns = [
            # Standard format: <TOOLCALL-[...]</TOOLCALL>
            r'<TOOLCALL-\[(.*?)\]</TOOLCALL>',
            # Missing dash: <TOOLCALL[...]</TOOLCALL>
            r'<TOOLCALL\[(.*?)\]</TOOLCALL>',
            # Quote instead of dash: <TOOLCALL"[...]</TOOLCALL>
            r'<TOOLCALL"\[(.*?)\]</TOOLCALL>',
            # With spaces: <TOOLCALL - [...]</TOOLCALL>
            r'<TOOLCALL\s*-\s*\[(.*?)\]</TOOLCALL>',
            # Case insensitive: <toolcall-[...]</toolcall>
            r'<toolcall-\[(.*?)\]</toolcall>',
            # Any separator: <TOOLCALL[any char][...]</TOOLCALL>
            r'<TOOLCALL.?\[(.*?)\]</TOOLCALL>',
            # Just looking for the JSON array pattern within angle brackets
            r'<[^>]*?(\[\{.*?"name".*?\}\])[^<]*?>',
        ]

        parsed_calls = []

        for pattern in patterns:
            try:
                # Try case-insensitive matching for the more flexible patterns
                if 'toolcall' in pattern.lower():
                    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                else:
                    matches = re.findall(pattern, content, re.DOTALL)

                if matches:
                    logging.debug(f"Found tool calls using pattern: {pattern}")
                    for match in matches:
                        parsed_calls.extend(self._parse_tool_call_json(match))

                    # If we found matches, we can stop trying other patterns
                    if parsed_calls:
                        break

            except Exception as e:
                logging.debug(f"Pattern {pattern} failed: {e}")
                continue

        # If no patterns worked, try to find JSON arrays anywhere in the content
        if not parsed_calls:
            parsed_calls = self._extract_json_arrays_from_content(content)

        return parsed_calls

    def _parse_tool_call_json(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Parse a JSON string that should contain tool call data

        Args:
            json_str: JSON string to parse

        Returns:
            List of parsed tool call dictionaries
        """
        parsed_calls = []

        try:
            # Clean up the JSON string
            json_str = json_str.strip()

            # If it doesn't start with '[', wrap it in an array
            if not json_str.startswith('['):
                json_str = f'[{json_str}]'

            # Parse the JSON content
            tool_calls_json = json.loads(json_str)

            # Ensure it's a list
            if not isinstance(tool_calls_json, list):
                tool_calls_json = [tool_calls_json]

            for tool_call in tool_calls_json:
                if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                    parsed_calls.append(tool_call)
                    logging.debug(f"Parsed custom tool call: {tool_call['name']}")
                else:
                    logging.warning(f"Invalid tool call format: {tool_call}")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse tool call JSON: {json_str}, error: {e}")
        except Exception as e:
            logging.error(f"Error parsing tool call: {e}")

        return parsed_calls

    def _extract_json_arrays_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Last resort: try to find JSON arrays with tool call structure anywhere in the content

        Args:
            content: The full content to search

        Returns:
            List of parsed tool call dictionaries
        """
        parsed_calls = []

        try:
            # Look for JSON arrays that contain objects with "name" and "arguments" keys
            # This is a more aggressive pattern that looks for the structure we want
            pattern = r'\[\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}\s*\]'
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                try:
                    tool_calls_json = json.loads(match)
                    for tool_call in tool_calls_json:
                        if isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
                            parsed_calls.append(tool_call)
                            logging.debug(f"Extracted tool call from content: {tool_call['name']}")
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logging.debug(f"Failed to extract JSON arrays from content: {e}")

        return parsed_calls

    def _normalize_tool_calls(self, openai_tool_calls=None, custom_tool_calls=None) -> List[Dict[str, Any]]:
        """
        Normalize both standard OpenAI tool calls and custom tool calls to a unified format

        Args:
            openai_tool_calls: List of OpenAI tool call objects (optional)
            custom_tool_calls: List of custom tool call dictionaries (optional)

        Returns:
            List of normalized tool call dictionaries
        """
        normalized_calls = []

        # Process standard OpenAI tool calls
        if openai_tool_calls:
            for tool_call in openai_tool_calls:
                try:
                    normalized_calls.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                            "source": "openai_standard",
                        }
                    )
                except Exception as e:
                    logging.error(f"Error normalizing OpenAI tool call: {e}")
                    normalized_calls.append(
                        {
                            "name": "error",
                            "arguments": {},
                            "source": "openai_standard",
                            "error": f"Failed to parse tool call: {str(e)}",
                        }
                    )

        # Process custom tool calls
        if custom_tool_calls:
            for tool_call in custom_tool_calls:
                normalized_calls.append(
                    {
                        "name": tool_call.get("name"),
                        "arguments": tool_call.get("arguments", {}),
                        "source": "custom_format",
                    }
                )

        return normalized_calls

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tool calls asynchronously using unified format

        Args:
            tool_calls: List of normalized tool call dictionaries

        Returns:
            List of tool response messages
        """

        async def execute_single_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
            """Execute a single tool call"""
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            source = tool_call.get("source", "unknown")

            # Handle pre-existing errors from normalization
            if "error" in tool_call:
                logging.error(f"Tool call error from {source}: {tool_call['error']}")
                return {"role": "tool", "content": f"Error: {tool_call['error']}"}

            if tool_name not in tools:
                logging.error(f"Unknown tool: {tool_name} (from {source})")
                return {"role": "tool", "content": f"Error: Unknown tool '{tool_name}'"}

            try:
                logging.debug(f"Executing tool call: {tool_name} with args: {tool_args} (from {source})")
                tool_function = tools[tool_name]

                # Run the tool function in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                tool_response = await loop.run_in_executor(None, lambda: tool_function(tool_args).json())

                logging.debug(f"Tool {tool_name} executed successfully (from {source})")
                return {"role": "tool", "content": tool_response}
            except Exception as e:
                logging.error(f"Error executing tool {tool_name} (from {source}): {e}")
                return {"role": "tool", "content": f"Error executing {tool_name}: {str(e)}"}

        # Execute all tool calls concurrently
        if not tool_calls:
            return []

        logging.info(f"Executing {len(tool_calls)} tool calls concurrently")
        tool_responses = await asyncio.gather(*[execute_single_tool(tool_call) for tool_call in tool_calls])
        return tool_responses

    def _apply_sliding_window(self, messages: List[Dict[str, Any]], max_turns: int = 3) -> List[Dict[str, Any]]:
        """
        Apply sliding window to limit conversation history and prevent bias in tool decisions

        Args:
            messages: List of all messages
            max_turns: Maximum number of conversation turns to keep (default: 3)

        Returns:
            Filtered messages with sliding window applied
        """
        if not messages:
            return messages

        # Always keep the system message
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        conversation_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Count conversation turns (user-assistant pairs)
        if len(conversation_messages) <= max_turns * 2:  # Each turn = user + assistant
            # Not enough messages to need windowing
            windowed_messages = system_messages + conversation_messages
        else:
            # Keep only the last max_turns conversation turns
            keep_count = max_turns * 2
            recent_conversation = conversation_messages[-keep_count:]
            windowed_messages = system_messages + recent_conversation

        logging.debug(
            f"Applied sliding window: {len(messages)} -> {len(windowed_messages)} messages (keeping {max_turns} turns)"
        )
        return windowed_messages

    def _contains_custom_tool_calls(self, content: str) -> bool:
        """
        Check if content contains custom tool call instructions

        Args:
            content: The response content to check

        Returns:
            True if content contains custom tool calls, False otherwise
        """
        if not content:
            return False

        # Check for various custom tool call patterns
        patterns = [
            r'<TOOLCALL-\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\[.*?\]</TOOLCALL>',
            r'<TOOLCALL"\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\s*-\s*\[.*?\]</TOOLCALL>',
            r'<toolcall-\[.*?\]</toolcall>',
            r'<TOOLCALL.?\[.*?\]</TOOLCALL>',
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                logging.debug(f"Found custom tool call pattern: {pattern}")
                return True

        return False

    def _remove_tool_call_instructions(self, content: str) -> str:
        """
        Remove all custom tool call instructions from content

        Args:
            content: The response content to clean

        Returns:
            Content with tool call instructions removed
        """
        if not content:
            return content

        # Remove various custom tool call patterns
        patterns = [
            r'<TOOLCALL-\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\[.*?\]</TOOLCALL>',
            r'<TOOLCALL"\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\s*-\s*\[.*?\]</TOOLCALL>',
            r'<toolcall-\[.*?\]</toolcall>',
            r'<TOOLCALL.?\[.*?\]</TOOLCALL>',
        ]

        cleaned_content = content
        for pattern in patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)

        return cleaned_content.strip()

    def _validate_and_clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and clean messages, removing any with empty content or tool call instructions

        Args:
            messages: List of message dictionaries

        Returns:
            Cleaned list of messages with non-empty content and no tool call instructions
        """
        cleaned_messages = []

        for msg in messages:
            content = msg.get("content", "")

            # Handle different content types
            if isinstance(content, str):
                # Check if content contains custom tool call instructions
                if self._contains_custom_tool_calls(content):
                    logging.warning(
                        f"Removing message with custom tool call instructions: {msg.get('role', 'unknown')}"
                    )
                    continue

                # String content - check if it's empty after stripping
                if content.strip():
                    cleaned_messages.append(msg)
                else:
                    logging.warning(f"Removing message with empty string content: {msg.get('role', 'unknown')}")
            elif isinstance(content, dict):
                # Dict content (like image messages) - keep if it has meaningful data
                if content:
                    cleaned_messages.append(msg)
                else:
                    logging.warning(f"Removing message with empty dict content: {msg.get('role', 'unknown')}")
            elif isinstance(content, list):
                # List content - keep if it has items
                if content:
                    cleaned_messages.append(msg)
                else:
                    logging.warning(f"Removing message with empty list content: {msg.get('role', 'unknown')}")
            else:
                # Other content types - be conservative and keep if truthy
                if content:
                    cleaned_messages.append(msg)
                else:
                    logging.warning(
                        f"Removing message with empty content: {msg.get('role', 'unknown')}, type: {type(content)}"
                    )

        if len(cleaned_messages) != len(messages):
            logging.debug(
                f"Cleaned messages: {len(messages)} -> {len(cleaned_messages)} (removed {len(messages) - len(cleaned_messages)} messages)"
            )

        return cleaned_messages

    async def generate_streaming_response(
        self, messages: List[Dict[str, Any]], model: str
    ) -> AsyncGenerator[str, str]:
        """
        Generate streaming response from LLM

        Args:
            messages: List of messages for the conversation
            model: Model name to use

        Yields:
            Filtered content chunks without <think> tags
        """
        try:
            # First, validate and clean all messages to remove any with empty content
            messages = self._validate_and_clean_messages(messages)

            # Filter out any tool messages from previous turns to prevent bias
            # Only keep system and user/assistant messages for clean conversation history
            clean_messages = []
            for msg in messages:
                if msg.get("role") in ["system", "user", "assistant"]:
                    clean_messages.append(msg)
                elif msg.get("role") == "tool":
                    logging.debug(f"Filtered out tool message from LLM input: {str(msg.get('content', ''))[:100]}...")

            # Validate clean messages again after filtering
            clean_messages = self._validate_and_clean_messages(clean_messages)

            # PHASE 1: Tool Decision Making - Use minimal context (system + current user message only)
            system_message = next((msg for msg in clean_messages if msg.get("role") == "system"), None)
            current_user_message = next((msg for msg in reversed(clean_messages) if msg.get("role") == "user"), None)

            if not current_user_message:
                logging.error("No user message found for tool decision")
                async for chunk in self._generate_simple_response("I didn't receive a message from you."):
                    yield chunk
                return

            # Create minimal context for tool decision
            tool_decision_messages = []
            if system_message:
                tool_decision_messages.append({"role": "system", "content": TOOL_PROMPT})
            tool_decision_messages.append(current_user_message)

            # Final validation of tool decision messages
            tool_decision_messages = self._validate_and_clean_messages(tool_decision_messages)

            logging.info(f"Tool decision context: {len(tool_decision_messages)} messages (system + current user)")

            # Make tool decision with minimal context
            tool_decision_params = {
                "model": model,
                "messages": tool_decision_messages,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 512,
                "tools": ALL_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }
            logging.debug(f"Tool decision params: {tool_decision_params}")
            initial_response = self.client.chat.completions.create(**tool_decision_params)

            # PHASE 2: Tool Detection and Execution (unified approach)
            tool_responses = []

            # Check for both standard OpenAI tool calls and custom tool call format
            openai_tool_calls = initial_response.choices[0].message.tool_calls
            response_content = initial_response.choices[0].message.content
            custom_tool_calls = self._parse_custom_tool_calls(response_content) if response_content else []

            # Normalize all tool calls to unified format
            all_tool_calls = self._normalize_tool_calls(
                openai_tool_calls=openai_tool_calls, custom_tool_calls=custom_tool_calls
            )

            if all_tool_calls:
                tool_count = len(all_tool_calls)
                openai_count = len(openai_tool_calls) if openai_tool_calls else 0
                custom_count = len(custom_tool_calls)

                logging.info(
                    f"Found {tool_count} total tool calls: {openai_count} standard OpenAI, {custom_count} custom format"
                )

                # Execute all tool calls concurrently using unified approach
                tool_responses = await self._execute_tool_calls(all_tool_calls)

                # Store tool responses for context extraction by streamlit app
                self.last_tool_responses = tool_responses
                logging.debug(f"Stored {len(tool_responses)} tool responses for context extraction")

                # Add to original messages for context extraction
                messages.extend(tool_responses)
            else:
                logging.info("No tool calls detected in response")
                # Clear previous tool responses if no tools were used
                self.last_tool_responses = []

            # PHASE 3: Response Generation with Full Context
            if tool_responses:
                logging.debug(
                    f"Generating response with tool results ({len(tool_responses)} tool responses from {len(all_tool_calls)} tool calls)"
                )

                # Apply sliding window to conversation history for response generation
                windowed_messages = self._apply_sliding_window(clean_messages, max_turns=MAX_TURNS)

                # Create full context: windowed conversation + tool results
                response_messages = windowed_messages + tool_responses

                # Final validation before API call
                response_messages = self._validate_and_clean_messages(response_messages)

                # Generate final response with full context
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=response_messages,
                    stream=True,
                    temperature=0.6,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                async for chunk in self._process_streaming_response(stream):
                    yield chunk
                return
            else:
                logging.info("No tools needed, generating response with conversation context")

                # Apply sliding window for response generation
                windowed_messages = self._apply_sliding_window(clean_messages, max_turns=MAX_TURNS)

                # Final validation before API call
                windowed_messages = self._validate_and_clean_messages(windowed_messages)

                # Generate response with conversation context (no tools)
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=windowed_messages,
                    stream=True,
                    temperature=0.6,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                async for chunk in self._process_streaming_response(stream):
                    yield chunk
                return

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

    async def _generate_simple_response(self, message: str) -> AsyncGenerator[str, str]:
        """
        Generate a simple response without streaming for error cases

        Args:
            message: The message to return

        Yields:
            The message
        """
        yield message

    async def _process_streaming_response(self, stream) -> AsyncGenerator[str, str]:
        """
        Process streaming response and filter think tags and tool call instructions

        Args:
            stream: OpenAI streaming response

        Yields:
            Filtered content chunks without think tags or tool call instructions
        """
        # Filter out <think> tags and tool call instructions from streaming response
        logging.debug("Starting to process streaming response...")
        full_response = ""
        thinking_mode = True  # Start in thinking mode
        buffer = ""
        has_yielded_content = False

        for chunk in stream:
            # Skip empty chunks
            if not hasattr(chunk.choices[0].delta, "content") or chunk.choices[0].delta.content is None:
                continue

            # Get content from chunk
            content = chunk.choices[0].delta.content
            full_response += content
            logging.debug(f"Received chunk: '{content}'")

            # Check if chunk contains tool call instructions - if so, skip it entirely
            if self._contains_custom_tool_calls(content):
                logging.warning("Skipping chunk containing tool call instructions")
                continue

            # Simple approach: if no think tags are present, yield immediately
            if "<think>" not in content and "</think>" not in content and not thinking_mode:
                logging.debug(f"No think tags, yielding: '{content}'")
                yield content.replace("$", "\\$").replace("\\${", "${")
                has_yielded_content = True
                continue

            # Complex think tag processing for content with think tags
            buffer += content
            output = ""

            # Check for think tags in accumulated buffer
            while "<think>" in buffer or "</think>" in buffer:
                if not thinking_mode and "<think>" in buffer:
                    # Found opening tag, yield content before tag
                    tag_pos = buffer.find("<think>")
                    output += buffer[:tag_pos]
                    buffer = buffer[tag_pos + 7 :]  # +7 to skip "<think>"
                    thinking_mode = True
                    logging.debug("Entered thinking mode")
                elif thinking_mode and "</think>" in buffer:
                    # Found closing tag, skip content in think tags
                    tag_pos = buffer.find("</think>")
                    buffer = buffer[tag_pos + 8 :]  # +8 to skip "</think>"
                    thinking_mode = False
                    logging.debug("Exited thinking mode")
                else:
                    # Only one tag found but not its pair, wait for more content
                    break

            # If not in thinking mode, yield remaining buffer
            if not thinking_mode and buffer:
                output += buffer
                buffer = ""

            # Before yielding, ensure no tool call instructions remain
            if output:
                output = self._remove_tool_call_instructions(output)
                if output:  # Only yield if there's content after removing tool calls
                    logging.debug(f"Yielding processed output: '{output}'")
                    yield output.replace("$", "\\$").replace("\\${", "${")
                    has_yielded_content = True

        logging.debug("Finished processing streaming response")

        # Final fallback: if no content was yielded and we have a response, process it
        if not has_yielded_content and full_response:
            # Remove think tags and tool call instructions
            final_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)
            final_response = self._remove_tool_call_instructions(final_response).strip()

            if final_response:
                logging.debug("Yielding final processed response after cleaning")
                yield final_response.replace("$", "\\$").replace("\\${", "${")
            else:
                logging.warning("Response was entirely think tags or tool calls, providing fallback")
                yield "I understand. Is there anything else I can help you with?"
        elif not has_yielded_content:
            # No response at all - provide fallback
            logging.warning("No response content received, providing fallback")
            yield "I understand. Is there anything else I can help you with?"

    async def _process_non_streaming_response(self, response) -> AsyncGenerator[str, str]:
        """
        Process non-streaming response and filter think tags and tool call instructions

        Args:
            response: OpenAI ChatCompletion object

        Yields:
            Filtered content chunks (for compatibility with streaming interface)
        """
        logging.debug("Starting to process non-streaming response...")

        # Extract content from ChatCompletion object
        response_content = response.choices[0].message.content

        if not response_content:
            logging.warning("No content found in response, providing fallback")
            yield "I understand. Is there anything else I can help you with?"
            return

        # Check if response contains only tool call instructions
        if self._contains_custom_tool_calls(response_content):
            # Remove tool call instructions
            cleaned_content = self._remove_tool_call_instructions(response_content)
            if not cleaned_content.strip():
                logging.warning("Response contained only tool call instructions, providing fallback")
                yield "I understand. Is there anything else I can help you with?"
                return
            response_content = cleaned_content

        # Remove any remaining custom tool call tags from the response
        cleaned_content = self._remove_tool_call_instructions(response_content)

        # Check if response contains think tags
        if "<think>" not in cleaned_content and "</think>" not in cleaned_content:
            logging.debug("No think tags found, returning response with escaped dollar signs")
            # Just escape dollar signs for markdown and yield
            final_response = cleaned_content.replace("$", "\\$").replace("\\${", "${").strip()
            if final_response:
                yield final_response
            else:
                logging.warning("Response was empty after cleaning, providing fallback")
                yield "I understand. Is there anything else I can help you with?"
            return

        # Filter out think tags using regex
        filtered_response = re.sub(r"<think>.*?</think>", "", cleaned_content, flags=re.DOTALL).strip()

        # Escape dollar signs for markdown
        final_response = filtered_response.replace("$", "\\$").replace("\\${", "${")

        logging.debug(
            f"Finished processing non-streaming response. " f"Filtered length: {len(final_response)} characters"
        )

        # Ensure we don't yield empty responses
        if final_response:
            logging.debug("Think tags and tool calls successfully removed from response")
            yield final_response
        else:
            logging.warning("Response was entirely within think tags or tool calls, providing fallback")
            yield "I understand. Is there anything else I can help you with?"

    def _filter_think_tags(self, content: str) -> str:
        """
        Filter out <think> tags from content

        Args:
            content: Raw content with potential think tags

        Returns:
            Filtered content without think tags
        """
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
