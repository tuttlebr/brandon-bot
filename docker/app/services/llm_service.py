import json
import logging
import re
from typing import Any, Dict, Generator, List

from models.chat_config import ChatConfig
from openai import OpenAI
from tools import (
    execute_retrieval_with_dict,
    execute_tavily_with_dict,
    execute_weather_with_dict,
    get_retrieval_tool_definition,
    get_tavily_tool_definition,
    get_weather_tool_definition,
)

tavily_tool_def = get_tavily_tool_definition()
weather_tool_def = get_weather_tool_definition()
retrieval_tool_def = get_retrieval_tool_definition()

tools = {
    "tavily_internet_search": execute_tavily_with_dict,
    "get_weather": execute_weather_with_dict,
    "retrieval_search": execute_retrieval_with_dict,
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
                    logging.info(f"Found tool calls using pattern: {pattern}")
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
                    logging.info(f"Parsed custom tool call: {tool_call['name']}")
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
                            logging.info(f"Extracted tool call from content: {tool_call['name']}")
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logging.debug(f"Failed to extract JSON arrays from content: {e}")

        return parsed_calls

    def _execute_custom_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute custom tool calls and return tool responses

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            List of tool response messages
        """
        tool_responses = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            if tool_name not in tools:
                logging.error(f"Unknown tool: {tool_name}")
                tool_responses.append({"role": "tool", "content": f"Error: Unknown tool '{tool_name}'"})
                continue

            try:
                logging.info(f"Executing custom tool call: {tool_name} with args: {tool_args}")
                tool_function = tools[tool_name]
                tool_response = tool_function(tool_args).json()
                tool_responses.append({"role": "tool", "content": tool_response})
                logging.info(f"Tool {tool_name} executed successfully")
            except Exception as e:
                logging.error(f"Error executing tool {tool_name}: {e}")
                tool_responses.append(
                    {"role": "tool", "content": f"Error executing {tool_name}: {str(e)}",}
                )

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

        logging.info(
            f"Applied sliding window: {len(messages)} -> {len(windowed_messages)} messages (keeping {max_turns} turns)"
        )
        return windowed_messages

    def generate_streaming_response(self, messages: List[Dict[str, Any]], model: str) -> Generator[str, None, str]:
        """
        Generate streaming response from LLM

        Args:
            messages: List of messages for the conversation
            model: Model name to use

        Yields:
            Filtered content chunks without <think> tags

        Returns:
            Final complete response text
        """
        try:
            # Filter out any tool messages from previous turns to prevent bias
            # Only keep system and user/assistant messages for clean conversation history
            clean_messages = []
            for msg in messages:
                if msg.get("role") in ["system", "user", "assistant"]:
                    clean_messages.append(msg)
                elif msg.get("role") == "tool":
                    logging.info(f"Filtered out tool message from LLM input: {str(msg.get('content', ''))[:100]}...")

            # PHASE 1: Tool Decision Making - Use minimal context (system + current user message only)
            system_message = next((msg for msg in clean_messages if msg.get("role") == "system"), None)
            current_user_message = next((msg for msg in reversed(clean_messages) if msg.get("role") == "user"), None)

            if not current_user_message:
                logging.error("No user message found for tool decision")
                return self._generate_simple_response("I didn't receive a message from you.")

            # Create minimal context for tool decision
            tool_decision_messages = []
            if system_message:
                tool_decision_messages.append(system_message)
            tool_decision_messages.append(current_user_message)

            logging.info(f"Tool decision context: {len(tool_decision_messages)} messages (system + current user)")

            # Make tool decision with minimal context
            tool_decision_params = {
                "model": model,
                "messages": tool_decision_messages,
                "stream": False,
                "temperature": 0.6,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "tools": [tavily_tool_def, weather_tool_def, retrieval_tool_def],
                "tool_choice": "auto",
            }

            initial_response = self.client.chat.completions.create(**tool_decision_params)

            # PHASE 2: Tool Execution (if needed)
            tool_responses = []

            # Check for standard OpenAI tool calls
            if initial_response.choices[0].message.tool_calls:
                logging.info("Standard tool calls found, executing tools")
                for tool_call in initial_response.choices[0].message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    tool_function = tools[tool_call.function.name]
                    tool_response = tool_function(args).json()
                    tool_responses.append({"role": "tool", "content": tool_response})
                    # Also add to original messages for context extraction
                    messages.append({"role": "tool", "content": tool_response})
            else:
                # Check for custom tool call format
                response_content = initial_response.choices[0].message.content
                custom_tool_calls = self._parse_custom_tool_calls(response_content)

                if custom_tool_calls:
                    logging.info(f"Custom tool calls found: {len(custom_tool_calls)}")
                    tool_responses = self._execute_custom_tool_calls(custom_tool_calls)
                    # Also add to original messages for context extraction
                    messages.extend(tool_responses)

            # PHASE 3: Response Generation with Full Context
            if tool_responses:
                logging.info(f"Generating response with tool results ({len(tool_responses)} tools used)")

                # Apply sliding window to conversation history for response generation
                windowed_messages = self._apply_sliding_window(clean_messages, max_turns=3)

                # Create full context: windowed conversation + tool results
                response_messages = windowed_messages + tool_responses

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
                return self._process_streaming_response(stream)
            else:
                logging.info("No tools needed, generating response with conversation context")

                # Apply sliding window for response generation
                windowed_messages = self._apply_sliding_window(clean_messages, max_turns=3)

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
                return self._process_streaming_response(stream)

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def _generate_simple_response(self, message: str) -> Generator[str, None, str]:
        """
        Generate a simple response without streaming for error cases

        Args:
            message: The message to return

        Yields:
            The message

        Returns:
            The message
        """
        yield message
        return message

    def _process_streaming_response(self, stream) -> Generator[str, None, str]:
        """
        Process streaming response and filter think tags

        Args:
            stream: OpenAI streaming response

        Yields:
            Filtered content chunks

        Returns:
            Complete filtered response
        """
        # Filter out <think> tags from streaming response
        logging.info("Starting to process streaming response...")
        full_response = ""
        thinking_mode = True  # Start in thinking mode
        buffer = ""

        for chunk in stream:
            # Skip empty chunks
            if not hasattr(chunk.choices[0].delta, "content") or chunk.choices[0].delta.content is None:
                continue

            # Get content from chunk
            content = chunk.choices[0].delta.content
            full_response += content
            logging.debug(f"Received chunk: '{content}'")

            # Simple approach: if no think tags are present, yield immediately
            if "<think>" not in content and "</think>" not in content and not thinking_mode:
                logging.debug(f"No think tags, yielding: '{content}'")
                yield content.replace("$", "\\$").replace("\\${", "${")
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

            # Yield the processed output (escape dollar signs for markdown)
            if output:
                logging.debug(f"Yielding processed output: '{output}'")
                yield output.replace("$", "\\$").replace("\\${", "${")

        logging.info("Finished processing streaming response")

        # Return the complete response with think tags filtered
        final_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)
        return final_response

    def _process_non_streaming_response(self, response) -> Generator[str, None, str]:
        """
        Process non-streaming response and filter think tags

        Args:
            response: OpenAI ChatCompletion object

        Yields:
            Filtered content chunks (for compatibility with streaming interface)

        Returns:
            Complete filtered response with think tags removed
        """
        logging.info("Starting to process non-streaming response...")

        # Extract content from ChatCompletion object
        response_content = response.choices[0].message.content

        if not response_content:
            logging.warning("No content found in response")
            return ""

        # Remove custom tool call tags from the response
        cleaned_content = re.sub(r"<TOOLCALL-\[.*?\]</TOOLCALL>", "", response_content, flags=re.DOTALL)

        # Check if response contains think tags
        if "<think>" not in cleaned_content and "</think>" not in cleaned_content:
            logging.debug("No think tags found, returning response with escaped dollar signs")
            # Just escape dollar signs for markdown and return
            final_response = cleaned_content.replace("$", "\\$").replace("\\${", "${")
            yield final_response
            return final_response

        # Filter out think tags using regex
        filtered_response = re.sub(r"<think>.*?</think>", "", cleaned_content, flags=re.DOTALL)

        # Escape dollar signs for markdown
        final_response = filtered_response.replace("$", "\\$").replace("\\${", "${")

        logging.info(
            f"Finished processing non-streaming response. " f"Filtered length: {len(final_response)} characters"
        )
        logging.debug("Think tags successfully removed from response")

        # Yield the complete filtered response
        yield final_response
        return final_response

    def _filter_think_tags(self, content: str) -> str:
        """
        Filter out <think> tags from content

        Args:
            content: Raw content with potential think tags

        Returns:
            Filtered content without think tags
        """
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
