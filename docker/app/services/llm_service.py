import json
import logging
import re
from typing import Any, Dict, Generator, List

from models.chat_config import ChatConfig
from openai import OpenAI
from tools import (
    execute_tavily_with_dict,
    execute_weather_with_dict,
    get_tavily_tool_definition,
    get_weather_tool_definition,
)

tavily_tool_def = get_tavily_tool_definition()
weather_tool_def = get_weather_tool_definition()

tools = {
    "tavily_internet_search": execute_tavily_with_dict,
    "get_weather": execute_weather_with_dict,
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
            # Create streaming completion
            initial_response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=0.6,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                tools=[tavily_tool_def, weather_tool_def],
                tool_choice="auto",
            )

            if initial_response.choices[0].finish_reason == "tool_calls":
                for tool_call in initial_response.choices[0].message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    tool_function = tools[tool_call.function.name]
                    tool_response = tool_function(args).json()
                    messages.append({"role": "tool", "content": tool_response})

                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=0.6,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return self._process_streaming_response(stream)

            else:
                return self._process_non_streaming_response(initial_response)

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)

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

        # Check if response contains think tags
        if "<think>" not in response_content and "</think>" not in response_content:
            logging.debug("No think tags found, returning response with escaped dollar signs")
            # Just escape dollar signs for markdown and return
            final_response = response_content.replace("$", "\\$").replace("\\${", "${")
            yield final_response
            return final_response

        # Filter out think tags using regex
        filtered_response = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL)

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
