import logging
import re
from typing import Any, Dict, Generator, List

from models.chat_config import ChatConfig
from openai import OpenAI


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
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.6,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )

            # Filter out <think> tags from streaming response
            full_response = ""
            thinking_mode = True
            buffer = ""

            for chunk in stream:
                # Skip empty chunks
                if not hasattr(chunk.choices[0].delta, "content") or chunk.choices[0].delta.content is None:
                    yield ""
                    continue

                # Get content from chunk
                content = chunk.choices[0].delta.content
                full_response += content
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
                    elif thinking_mode and "</think>" in buffer:
                        # Found closing tag, skip content in think tags
                        tag_pos = buffer.find("</think>")
                        buffer = buffer[tag_pos + 8 :]  # +8 to skip "</think>"
                        thinking_mode = False
                    else:
                        # Only one tag found but not its pair, wait for more content
                        break

                # If not in thinking mode, yield remaining buffer
                if not thinking_mode and buffer:
                    output += buffer
                    buffer = ""

                yield output.replace("$", "\\$").replace("\\${", "${")

            # Return the complete response with think tags filtered
            final_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)
            return final_response

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            raise Exception(error_msg)
