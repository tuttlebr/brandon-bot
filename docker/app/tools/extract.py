"""
Web Extract Tool - MVC Pattern Implementation

This tool extracts content from URLs using LLM-based HTML processing,
following the Model-View-Controller pattern.
"""

import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import (
    BaseTool,
    BaseToolResponse,
    ExecutionMode,
    StreamingToolResponse,
    ToolController,
    ToolView,
)

# Configure logger
from utils.logging_config import get_logger
from utils.text_processing import StreamingThinkTagFilter
from utils.web_extractor import WebDataExtractor

logger = get_logger(__name__)


class WebExtractResponse(BaseToolResponse):
    """Response from URL extraction"""

    url: str = Field(description="The extracted URL")
    content: str = Field(description="The extracted and processed content")
    title: Optional[str] = Field(None, description="Page title if found")
    raw_content: Optional[str] = Field(
        None, description="Raw content before processing"
    )
    direct_response: bool = Field(
        default=True,
        description=(
            "Flag indicating this response should be returned directly to user"
        ),
    )


class StreamingExtractResponse(StreamingToolResponse):
    """Streaming response from URL extraction"""

    url: str = Field(description="The extracted URL")
    title: Optional[str] = Field(None, description="Page title if found")
    direct_response: bool = Field(
        default=True,
        description=(
            "Flag indicating this response should be returned directly to user"
        ),
    )


class WebExtractController(ToolController):
    """Controller handling web extraction business logic"""

    def __init__(self, llm_type: str = None):
        # llm_type is kept for backward compatibility but not used
        # The actual LLM type is determined from tool_llm_config
        pass

        # Request headers to mimic a real browser
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the web extraction request"""
        url = params["url"]
        request = params.get("request", "")

        try:
            # Extract content from URL
            extractor = WebDataExtractor(
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                        " AppleWebKit/537.36 (KHTML, like Gecko)"
                        " Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
            )

            # Extract content from URL synchronously
            result = extractor.extract_sync(url)

            if not result["success"]:
                raise RuntimeError(
                    f"Failed to extract content from {url}:"
                    f" {result.get('error', 'Unknown error')}"
                )

            content = result["content"]
            title = result.get("title", "")
            raw_content = content  # Store raw content for reference

            # If no specific request, return raw content
            if not request:
                formatted_content = (
                    f"# {title}\n\n{content}" if title else content
                )
                return {
                    "url": url,
                    "content": formatted_content,
                    "title": title,
                    "raw_content": raw_content,
                }

            # Process with LLM using streaming internally for faster latency
            processed_content = self._extract_with_llm(
                url, content, request, title
            )

            return {
                "url": url,
                "content": processed_content,
                "title": title,
                "raw_content": raw_content,
            }

        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise RuntimeError(f"Failed to process {url}: {str(e)}")

    async def process_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process the URL extraction request asynchronously with streaming"""
        url = params["url"]
        request = params.get("request", "")

        try:
            # Extract content from URL
            extractor = WebDataExtractor(
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                        " AppleWebKit/537.36 (KHTML, like Gecko)"
                        " Chrome/91.0.4472.124 Safari/537.36"
                    )
                }
            )

            # Extract content from URL with timeout
            import asyncio

            try:
                extraction_task = asyncio.create_task(extractor.extract(url))
                result = await asyncio.wait_for(extraction_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.error(f"Timeout extracting content from {url}")

                # Return error with streaming response
                async def error_generator():
                    yield (
                        f"Failed to extract content from {url}: Request timed"
                        " out after 30 seconds"
                    )

                return {
                    "url": url,
                    "content_generator": error_generator(),
                    "title": None,
                    "success": False,
                    "error_message": "Request timed out",
                    "is_streaming": True,
                    "direct_response": True,
                }

            if not result["success"]:
                # Return error with streaming response
                async def error_generator():
                    yield (
                        f"Failed to extract content from {url}:"
                        f" {result.get('error', 'Unknown error')}"
                    )

                return {
                    "url": url,
                    "content_generator": error_generator(),
                    "title": result.get("title"),
                    "success": False,
                    "error_message": result.get("error", "Unknown error"),
                    "is_streaming": True,
                    "direct_response": True,
                }

            content = result["content"]
            title = result.get("title", "")

            # If no specific request, create streaming response
            if not request:

                async def content_generator():
                    yield f"# {title}\n\n" if title else ""
                    yield content

                return {
                    "url": url,
                    "content_generator": content_generator(),
                    "title": title,
                    "success": True,
                    "is_streaming": True,
                    "direct_response": True,
                }

            # Process with LLM using streaming
            content_generator = self._extract_with_llm_streaming(
                url, content, request, title
            )

            return {
                "url": url,
                "content_generator": content_generator,
                "title": title,
                "success": True,
                "is_streaming": True,
                "direct_response": True,
            }

        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            error_msg = str(e)

            # Return error with streaming response
            async def error_generator():
                yield f"Failed to process {url}: {error_msg}"

            return {
                "url": url,
                "content_generator": error_generator(),
                "title": None,
                "success": False,
                "error_message": error_msg,
                "is_streaming": True,
                "direct_response": True,
            }

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text using regex"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls

    def _extract_with_llm(
        self, url: str, content: str, request: str, title: str = ""
    ) -> str:
        """
        Extract content using sync LLM with streaming for faster first token

        Args:
            url: The URL being processed
            content: The extracted content
            request: The user's request
            title: The page title

        Returns:
            Processed content as string
        """
        try:
            # Get sync LLM client and model using the configured LLM type
            from tools.tool_llm_config import get_tool_llm_type

            llm_type = get_tool_llm_type("extract_web_content")
            client = llm_client_service.get_client(llm_type)
            model_name = llm_client_service.get_model_name(llm_type)

            # Create system prompt for extraction
            system_prompt = (
                "You are a helpful assistant that processes web content.\n"
                "Your task is to respond to the user's request based on the "
                "provided web content.\n\n"
                f"Content URL: {url}\n"
                f"Content Title: {title}\n\n"
                "Instructions:\n"
                "- Focus on the user's specific request\n"
                "- Provide clear, well-structured responses\n"
                "- Maintain accuracy to the source content\n"
                "- Include relevant quotes when appropriate\n"
                "- If the content doesn't contain information to answer the "
                "request, say so clearly"
            )

            # Prepare messages
            user_message = f"""Based on the following web content, {request}:

{content}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(f"Processing web content with {model_name} for {url}")

            # Generate response with streaming for faster first token
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.28,
                top_p=0.95,
                frequency_penalty=0.002,
                presence_penalty=0.9,
                stream=True,  # Enable streaming for faster latency
            )

            # Create think tag filter for streaming
            think_filter = StreamingThinkTagFilter()
            collected_response = ""

            # Process stream with think tag filtering
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        collected_response += filtered_content

            # Get any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                collected_response += final_content

            return collected_response.strip()

        except Exception as e:
            logger.error(f"Error extracting content with LLM: {e}")
            raise

    async def _extract_with_llm_streaming(
        self, url: str, content: str, request: str, title: str = ""
    ) -> AsyncGenerator[str, None]:
        """
        Stream extraction using LLM to transform/summarize/answer questions

        Args:
            url: The URL being processed
            content: The extracted content
            request: The user's request
            title: The page title

        Yields:
            Processed content chunks
        """
        try:
            # Get async LLM client and model using the configured LLM type
            from tools.tool_llm_config import get_tool_llm_type

            llm_type = get_tool_llm_type("extract_web_content")
            client = llm_client_service.get_async_client(llm_type)
            model_name = llm_client_service.get_model_name(llm_type)

            # Create system prompt for extraction
            system_prompt = (
                "You are a helpful assistant that processes web content.\n"
                "Your task is to respond to the user's request based on the "
                "provided web content.\n\n"
                f"Content URL: {url}\n"
                f"Content Title: {title}\n\n"
                "Instructions:\n"
                "- Focus on the user's specific request\n"
                "- Provide clear, well-structured responses\n"
                "- Maintain accuracy to the source content\n"
                "- Include relevant quotes when appropriate\n"
                "- If the content doesn't contain information to answer the "
                "request, say so clearly"
            )

            # Prepare messages
            user_message = f"""Based on the following web content, {request}:

{content}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(
                f"Streaming web content transformation with {model_name} for"
                f" {url}"
            )

            # Generate response with streaming
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.28,
                top_p=0.95,
                frequency_penalty=0.002,
                presence_penalty=0.9,
                stream=True,  # Enable streaming
            )

            # Create think tag filter for streaming
            think_filter = StreamingThinkTagFilter()

            # Process stream with think tag filtering and yield chunks
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        yield filtered_content

            # Yield any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                yield final_content

        except Exception as e:
            logger.error(f"Error in streaming extraction: {e}")
            yield f"Error processing content: {str(e)}"


class WebExtractView(ToolView):
    """View for formatting web extraction responses"""

    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format raw data into WebExtractResponse"""
        try:
            # Check if this is a streaming response
            if data.get("is_streaming") and data.get("content_generator"):
                return StreamingExtractResponse(**data)
            else:
                return WebExtractResponse(**data)
        except Exception as e:
            logger.error(f"Error formatting extraction response: {e}")
            return WebExtractResponse(
                url=data.get("url", ""),
                content="",
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
            )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into WebExtractResponse"""
        error_code = "UNKNOWN_ERROR"
        error_message = str(error)

        if isinstance(error, ValueError):
            if "Invalid URL" in str(error):
                error_code = "INVALID_URL"
            elif "only extract content from URLs" in str(error):
                error_code = "UNAUTHORIZED_URL"
            else:
                error_code = "CONTENT_ERROR"
        elif isinstance(error, TimeoutError):
            error_code = "TIMEOUT_ERROR"
        elif isinstance(error, ConnectionError):
            if "HTTP error" in str(error):
                error_code = "HTTP_ERROR"
            else:
                error_code = "NETWORK_ERROR"

        return WebExtractResponse(
            url="",
            content="",
            success=False,
            error_message=error_message,
            error_code=error_code,
            response_time=0.0,
        )


class WebExtractTool(BaseTool):
    """Tool for extracting content from web URLs using
    LLM-based HTML processing"""

    def __init__(self):
        super().__init__()
        self.name = "extract_web_content"
        self.description = (
            "Extract and read content from a specific URL. Use when user "
            "provides a URL AND asks to read or analyze it."
        )
        self.execution_mode = (
            ExecutionMode.AUTO
        )  # Changed to AUTO to support both sync and async
        self.timeout = 256.0

    def _initialize_mvc(self):
        """Initialize MVC components"""
        self._controller = WebExtractController()
        self._view = WebExtractView()

    def get_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": (
                                "The web URL to extract content from. Must be"
                                " a valid HTTP or HTTPS URL."
                            ),
                        },
                        "request": {
                            "type": "string",
                            "description": (
                                "Optional specific request about what to"
                                " extract or how to process the content (e.g.,"
                                " 'summarize the main points', 'extract"
                                " pricing information'). If empty, returns the"
                                " full extracted content."
                            ),
                        },
                        "but_why": {
                            "type": "integer",
                            "description": (
                                "An integer from 1-5 where a larger number"
                                " indicates confidence this is the right tool"
                                " to help the user."
                            ),
                        },
                    },
                    "required": ["url", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get the response type for this tool"""
        return WebExtractResponse


# Helper functions for backward compatibility
def get_web_extract_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for web extract tool"""
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("extract_web_content", WebExtractTool)

    # Get the tool instance and return its definition
    tool = get_tool("extract_web_content")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get web extract tool definition")


# Internal helper function used by other tools
def execute_web_extract_batch(
    urls: List[str], messages: Optional[List[Dict[str, Any]]] = None
) -> List[WebExtractResponse]:
    """
    Execute batch web content extraction

    Args:
        urls: List of URLs to extract content from
        messages: Optional conversation messages to verify URL sources

    Returns:
        List of WebExtractResponse objects, one for each URL
    """
    from tools.registry import execute_tool

    results = []
    for url in urls:
        try:
            result = execute_tool(
                "extract_web_content",
                {"url": url, "messages": messages, "but_why": 5},
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to extract {url}: {e}")
            results.append(
                WebExtractResponse(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"Extraction failed: {str(e)}",
                    error_code="EXTRACTION_ERROR",
                    response_time=0.0,
                )
            )

    return results
