"""
Web Extract Tool - MVC Pattern Implementation

This tool extracts content from URLs using LLM-based HTML processing,
following the Model-View-Controller pattern.
"""

import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
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
from utils.text_processing import StreamingThinkTagFilter, strip_think_tags
from utils.web_extractor import WebDataExtractor

# Configure logger
logger = logging.getLogger(__name__)


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
        description="Flag indicating this response should be returned directly to user",
    )


class StreamingExtractResponse(StreamingToolResponse):
    """Streaming response from URL extraction"""

    url: str = Field(description="The extracted URL")
    title: Optional[str] = Field(None, description="Page title if found")
    direct_response: bool = Field(
        default=True,
        description="Flag indicating this response should be returned directly to user",
    )


class WebExtractController(ToolController):
    """Controller handling web extraction business logic"""

    def __init__(self, llm_type: str):
        self.llm_type = llm_type

        # Request headers to mimic a real browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
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
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )

            # Extract content from URL synchronously
            result = extractor.extract_sync(url)

            if not result["success"]:
                raise RuntimeError(
                    f"Failed to extract content from {url}: {result.get('error', 'Unknown error')}"
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
        stream = params.get("stream", True)  # Default to streaming

        try:
            # Extract content from URL
            extractor = WebDataExtractor(
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
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
                    yield f"Failed to extract content from {url}: Request timed out after 30 seconds"

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
                    yield f"Failed to extract content from {url}: {result.get('error', 'Unknown error')}"

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
            raw_content = content  # Store raw content for reference

            # If no specific request, create streaming response with raw content
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

            # Return error with streaming response
            async def error_generator():
                yield f"Failed to process {url}: {str(e)}"

            return {
                "url": url,
                "content_generator": error_generator(),
                "title": None,
                "success": False,
                "error_message": str(e),
                "is_streaming": True,
                "direct_response": True,
            }

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text using regex"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls

    def _validate_url(self, url: str) -> bool:
        """Validate that the URL is properly formatted"""
        try:
            if not url.startswith(("http://", "https://")):
                return False

            parsed = urlparse(url)
            if not parsed.netloc:
                return False

            return True
        except Exception:
            return False

    def _check_user_provided_url(
        self, url: str, messages: List[Dict[str, Any]]
    ) -> bool:
        """Check if the URL was provided by the user"""
        if not messages:
            return True

        for message in reversed(messages):
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                if url in content:
                    logger.info(f"Found URL '{url}' in user message")
                    return True

                urls_in_message = self._extract_urls_from_text(content)
                if url in urls_in_message:
                    logger.info(f"Found URL '{url}' in user message content")
                    return True

        logger.warning(f"URL '{url}' was not found in any user message")
        return False

    def _fetch_html_content(self, url: str) -> tuple[str, float]:
        """Fetch HTML content from URL"""
        import time

        start_time = time.time()

        try:
            response = requests.get(
                url, headers=self.headers, timeout=5, allow_redirects=True
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if (
                "text/html" not in content_type
                and "application/xhtml" not in content_type
            ):
                raise ValueError(
                    f"URL does not return HTML content. Content-Type: {content_type}"
                )

            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "noscript", "iframe"]):
                script.decompose()
            body = str(soup.find("body"))

            response_time = time.time() - start_time

            return body, response_time

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            raise

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
            # Get sync LLM client and model
            from models.chat_config import ChatConfig
            from openai import OpenAI

            config_obj = ChatConfig.from_environment()

            # Get appropriate client based on tool's llm_type
            if self.llm_type == "fast":
                client = OpenAI(
                    api_key=config_obj.fast_api_key,
                    base_url=config_obj.fast_endpoint,
                )
                model_name = config_obj.fast_model_name
            else:  # default
                client = OpenAI(
                    api_key=config_obj.api_key, base_url=config_obj.endpoint
                )
                model_name = config_obj.model_name

            # Create system prompt for extraction
            system_prompt = f"""You are a helpful assistant that processes web content.
Your task is to respond to the user's request based on the provided web content.

Content URL: {url}
Content Title: {title}

Instructions:
- Focus on the user's specific request
- Provide clear, well-structured responses
- Maintain accuracy to the source content
- Include relevant quotes when appropriate
- If the content doesn't contain information to answer the request, say so clearly"""

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
                temperature=0.3,
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

    async def _extract_with_llm_async(
        self, html_content: str, url: str
    ) -> str:
        """Extract content from HTML using LLM asynchronously"""
        try:
            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            logger.debug(
                f"Using LLM type '{self.llm_type}' for async web extraction (configured in tool_llm_config.py)"
            )

            if len(html_content) > 200000:
                html_content = (
                    html_content[:200000]
                    + "\n[Content truncated due to length]"
                )
                logger.info(
                    f"Truncated HTML content to 200k characters for LLM processing"
                )

            # Get system prompt from centralized configuration
            from tools.tool_llm_config import get_tool_system_prompt

            system_prompt = get_tool_system_prompt("extract_web_content", "")

            user_message = f"Extract the main content from this webpage and convert it to markdown:\n\nURL: {url}\n\nHTML Content:\n{html_content}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.info(
                f"Extracting content from URL using {model_name} (async non-streaming)"
            )

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=65536,
            )

            result = response.choices[0].message.content.strip()

            # Clean and validate the extracted content
            cleaned_result = self._clean_extracted_content(result)

            logger.info(
                f"Successfully extracted {len(cleaned_result)} characters from URL (async)"
            )

            return cleaned_result

        except Exception as e:
            logger.error(f"Error extracting content with async LLM: {e}")
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
            # Get async LLM client and model
            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Create system prompt for extraction
            system_prompt = f"""You are a helpful assistant that processes web content.
Your task is to respond to the user's request based on the provided web content.

Content URL: {url}
Content Title: {title}

Instructions:
- Focus on the user's specific request
- Provide clear, well-structured responses
- Maintain accuracy to the source content
- Include relevant quotes when appropriate
- If the content doesn't contain information to answer the request, say so clearly"""

            # Prepare messages
            user_message = f"""Based on the following web content, {request}:

{content}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(
                f"Streaming web content transformation with {model_name} for {url}"
            )

            # Generate response with streaming
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
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

    async def _extract_with_llm_streaming_collected(
        self, url: str, content: str, request: str, title: str = ""
    ) -> str:
        """
        Process extraction using LLM with streaming but collect the full response.
        This method is kept for compatibility.
        """
        collected_result = ""
        async for chunk in self._extract_with_llm_streaming(
            url, content, request, title
        ):
            collected_result += chunk
        return collected_result

    def _clean_extracted_content(self, content: str) -> str:
        """Clean up extracted content for better readability"""
        if not content:
            return ""

        # Remove excessive newlines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Remove any remaining HTML tags that might have slipped through
        content = re.sub(r"<[^>]+>", "", content)

        # Remove excessive whitespace
        content = re.sub(r" {2,}", " ", content)
        content = content.strip()

        # Strip think tags from extracted content
        content = strip_think_tags(content)

        # Add a header with the extraction notice
        header = "## Web Content Extract\n\n"

        return header + content


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
    """Tool for extracting content from web URLs using LLM-based HTML processing"""

    def __init__(self):
        super().__init__()
        self.name = "extract_web_content"
        self.description = "Extract and read content from a specific URL. Use when user provides a URL AND asks to read or analyze it."
        self.execution_mode = (
            ExecutionMode.AUTO
        )  # Changed to AUTO to support both sync and async
        self.timeout = 60.0

    def _initialize_mvc(self):
        """Initialize MVC components"""
        self._controller = WebExtractController(self.llm_type)
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
                            "description": "The web URL to extract content from. Must be a valid HTTP or HTTPS URL.",
                        },
                        "request": {
                            "type": "string",
                            "description": "Optional specific request about what to extract or how to process the content (e.g., 'summarize the main points', 'extract pricing information'). If empty, returns the full extracted content.",
                        },
                        "but_why": {
                            "type": "integer",
                            "description": "An integer from 1-5 where a larger number indicates confidence this is the right tool to help the user.",
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
