"""
Web Extract Tool - MVC Pattern Implementation

This tool extracts content from URLs using LLM-based HTML processing,
following the Model-View-Controller pattern.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Type
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import (
    BaseTool,
    BaseToolResponse,
    ExecutionMode,
    ToolController,
    ToolView,
)
from utils.text_processing import strip_think_tags

# Configure logger
logger = logging.getLogger(__name__)


class ExtractResult(BaseToolResponse):
    """Result from URL extraction"""

    url: str = Field(description="The URL that was extracted")
    content: str = Field(description="Extracted content in markdown format")
    raw_content: Optional[str] = Field(
        None, description="Raw HTML content if available"
    )
    response_time: float = Field(description="Response time for the extraction")


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
        url = params['url']
        messages = params.get('messages', [])

        # Validate URL format
        if not self._validate_url(url):
            raise ValueError(
                f"Invalid URL format: '{url}'. Please provide a valid HTTP or HTTPS URL."
            )

        # Check if URL was provided by user (if messages are available)
        if messages and not self._check_user_provided_url(url, messages):
            raise ValueError(
                "I can only extract content from URLs that you directly provide in your message."
            )

        try:
            # Fetch HTML content
            logger.debug(f"Fetching HTML content from: '{url}'")
            html_content, response_time = self._fetch_html_content(url)

            # Extract content using LLM
            logger.debug(f"Extracting content using LLM for: '{url}'")
            extracted_content = self._extract_with_llm(html_content, url)

            return {
                "url": url,
                "content": extracted_content,
                "raw_content": html_content,
                "response_time": response_time,
            }

        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request timeout: The webpage at '{url}' took too long to respond."
            )
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(
                f"HTTP error {e.response.status_code}: Unable to access '{url}'."
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error: Unable to connect to '{url}'.")
        except ValueError:
            # Re-raise ValueError for content type errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error during web extraction: {e}")
            raise RuntimeError(f"An unexpected error occurred: {str(e)}")

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
                url, headers=self.headers, timeout=30, allow_redirects=True
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

    def _extract_with_llm(self, html_content: str, url: str) -> str:
        """Extract content from HTML using LLM"""
        try:
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            logger.debug(
                f"Using LLM type '{self.llm_type}' for web extraction (configured in tool_llm_config.py)"
            )

            if len(html_content) > 200000:
                html_content = (
                    html_content[:200000] + "\n[Content truncated due to length]"
                )
                logger.info(
                    f"Truncated HTML content to 300k characters for LLM processing"
                )

            system_prompt = """You are an expert HTML to markdown converter. Your task is to extract the main content from web pages and convert it to clean, readable markdown format.

Instructions:
1. Extract ONLY the main article/content from the webpage
2. Ignore navigation menus, sidebars, advertisements, headers, footers, and other peripheral content
3. Convert the content to clean markdown format
4. Preserve the structure and hierarchy of the content using appropriate markdown headers
5. Include any relevant images, links, and formatting from the main content
6. Do not add any commentary, explanations, or meta-information about the extraction process
7. Never mention that you extracted content or reference the source URL in the output
8. If the page contains mostly navigation or non-content elements, extract what meaningful content you can find
9. Remove any JavaScript, CSS, or other non-content elements
10. The content should be returned verbatim, not summarized

Output only the extracted markdown content with no additional commentary."""

            user_message = f"Extract the main content from this webpage and convert it to markdown:\n\nURL: {url}\n\nHTML Content:\n{html_content}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.debug(f"Processing HTML content with LLM model: {model_name}")

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                stream=False,
            )

            extracted_content = response.choices[0].message.content.strip()

            # Strip think tags immediately after LLM response
            extracted_content = strip_think_tags(extracted_content)

            # Clean up any remaining HTML artifacts and thinking tags
            extracted_content = self._clean_extracted_content(extracted_content)

            return extracted_content

        except Exception as e:
            logger.error(f"Error extracting content with LLM: {e}")
            raise

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
        """Format raw data into ExtractResult"""
        try:
            return ExtractResult(**data)
        except Exception as e:
            logger.error(f"Error formatting extract response: {e}")
            return ExtractResult(
                url=data.get("url", ""),
                content="",
                success=False,
                error_message=f"Response formatting error: {str(e)}",
                error_code="FORMAT_ERROR",
                response_time=0.0,
            )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error into ExtractResult"""
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

        return ExtractResult(
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
        self.description = "ONLY use when explicitly asked to read, extract, or analyze content from a specific web URL that the user provides. Extracts and reads content from web URLs provided by the user, converting web pages into clean, readable markdown format using LLM-based processing. DO NOT use for general questions, information lookup, or when no specific URL is provided."
        self.execution_mode = ExecutionMode.SYNC
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
                        "but_why": {
                            "type": "string",
                            "description": "A single sentence explaining why this tool was selected for the query.",
                        },
                    },
                    "required": ["url", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get the response type for this tool"""
        return ExtractResult


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
) -> List[ExtractResult]:
    """
    Execute batch web content extraction

    Args:
        urls: List of URLs to extract content from
        messages: Optional conversation messages to verify URL sources

    Returns:
        List of ExtractResult objects, one for each URL
    """
    from tools.registry import execute_tool

    results = []
    for url in urls:
        try:
            result = execute_tool(
                "extract_web_content",
                {
                    "url": url,
                    "messages": messages,
                    "but_why": "Extracting web content as part of search result enrichment",
                },
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to extract {url}: {e}")
            results.append(
                ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"Extraction failed: {str(e)}",
                    error_code="EXTRACTION_ERROR",
                    response_time=0.0,
                )
            )

    return results
