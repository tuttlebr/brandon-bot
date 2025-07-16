"""
Web Extract Tool

This tool extracts content from URLs using LLM-based HTML processing.
It fetches HTML content directly and uses an LLM to convert it to clean markdown.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pydantic import Field
from services.llm_client_service import llm_client_service
from tools.base import BaseTool, BaseToolResponse
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
    success: bool = Field(default=True, description="Whether extraction was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if extraction failed"
    )
    response_time: float = Field(description="Response time for the extraction")


class WebExtractTool(BaseTool):
    """Tool for extracting content from web URLs using LLM-based HTML processing"""

    def __init__(self):
        super().__init__()
        self.name = "extract_web_content"
        self.description = "ONLY use when explicitly asked to read, extract, or analyze content from a specific web URL that the user provides. Extracts and reads content from web URLs provided by the user, converting web pages into clean, readable markdown format using LLM-based processing. DO NOT use for general questions, information lookup, or when no specific URL is provided."

        # Request headers to mimic a real browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format

        Returns:
            Dict containing the OpenAI-compatible tool definition
        """
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

    def execute(self, params: Dict[str, Any]) -> ExtractResult:
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from text using regex

        Args:
            text: Text to search for URLs

        Returns:
            List of found URLs
        """
        # URL regex pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls

    def _validate_url(self, url: str) -> bool:
        """
        Validate that the URL is properly formatted

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic URL validation
            if not url.startswith(("http://", "https://")):
                return False

            # Parse URL to validate structure
            parsed = urlparse(url)
            if not parsed.netloc:
                return False

            return True
        except Exception:
            return False

    def _check_user_provided_url(
        self, url: str, messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if the URL was provided by the user (not by tools or system)

        Args:
            url: URL to check
            messages: Conversation messages

        Returns:
            True if URL was provided by user, False otherwise
        """
        if not messages:
            # If no messages provided, assume it's from user (for direct tool calls)
            return True

        # Look for the URL in user messages only
        for message in reversed(messages):  # Check from most recent
            if message.get("role") == "user":
                content = str(message.get("content", ""))
                if url in content:
                    logger.info(f"Found URL '{url}' in user message")
                    return True

                # Also check if URL is part of the content
                urls_in_message = self._extract_urls_from_text(content)
                if url in urls_in_message:
                    logger.info(f"Found URL '{url}' in user message content")
                    return True

        logger.warning(f"URL '{url}' was not found in any user message")
        return False

    def _fetch_html_content(self, url: str) -> tuple[str, float]:
        """
        Fetch HTML content from URL

        Args:
            url: URL to fetch

        Returns:
            Tuple of (html_content, response_time)

        Raises:
            requests.RequestException: If the request fails
        """
        import time

        start_time = time.time()

        try:
            # Make request with timeout and proper headers
            response = requests.get(
                url, headers=self.headers, timeout=30, allow_redirects=True
            )
            response.raise_for_status()

            # Check if content is HTML
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
        """
        Extract content from HTML using LLM

        Args:
            html_content: Raw HTML content
            url: Original URL for context

        Returns:
            Extracted markdown content
        """
        try:
            # Get LLM client based on tool configuration from tool_llm_config.py
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

            system_prompt = """detailed thinking off

You are an expert HTML to markdown converter. Your task is to extract the main content from web pages and convert it to clean, readable markdown format.

Instructions:
1. Extract ONLY the main article/content from the webpage
2. Ignore navigation menus, sidebars, advertisements, headers, footers, and other peripheral content
3. Convert the content to clean markdown format
4. Preserve the structure and hierarchy of the content using appropriate markdown headers
5. Include any relevant images, links, and formatting from the main content
6. Do not add any commentary or explanations - just output the extracted markdown content
7. If the page contains mostly navigation or non-content elements, extract what meaningful content you can find
8. Remove any JavaScript, CSS, or other non-content elements
9. The content should be returned verbatim, not summarized.

Output only the extracted markdown content, nothing else."""

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

    def extract_urls_batch(
        self, urls: List[str], messages: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExtractResult]:
        """
        Extract content from multiple URLs

        Args:
            urls: List of URLs to extract content from
            messages: Optional conversation messages to verify URL sources

        Returns:
            List of ExtractResult objects, one for each URL
        """
        if not urls:
            logger.error("No URLs provided for batch extraction")
            return []

        logger.info(f"Starting batch extraction for {len(urls)} URLs")

        results = []
        for url in urls:
            try:
                result = self.extract_url_content(url, messages)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {url}: {e}")
                results.append(
                    ExtractResult(
                        url=url,
                        content="",
                        success=False,
                        error_message=f"Extraction failed: {str(e)}",
                        response_time=0.0,
                    )
                )

        successful_extractions = len([r for r in results if r.success])
        logger.info(
            f"Batch extraction completed. {successful_extractions} successful out of {len(results)} URLs"
        )
        return results

    def extract_url_content(
        self, url: str, messages: Optional[List[Dict[str, Any]]] = None
    ) -> ExtractResult:
        """
        Extract content from a URL using LLM-based processing

        Args:
            url: The URL to extract content from
            messages: Optional conversation messages to verify URL source

        Returns:
            ExtractResult: The extraction result
        """
        logger.info(f"Starting web extraction for URL: '{url}'")

        # Validate URL format
        if not self._validate_url(url):
            logger.error(f"Invalid URL format: '{url}'")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"Invalid URL format: '{url}'. Please provide a valid HTTP or HTTPS URL.",
                response_time=0.0,
            )

        # Check if URL was provided by user (if messages are available)
        if messages and not self._check_user_provided_url(url, messages):
            logger.error(f"URL was not provided by user: '{url}'")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message="I can only extract content from URLs that you directly provide in your message.",
                response_time=0.0,
            )

        try:
            # Fetch HTML content
            logger.debug(f"Fetching HTML content from: '{url}'")
            html_content, response_time = self._fetch_html_content(url)

            # Extract content using LLM
            logger.debug(f"Extracting content using LLM for: '{url}'")
            extracted_content = self._extract_with_llm(html_content, url)

            return ExtractResult(
                url=url,
                content=extracted_content,
                raw_content=html_content,
                success=True,
                response_time=response_time,
            )

        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching URL: '{url}'")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"Request timeout: The webpage at '{url}' took too long to respond.",
                response_time=0.0,
            )
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching URL {url}: {e}")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"HTTP error {e.response.status_code}: Unable to access '{url}'.",
                response_time=0.0,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching URL {url}: {e}")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"Network error: Unable to connect to '{url}'.",
                response_time=0.0,
            )
        except ValueError as e:
            logger.error(f"Content type error for URL {url}: {e}")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=str(e),
                response_time=0.0,
            )
        except Exception as e:
            logger.error(f"Unexpected error during web extraction: {e}")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"An unexpected error occurred: {str(e)}",
                response_time=0.0,
            )

    def _clean_extracted_content(self, content: str) -> str:
        """
        Clean up extracted content for better readability

        Args:
            content: Raw extracted content

        Returns:
            Cleaned content
        """
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

    def run_with_dict(self, params: Dict[str, Any]) -> ExtractResult:
        """
        Execute web extraction with parameters provided as a dictionary

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'url', optionally 'messages'

        Returns:
            ExtractResult: The extraction result
        """
        if "url" not in params:
            raise ValueError("'url' key is required in parameters dictionary")

        url = params["url"]
        messages = params.get("messages", [])

        logger.debug(f"run_with_dict called with URL: '{url}'")
        return self.extract_url_content(url, messages)


# Create global instance and helper functions
web_extract_tool = WebExtractTool()


def get_web_extract_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition for web extraction"""
    return web_extract_tool.to_openai_format()


def execute_web_extract(
    url: str, messages: Optional[List[Dict[str, Any]]] = None
) -> ExtractResult:
    """
    Execute web content extraction

    Args:
        url: The URL to extract content from
        messages: Optional conversation messages to verify URL source

    Returns:
        ExtractResult: The extraction result
    """
    return web_extract_tool.extract_url_content(url, messages)


def execute_web_extract_with_dict(params: Dict[str, Any]) -> ExtractResult:
    """
    Execute web extraction with parameters as dictionary

    Args:
        params: Dictionary containing 'url' and optionally 'messages'

    Returns:
        ExtractResult: The extraction result
    """
    return web_extract_tool.run_with_dict(params)


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
    return web_extract_tool.extract_urls_batch(urls, messages)
