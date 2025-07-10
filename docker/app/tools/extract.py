"""
Web Extract Tool

This tool extracts content from URLs using the Tavily Extract API.
It only processes URLs provided directly by users (not from other tools or system messages).
"""

import logging
import re
from typing import Any, Dict, List, Optional

import requests
from pydantic import Field
from tools.base import BaseTool, BaseToolResponse

# Configure logger
logger = logging.getLogger(__name__)


class ExtractResult(BaseToolResponse):
    """Result from URL extraction"""

    url: str = Field(description="The URL that was extracted")
    content: str = Field(description="Extracted content in markdown format")
    raw_content: Optional[str] = Field(None, description="Raw content if available")
    success: bool = Field(default=True, description="Whether extraction was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if extraction failed"
    )
    response_time: float = Field(description="Response time for the extraction")


class WebExtractTool(BaseTool):
    """Tool for extracting content from web URLs using Tavily Extract API"""

    def __init__(self):
        super().__init__()
        self.name = "extract_web_content"
        self.description = "ONLY use when explicitly asked to read, extract, or analyze content from a specific web URL that the user provides. Extracts and reads content from web URLs provided by the user, converting web pages into clean, readable markdown format. DO NOT use for general questions, information lookup, or when no specific URL is provided."
        self.extract_url = "https://api.tavily.com/extract"

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
            if not url.startswith(('http://', 'https://')):
                return False

            # Check for basic URL structure
            parts = url.split('/', 3)
            if len(parts) < 3:
                return False

            domain = parts[2]
            if '.' not in domain:
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

    def extract_urls_batch(
        self, urls: List[str], messages: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExtractResult]:
        """
        Extract content from multiple URLs in a single API call

        Args:
            urls: List of URLs to extract content from
            messages: Optional conversation messages to verify URL sources

        Returns:
            List of ExtractResult objects, one for each URL

        Raises:
            ValueError: If URLs list is empty or invalid
            requests.RequestException: If the API request fails
        """
        if not urls:
            logger.error("No URLs provided for batch extraction")
            return []

        logger.info(f"Starting batch extraction for {len(urls)} URLs")

        # Validate all URLs
        valid_urls = []
        for url in urls:
            if self._validate_url(url):
                valid_urls.append(url)
            else:
                logger.warning(f"Invalid URL format: '{url}'")

        if not valid_urls:
            logger.error("No valid URLs provided for batch extraction")
            return [
                ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"Invalid URL format: '{url}'",
                    response_time=0.0,
                )
                for url in urls
            ]

        # Get API key from environment
        from utils.config import config

        api_key = config.env.TAVILY_API_KEY
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable is not set")
            return [
                ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message="Web extraction service is not configured.",
                    response_time=0.0,
                )
                for url in valid_urls
            ]

        # Prepare request payload
        payload = {
            "urls": valid_urls,  # API expects a list
            "include_images": False,
            "include_favicon": False,
            "extract_depth": "basic",
            "format": "text",
        }

        # API headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            logger.debug(f"Making batch extract API request for {len(valid_urls)} URLs")

            # Make the API request
            response = requests.post(self.extract_url, headers=headers, json=payload)
            response.raise_for_status()

            logger.debug(
                f"Batch extract API request successful (HTTP {response.status_code})"
            )

            # Parse the response
            data = response.json()

            # Initialize results list
            results = []

            # Process successful extractions
            if data.get("results"):
                successful_extractions = {
                    result.get("url"): result for result in data["results"]
                }

                for url in urls:
                    if url in successful_extractions:
                        result = successful_extractions[url]
                        raw_content = result.get("raw_content", "")
                        content = self._clean_extracted_content(raw_content)

                        results.append(
                            ExtractResult(
                                url=url,
                                content=content,
                                raw_content=raw_content,
                                success=True,
                                response_time=data.get("response_time", 0.0),
                            )
                        )
                    else:
                        # URL was not successfully extracted
                        results.append(
                            ExtractResult(
                                url=url,
                                content="",
                                success=False,
                                error_message=f"Failed to extract content from '{url}'",
                                response_time=data.get("response_time", 0.0),
                            )
                        )
            else:
                # No successful extractions
                for url in urls:
                    results.append(
                        ExtractResult(
                            url=url,
                            content="",
                            success=False,
                            error_message="No content could be extracted from the URL.",
                            response_time=data.get("response_time", 0.0),
                        )
                    )

            logger.info(
                f"Batch extraction completed. {len([r for r in results if r.success])} successful out of {len(results)} URLs"
            )
            return results

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during batch extract API request: {e}")
            return [
                ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"Failed to extract content: HTTP error {e.response.status_code}",
                    response_time=0.0,
                )
                for url in urls
            ]
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception during batch extract API call: {e}")
            return [
                ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"Failed to extract content: {str(e)}",
                    response_time=0.0,
                )
                for url in urls
            ]
        except Exception as e:
            logger.error(f"Unexpected error during batch web extraction: {e}")
            return [
                ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"An unexpected error occurred: {str(e)}",
                    response_time=0.0,
                )
                for url in urls
            ]

    def extract_url_content(
        self, url: str, messages: Optional[List[Dict[str, Any]]] = None
    ) -> ExtractResult:
        """
        Extract content from a URL using Tavily Extract API

        Args:
            url: The URL to extract content from
            messages: Optional conversation messages to verify URL source

        Returns:
            ExtractResult: The extraction result

        Raises:
            ValueError: If URL is invalid or not user-provided
            requests.RequestException: If the API request fails
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

        # Get API key from environment
        from utils.config import config

        api_key = config.env.TAVILY_API_KEY
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable is not set")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message="Web extraction service is not configured.",
                response_time=0.0,
            )

        # Prepare request payload
        payload = {
            "urls": [url],  # API expects a list
            "include_images": False,
            "include_favicon": False,
            "extract_depth": "basic",
            "format": "markdown",
        }

        # API headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            logger.debug(f"Making extract API request for URL: '{url}'")

            # Make the API request
            response = requests.post(self.extract_url, headers=headers, json=payload)
            response.raise_for_status()

            logger.debug(
                f"Extract API request successful (HTTP {response.status_code})"
            )

            # Parse the response
            data = response.json()

            # Check for successful extractions
            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]

                # Get the raw content
                raw_content = result.get("raw_content", "")

                # Clean up the content for better readability
                content = self._clean_extracted_content(raw_content)

                return ExtractResult(
                    url=url,
                    content=content,
                    raw_content=raw_content,
                    success=True,
                    response_time=data.get("response_time", 0.0),
                )

            # Check for failed extractions
            elif data.get("failed_results") and len(data["failed_results"]) > 0:
                logger.error(f"URL extraction failed for: '{url}'")
                return ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message=f"Failed to extract content from '{url}'. The page may be inaccessible or blocked.",
                    response_time=data.get("response_time", 0.0),
                )

            else:
                logger.error("No results returned from extract API")
                return ExtractResult(
                    url=url,
                    content="",
                    success=False,
                    error_message="No content could be extracted from the URL.",
                    response_time=data.get("response_time", 0.0),
                )

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during extract API request: {e}")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"Failed to extract content: HTTP error {e.response.status_code}",
                response_time=0.0,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception during extract API call: {e}")
            return ExtractResult(
                url=url,
                content="",
                success=False,
                error_message=f"Failed to extract content: {str(e)}",
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
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Remove navigation artifacts
        navigation_patterns = [
            r'Jump to content',
            r'Main menu',
            r'Toggle.*menu',
            r'Search\s*Appearance',
            r'Create account.*Log in',
            r'Personal tools',
        ]

        for pattern in navigation_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Clean up whitespace
        content = re.sub(r' {2,}', ' ', content)
        content = content.strip()

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
