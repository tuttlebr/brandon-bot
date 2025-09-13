"""
Web Data Extractor

Utility for extracting content from web pages
"""

import asyncio
import logging
import time
from typing import Dict, Optional

import aiohttp
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebDataExtractor:
    """Extract and process content from web pages"""

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        """Initialize the web data extractor

        Args:
            headers: Optional custom headers for requests
        """
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }

    def extract_sync(self, url: str) -> Dict[str, any]:
        """Extract content from URL synchronously

        Args:
            url: The URL to extract content from

        Returns:
            Dict with success, content, title, and optional error
        """
        try:
            start_time = time.time()

            # Make request
            response = requests.get(
                url, headers=self.headers, timeout=30, allow_redirects=True
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if (
                "text/html" not in content_type
                and "application/xhtml" not in content_type
            ):
                return {
                    "success": False,
                    "error": (
                        "URL does not return HTML content. Content-Type:"
                        f" {content_type}"
                    ),
                    "content": "",
                    "title": None,
                }

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else ""

            # Remove script and style elements
            for script in soup(["script", "style", "noscript", "iframe"]):
                script.decompose()

            # Try to get main content
            content = ""

            # Try different content containers
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find(id="main")
                or soup.find(id="content")
                or soup.find(class_="content")
                or soup.find("body")
            )

            if main_content:
                # Get text content
                content = main_content.get_text(separator="\n", strip=True)
            else:
                # Fallback to body text
                content = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = content.split("\n")
            lines = [line.strip() for line in lines if line.strip()]
            content = "\n".join(lines)

            elapsed_time = time.time() - start_time
            logger.info(
                f"Successfully extracted content from {url} in"
                f" {elapsed_time:.2f}s"
            )

            return {
                "success": True,
                "content": content,
                "title": title,
                "response_time": elapsed_time,
            }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out",
                "content": "",
                "title": None,
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP error {e.response.status_code}",
                "content": "",
                "title": None,
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "content": "",
                "title": None,
            }
        except Exception as e:
            logger.error(f"Unexpected error extracting {url}: {e}")
            return {
                "success": False,
                "error": f"Extraction error: {str(e)}",
                "content": "",
                "title": None,
            }

    async def extract(self, url: str) -> Dict[str, any]:
        """Extract content from URL asynchronously

        Args:
            url: The URL to extract content from

        Returns:
            Dict with success, content, title, and optional error
        """
        try:
            start_time = time.time()

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, headers=self.headers, allow_redirects=True
                ) as response:
                    response.raise_for_status()

                    # Check content type
                    content_type = response.headers.get(
                        "content-type", ""
                    ).lower()
                    if (
                        "text/html" not in content_type
                        and "application/xhtml" not in content_type
                    ):
                        return {
                            "success": False,
                            "error": (
                                "URL does not return HTML content."
                                f" Content-Type: {content_type}"
                            ),
                            "content": "",
                            "title": None,
                        }

                    # Get HTML text
                    html_text = await response.text()

            # Parse HTML in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._parse_html, html_text
            )

            elapsed_time = time.time() - start_time
            logger.info(
                f"Successfully extracted content from {url} in"
                f" {elapsed_time:.2f}s"
            )

            result["response_time"] = elapsed_time
            return result

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timed out",
                "content": "",
                "title": None,
            }
        except aiohttp.ClientResponseError as e:
            return {
                "success": False,
                "error": f"HTTP error {e.status}",
                "content": "",
                "title": None,
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"Network error: {str(e)}",
                "content": "",
                "title": None,
            }
        except Exception as e:
            logger.error(f"Unexpected error extracting {url}: {e}")
            return {
                "success": False,
                "error": f"Extraction error: {str(e)}",
                "content": "",
                "title": None,
            }

    def _parse_html(self, html_text: str) -> Dict[str, any]:
        """Parse HTML text and extract content

        Args:
            html_text: Raw HTML text

        Returns:
            Dict with success, content, and title
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_text, "html.parser")

            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else ""

            # Remove script and style elements
            for script in soup(["script", "style", "noscript", "iframe"]):
                script.decompose()

            # Try to get main content
            content = ""

            # Try different content containers
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find(id="main")
                or soup.find(id="content")
                or soup.find(class_="content")
                or soup.find("body")
            )

            if main_content:
                # Get text content
                content = main_content.get_text(separator="\n", strip=True)
            else:
                # Fallback to body text
                content = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = content.split("\n")
            lines = [line.strip() for line in lines if line.strip()]
            content = "\n".join(lines)

            return {
                "success": True,
                "content": content,
                "title": title,
            }

        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return {
                "success": False,
                "content": "",
                "title": None,
                "error": f"HTML parsing error: {str(e)}",
            }
