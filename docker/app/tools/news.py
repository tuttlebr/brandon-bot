import logging
from typing import Any, Dict, List, Optional, Type

import requests
from pydantic import BaseModel, Field
from tools.base import BaseTool, BaseToolResponse
from utils.text_processing import strip_think_tags

# Configure logger
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Individual search result from Tavily API"""

    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None
    extracted_content: Optional[str] = Field(
        None, description="Extracted content from URL"
    )


class TavilyResponse(BaseToolResponse):
    """Complete response from Tavily API"""

    query: str
    follow_up_questions: Optional[List[str]] = None
    answer: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    results: List[SearchResult] = Field(default_factory=list)
    formatted_results: str = Field(
        default="", description="Formatted results for display"
    )
    response_time: float


class NewsTool(BaseTool):
    """Tool for performing Tavily news searches"""

    def __init__(self):
        super().__init__()
        self.name = "tavily_news_search"
        self.description = "Search specifically for news articles and breaking events. Use when user explicitly asks for news, headlines, or current events."

    def _initialize_mvc(self):
        """Initialize MVC components"""
        # This tool doesn't need separate MVC components as it's simple
        self._controller = None
        self._view = None

    def get_definition(self) -> Dict[str, Any]:
        """
        Return OpenAI-compatible tool definition

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
                        "query": {
                            "type": "string",
                            "description": "The search query for recent news, breaking news, or current events (only use for news-related topics)",
                        },
                        "but_why": {
                            "type": "integer",
                            "description": "An integer from 1-5 where a larger number indicates confidence this is the right tool to help the user.",
                        },
                    },
                    "required": ["query", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[TavilyResponse]:
        """Get the response type for this tool"""
        return TavilyResponse

    def execute(self, params: Dict[str, Any]):
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def _extract_content_for_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Extract content for high-scoring results using batch extraction

        Args:
            results: List of search results

        Returns:
            List of high-scoring search results with extracted content added
        """
        # Filter results with score >= 0.4
        high_scoring_results = [result for result in results if result.score >= 0.15]

        if not high_scoring_results:
            logger.warning(
                "No high-scoring results found, attempting extraction for top result only"
            )
            if not results:
                return []

            # Use the top result as fallback and attempt extraction
            top_result = results[0]
            fallback_results = [top_result]

            # Import extract tool
            from tools.extract import execute_web_extract_batch

            try:
                # Attempt extraction for the top result
                extract_results = execute_web_extract_batch([top_result.url])

                # Update with extracted content if successful
                for extract_result in extract_results:
                    # Handle both WebExtractResponse and StreamingExtractResponse
                    content = None
                    if extract_result.success:
                        if (
                            hasattr(extract_result, 'content')
                            and extract_result.content
                        ):
                            # Regular WebExtractResponse
                            content = extract_result.content
                        elif (
                            hasattr(extract_result, 'content_generator')
                            and extract_result.content_generator
                        ):
                            # StreamingExtractResponse - we need to collect the content
                            try:
                                import asyncio

                                # Create a new event loop if one doesn't exist
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)

                                # Collect content from the async generator
                                async def collect_content():
                                    collected = ""
                                    async for chunk in extract_result.content_generator:
                                        collected += chunk
                                    return collected

                                content = loop.run_until_complete(collect_content())
                            except Exception as e:
                                logger.error(
                                    f"Failed to collect streaming content from fallback result {extract_result.url}: {e}"
                                )
                                content = None

                    if content:
                        top_result.extracted_content = content
                        logger.debug(
                            f"Successfully extracted content from fallback result: {extract_result.url}"
                        )
                    else:
                        logger.warning(
                            f"Failed to extract content from fallback result: {extract_result.url}"
                        )

            except Exception as e:
                logger.error(f"Error in fallback extraction: {e}")
                # Continue without extracted content - don't fail the entire search

            return fallback_results

        logger.info(
            f"Extracting content for {len(high_scoring_results)} high-scoring results"
        )

        # Import extract tool
        from tools.extract import execute_web_extract_batch

        try:
            # Get URLs for high-scoring results
            urls = [result.url for result in high_scoring_results]

            # Perform batch extraction
            extract_results = execute_web_extract_batch(urls)

            # Create a mapping of URL to extracted content
            url_to_content = {}
            for extract_result in extract_results:
                logger.debug(f"Extract result: {extract_result}")

                # Handle both WebExtractResponse and StreamingExtractResponse
                content = None
                if extract_result.success:
                    if hasattr(extract_result, 'content') and extract_result.content:
                        # Regular WebExtractResponse
                        content = extract_result.content
                    elif (
                        hasattr(extract_result, 'content_generator')
                        and extract_result.content_generator
                    ):
                        # StreamingExtractResponse - we need to collect the content
                        try:
                            import asyncio

                            # Create a new event loop if one doesn't exist
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                            # Collect content from the async generator
                            async def collect_content():
                                collected = ""
                                async for chunk in extract_result.content_generator:
                                    collected += chunk
                                return collected

                            content = loop.run_until_complete(collect_content())
                        except Exception as e:
                            logger.error(
                                f"Failed to collect streaming content from {extract_result.url}: {e}"
                            )
                            content = None

                if content:
                    url_to_content[extract_result.url] = content
                    logger.debug(
                        f"Successfully extracted content from {extract_result.url}"
                    )
                else:
                    logger.warning(
                        f"Failed to extract content from {extract_result.url}: {extract_result.error_message}"
                    )

            # Update only the high-scoring results with extracted content
            for result in high_scoring_results:
                if result.url in url_to_content:
                    result.extracted_content = url_to_content[result.url]

            logger.info(
                f"Batch extraction completed. {len(url_to_content)} successful extractions"
            )

        except Exception as e:
            logger.error(f"Error in batch extraction: {e}")
            # Continue without extracted content - don't fail the entire search

        return high_scoring_results

    def format_results(self, results: List[SearchResult]) -> str:
        """
        Format search results for display as numbered markdown list

        Args:
            results: List of SearchResult objects

        Returns:
            str: Formatted results as markdown
        """
        if not results:
            return ""

        formatted_entries = []
        for i, result in enumerate(results, 1):
            # Clean up content text to remove formatting artifacts
            clean_content = self._clean_content(result.content)
            # Format as: 1. [title](url): content
            entry = f"{i}. [{result.title}]({result.url}): {clean_content}\n___"

            # Add extracted content if available
            if result.extracted_content:
                # Strip think tags from extracted content before display
                cleaned_extract = strip_think_tags(result.extracted_content)

                entry += f"\n\n**Extracted Content:**\n{cleaned_extract}"

            formatted_entries.append(entry)

        return "\n".join(formatted_entries)

    def _clean_content(self, content: str) -> str:
        """
        Clean content text by removing formatting artifacts and ensuring plain text display

        Args:
            content: Raw content string from search results

        Returns:
            str: Cleaned content suitable for markdown display
        """
        if not content:
            return ""

        # Remove common markdown formatting artifacts
        import re

        # Remove markdown headers (# ## ###)
        content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)

        # Remove markdown bold/italic formatting (**text**, *text*, __text__, _text_)
        content = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", content)
        content = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", content)

        # Remove markdown links but keep the text [text](url) -> text
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", "", content)

        # Remove excessive whitespace and normalize line breaks
        content = re.sub(r"\s+", " ", content)
        content = content.strip()

        # Remove leading/trailing quotes that might be artifacts
        content = content.strip("\"'")

        return content

    def search_tavily(self, query: str, **kwargs) -> TavilyResponse:
        """
        Search using Tavily API with batch content extraction for high-scoring results.

        Args:
            query (str): The search query
            **kwargs: Additional search parameters to override defaults

        Returns:
            TavilyResponse: The search results in a validated Pydantic model

        Raises:
            ValueError: If TAVILY_API_KEY environment variable is not set
            requests.RequestException: If the API request fails
        """
        logger.info(f"Starting Tavily news search for query: '{query}'")

        # Get API key from environment
        from utils.config import config

        api_key = config.env.TAVILY_API_KEY
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable is not set")
            raise ValueError("TAVILY_API_KEY environment variable is not set")

        logger.debug("API key found, preparing search parameters")

        # Default search parameters matching the shell script
        # TODO: add more tavily API friendly domains to the include_domains list
        default_params = {
            "query": query,
            "topic": "news",
            "auto_parameters": True,
            "include_raw_content": False,
            "max_results": 10,
            "include_images": False,
        }

        # Update with any provided kwargs
        search_params = {**default_params, **kwargs}
        logger.debug(f"Search parameters: {search_params}")

        # API endpoint and headers
        url = "https://api.tavily.com/search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            logger.debug(
                f"Making API request to Tavily news search for query: '{query}'"
            )

            # Make the API request
            response = requests.post(url, headers=headers, json=search_params)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            logger.debug(
                f"Tavily news API request successful (HTTP {response.status_code})"
            )

            # Parse the JSON response and validate with Pydantic
            response_data = response.json()
            tavily_response = TavilyResponse(**response_data)

            # Extract content for high-scoring results
            tavily_response.results = self._extract_content_for_results(
                tavily_response.results
            )

            # Format the results for display
            tavily_response.formatted_results = self.format_results(
                tavily_response.results
            )

            logger.info(
                f"Search completed successfully. Found {len(tavily_response.results)} results in {tavily_response.response_time:.2f}s"
            )
            logger.debug(
                f"Search results: {[result.title for result in tavily_response.results]}"
            )

            return tavily_response

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during Tavily API request: {e}")
            logger.error(f"Response status: {response.status_code}")
            if hasattr(response, "text"):
                logger.error(f"Response body: {response.text}")
            raise requests.RequestException(f"Tavily API HTTP error: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception during Tavily API call: {e}")
            raise requests.RequestException(f"Tavily API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Tavily search: {e}")
            raise

    def run_with_dict(self, params: Dict[str, Any]) -> TavilyResponse:
        """
        Execute a Tavily search with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'query'

        Returns:
            TavilyResponse: The search results in a validated Pydantic model
        """
        if "query" not in params:
            raise ValueError("'query' key is required in parameters dictionary")

        query = params["query"]
        logger.debug(f"run_with_dict method called with query: '{query}'")
        return self.search_tavily(query)


# Helper functions for backward compatibility
def get_news_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for news search

    Returns:
        Dict containing the OpenAI tool definition
    """
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("tavily_news_search", NewsTool)

    # Get the tool instance and return its definition
    tool = get_tool("tavily_news_search")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get news tool definition")
