import logging
from typing import Any, Dict, List, Optional, Type

import serpapi
from pydantic import BaseModel, Field
from tools.base import BaseTool, BaseToolResponse
from utils.text_processing import clean_content, strip_think_tags

# Configure logger
logger = logging.getLogger(__name__)


class NewsResult(BaseModel):
    """Individual news result from SerpAPI"""

    position: int
    title: str
    link: str
    source: str
    thumbnail: Optional[str] = None
    snippet: str
    date: str
    extracted_content: Optional[str] = Field(
        None, description="Extracted content from URL"
    )


class SerpAPINewsResponse(BaseToolResponse):
    """Complete response from SerpAPI News Search"""

    query: str
    news_results: List[NewsResult] = Field(default_factory=list)
    formatted_results: str = Field(
        default="", description="Formatted results for display"
    )


class NewsTool(BaseTool):
    """Tool for performing SerpAPI news searches"""

    def __init__(self):
        super().__init__()
        self.name = "serpapi_news_search"
        self.description = (
            "Up-to-date news articles and breaking events. "
            "Use when the user explicitly asks for news, headlines, "
            "or current events."
        )

    def _initialize_mvc(self):
        """Initialize MVC components (not needed for this tool)"""
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
                            "description": (
                                "The search query for recent news, breaking "
                                "news, or current events (only use for "
                                "news-related topics)"
                            ),
                        },
                        "but_why": {
                            "type": "integer",
                            "description": (
                                "An integer from 1-5 where a larger number "
                                "indicates confidence this is the right tool "
                                "to help the user."
                            ),
                        },
                    },
                    "required": ["query", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[SerpAPINewsResponse]:
        """Get the response type for this tool"""
        return SerpAPINewsResponse

    def execute(self, params: Dict[str, Any]):
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def _extract_content_for_results(
        self, results: List[NewsResult]
    ) -> List[NewsResult]:
        """
        Extract content for news results using batch extraction

        Args:
            results: List of news results

        Returns:
            List of news results with extracted content added
        """
        if not results:
            logger.warning("No news results found for extraction")
            return []

        logger.info(
            "Extracting content for %d news results",
            len(results),
        )

        # Import extract tool
        from tools.extract import execute_web_extract_batch

        try:
            # Get URLs for all results
            urls = [result.link for result in results]

            # Perform batch extraction
            extract_results = execute_web_extract_batch(urls)

            # Create a mapping of URL to extracted content
            url_to_content = {}
            for extract_result in extract_results:
                logger.debug("Extract result: %s", extract_result)

                # Handle both WebExtractResponse and StreamingExtractResponse
                content = None
                if extract_result.success:
                    if (
                        hasattr(extract_result, "content")
                        and extract_result.content
                    ):
                        # Regular WebExtractResponse
                        content = extract_result.content
                    elif (
                        hasattr(extract_result, "content_generator")
                        and extract_result.content_generator
                    ):
                        # StreamingExtractResponse - collect the content
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
                                async for (
                                    chunk
                                ) in extract_result.content_generator:
                                    collected += chunk
                                return collected

                            content = loop.run_until_complete(
                                collect_content()
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to collect streaming content "
                                "from %s: %s",
                                extract_result.url,
                                e,
                            )
                            content = None

                if content:
                    url_to_content[extract_result.url] = content
                    logger.debug(
                        "Successfully extracted content from %s",
                        extract_result.url,
                    )
                else:
                    logger.warning(
                        "Failed to extract content from %s: %s",
                        extract_result.url,
                        extract_result.error_message,
                    )

            # Update results with extracted content
            for result in results:
                if result.link in url_to_content:
                    result.extracted_content = url_to_content[result.link]

            logger.info(
                "Batch extraction completed. %d successful extractions",
                len(url_to_content),
            )

        except Exception as e:
            logger.error("Error in batch extraction: %s", e)
            # Continue without extracted content - don't fail the entire search

        return results

    def format_results(self, results: List[NewsResult]) -> str:
        """
        Format search results for display as numbered markdown list

        Args:
            results: List of NewsResult objects

        Returns:
            str: Formatted results as markdown
        """
        if not results:
            return ""

        formatted_entries = []
        for i, result in enumerate(results, 1):
            # Clean up snippet text to remove formatting artifacts
            clean_snippet = clean_content(result.snippet)
            # Format as: 1. [title](link) - source (date): snippet
            entry = (
                f"{i}. [{result.title}]({result.link}) - "
                f"{result.source} ({result.date}): {clean_snippet}\n\n"
            )

            # Add extracted content if available
            if result.extracted_content:
                # Strip think tags from extracted content before display
                cleaned_extract = strip_think_tags(result.extracted_content)

                entry += f"\n\n**Extracted Content:**\n{cleaned_extract}"
            entry += "\n\n___"
            formatted_entries.append(entry)

        return "\n".join(formatted_entries)

    def search_serpapi_news(self, query: str, **kwargs) -> SerpAPINewsResponse:
        """
        Search for news using SerpAPI with google_news_light engine.

        Args:
            query (str): The search query
            **kwargs: Additional search parameters to override defaults

        Returns:
            SerpAPINewsResponse: The search results in a validated Pydantic
                                model

        Raises:
            ValueError: If SERPAPI_KEY environment variable is not set
                       or if the SerpAPI request fails
        """
        logger.info("Starting SerpAPI news search for query: '%s'", query)

        # Get API key from environment
        from utils.config import config

        api_key = config.env.SERPAPI_KEY
        if not api_key:
            logger.error("SERPAPI_KEY environment variable is not set")
            raise ValueError("SERPAPI_KEY environment variable is not set")

        logger.debug("API key found, preparing search parameters")

        # Default search parameters for SerpAPI news search
        default_params = {
            "q": query,
            "engine": "google_news_light",
            "num": 5,  # Always set to 5 as per requirement
            "hl": "en",
            "gl": "us",
        }

        # Update with any provided kwargs
        search_params = {**default_params, **kwargs}
        logger.debug("Search parameters: %s", search_params)

        try:
            logger.debug(
                "Making API request to SerpAPI news search for query: '%s'",
                query,
            )

            # Create SerpAPI client and perform search
            client = serpapi.Client(api_key=api_key)
            response_data = client.search(search_params)

            # Extract news_results from response
            news_results = response_data.get("news_results", [])
            # Convert news_results to NewsResult objects
            result_objects = []
            for news_item in news_results:
                result = NewsResult(
                    position=news_item.get("position", 0),
                    title=news_item.get("title", ""),
                    link=news_item.get("link", ""),
                    source=news_item.get("source", ""),
                    thumbnail=news_item.get("thumbnail"),
                    snippet=news_item.get("snippet", ""),
                    date=news_item.get("date", ""),
                )
                result_objects.append(result)

            # Create response object
            serpapi_response = SerpAPINewsResponse(
                query=query, news_results=result_objects
            )

            # Extract content for news results
            serpapi_response.news_results = self._extract_content_for_results(
                serpapi_response.news_results
            )

            # Format the results for display
            serpapi_response.formatted_results = self.format_results(
                serpapi_response.news_results
            )

            logger.info(
                "Search completed successfully. Found %d news results",
                len(serpapi_response.news_results),
            )
            logger.debug(
                "Search results: %s",
                [result.title for result in serpapi_response.news_results],
            )

            return serpapi_response

        except KeyError as e:
            logger.error("Missing key in SerpAPI response: %s", e)
            raise ValueError(
                f"Invalid SerpAPI response format: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("Value error during SerpAPI news search: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during SerpAPI news search: %s", e)
            raise ValueError(f"SerpAPI news search failed: {str(e)}") from e

    def run_with_dict(self, params: Dict[str, Any]) -> SerpAPINewsResponse:
        """
        Execute a SerpAPI news search with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys:
                   - 'query': Search query
                   - 'but_why': Confidence level (1-5)

        Returns:
            SerpAPINewsResponse: The search results in a validated Pydantic
                                model
        """
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )

        query = params["query"]

        logger.debug("run_with_dict method called with query: '%s'", query)
        return self.search_serpapi_news(query)


# Helper functions for backward compatibility
def get_news_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for news search

    Returns:
        Dict containing the OpenAI tool definition
    """
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("serpapi_news_search", NewsTool)

    # Get the tool instance and return its definition
    tool = get_tool("serpapi_news_search")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get news tool definition")
