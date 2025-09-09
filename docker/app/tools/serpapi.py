import logging
from typing import Any, Dict, List, Optional, Type

import serpapi
from pydantic import BaseModel, Field
from tools.base import BaseTool, BaseToolResponse
from utils.text_processing import clean_content, strip_think_tags

# Configure logger
logger = logging.getLogger(__name__)


class SearchMetadata(BaseModel):
    """Search metadata from SerpAPI"""

    id: str
    status: str
    json_endpoint: str
    created_at: str
    processed_at: str
    google_light_url: str
    raw_html_file: str
    total_time_taken: float


class SearchParameters(BaseModel):
    """Search parameters used in SerpAPI query"""

    engine: str
    q: str
    location_requested: str
    location_used: str
    google_domain: str
    hl: str
    gl: str
    device: str


class SearchInformation(BaseModel):
    """Search information from SerpAPI"""

    query_displayed: str
    organic_results_state: str


class OrganicResult(BaseModel):
    """Individual organic search result from SerpAPI"""

    position: int
    title: str
    link: str
    displayed_link: str
    snippet: str
    missing: Optional[List[str]] = None
    must_include: Optional[Dict[str, str]] = None
    extensions: Optional[List[str]] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    extracted_content: Optional[str] = Field(
        None, description="Extracted content from URL or snippet fallback"
    )


class RelatedQuestion(BaseModel):
    """Related question from SerpAPI"""

    snippet: Optional[str] = None
    more_results_link: Optional[str] = ""


class RelatedSearch(BaseModel):
    """Related search from SerpAPI"""

    query: str
    link: str
    serpapi_link: str


class SerpAPIResponse(BaseToolResponse):
    """Complete response from SerpAPI"""

    search_metadata: SearchMetadata
    search_parameters: SearchParameters
    search_information: SearchInformation
    organic_results: List[OrganicResult] = Field(default_factory=list)
    related_questions: Optional[List[RelatedQuestion]] = None
    related_searches: Optional[List[RelatedSearch]] = None
    serpapi_pagination: Optional[Dict[str, Any]] = None
    formatted_results: str = Field(
        default="", description="Formatted results for display"
    )


class SerpAPITool(BaseTool):
    """Tool for performing SerpAPI internet searches"""

    def __init__(self):
        super().__init__()
        self.name = "serpapi_internet_search"
        self.description = (
            "Do not use this tool for weather or news."
            "Search the internet for current information using Google. "
            "Query MUST be in the form of a question for best results. "
            "Returns search parameters and top 1 organic results "
            "with extracted webpage content (or snippet if extraction fails)."
            "When helpful, provide the results links in markdown format."
            "NEVER make up or guess URLs."
        )

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
                            "description": (
                                "Query used to search the internet. Required "
                                "to be in the form of a question. "
                                "Examples: 'What is the weather today?', "
                                "'How does photosynthesis work?', "
                                "'When was the Eiffel Tower built?'"
                            ),
                        },
                        "but_why": {
                            "type": "integer",
                            "description": (
                                "An integer from 1-5 where a larger number "
                                "indicates confidence this is the right "
                                "tool to help the user."
                            ),
                        },
                        "location_requested": {
                            "type": "string",
                            "description": (
                                "location for the search "
                                "MUST be in the form of a 'City, State, "
                                "Country' (e.g., 'New York, NY, "
                                "United States')."
                            ),
                            "default": "Saline, Michigan, United States",
                        },
                    },
                    "required": ["query", "but_why"],
                },
            },
        }

    def get_response_type(self) -> Type[SerpAPIResponse]:
        """Get the response type for this tool"""
        return SerpAPIResponse

    def execute(self, params: Dict[str, Any]):
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def _extract_top_results(
        self, results: List[OrganicResult]
    ) -> List[OrganicResult]:
        """
        Return top 1 organic results with extracted content

        Args:
            results: List of organic search results

        Returns:
            List containing top 1 results with extracted content
        """
        if not results:
            logger.warning("No organic results found")
            return []

        # Get top 1 results
        top_results = results[:1]
        logger.info("Processing top %d organic results", len(top_results))

        # Initialize all results with snippet as extracted_content
        # This ensures we always have content to return
        for result in top_results:
            result.extracted_content = result.snippet

        # Try to extract content from URLs for all top results
        try:
            from tools.extract import execute_web_extract_batch

            # Get URLs for all top results
            urls = [result.link for result in top_results]
            logger.debug(
                "Attempting to extract content from %d URLs", len(urls)
            )

            try:
                extract_results = execute_web_extract_batch(urls)
            except Exception as e:
                logger.error(
                    "execute_web_extract_batch failed: %s. "
                    "Using snippets for all results.",
                    e,
                )
                # Already initialized with snippets, so just return
                return top_results

            # Create URL to extract result mapping
            url_to_extract = {}
            for extract_result in extract_results:
                url_to_extract[extract_result.url] = extract_result

            # Process each top result
            for result in top_results:
                try:
                    extract_result = url_to_extract.get(result.link)

                    if extract_result and extract_result.success:
                        # Handle WebExtractResponse and
                        # StreamingExtractResponse
                        content = None
                        if (
                            hasattr(extract_result, "content")
                            and extract_result.content
                        ):
                            content = extract_result.content
                        elif (
                            hasattr(extract_result, "content_generator")
                            and extract_result.content_generator
                        ):
                            # Handle streaming response
                            try:
                                import asyncio

                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)

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
                                    "for %s: %s. Using snippet instead.",
                                    result.link,
                                    e,
                                )
                                content = None

                        if content:
                            result.extracted_content = content
                            logger.info(
                                "Successfully extracted content from: %s",
                                result.link,
                            )
                        else:
                            logger.warning(
                                "No content extracted for %s, keeping snippet",
                                result.link,
                            )
                            # Already has snippet as extracted_content
                    else:
                        logger.warning(
                            "Extraction failed or not successful for %s, "
                            "keeping snippet",
                            result.link,
                        )
                        # Already has snippet as extracted_content

                except Exception as e:
                    logger.error(
                        "Error processing extraction result for %s: %s. "
                        "Keeping snippet.",
                        result.link,
                        e,
                    )
                    # Already has snippet as extracted_content

        except ImportError as e:
            logger.error(
                "Failed to import tools.extract module: %s. "
                "Using snippets for all results.",
                e,
            )
            # Already initialized with snippets, so just return
        except Exception as e:
            logger.error(
                "Unexpected error during content extraction: %s. "
                "Using snippets for all results.",
                e,
            )
            # Already initialized with snippets, so just return

        # Ensure all results have extracted_content (should already be set)
        for result in top_results:
            if not result.extracted_content:
                result.extracted_content = result.snippet

        return top_results

    def _validate_query_format(self, query: str) -> str:
        """
        Clean and validate the query

        Args:
            query: The search query

        Returns:
            The cleaned query
        """
        query = query.strip()

        return query

    def format_results(self, results: List[OrganicResult]) -> str:
        """
        Format search results for display as numbered markdown list

        Args:
            results: List of OrganicResult objects

        Returns:
            str: Formatted results as markdown
        """
        if not results:
            return ""

        formatted_entries = []
        for i, result in enumerate(results, 1):
            # Clean up snippet text to remove formatting artifacts
            clean_snippet = clean_content(result.snippet)
            # Format as: 1. [title](link): snippet
            entry = (
                f"{i}. [{result.title}]({result.link}) - "
                f"{result.link}: {clean_snippet}\n\n"
            )

            # Add extracted content if available
            if result.extracted_content:
                # Strip think tags from extracted content before display
                cleaned_extract = strip_think_tags(result.extracted_content)
                entry += f"\n\n**Extracted Content:**\n{cleaned_extract}"
            entry += "\n\n___"
            formatted_entries.append(entry)

        return "\n".join(formatted_entries)

    def search_serpapi(
        self, query: str, location_requested: str, **kwargs
    ) -> SerpAPIResponse:
        """
        Search using SerpAPI and return top 1 organic results with
        extracted content from the webpages (falls back to snippet if
        extraction fails).

        Args:
            query (str): The search query
            location_requested (str): Location for the search
            **kwargs: Additional search parameters to override defaults

        Returns:
            SerpAPIResponse: The search results with top 1 results
                            including extracted content

        Raises:
            ValueError: If SERPAPI_KEY environment variable is not set
                       or if the SerpAPI request fails
        """
        # Validate query format
        query = self._validate_query_format(query)

        logger.info("Starting SerpAPI search for query: '%s'", query)

        # Get API key from environment
        from utils.config import config

        api_key = config.env.SERPAPI_KEY
        if not api_key:
            logger.error("SERPAPI_KEY environment variable is not set")
            raise ValueError("SERPAPI_KEY environment variable is not set")

        logger.debug("API key found, preparing search parameters")

        logger.debug("Using location for search: '%s'", location_requested)

        # Default search parameters for SerpAPI
        default_params = {
            "q": query,
            "engine": "google_light",
            "location": location_requested,
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "device": "desktop",
        }

        # Update with any provided kwargs
        search_params = {**default_params, **kwargs}
        logger.debug("Search parameters: %s", search_params)

        try:
            logger.debug(
                "Making API request to SerpAPI for query: '%s'", query
            )

            # Create SerpAPI client and perform search
            client = serpapi.Client(api_key=api_key)
            response_data = client.search(search_params)

            # Parse the JSON response and validate with Pydantic
            serpapi_response = SerpAPIResponse(**response_data)

            # Extract content for top 1 results
            serpapi_response.organic_results = self._extract_top_results(
                serpapi_response.organic_results
            )

            # Format the results for display
            serpapi_response.formatted_results = self.format_results(
                serpapi_response.organic_results
            )

            logger.info(
                "Search completed successfully. "
                "Returning %d results with content",
                len(serpapi_response.organic_results),
            )
            if serpapi_response.organic_results:
                for idx, result in enumerate(serpapi_response.organic_results):
                    logger.debug(
                        "Result %d: %s (content length: %d chars)",
                        idx + 1,
                        result.title,
                        (
                            len(result.extracted_content)
                            if result.extracted_content
                            else 0
                        ),
                    )

            return serpapi_response

        except KeyError as e:
            logger.error("Missing key in SerpAPI response: %s", e)
            raise ValueError(
                f"Invalid SerpAPI response format: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("Value error during SerpAPI search: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error during SerpAPI search: %s", e)
            raise ValueError(f"SerpAPI search failed: {str(e)}") from e

    def run_with_dict(self, params: Dict[str, Any]) -> SerpAPIResponse:
        """
        Execute a SerpAPI search with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys:
                   - 'query': Search query (MUST be in question form)
                   - 'but_why': Confidence level (1-5)
                   - 'location_requested': Location for search (defaults
                     to 'Saline, Michigan, United States' if not
                     provided or empty)

        Returns:
            SerpAPIResponse: The search results in a validated Pydantic model
        """
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )

        query = params["query"]

        # Use location_requested if provided and not empty, otherwise default
        location_requested = params.get("location_requested", "").strip()
        if not location_requested:
            location_requested = "Saline, Michigan, United States"
            logger.debug(
                "No location provided, using default: '%s'", location_requested
            )
        else:
            logger.debug("Using provided location: '%s'", location_requested)

        logger.debug("run_with_dict method called with query: '%s'", query)
        return self.search_serpapi(
            query, location_requested=location_requested
        )


# Helper functions for backward compatibility
def get_serpapi_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for SerpAPI search

    Returns:
        Dict containing the OpenAI tool definition
    """
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("serpapi_internet_search", SerpAPITool)

    # Get the tool instance and return its definition
    tool = get_tool("serpapi_internet_search")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get serpapi tool definition")
