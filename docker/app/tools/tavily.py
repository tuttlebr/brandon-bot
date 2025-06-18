import logging
import os
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

# Configure logger
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Individual search result from Tavily API"""

    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None


class TavilyResponse(BaseModel):
    """Complete response from Tavily API"""

    query: str
    follow_up_questions: Optional[List[str]] = None
    answer: Optional[str] = None
    images: List[str] = Field(default_factory=list)
    results: List[SearchResult] = Field(default_factory=list)
    response_time: float


class TavilyTool:
    """Tool for performing Tavily internet searches"""

    def __init__(self):
        self.name = "tavily_internet_search"
        self.description = (
            "A search engine optimized for comprehensive, accurate, and trusted news results. "
            "Useful for when you need to answer questions about local business, current events, general internet search or news. "
            "It not only retrieves URLs and snippets, but offers advanced search depths, "
            "domain management, same-day search filtering this tool delivers "
            "real-time, accurate, and citation-backed results."
            "Input should be a search query."
        )

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
                        "query": {"type": "string", "description": "The search query to look up current information"}
                    },
                    "required": ["query"],
                },
            },
        }

    def search_tavily(self, query: str, **kwargs) -> TavilyResponse:
        """
        Search using Tavily API with the same parameters as the shell script.

        Args:
            query (str): The search query
            **kwargs: Additional search parameters to override defaults

        Returns:
            TavilyResponse: The search results in a validated Pydantic model

        Raises:
            ValueError: If TAVILY_API_KEY environment variable is not set
            requests.RequestException: If the API request fails
        """
        logger.info(f"Starting Tavily search for query: '{query}'")

        # Get API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable is not set")
            raise ValueError("TAVILY_API_KEY environment variable is not set")

        logger.debug("API key found, preparing search parameters")

        # Default search parameters matching the shell script
        default_params = {
            "query": query,
            "topic": "general",
            "search_depth": "advanced",
            "chunks_per_source": 3,
            "max_results": 3,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_image_descriptions": False,
            "time_range": "month",
            "include_domains": [],
            "exclude_domains": [],
            "country": "united states",
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
            logger.info(f"Making API request to Tavily for query: '{query}'")

            # Make the API request
            response = requests.post(url, headers=headers, json=search_params)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            logger.info(f"Tavily API request successful (HTTP {response.status_code})")

            # Parse the JSON response and validate with Pydantic
            response_data = response.json()
            tavily_response = TavilyResponse(**response_data)

            logger.info(
                f"Search completed successfully. Found {len(tavily_response.results)} results in {tavily_response.response_time:.2f}s"
            )
            logger.debug(f"Search results: {[result.title for result in tavily_response.results]}")

            return tavily_response

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during Tavily API request: {e}")
            logger.error(f"Response status: {response.status_code}")
            if hasattr(response, 'text'):
                logger.error(f"Response body: {response.text}")
            raise requests.RequestException(f"Tavily API HTTP error: {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception during Tavily API call: {e}")
            raise requests.RequestException(f"Tavily API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Tavily search: {e}")
            raise

    def _run(self, query: str = None, **kwargs) -> TavilyResponse:
        """
        Execute a Tavily search with the given query.

        Args:
            query (str): The search query (for backward compatibility)
            **kwargs: Can accept a dictionary with 'query' key

        Returns:
            TavilyResponse: The search results in a validated Pydantic model
        """
        # Support both direct parameter and dictionary input
        if query is None and 'query' in kwargs:
            query = kwargs['query']
        elif query is None:
            raise ValueError("Query parameter is required")

        logger.debug(f"_run method called with query: '{query}'")
        return self.search_tavily(query)

    def run_with_dict(self, params: Dict[str, Any]) -> TavilyResponse:
        """
        Execute a Tavily search with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'query'

        Returns:
            TavilyResponse: The search results in a validated Pydantic model
        """
        if 'query' not in params:
            raise ValueError("'query' key is required in parameters dictionary")

        query = params['query']
        logger.debug(f"run_with_dict method called with query: '{query}'")
        return self.search_tavily(query)


# Create a global instance and helper function for easy access
tavily_tool = TavilyTool()


def get_tavily_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for Tavily search

    Returns:
        Dict containing the OpenAI tool definition
    """
    return tavily_tool.to_openai_format()


def execute_tavily_search(query: str) -> TavilyResponse:
    """
    Execute a Tavily search with the given query

    Args:
        query: The search query

    Returns:
        TavilyResponse: The search results
    """
    return tavily_tool.search_tavily(query)


def execute_tavily_with_dict(params: Dict[str, Any]) -> TavilyResponse:
    """
    Execute a Tavily search with parameters provided as a dictionary

    Args:
        params: Dictionary containing the required parameters
               Expected keys: 'query'

    Returns:
        TavilyResponse: The search results
    """
    return tavily_tool.run_with_dict(params)
