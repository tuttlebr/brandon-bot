import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

import serpapi
from pydantic import BaseModel, ConfigDict, Field
from tools.base import BaseTool, BaseToolResponse

# Configure logger
from utils.logging_config import get_logger

logger = get_logger(__name__)


class NewsSource(BaseModel):
    """News source information from SerpAPI"""

    name: str = Field(default="", description="Name of the source")
    title: Optional[str] = Field(None, description="Title of the source")
    icon: Optional[str] = Field(None, description="Link to the source icon")
    authors: Optional[List[str]] = Field(None, description="List of authors")


class NewsAuthor(BaseModel):
    """News author information from SerpAPI"""

    name: str = Field(default="", description="Name of the author")
    thumbnail: Optional[str] = Field(None, description="Author's thumbnail")
    handle: Optional[str] = Field(None, description="X/Twitter handle")


class NewsResult(BaseModel):
    """Individual news result from SerpAPI"""

    position: int
    title: str
    link: str
    source: NewsSource
    author: Optional[NewsAuthor] = None
    thumbnail: Optional[str] = None
    thumbnail_small: Optional[str] = Field(
        None, description="Low-resolution thumbnail"
    )
    snippet: str
    date: str
    type: Optional[str] = Field(
        None, description="Type of news (e.g., 'Opinion', 'Local coverage')"
    )
    video: Optional[bool] = Field(
        None, description="True if the result is a video"
    )
    extracted_content: Optional[str] = Field(
        None, description="Extracted content from URL or snippet fallback"
    )
    extraction_decision: Optional[str] = Field(
        None, description="LLM's decision on extraction necessity"
    )
    extraction_confidence: Optional[float] = Field(
        None, description="Confidence score for the extraction decision"
    )
    extraction_method: Optional[str] = Field(
        None, description="Method used for content (snippet/extracted/cached)"
    )


class SerpAPINewsResponse(BaseToolResponse):
    """Complete response from SerpAPI News Search"""

    model_config = ConfigDict(extra="allow")

    query: str
    news_results: List[NewsResult] = Field(default_factory=list)
    formatted_results: str = Field(
        default="", description="Formatted results for display"
    )
    extraction_stats: Optional[Dict[str, Any]] = Field(
        None, description="Statistics about the extraction process"
    )


class ExtractionConfig(BaseModel):
    """Configuration for extraction behavior"""

    max_parallel_extractions: int = Field(
        default=5, description="Max parallel web extractions"
    )
    extraction_timeout: float = Field(
        default=15.0, description="Timeout for each extraction"
    )
    retry_on_failure: bool = Field(
        default=True, description="Retry failed extractions"
    )
    max_retries: int = Field(default=15, description="Max retry attempts")
    always_extract_top_n: int = Field(
        default=1,
        description="Always extract top N results regardless of analysis",
    )
    use_cached_content: bool = Field(
        default=True, description="Use cached extracted content"
    )
    cache_ttl_seconds: int = Field(
        default=3600, description="Cache TTL in seconds"
    )


class NewsTool(BaseTool):
    """Tool for performing SerpAPI news searches with intelligent extraction"""

    def __init__(self, extraction_config: Optional[ExtractionConfig] = None):
        super().__init__()
        self.name = "serpapi_news_search"
        self.description = (
            "Up-to-date news articles and breaking events with intelligently "
            "extracted article content (or snippet if extraction not needed). "
            "Use when the user explicitly asks for news, headlines, "
            "or current events."
        )
        self.extraction_config = extraction_config or ExtractionConfig()
        self._content_cache = {}  # Simple in-memory cache

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
                        "top_n": {
                            "type": "integer",
                            "description": (
                                "Number of top results to process (default: 2)"
                            ),
                            "default": 2,
                            "maximum": 3,
                            "minimum": 1,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def get_response_type(self) -> Type[SerpAPINewsResponse]:
        """Get the response type for this tool"""
        return SerpAPINewsResponse

    def execute(self, params: Dict[str, Any]):
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def _extract_news_results(
        self, results: List[NewsResult], user_query: str = "", top_n: int = 2
    ) -> List[NewsResult]:
        """
        Extract content for news results with intelligent decision making.
        Uses LLM-based analysis to determine if web extraction is needed.
        Implements parallel processing for efficiency.
        Deduplicates results by URL to avoid redundant extractions.

        Args:
            results: List of news results
            user_query: The original user query for context

        Returns:
            List of news results with extracted content
        """
        if not results:
            logger.warning("No news results found")
            return []

        # Deduplicate by URL while preserving order and merging snippets
        url_to_results = {}  # Map URL to list of results with that URL
        deduplicated_results = []

        for result in results:
            if result.link not in url_to_results:
                url_to_results[result.link] = []
                deduplicated_results.append(result)
            url_to_results[result.link].append(result)

        # Merge snippets for duplicate URLs
        duplicate_count = 0
        for url, results_list in url_to_results.items():
            if len(results_list) > 1:
                duplicate_count += len(results_list) - 1
                # Merge snippets into the first result
                primary_result = results_list[0]
                merged_snippets = [primary_result.snippet]

                for duplicate in results_list[1:]:
                    if duplicate.snippet not in merged_snippets:
                        merged_snippets.append(duplicate.snippet)

                # Update the primary result with merged snippets
                if len(merged_snippets) > 1:
                    primary_result.snippet = "\n\n".join(merged_snippets)
                    logger.info(
                        "Merged %d snippets for URL: %s",
                        len(merged_snippets),
                        url[:80] + "..." if len(url) > 80 else url,
                    )

        if duplicate_count > 0:
            logger.info(
                "Found %d duplicate URLs, merged into %d unique results",
                duplicate_count,
                len(deduplicated_results),
            )

        results = deduplicated_results

        logger.info(
            "Processing %d unique news results with intelligent extraction",
            len(results),
        )

        # Initialize all results with snippet as extracted_content
        # This ensures we always have content to return
        for result in results:
            result.extracted_content = result.snippet
            result.extraction_method = "snippet"

        # If no user query provided, skip intelligent analysis and use snippets
        if not user_query:
            logger.info(
                "No user query provided, using snippets for all %d results",
                len(results),
            )
            return results

        # Use thread-based parallel processing
        logger.info(
            "Using thread-based parallel processing for %d results",
            len(results),
        )

        # First, analyze all snippets in parallel
        analysis_results = self._analyze_snippets_parallel(
            results, user_query, top_n
        )

        # Collect results that need extraction
        extraction_candidates = []
        for i, (result, (needs_extraction, _)) in enumerate(
            zip(results, analysis_results)
        ):
            result.extraction_decision = None
            result.extraction_confidence = (
                None  # No longer using confidence scores
            )

            # Determine if we should extract
            should_extract = needs_extraction or (
                i < self.extraction_config.always_extract_top_n
            )

            if should_extract:
                extraction_candidates.append((i, result))
                logger.info(
                    "Result %d will be extracted: %s",
                    i + 1,
                    result.title[:50] + "...",
                )
            else:
                logger.info(
                    "Result %d snippet is sufficient: %s",
                    i + 1,
                    result.title[:50] + "...",
                )

        # Extract content in parallel with rate limiting
        if extraction_candidates:
            self._extract_content_parallel(extraction_candidates)

        return results

    def _analyze_snippets_parallel(
        self, results: List[NewsResult], user_query: str, top_n: int
    ) -> List[Tuple[bool, str]]:
        """
        Analyze multiple snippets in parallel to determine extraction needs.

        Returns:
            List of tuples (needs_extraction, decision_reason)
        """
        with ThreadPoolExecutor(max_workers=min(len(results), 5)) as executor:
            futures = []
            for result in results:
                future = executor.submit(
                    self._analyze_snippet_sync_wrapper,
                    result.snippet,
                    user_query,
                    result.link,
                    top_n,
                )
                futures.append(future)

            analysis_results = []
            for future in futures:
                try:
                    result = future.result(timeout=30.0)
                    analysis_results.append(result)
                except Exception as e:
                    logger.warning("Snippet analysis failed: %s", e)
                    # Default to extraction on failure
                    analysis_results.append((True, ""))

            return analysis_results

    def _extract_content_parallel(
        self, extraction_candidates: List[Tuple[int, NewsResult]]
    ):
        """
        Extract content from multiple URLs in parallel with rate limiting.
        Deduplicates extraction requests by URL to avoid redundant work.
        """
        # Check cache first
        if self.extraction_config.use_cached_content:
            remaining_candidates = []
            for idx, result in extraction_candidates:
                cached_content = self._get_cached_content(result.link)
                if cached_content:
                    result.extracted_content = cached_content
                    result.extraction_method = "cached"
                    logger.info(
                        "Using cached content for result %d: %s",
                        idx + 1,
                        result.title[:50] + "...",
                    )
                else:
                    remaining_candidates.append((idx, result))
            extraction_candidates = remaining_candidates

        if not extraction_candidates:
            return

        # Group candidates by URL to avoid duplicate extractions
        url_to_candidates = {}
        for idx, result in extraction_candidates:
            if result.link not in url_to_candidates:
                url_to_candidates[result.link] = []
            url_to_candidates[result.link].append((idx, result))

        # Limit parallel extractions
        max_parallel = self.extraction_config.max_parallel_extractions

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            # Submit one extraction per unique URL
            for url, candidates in url_to_candidates.items():
                future = executor.submit(self._extract_with_retry, url)
                futures[future] = candidates

            for future in futures:
                candidates = futures[future]
                try:
                    extracted_content = future.result(
                        timeout=self.extraction_config.extraction_timeout
                    )
                    if extracted_content:
                        # Share extracted content among all results with URL
                        for idx, result in candidates:
                            result.extracted_content = extracted_content
                            result.extraction_method = "extracted"
                            logger.info(
                                "Successfully extracted content for "
                                "result %d: %s",
                                idx + 1,
                                result.title[:50] + "...",
                            )

                        # Cache the content once
                        if self.extraction_config.use_cached_content:
                            self._cache_content(
                                candidates[0][1].link, extracted_content
                            )
                    else:
                        for idx, result in candidates:
                            logger.info(
                                "Extraction returned empty for result %d, "
                                "keeping snippet",
                                idx + 1,
                            )
                except Exception as e:
                    for idx, result in candidates:
                        logger.error(
                            "Extraction failed for result %d (%s): %s",
                            idx + 1,
                            result.link,
                            e,
                        )

    def _analyze_snippet_sync_wrapper(
        self, snippet: str, user_query: str, url: str, top_n: int
    ) -> Tuple[bool, str]:
        """
        Synchronous wrapper for async snippet analysis.

        Returns:
            Tuple of (needs_extraction, decision_reason)
        """
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._analyze_snippet_completeness_enhanced(
                        snippet, user_query, url, top_n
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.warning("Failed to analyze snippet: %s", e)
            return (True, "")

    def _extract_with_retry(self, url: str) -> Optional[str]:
        """
        Extract content with retry logic.
        """
        max_retries = (
            self.extraction_config.max_retries
            if self.extraction_config.retry_on_failure
            else 1
        )

        for attempt in range(max_retries):
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    content = loop.run_until_complete(
                        self._extract_web_content_safely(url)
                    )
                    if content:
                        return content
                finally:
                    loop.close()

            except Exception as e:
                logger.warning(
                    "Extraction attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    max_retries,
                    url,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief delay before retry

        return None

    async def _analyze_snippet_completeness_enhanced(
        self, snippet: str, user_query: str, url: str, top_n: int
    ) -> Tuple[bool, str]:
        """
        Enhanced snippet analysis for news articles.

        Args:
            snippet: The search result snippet
            user_query: The original user query
            url: The URL of the search result

        Returns:
            Tuple of (needs_extraction, empty_string)
        """
        COMPLETENESS_SCHEMA = {
            "type": "object",
            "properties": {
                "needs_extraction": {"type": "boolean"},
            },
            "required": ["needs_extraction"],
        }

        try:
            from services.llm_client_service import llm_client_service

            # Use the configured LLM for this tool
            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            system_prompt = """/no_think
You analyze news snippets to determine if they contain sufficient information.

Task: Determine if the news snippet fully answers the user's question about news.

You must respond with valid JSON in this exact format:
{
    "needs_extraction": true
}
 - OR -
{
    "needs_extraction": false
}
Set needs_extraction to true if:
- The snippet is truncated or incomplete
- Important details about the news event are missing
- The snippet references information not shown
- The user needs more context about the news

Set needs_extraction to false if:
- The snippet completely answers the question
- All key facts are present in the snippet
- No additional information is needed

Important: Respond ONLY with the JSON object, no other text."""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User Question: {user_query}\n\nNews Article"
                        f" Snippet: {snippet}"
                    ),
                },
            ]

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                top_p=1,
                extra_body={"nvext": {"guided_json": COMPLETENESS_SCHEMA}},
            )

            # Parse the JSON response
            import json

            raw_content = response.choices[0].message.content
            if not raw_content:
                logger.error("Empty response from LLM for snippet analysis")
                return (True, "")

            # Log raw response for debugging
            logger.debug(
                "Raw LLM response for snippet analysis: %s", raw_content
            )

            try:
                decision_data = json.loads(raw_content)
            except json.JSONDecodeError as json_err:
                logger.error(
                    "Failed to parse JSON from LLM response: %s. "
                    "Raw content: %s",
                    json_err,
                    raw_content[:200],
                )
                return (True, "")

            # Validate response is a dictionary
            if not isinstance(decision_data, dict):
                logger.error(
                    "LLM response is not a dictionary. Type: %s, Content: %s",
                    type(decision_data).__name__,
                    str(decision_data)[:200],
                )
                return (True, "")

            # Handle both boolean and string representations
            needs_extraction_raw = decision_data.get("needs_extraction", True)
            if isinstance(needs_extraction_raw, str):
                needs_extraction = needs_extraction_raw.lower() in [
                    "true",
                    "yes",
                    "1",
                ]
            else:
                needs_extraction = bool(needs_extraction_raw)

            logger.info(
                "News snippet analysis for '%s': needs_extraction=%s",
                url[:80] + "..." if len(url) > 80 else url,
                needs_extraction,
            )

            return (needs_extraction, "")

        except Exception as e:
            logger.error(
                "Failed to analyze snippet for %s: %s. "
                "Defaulting to extraction.",
                url,
                e,
            )
            # If analysis fails, default to extraction to be safe
            return (True, "")

    async def _extract_web_content_safely(self, url: str) -> Optional[str]:
        """
        Safely extract web content using the web extraction tool
        without causing deadlocks.

        Args:
            url: The URL to extract content from

        Returns:
            Extracted content or None if extraction fails
        """
        try:
            from tools.registry import get_tool

            # Get the web extraction tool
            extract_tool = get_tool("extract_web_content")
            if not extract_tool:
                logger.warning("Web extraction tool not available")
                return None

            # Use the tool's async execution method to avoid deadlock
            params = {
                "url": url,
                "request": "",  # No specific request, just get content
            }

            # Execute the tool asynchronously
            result = await extract_tool._execute_controller_async(params)

            if result and result.get("success", False):
                # Handle streaming response
                if result.get("is_streaming") and result.get(
                    "content_generator"
                ):
                    # Collect content from the async generator
                    content = ""
                    async for chunk in result["content_generator"]:
                        content += chunk
                else:
                    # Handle non-streaming response
                    content = result.get("content", "")

                logger.info(
                    "Successfully extracted %d characters from %s",
                    len(content),
                    url,
                )
                return content
            else:
                logger.warning(
                    "Web extraction failed for %s: %s",
                    url,
                    (
                        result.get("error_message", "Unknown error")
                        if result
                        else "No result"
                    ),
                )
                return None

        except Exception as e:
            logger.warning("Error extracting content from %s: %s", url, e)
            return None

    def _get_cached_content(self, url: str) -> Optional[str]:
        """Get cached content if available and not expired."""
        if url in self._content_cache:
            cached_data = self._content_cache[url]
            if (
                time.time() - cached_data["timestamp"]
                < self.extraction_config.cache_ttl_seconds
            ):
                return cached_data["content"]
            else:
                # Remove expired cache entry
                del self._content_cache[url]
        return None

    def _cache_content(self, url: str, content: str):
        """Cache extracted content."""
        self._content_cache[url] = {
            "content": content,
            "timestamp": time.time(),
        }

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

            # Build metadata components
            metadata_parts = [result.source.name]

            # Add author if available
            if result.author and result.author.name:
                author_str = result.author.name
                if result.author.handle:
                    author_str += f" (@{result.author.handle})"
                metadata_parts.append(f"by {author_str}")

            # Add type if it's opinion or special coverage
            if result.type:
                metadata_parts.append(f"[{result.type}]")

            # Add video indicator
            if result.video:
                metadata_parts.append("📹 Video")

            metadata_parts.append(result.date)

            # Build the entry with metadata
            entry = f"{i}. [{result.source.name}]({result.link})"
            # Format authors properly
            if result.source.authors:
                authors = result.source.authors
                if len(authors) == 1:
                    entry += f" by {authors[0]}"
                elif len(authors) == 2:
                    entry += f" by {authors[0]} and {authors[1]}"
                else:
                    # 3 or more authors
                    entry += f" by {', '.join(authors[:-1])} and {authors[-1]}"
            entry += f" - {result.title}\n\n"
            formatted_entries.append(entry)

        return "\n".join(formatted_entries)

    def search_serpapi_news(
        self, query: str, top_n: int = 2, **kwargs
    ) -> SerpAPINewsResponse:
        """
        Search for news using SerpAPI with google_news_light engine.

        Args:
            query (str): The search query
            top_n (int): Number of news results to return (default: 2)
            **kwargs: Additional search parameters to override defaults

        Returns:
            SerpAPINewsResponse: The search results in a validated Pydantic
                                model

        Raises:
            ValueError: If SERPAPI_KEY environment variable is not set
                       or if the SerpAPI request fails
        """
        logger.info(
            "Starting SerpAPI news search for query: '%s' with top_n: %d",
            query,
            top_n,
        )

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
            "engine": "google_news",
            "num": top_n,  # Use the top_n parameter to limit results
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

            # Limit results to top_n immediately
            if len(news_results) > top_n:
                logger.info(
                    "SerpAPI returned %d results, limiting to top_n=%d",
                    len(news_results),
                    top_n,
                )
                news_results = news_results[:top_n]

            # Convert news_results to NewsResult objects
            result_objects = []
            for news_item in news_results:
                # Handle source field - it can be either a dict or string
                source_data = news_item.get("source", {})
                if isinstance(source_data, dict):
                    source = NewsSource(
                        name=source_data.get("name", ""),
                        title=source_data.get("title"),
                        icon=source_data.get("icon"),
                        authors=source_data.get("authors"),
                    )
                else:
                    # Fallback for string source (backward compatibility)
                    source = NewsSource(
                        name=str(source_data) if source_data else ""
                    )

                # Handle author field if present
                author_data = news_item.get("author")
                author = None
                if author_data and isinstance(author_data, dict):
                    author = NewsAuthor(
                        name=author_data.get("name", ""),
                        thumbnail=author_data.get("thumbnail"),
                        handle=author_data.get("handle"),
                    )

                result = NewsResult(
                    position=news_item.get("position", 0),
                    title=news_item.get("title", ""),
                    link=news_item.get("link", ""),
                    source=source,
                    author=author,
                    thumbnail=news_item.get("thumbnail"),
                    thumbnail_small=news_item.get("thumbnail_small"),
                    snippet=news_item.get("snippet", ""),
                    date=news_item.get("date", ""),
                    type=news_item.get("type"),
                    video=news_item.get("video"),
                )
                result_objects.append(result)

            # Create response object
            serpapi_response = SerpAPINewsResponse(
                query=query, news_results=result_objects
            )

            # Track extraction statistics
            start_time = time.time()

            # Extract content for news results with intelligent decision making
            serpapi_response.news_results = self._extract_news_results(
                serpapi_response.news_results, user_query=query, top_n=top_n
            )

            # Calculate extraction statistics
            extraction_time = time.time() - start_time
            extraction_stats = {
                "total_results": len(serpapi_response.news_results),
                "extraction_time_seconds": round(extraction_time, 2),
                "extraction_methods": {},
            }

            for result in serpapi_response.news_results:
                method = result.extraction_method or "unknown"
                extraction_stats["extraction_methods"][method] = (
                    extraction_stats["extraction_methods"].get(method, 0) + 1
                )

            serpapi_response.extraction_stats = extraction_stats

            # Format the results for display
            serpapi_response.formatted_results = self.format_results(
                serpapi_response.news_results
            )

            logger.info(
                "Search completed successfully. "
                "Found %d news results with content (extraction took %.2fs)",
                len(serpapi_response.news_results),
                extraction_time,
            )

            # Log extraction statistics
            logger.info(
                "Extraction methods used: %s",
                extraction_stats["extraction_methods"],
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
                   - 'top_n': Number of results to return (optional, default: 2)

        Returns:
            SerpAPINewsResponse: The search results in a validated Pydantic
                                model
        """
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )

        query = params["query"]
        top_n = params.get("top_n", 2)  # Default to 2 if not provided

        logger.debug(
            "run_with_dict method called with query: '%s', top_n: %d",
            query,
            top_n,
        )
        return self.search_serpapi_news(query, top_n=top_n)


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
