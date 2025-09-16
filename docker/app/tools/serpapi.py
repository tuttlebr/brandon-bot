import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Type

import serpapi
from pydantic import BaseModel, ConfigDict, Field
from tools.base import BaseTool, BaseToolResponse

# Configure logger
from utils.logging_config import get_logger
from utils.text_processing import clean_content

logger = get_logger(__name__)


class SearchMetadata(BaseModel):
    """Search metadata from SerpAPI"""

    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    json_endpoint: Optional[str] = None
    created_at: str
    processed_at: str
    search_url: Optional[str] = None
    raw_html_file: Optional[str] = None
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

    model_config = ConfigDict(extra="allow")

    query_displayed: str
    organic_results_state: str


class OrganicResult(BaseModel):
    """Individual organic search result from SerpAPI"""

    model_config = ConfigDict(extra="allow")

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
    extraction_decision: Optional[str] = Field(
        None, description="LLM's decision on extraction necessity"
    )
    extraction_confidence: Optional[float] = Field(
        None, description="Confidence score for the extraction decision"
    )
    extraction_method: Optional[str] = Field(
        None, description="Method used for content (snippet/extracted/cached)"
    )


class RelatedQuestion(BaseModel):
    """Related question from SerpAPI"""

    snippet: Optional[str] = None
    more_results_link: Optional[str] = ""


class RelatedSearch(BaseModel):
    """Related search from SerpAPI"""

    model_config = ConfigDict(extra="allow")

    query: Optional[str] = None
    link: Optional[str] = None
    serpapi_link: Optional[str] = None
    # Additional fields that might be in the response
    block_position: Optional[int] = None


class SerpAPIResponse(BaseToolResponse):
    """Complete response from SerpAPI"""

    model_config = ConfigDict(extra="allow")

    search_metadata: SearchMetadata
    search_parameters: SearchParameters
    search_information: SearchInformation
    organic_results: List[OrganicResult] = Field(default_factory=list)
    related_questions: Optional[List[RelatedQuestion]] = None
    related_searches: Optional[List[RelatedSearch]] = None
    serpapi_pagination: Optional[Dict[str, Any]] = None
    formatted_results: Optional[str] = Field(
        default="", description="Formatted results for display"
    )
    extraction_stats: Optional[Dict[str, Any]] = Field(
        None, description="Statistics about the extraction process"
    )


class ExtractionConfig(BaseModel):
    """Configuration for extraction behavior"""

    max_parallel_extractions: int = Field(
        default=3, description="Max parallel web extractions"
    )
    extraction_timeout: float = Field(
        default=10.0, description="Timeout for each extraction"
    )
    retry_on_failure: bool = Field(
        default=True, description="Retry failed extractions"
    )
    max_retries: int = Field(default=2, description="Max retry attempts")
    always_extract_top_n: int = Field(
        default=0,
        description="Always extract top N results regardless of analysis",
    )
    use_cached_content: bool = Field(
        default=True, description="Use cached extracted content"
    )
    cache_ttl_seconds: int = Field(
        default=3600, description="Cache TTL in seconds"
    )


class SerpAPITool(BaseTool):
    """Tool for performing SerpAPI internet searches with intelligent extraction"""

    def __init__(self, extraction_config: Optional[ExtractionConfig] = None):
        super().__init__()
        self.name = "serpapi_internet_search"
        self.description = (
            "Search the internet for current information using Google. May be"
            " used in conjunction with the weather or news tools. Required for"
            " queries which cannot be answered using first principles, logic"
            " or well-known facts. Query MUST be in the form of a question for"
            " best results. Returns search parameters and top organic results"
            " with intelligently extracted webpage content (or snippet if"
            " extraction not needed).When helpful, provide the results links"
            " in markdown format.NEVER make up or guess URLs."
        )
        self.extraction_config = extraction_config or ExtractionConfig()
        self._content_cache = {}  # Simple in-memory cache

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
                                "Examples: 'How does photosynthesis work?', "
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
        self,
        results: List[OrganicResult],
        top_n: int = 3,
        user_query: str = "",
    ) -> List[OrganicResult]:
        """
        Return top_n organic results with extracted content.
        Uses intelligent LLM-based decision making to determine if web
        extraction is needed. Implements parallel processing for efficiency.
        Deduplicates results by URL to avoid redundant extractions.

        Args:
            results: List of organic search results
            top_n: Number of top results to process
            user_query: The original user query for context

        Returns:
            List containing top_n results with extracted content
        """
        if not results:
            logger.warning("No organic results found")
            return []

        # Get top_n results
        top_results = results[:top_n]

        # Deduplicate by URL while preserving order and merging snippets
        url_to_results = {}  # Map URL to list of results with that URL
        deduplicated_results = []

        for result in top_results:
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

        top_results = deduplicated_results

        logger.info(
            "Processing %d unique organic results with intelligent extraction",
            len(top_results),
        )

        # Initialize all results with snippet as extracted_content
        # This ensures we always have content to return
        for result in top_results:
            result.extracted_content = result.snippet
            result.extraction_method = "snippet"

        # If no user query provided, skip intelligent analysis and use snippets
        if not user_query:
            logger.info(
                "No user query provided, using search snippets for all %d"
                " results",
                len(top_results),
            )
            return top_results

        # Note: The implementation uses synchronous processing even in async contexts
        # to avoid complexity and potential deadlocks

        # Use thread-based parallel processing for synchronous context
        logger.info(
            "Using thread-based parallel processing for %d results",
            len(top_results),
        )

        # First, analyze all snippets in parallel
        analysis_results = self._analyze_snippets_parallel(
            top_results, user_query
        )

        # Collect results that need extraction
        extraction_candidates = []
        for i, (result, (needs_extraction, _)) in enumerate(
            zip(top_results, analysis_results)
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

        return top_results

    def _analyze_snippets_parallel(
        self, results: List[OrganicResult], user_query: str
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
                )
                futures.append(future)

            analysis_results = []
            for future in futures:
                try:
                    result = future.result(timeout=30.0)
                    analysis_results.append(result)
                except Exception as e:
                    logger.error("Snippet analysis failed: %s", e)
                    # Default to extraction on failure
                    analysis_results.append((True, ""))

            return analysis_results

    def _extract_content_parallel(
        self, extraction_candidates: List[Tuple[int, OrganicResult]]
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
                        # Share the extracted content among all results with this URL
                        for idx, result in candidates:
                            result.extracted_content = extracted_content
                            result.extraction_method = "extracted"
                            logger.info(
                                "Successfully extracted content for result"
                                " %d: %s",
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
                                "Extraction returned empty for result %d,"
                                " keeping snippet",
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
        self, snippet: str, user_query: str, url: str
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
                        snippet, user_query, url
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
                logger.error(
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
        self, snippet: str, user_query: str, url: str
    ) -> Tuple[bool, str]:
        """
        Enhanced snippet analysis.

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
You are analyzing search result snippets to determine if they contain sufficient information.

Task: Determine if the provided snippet fully answers the user's question.

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
- Important details are missing
- The snippet references information not shown

Set needs_extraction to false if:
- The snippet completely answers the question
- No additional information is needed

Important: Respond ONLY with the JSON object, no other text."""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User Question: {user_query}\n\nSearch Result"
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
                    "Failed to parse JSON from LLM response: %s. Raw"
                    " content: %s",
                    json_err,
                    raw_content[:200],  # Log first 200 chars
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
                "Snippet analysis for '%s': needs_extraction=%s",
                url[:80] + "..." if len(url) > 80 else url,
                needs_extraction,
            )

            return (needs_extraction, "")

        except Exception as e:
            logger.error(
                "Failed to analyze snippet for %s: %s. Defaulting to"
                " extraction.",
                url,
                e,
            )
            # If analysis fails, default to extraction to be safe
            return (True, "")

    async def _analyze_snippet_completeness(
        self, snippet: str, user_query: str, url: str
    ) -> bool:
        """
        Original snippet analysis for backward compatibility.
        """
        needs_extraction, _ = (
            await self._analyze_snippet_completeness_enhanced(
                snippet, user_query, url
            )
        )
        return (
            not needs_extraction
        )  # Invert because original returns True if sufficient

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
                "but_why": 5,  # High confidence
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
                logger.error(
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
            logger.error("Error extracting content from %s: %s", url, e)
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

    def format_results(self, results: List[OrganicResult]) -> str:
        """
        Format search results for display as numbered markdown list
        with extraction metadata.

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

            # Build the entry with metadata
            entry = f"{i}. [{result.title}]({result.link}) - "

            # Check if snippet contains merged content
            if "\n\n" in result.snippet:
                snippet_parts = result.snippet.split("\n\n")
                entry += f"[Merged {len(snippet_parts)} snippets] "
                # Show first snippet only in summary
                entry += f"{clean_content(snippet_parts[0])}\n\n"
            else:
                entry += f"{clean_snippet}\n\n"

            formatted_entries.append(entry)

        return "\n".join(formatted_entries)

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

    def search_serpapi(
        self, query: str, location_requested: str, top_n: int = 3, **kwargs
    ) -> SerpAPIResponse:
        """
        Search using SerpAPI and return top_n organic results with
        intelligently extracted content.

        Args:
            query (str): The search query
            location_requested (str): Location for the search
            top_n (int): Number of top results to process
            **kwargs: Additional search parameters to override defaults

        Returns:
            SerpAPIResponse: The search results with top_n results
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
            "engine": "google",
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

            # Log the raw response for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Raw SerpAPI response keys: %s", list(response_data.keys())
                )
                # Log related_searches structure if present
                if "related_searches" in response_data:
                    related = response_data["related_searches"]
                    logger.debug(
                        "Related searches structure: %s",
                        related[:1] if related else "Empty",
                    )

            # Parse the JSON response and validate with Pydantic
            serpapi_response = SerpAPIResponse(**response_data)

            # Track extraction statistics
            start_time = time.time()

            # Extract content for top_n results with intelligent decision making
            serpapi_response.organic_results = self._extract_top_results(
                serpapi_response.organic_results, top_n=top_n, user_query=query
            )

            # Calculate extraction statistics
            extraction_time = time.time() - start_time
            extraction_stats = {
                "total_results": len(serpapi_response.organic_results),
                "extraction_time_seconds": round(extraction_time, 2),
                "extraction_methods": {},
            }

            for result in serpapi_response.organic_results:
                method = result.extraction_method or "unknown"
                extraction_stats["extraction_methods"][method] = (
                    extraction_stats["extraction_methods"].get(method, 0) + 1
                )

            serpapi_response.extraction_stats = extraction_stats

            # Format the results for display
            serpapi_response.formatted_results = self.format_results(
                serpapi_response.organic_results
            )

            logger.info(
                "Search completed successfully. "
                "Returning %d results with content (extraction took %.2fs)",
                len(serpapi_response.organic_results),
                extraction_time,
            )

            # Log extraction statistics
            logger.info(
                "Extraction methods used: %s",
                extraction_stats["extraction_methods"],
            )

            return serpapi_response

        except KeyError as e:
            logger.error("Missing key in SerpAPI response: %s", e)
            raise ValueError(
                f"Invalid SerpAPI response format: {str(e)}"
            ) from e
        except ValueError as e:
            logger.error("Value error during SerpAPI search: %s", e)
            # Log additional context if it's a Pydantic validation error
            if "validation error" in str(e).lower():
                logger.error(
                    "This may be due to unexpected API response structure. "
                    "Enable debug logging to see the raw response."
                )
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
                   - 'top_n': Number of top results to process (optional)

        Returns:
            SerpAPIResponse: The search results in a validated Pydantic model
        """
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )

        query = params["query"]
        top_n = params.get("top_n", 3)

        # Use location_requested if provided and not empty, otherwise default
        location_requested = params.get("location_requested", "").strip()
        if not location_requested:
            location_requested = "Saline, Michigan, United States"
            logger.debug(
                "No location provided, using default: '%s'", location_requested
            )
        else:
            logger.debug("Using provided location: '%s'", location_requested)

        logger.debug(
            "run_with_dict method called with query: '%s', top_n: %d",
            query,
            top_n,
        )
        return self.search_serpapi(
            query, location_requested=location_requested, top_n=top_n
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
