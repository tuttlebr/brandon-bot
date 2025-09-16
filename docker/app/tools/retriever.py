import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from pymilvus import MilvusClient
from tools.base import BaseTool, BaseToolResponse
from utils.config import config

# Configure logger
from utils.logging_config import get_logger
from utils.text_processing import strip_all_thinking_formats

logger = get_logger(__name__)

MAX_RESULTS = 25


@dataclass
class SearchConfig:
    """Configuration for similarity search parameters"""

    collection_name: str
    uri: str
    db_name: str
    vector_field: str = "embedding"
    radius: float = 1.6
    range_filter: float = 0.001
    topk: int = MAX_RESULTS
    output_fields: List[str] = None

    def __post_init__(self):
        if self.output_fields is None:
            self.output_fields = [
                "reference_id",
                "title",
                "source",
                "text",
                "creation_date",
            ]


class EmbeddingCreator:
    """Handles creation of embeddings using OpenAI API"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        self.embed_model = OpenAI(api_key=api_key, base_url=base_url)

    def create_query(self, input_text: str) -> Dict[str, Any]:
        """Create embedding for a query"""
        return self.embed_model.embeddings.create(
            input=input_text,
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )

    def create_formatted_query(self, input_text: str) -> List[float]:
        """Create and format query embedding"""
        embedding_info = self.create_query(input_text)
        return [embedding_info.data[0].embedding]


class SimilaritySearch:
    """Handles vector similarity search operations"""

    def __init__(
        self,
        collection_name: str,
        uri: str,
        db_name: str,
        vector_field: str = "embedding",
        output_fields: List[str] = None,
    ):
        self.config = SearchConfig(
            collection_name=collection_name,
            uri=uri,
            db_name=db_name,
            vector_field=vector_field,
            output_fields=output_fields,
        )
        self.search_params = self._initialize_search_params()
        self.client = self._initialize_client()

    def _initialize_search_params(self) -> Dict[str, Any]:
        """Initialize search parameters"""
        return {
            "metric_type": "L2",
            "index_type": "FLAT",
            "params": {
                "radius": self.config.radius,
                "range_filter": self.config.range_filter,
            },
        }

    def _initialize_client(self) -> MilvusClient:
        """Initialize and configure Milvus client"""
        client = MilvusClient(
            collection_name=self.config.collection_name,
            uri=self.config.uri,
            vector_field=self.config.vector_field,
            db_name=self.config.db_name,
            overwrite=False,
        )
        client.load_collection(collection_name=self.config.collection_name)
        return client

    def search(self, data: List[float]) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        results = self.client.search(
            data=data,
            limit=self.config.topk,
            collection_name=self.config.collection_name,
            search_params=self.search_params,
            output_fields=self.config.output_fields,
        )
        logging.debug("Vector search results: %s", results[0])

        return results

    def reranker(
        self, query: str, embedding_response: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Rerank search results using external reranker service"""
        payload = self._prepare_reranker_payload(
            query, embedding_response, config.env.RERANKER_MODEL
        )

        if not payload["passages"]:
            return None

        try:
            reranker_response = self._call_reranker_service(
                payload,
                config.env.RERANKER_ENDPOINT,
                config.env.RERANKER_API_KEY,
            )
            combined_results = self._combine_results(
                embedding_response, reranker_response
            )

            # Check if we have valid results before trying to remove outliers
            if not combined_results or not combined_results[0]:
                logging.warning(
                    "No valid results after combining with reranker response"
                )
                return None

            # Check if results have logit values
            results_with_logits = [
                r
                for r in combined_results[0]
                if "logit" in r and r["logit"] is not None
            ]
            if not results_with_logits:
                logging.warning(
                    "No results with valid logit scores after reranking"
                )
                return None

            validated_results = self._remove_outliers(
                results_with_logits, key="logit"
            )
            return [validated_results]
        except Exception as e:
            logging.error("Reranking failed: %s", str(e))
            return None

    def _prepare_reranker_payload(
        self, query: str, embedding_response: List[Dict[str, Any]], model: str
    ) -> Dict[str, Any]:
        """Prepare payload for reranker service"""
        return {
            "model": model,
            "query": {"text": query},
            "passages": [
                {"text": result["entity"]["text"]}
                for result in embedding_response[0]
                if result.get("entity", {}).get("text")
            ],
            "truncate": "END",
        }

    def _call_reranker_service(
        self, payload: Dict[str, Any], endpoint: str, api_key: str
    ) -> Dict[str, Any]:
        """Call reranker service with prepared payload"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }

        logging.debug("Payload: %s", payload)
        with requests.Session() as session:
            response = session.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            logging.info('HTTP Request: POST %s "HTTP/1.1 200 OK"', endpoint)
            return response.json()

    def _combine_results(
        self,
        embedding_response: List[Dict[str, Any]],
        reranker_response: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Combine embedding and reranker results"""
        # Check if reranker response has the expected structure
        if not reranker_response or "rankings" not in reranker_response:
            logging.warning("Reranker response missing 'rankings' field")
            return []

        # Update results with reranker scores
        for rank in reranker_response.get("rankings", []):
            index = rank.get("index")
            logit = rank.get("logit")

            # Validate index and logit
            if index is None or logit is None:
                logging.debug(
                    "Skipping ranking with missing index or logit: %s", rank
                )
                continue

            # Check if index is within bounds
            if index < 0 or index >= len(embedding_response[0]):
                logging.debug(
                    "Skipping ranking with out-of-bounds index: %s", index
                )
                continue

            embedding_response[0][index].update(
                {"logit": logit, "index": index}
            )

        rerankable_results = [
            result for result in embedding_response[0] if "logit" in result
        ]

        return [rerankable_results]

    def _remove_outliers(
        self,
        data: Union[List[float], List[Dict[str, Any]]],
        key: Optional[str] = None,
    ) -> Union[List[float], List[Dict[str, Any]]]:
        """
        Automatically removes outliers by finding natural breaks in the data distribution.
        For logit scores, lower values are considered more relevant and the lowest value is always kept.
        Finds the largest gap in sorted values to determine the natural cutoff point.
        'How many GPUs are in a single compute tray of the NVL72 GB200?'

        Parameters:
        data (Union[List[float], List[Dict[str, Any]]]): Either a list of scores or a list of dictionaries with scores
        key (str, optional): If data is a list of dictionaries, the key for the score values

        Returns:
        Union[List[float], List[Dict[str, Any]]]: Filtered list with outliers removed
        """
        # Input validation
        if not data:
            logging.error("Input data cannot be empty")
            return data

        # Debug logging
        logging.debug(
            "_remove_outliers called with %s items, key='%s'", len(data), key
        )
        if data and len(data) > 0:
            logging.debug("First item type: %s", type(data[0]))
            logging.debug("First item: %s", data[0])
            if key and isinstance(data[0], dict):
                logging.debug(
                    "First item has key '%s': %s", key, key in data[0]
                )
                if key in data[0]:
                    logging.info(
                        "Value for key '%s': %s (type: %s)",
                        key,
                        data[0][key],
                        type(data[0][key]),
                    )

        try:
            # Extract scores if data is a list of dictionaries or Pymilvus Hit objects
            if key is not None and (
                isinstance(data[0], dict) or hasattr(data[0], key)
            ):
                try:
                    # Extract values and validate they are numeric, keeping track of valid items
                    raw_scores = []
                    valid_items = []

                    for item in data:
                        # Handle both dictionaries and Pymilvus Hit objects
                        try:
                            if isinstance(item, dict):
                                if key not in item:
                                    logging.error(
                                        "Key '%s' not found in dict item: %s",
                                        key,
                                        item,
                                    )
                                    continue
                                value = item[key]
                            else:
                                # Handle Pymilvus Hit objects or other objects with attributes
                                if hasattr(item, key):
                                    value = getattr(item, key)
                                else:
                                    logging.error(
                                        "Attribute '%s' not found in item: %s",
                                        key,
                                        type(item),
                                    )
                                    continue

                            # Check if value is numeric (int, float) and not None/NaN
                            if value is None or (
                                isinstance(value, float) and np.isnan(value)
                            ):
                                logging.warning(
                                    "Skipping non-numeric value: %s", value
                                )
                                continue

                            if not isinstance(value, (int, float)):
                                logging.debug(
                                    "Skipping non-numeric value of type"
                                    " %s: %s",
                                    type(value),
                                    value,
                                )
                                continue

                            raw_scores.append(float(value))
                            valid_items.append(item)

                        except Exception as e:
                            logging.warning("Error processing item: %s", e)
                            continue

                    if not raw_scores:
                        logging.warning(
                            "No valid numeric values found in data"
                        )
                        return data

                    scores = np.array(raw_scores, dtype=np.float64)
                    original_data = (
                        valid_items  # Use only the items with valid scores
                    )

                except Exception as e:
                    logging.error("Error extracting scores from data: %s", e)
                    return data
            else:
                original_data = None
                try:
                    # Validate and clean raw data
                    cleaned_data = []
                    for value in data:
                        if value is None or (
                            isinstance(value, float) and np.isnan(value)
                        ):
                            logging.warning(
                                "Skipping non-numeric value: %s", value
                            )
                            continue
                        if not isinstance(value, (int, float)):
                            logging.debug(
                                "Skipping non-numeric value of type %s: %s",
                                type(value),
                                value,
                            )
                            continue
                        cleaned_data.append(float(value))

                    if not cleaned_data:
                        logging.error("No valid numeric values found in data")
                        return data

                    scores = np.array(cleaned_data, dtype=np.float64)
                except Exception as e:
                    logging.error("Error creating array from data: %s", e)
                    return data

            # Verify we have a valid numeric array
            if not np.issubdtype(scores.dtype, np.number):
                logging.error(
                    "All values must be numeric. Found non-numeric values in"
                    " data."
                )
                return data

            if len(scores) == 0:
                logging.error("No valid scores remain after filtering")
                return data

            original_count = len(scores)

            # If we only have 1-2 items, keep them all
            if original_count <= 2:
                logging.debug("Only 1-2 items, keeping all")
                return data

            # Sort scores to find natural breaks
            sorted_scores = np.sort(scores)

            # Find the largest gap between consecutive values
            gaps = []
            gap_positions = []

            for i in range(1, len(sorted_scores)):
                gap = sorted_scores[i] - sorted_scores[i - 1]
                gaps.append(gap)
                gap_positions.append(i)  # Position after which to cut

            # Find the largest gap
            max_gap_idx = np.argmax(gaps)
            max_gap_size = gaps[max_gap_idx]
            cut_position = gap_positions[max_gap_idx]

            # The cutoff is the value just before the largest gap
            cutoff = sorted_scores[cut_position - 1]

            # Always keep at least the lowest value (most relevant)
            min_score = np.min(scores)
            if cutoff < min_score:
                cutoff = min_score

            # Filter based on the cutoff
            if original_data is None:
                # Filter scores (keep values <= cutoff)
                filtered_scores = scores[scores <= cutoff]
                logging.info(
                    "Natural break filtering: kept %s/%s values, cutoff: %s,"
                    " largest gap: %s",
                    len(filtered_scores),
                    original_count,
                    cutoff,
                    max_gap_size,
                )
                return filtered_scores.tolist()
            else:
                # Filter dictionaries or Hit objects
                filtered_data = []
                for item in original_data:
                    try:
                        if isinstance(item, dict):
                            value = item[key]
                        else:
                            value = getattr(item, key)

                        if value <= cutoff:
                            filtered_data.append(item)
                    except Exception as e:
                        logging.warning("Error filtering item: %s", e)
                        continue

                logging.info(
                    "Natural break filtering: kept %s/%s items, cutoff: %s,"
                    " largest gap: %s",
                    len(filtered_data),
                    original_count,
                    cutoff,
                    max_gap_size,
                )
                return filtered_data

        except Exception as e:
            logging.error("Error in remove_outliers: %s", str(e))
            return data

    def format_results(self, results: Optional[List[Dict[str, Any]]]) -> str:
        """Format search results for display"""
        if not results:
            return ""

        formatted_entries = []
        seen_content = set()  # Track unique combinations of title + text
        duplicate_count = 0

        i = 1
        for idx, result in enumerate(results[0]):
            entity = result["entity"]
            # Create a unique key using both title and text to avoid false duplicates
            content_key = (entity.get("title", ""), entity.get("text", ""))

            if content_key in seen_content:
                duplicate_count += 1
                logger.debug(
                    "Skipping duplicate result %d: title='%s', text='%s...'",
                    idx,
                    entity.get("title", "")[:50],
                    entity.get("text", "")[:50],
                )
                continue

            seen_content.add(content_key)
            entry = self._format_single_result(i, entity)
            formatted_entries.append(entry)
            i += 1

        if duplicate_count > 0:
            logger.info(
                "Formatted %d unique results from %d total results (%d"
                " duplicates removed)",
                len(formatted_entries),
                len(results[0]),
                duplicate_count,
            )

        return "\n\n".join(formatted_entries)

    def _format_single_result(self, index: int, entity: Dict[str, Any]) -> str:
        """Format a single search result entry"""
        title = entity["title"].strip()
        text = entity["text"].replace(title, "").strip()
        # Strip think tags from text content before display
        text = strip_all_thinking_formats(text)
        source = entity["source"].strip()
        creation_date = entity["creation_date"].strip()
        base_text = (
            f"<small>{index}. [{title}]({source}), _'{text}'_, {creation_date}"
        )

        return f"{base_text}</small>"


class SearchResult(BaseModel):
    """Individual search result from vector similarity search"""

    reference_id: str = Field(description="Reference ID of the document")
    title: str = Field(description="Title of the document")
    source: str = Field(description="Source URL or location of the document")
    text: str = Field(description="Text content of the document")
    creation_date: str = Field(description="Creation date of the document")
    distance: Optional[float] = Field(
        None, description="Vector similarity distance"
    )
    logit: Optional[float] = Field(None, description="Reranker logit score")
    index: Optional[int] = Field(
        None, description="Original index in search results"
    )


class RetrievalResponse(BaseToolResponse):
    """Complete response from retrieval search"""

    query: str = Field(description="The original search query")
    results: List[SearchResult] = Field(
        default_factory=list, description="List of search results"
    )
    total_results: int = Field(description="Total number of results found")
    reranked: bool = Field(
        default=False, description="Whether results were reranked"
    )
    formatted_results: str = Field(
        default="", description="Formatted results for display"
    )


class RetrieverTool(BaseTool):
    """Tool for performing vector similarity search with optional reranking"""

    def __init__(self):
        super().__init__()
        self.name = "retrieval_search"
        self.description = (
            "NVIDIA-ONLY technical documentation knowledge base containing "
            "proprietary NVIDIA product information, GPU specifications, "
            "CUDA documentation, and internal technical resources. "
            "Use ONLY for questions specifically about NVIDIA hardware, "
            "software, or technologies. For general AI/ML questions or "
            "information about other companies' products (e.g., Anthropic, "
            "OpenAI, Google), use serpapi_internet_search instead."
        )

    def _initialize_mvc(self):
        """Initialize MVC components"""
        # Initialize embedding creator and similarity search using centralized config
        self.embedding_creator = EmbeddingCreator(
            base_url=config.env.EMBEDDING_ENDPOINT,
            api_key=config.env.EMBEDDING_API_KEY,
            model=config.env.EMBEDDING_MODEL,
        )

        self.similarity_search = SimilaritySearch(
            collection_name=config.env.COLLECTION_NAME,
            uri=config.env.DATABASE_URL,
            db_name=config.env.DEFAULT_DB,
            vector_field="embedding",
        )

        # This tool doesn't need separate MVC components
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
                                "User query to pass to the NVIDIA technical"
                                " documentation knowledge base. ONLY use this"
                                " for questions specifically about: NVIDIA"
                                " GPUs (e.g., H100, A100, RTX series), CUDA"
                                " programming, NVIDIA software (e.g.,"
                                " TensorRT, cuDNN), NVLink, DGX systems, or"
                                " other NVIDIA-specific technologies. DO NOT"
                                " use for questions about other companies'"
                                " products or general AI/ML topics."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def get_response_type(self) -> Type[RetrievalResponse]:
        """Get the response type for this tool"""
        return RetrievalResponse

    def execute(self, params: Dict[str, Any]):
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def search_documents(
        self, query: str, use_reranker: bool = True, max_results: int = 10
    ) -> RetrievalResponse:
        """
        Search documents using vector similarity with optional reranking

        Args:
            query: The search query string
            use_reranker: Whether to apply reranking to improve relevance
            max_results: Maximum number of results to return

        Returns:
            RetrievalResponse: The search results in a validated Pydantic model

        Raises:
            Exception: If the search operation fails
        """
        logger.info(f"Starting retrieval search for query: '{query}'")

        try:
            # Create query embedding
            logger.debug("Creating query embedding")
            query_embedding = self.embedding_creator.create_formatted_query(
                query
            )

            # Perform similarity search
            logger.debug("Performing vector similarity search")
            search_results = self.similarity_search.search(query_embedding)

            if not search_results or not search_results[0]:
                logger.warning("No search results found")
                return RetrievalResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    reranked=False,
                    formatted_results="",
                )

            # Apply reranking if requested
            reranked_results = None
            if use_reranker:
                logger.debug("Applying reranker to search results")
                reranked_results = self.similarity_search.reranker(
                    query, search_results
                )

            # Use reranked results if available, otherwise use original results
            final_results = (
                reranked_results if reranked_results else search_results
            )

            # Convert to Pydantic models
            results = []
            for result in final_results[0][:max_results]:
                entity = result["entity"]
                search_result = SearchResult(
                    reference_id=entity.get("reference_id", ""),
                    title=entity.get("title", ""),
                    source=entity.get("source", ""),
                    text=entity.get("text", ""),
                    creation_date=entity.get("creation_date", ""),
                    distance=result.get("distance"),
                    logit=result.get("logit"),
                    index=result.get("index"),
                )
                results.append(search_result)

            # Format results for display
            formatted_results = self.similarity_search.format_results(
                final_results
            )

            # Note: 'results' contains ALL results (including duplicates)
            # while 'formatted_results' shows only unique results for display
            response = RetrievalResponse(
                query=query,
                results=results,
                total_results=len(results),
                reranked=use_reranker and reranked_results is not None,
                formatted_results=formatted_results,
            )

            logger.info(
                f"Search completed successfully. Found {len(results)} results"
                f" (reranked: {response.reranked})"
            )
            return response

        except Exception as e:
            logger.error(f"Error during retrieval search: {e}")
            raise Exception(f"Retrieval search failed: {str(e)}")

    def run_with_dict(self, params: Dict[str, Any]) -> RetrievalResponse:
        """
        Execute a retrieval search with parameters provided as a dictionary.

        Args:
            params: Dictionary containing the required parameters
                   Expected keys: 'query', optionally 'use_reranker', 'max_results'

        Returns:
            RetrievalResponse: The search results in a validated Pydantic model
        """
        if "query" not in params:
            raise ValueError(
                "'query' key is required in parameters dictionary"
            )

        query = params["query"]
        use_reranker = params.get("use_reranker", True)
        max_results = params.get("max_results", MAX_RESULTS)

        logger.debug(
            "run_with_dict method called with query: '%s', use_reranker: %s,"
            " max_results: %s",
            query,
            use_reranker,
            max_results,
        )
        return self.search_documents(query, use_reranker, max_results)


# Helper functions for backward compatibility
def get_retrieval_tool_definition() -> Dict[str, Any]:
    """
    Get the OpenAI-compatible tool definition for retrieval search

    Returns:
        Dict containing the OpenAI tool definition
    """
    from tools.registry import get_tool, register_tool_class

    # Register the tool class if not already registered
    register_tool_class("retrieval_search", RetrieverTool)

    # Get the tool instance and return its definition
    tool = get_tool("retrieval_search")
    if tool:
        return tool.get_definition()
    else:
        raise RuntimeError("Failed to get retrieval tool definition")
