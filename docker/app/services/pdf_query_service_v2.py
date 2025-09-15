"""
PDFQueryService V2
------------------
Simpler wrapper around SimilaritySearch that is always used when a PDF is active.

Given a `pdf_id` and a natural-language query it:
1. Creates an embedding for the query (via EmbeddingCreator)
2. Searches Milvus `pdf_chunks` collection (filtered by `pdf_id`)
3. Returns top-k chunks with distance scores and formatted context string for prompt injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from tools.retriever import EmbeddingCreator, SearchConfig, SimilaritySearch
from utils.config import config as app_config

from utils.logging_config import get_logger

logger = get_logger(__name__)

_COLLECTION = "pdf_chunks"
# Use configured max chunks per query (default: 10)
MAX_PDF_RESULTS = app_config.file_processing.PDF_MAX_CHUNKS_PER_QUERY


@dataclass
class PDFSearchConfig(SearchConfig):
    """Configuration for PDF similarity search parameters"""

    def __init__(
        self,
        collection_name: str = _COLLECTION,
        uri: str = None,
        db_name: str = None,
        vector_field: str = "vector",
        radius: float = app_config.file_processing.PDF_SIMILARITY_THRESHOLD,  # L2 distance threshold (default: 3.0)
        range_filter: float = 0.001,  # Minimum distance filter
        topk: int = MAX_PDF_RESULTS,
        output_fields: List[str] = None,
    ):
        if output_fields is None:
            output_fields = ["id", "text", "metadata"]

        super().__init__(
            collection_name=collection_name,
            uri=uri or app_config.env.DATABASE_URL,
            db_name=db_name or app_config.env.DEFAULT_DB,
            vector_field=vector_field,
            radius=radius,
            range_filter=range_filter,
            topk=topk,
            output_fields=output_fields,
        )


@dataclass
class PDFChunkMatch:
    chunk_id: str
    page_range: str
    text: str
    distance: float


class PDFQueryServiceV2:
    """Lightweight similarity search for PDF chunks."""

    def __init__(
        self,
        config: ChatConfig | None = None,
        search_config: PDFSearchConfig | None = None,
        milvus_uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ):
        self.config = config or ChatConfig.from_environment()

        # Use provided search config or create default one
        self.search_config = search_config or PDFSearchConfig(
            uri=milvus_uri,
            db_name=db_name,
        )

        self.embedding_creator = EmbeddingCreator(
            base_url=app_config.env.EMBEDDING_ENDPOINT,
            api_key=app_config.env.EMBEDDING_API_KEY,
            model=app_config.env.EMBEDDING_MODEL,
        )

        # Initialize SimilaritySearch with the same pattern as retriever.py
        self.milvus = SimilaritySearch(
            collection_name=self.search_config.collection_name,
            uri=self.search_config.uri,
            db_name=self.search_config.db_name,
            vector_field=self.search_config.vector_field,
            output_fields=self.search_config.output_fields,
        )

    def query(
        self,
        pdf_id: str,
        query: str,
        top_k: Optional[int] = None,
        use_search_config: bool = True,
        pdf_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return best matching chunks for a query.

        Args:
            pdf_id: The PDF identifier to filter results
            query: The search query string
            top_k: Override for number of results (uses search_config.topk if None)
            use_search_config: Whether to use the configured search parameters
            pdf_filename: Optional filename for display (will try to extract from metadata if not provided)

        Returns:
            Dict containing chunks, used flag, formatted context, and tool_response
        """
        # Use configured topk or override
        limit = top_k if top_k is not None else self.search_config.topk

        # Create query embedding
        embedding_response = self.embedding_creator.create_formatted_query(
            query
        )
        logger.debug("Generated embedding for query")

        # Update search config to filter by pdf_id
        # Note: Milvus doesn't support JSON field filtering directly in community edition
        # So we'll search all and filter manually (keeping existing behavior)
        search_results = self.milvus.search(embedding_response)

        if not search_results or not search_results[0]:
            logger.info("No search results found for query")
            return {"chunks": [], "used": False, "formatted_context": ""}

        # Process results - filter by pdf_id and convert to PDFChunkMatch
        matches: List[PDFChunkMatch] = []
        for result in search_results[0]:
            # Extract entity data
            entity = result.get("entity", result)

            # Get metadata and check pdf_id
            metadata_str = entity.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
                if metadata.get("pdf_id") != pdf_id:
                    continue

                # Extract page range from metadata
                pages = metadata.get("pages", [])
                if pages:
                    page_range = (
                        f"{pages[0]}-{pages[-1]}"
                        if len(pages) > 1
                        else str(pages[0])
                    )
                else:
                    page_range = "unknown"

                # Create match object
                matches.append(
                    PDFChunkMatch(
                        chunk_id=str(entity.get("id", "")),
                        page_range=page_range,
                        text=entity.get("text", ""),
                        distance=result.get("distance", 0.0),
                    )
                )

                if len(matches) >= limit:
                    break

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse metadata: {e}")
                continue

        logger.info(
            "PDFQueryService: found %d matches for pdf_id '%s'",
            len(matches),
            pdf_id,
        )

        # Determine filename - use provided or try to extract from metadata
        if not pdf_filename and matches:
            try:
                # Get metadata from first match
                first_match_entity = (
                    search_results[0][0]
                    if search_results and search_results[0]
                    else None
                )
                if first_match_entity:
                    metadata_str = first_match_entity.get("entity", {}).get(
                        "metadata", "{}"
                    )
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    pdf_filename = metadata.get("filename", "Unknown PDF")
                else:
                    pdf_filename = "Unknown PDF"
            except Exception as e:
                logger.debug(f"Could not extract filename from metadata: {e}")
                pdf_filename = "Unknown PDF"
        elif not pdf_filename:
            pdf_filename = "Unknown PDF"

        # Count unique chunks (deduplication happens in _format_context)
        seen_texts = set()
        unique_count = 0
        for match in matches:
            if match.text not in seen_texts:
                seen_texts.add(match.text)
                unique_count += 1

        return {
            "chunks": matches,
            "used": bool(matches),
            "formatted_context": self._format_context(matches),
            "tool_response": self.format_as_tool_response(
                matches, pdf_filename
            ),
            "unique_chunks": unique_count,
        }

    # ------------------------------------------------------------------
    def _format_context(self, matches: List[PDFChunkMatch]) -> str:
        """Format PDF chunks for tool injection into LLM conversation."""
        if not matches:
            return ""

        formatted_entries = []
        seen_texts = set()

        # Track unique chunks while maintaining order
        unique_matches = []
        for match in matches:
            # Only add if text hasn't been seen before
            if match.text not in seen_texts:
                seen_texts.add(match.text)
                unique_matches.append(match)

        # Format unique chunks with proper numbering
        for i, match in enumerate(unique_matches, 1):
            entry = self._format_single_match(i, match, len(unique_matches))
            formatted_entries.append(entry)

        # Join all entries with double newlines
        return "\n\n".join(formatted_entries)

    def _format_single_match(
        self, index: int, match: PDFChunkMatch, total_matches: int
    ) -> str:
        """Format a single PDF chunk match."""
        # Clean text but NEVER truncate - data integrity is critical
        text = match.text.strip()

        # Format with clear chunk indicators
        formatted = (
            f"<small>{index}. [Page {match.page_range}], "
            f"_'{text}'_, "
            f"(relevance: {1.0 - match.distance:.3f})</small>"
        )

        return formatted

    def format_as_tool_response(
        self, matches: List[PDFChunkMatch], pdf_filename: str
    ) -> Dict[str, Any]:
        """Format PDF search results as a tool response for LLM injection."""
        if not matches:
            return {
                "role": "tool",
                "content": "No relevant PDF content found for the query.",
                "metadata": {
                    "tool_name": "pdf_search",
                    "pdf_filename": pdf_filename,
                    "chunks_found": 0,
                },
            }

        # Format the context using the same style as retriever
        formatted_content = self._format_context(matches)

        # Create tool response structure
        tool_response = {
            "role": "tool",
            "content": (
                "PDF Search Results from"
                f" '{pdf_filename}':\n\n{formatted_content}"
            ),
            "metadata": {
                "tool_name": "pdf_search",
                "pdf_filename": pdf_filename,
                "chunks_found": len(matches),
                "pages_covered": list(set(m.page_range for m in matches)),
            },
        }

        return tool_response
