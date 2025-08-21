"""
Document Analyzer Service

This service handles document analysis operations, particularly for PDFs.
Extracted from the monolithic AssistantTool to separate concerns.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from services.llm_client_service import llm_client_service
from utils.batch_processor import DocumentProcessor
from utils.config import config as app_config

logger = logging.getLogger(__name__)


class DocumentAnalyzerService:
    """Service for analyzing documents, especially PDFs"""

    def __init__(self, config: ChatConfig, llm_type: str):
        """
        Initialize document analyzer service

        Args:
            config: Chat configuration
            llm_type: Type of LLM to use
        """
        self.config = config
        self.llm_type = llm_type

    async def analyze_document(
        self,
        document_text: str,
        instructions: str,
        document_type: str = "document",
        filename: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Analyze a document with specific instructions

        Args:
            document_text: The document text to analyze
            instructions: Analysis instructions or questions
            document_type: Type of document (pdf, document, etc.)
            filename: Optional filename for context

        Returns:
            Analysis result dictionary
        """
        try:
            # Check if the document is too large for direct processing
            estimated_tokens = (
                len(document_text) // 4
            )  # Rough token estimation
            max_tokens = (
                126000  # Conservative limit to stay well under model limits
            )

            if estimated_tokens > max_tokens:
                logger.warning(
                    f"Document too large ({estimated_tokens} estimated "
                    f"tokens), processing in chunks"
                )
                return await self._analyze_large_document_chunked(
                    document_text, instructions, document_type, filename
                )

            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Get system prompt from centralized configuration
            from tools.tool_llm_config import get_tool_system_prompt

            base_prompt = get_tool_system_prompt("document_analysis", "")
            # Interpolate document type into the prompt
            system_prompt = base_prompt.replace(
                "a document", f"a {document_type}"
            )

            if filename:
                user_message = (
                    f"Based on the document '{filename}', please answer "
                    f"this question: {instructions}\n\nDocument:\n"
                    f"{document_text}"
                )
            else:
                user_message = (
                    f"Please analyze the following document and answer "
                    f"this question: {instructions}\n\nDocument:\n"
                    f"{document_text}"
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.info(f"Analyzing document with {model_name}")

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            return {
                "success": True,
                "result": result,
                "processing_notes": (
                    f"Document analysis completed for query: "
                    f"{instructions[:100]}"
                    f"{'...' if len(instructions) > 100 else ''}"
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"success": False, "error": str(e)}

    async def analyze_document_streaming(
        self,
        document_text: str,
        instructions: str,
        document_type: str = "document",
        filename: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Analyze a document using LLM with streaming response

        Args:
            document_text: The document text to analyze
            instructions: Analysis instructions or questions
            document_type: Type of document (default: "document")
            filename: Optional filename for context

        Returns:
            Dictionary with analysis results
        """
        from utils.text_processing import StreamingThinkTagFilter

        try:
            # Check if the document is too large
            estimated_tokens = len(document_text) // 4
            max_tokens = 50000  # Conservative limit for analysis

            if estimated_tokens > max_tokens:
                logger.warning(
                    f"Document too large ({estimated_tokens} estimated "
                    f"tokens), analyzing in chunks"
                )
                return await self._analyze_large_document_chunked(
                    document_text, instructions, document_type, filename
                )

            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Get system prompt from centralized configuration
            from tools.tool_llm_config import get_tool_system_prompt

            base_prompt = get_tool_system_prompt("document_analysis", "")
            # Interpolate document type into the prompt
            system_prompt = base_prompt.replace(
                "a document", f"a {document_type}"
            )

            if filename:
                user_message = (
                    f"Based on the document '{filename}', please answer "
                    f"this question: {instructions}\n\nDocument:\n"
                    f"{document_text}"
                )
            else:
                user_message = (
                    f"Please analyze the following document and answer "
                    f"this question: {instructions}\n\nDocument:\n"
                    f"{document_text}"
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            logger.info(f"Analyzing document with streaming {model_name}")

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
                stream=True,  # Enable streaming
            )

            # Create think tag filter for streaming
            think_filter = StreamingThinkTagFilter()
            collected_result = ""

            # Process stream with think tag filtering
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    # Filter think tags from the chunk
                    filtered_content = think_filter.process_chunk(
                        chunk_content
                    )
                    if filtered_content:
                        collected_result += filtered_content

            # Get any remaining content from the filter
            final_content = think_filter.flush()
            if final_content:
                collected_result += final_content

            return {
                "success": True,
                "result": collected_result,
                "processing_notes": (
                    f"Document analysis completed for query: "
                    f"{instructions[:100]}"
                    f"{'...' if len(instructions) > 100 else ''}"
                ),
            }

        except Exception as e:
            logger.error(f"Error analyzing document with streaming: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_large_document_chunked(
        self,
        document_text: str,
        instructions: str,
        document_type: str,
        filename: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Analyze a large document by splitting it into chunks and processing
        hierarchically

        Args:
            document_text: The document text to analyze
            instructions: Analysis instructions or questions
            document_type: Type of document
            filename: Optional filename for context

        Returns:
            Analysis result dictionary
        """
        try:
            # Split document into chunks (approximately 80K characters each)
            chunk_size = 80000
            chunks = []

            for i in range(0, len(document_text), chunk_size):
                chunk = document_text[i : i + chunk_size]
                chunks.append(chunk)

            logger.info(f"Processing large document in {len(chunks)} chunks")

            # Process all chunks concurrently
            async def process_chunk_async(i: int, chunk: str):
                try:
                    chunk_instructions = (
                        f"{instructions} "
                        f"(Processing section {i+1} of {len(chunks)})"
                    )
                    chunk_result = await self._analyze_single_chunk(
                        chunk, chunk_instructions, document_type, filename
                    )
                    if chunk_result["success"]:
                        return chunk_result["result"]
                    else:
                        return (
                            f"Section {i+1} processing failed: "
                            f"{chunk_result.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    return (
                        f"Section {i+1} processing failed due to error: "
                        f"{str(e)}"
                    )

            # Run all chunk processing tasks concurrently
            chunk_results_raw = await asyncio.gather(
                *[
                    process_chunk_async(i, chunk)
                    for i, chunk in enumerate(chunks)
                ],
                return_exceptions=True,
            )

            # Handle any exceptions in results
            chunk_results = []
            for i, result in enumerate(chunk_results_raw):
                if isinstance(result, Exception):
                    logger.error(
                        f"Chunk {i+1} failed with exception: {result}"
                    )
                    chunk_results.append(
                        f"Section {i+1} processing failed due to error: {str(result)}"
                    )
                else:
                    chunk_results.append(result)

            if not chunk_results:
                return {
                    "success": False,
                    "error": "No content could be processed from the document.",
                }

            # Combine chunk results
            if len(chunk_results) > 1:
                combined_text = "\n\n---\n\n".join(chunk_results)

                # Create final synthesis
                synthesis_instructions = (
                    f"Based on these analysis sections, provide a direct answer to: {instructions}. "
                    f"Combine all relevant information into a concise response."
                )

                return await self._analyze_single_chunk(
                    combined_text,
                    synthesis_instructions,
                    "analysis results",
                    filename,
                )
            else:
                return {"success": True, "result": chunk_results[0]}

        except Exception as e:
            logger.error(f"Error in chunked document analysis: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_single_chunk(
        self,
        chunk_text: str,
        instructions: str,
        document_type: str,
        filename: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Analyze a single chunk of text

        Args:
            chunk_text: The text chunk to analyze
            instructions: Analysis instructions
            document_type: Type of document
            filename: Optional filename for context

        Returns:
            Analysis result dictionary
        """
        try:
            client = llm_client_service.get_async_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            # Get system prompt from centralized configuration
            from tools.tool_llm_config import get_tool_system_prompt

            base_prompt = get_tool_system_prompt("document_analysis", "")
            # Interpolate document type into the prompt
            system_prompt = base_prompt.replace(
                "a document", f"a {document_type}"
            )

            if filename:
                user_message = f"Based on the document '{filename}', please answer this question: {instructions}\n\nDocument:\n{chunk_text}"
            else:
                user_message = f"Please analyze the following document and answer this question: {instructions}\n\nDocument:\n{chunk_text}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=app_config.llm.DEFAULT_TEMPERATURE,
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            return {
                "success": True,
                "result": result,
                "processing_notes": f"Chunk analysis completed for query: {instructions[:100]}{'...' if len(instructions) > 100 else ''}",
            }

        except Exception as e:
            logger.error(f"Error analyzing chunk: {e}")
            return {"success": False, "error": str(e)}

    async def analyze_pdf_pages(
        self, pages: List[Dict[str, Any]], instructions: str, filename: str
    ) -> Dict[str, any]:
        """
        Analyze PDF pages with intelligent routing based on document size

        Args:
            pages: List of page dictionaries with 'page' and 'text' keys
            instructions: Analysis instructions
            filename: PDF filename

        Returns:
            Analysis result dictionary
        """
        total_pages = len(pages)

        if total_pages == 0:
            return {
                "success": False,
                "error": "The document appears to be empty or contains no extractable text.",
            }

        # Categorize document and route appropriately
        doc_size = DocumentProcessor.categorize_document_size(total_pages)

        logger.info(
            f"Analyzing {doc_size} document '{filename}' with {total_pages} pages"
        )

        try:
            if doc_size == "small":
                return await self._analyze_small_document(
                    pages, instructions, filename
                )
            elif doc_size == "medium":
                return await self._analyze_medium_document(
                    pages, instructions, filename
                )
            else:  # large
                return await self._analyze_large_document(
                    pages, instructions, filename
                )
        except Exception as e:
            logger.error(f"Error in PDF analysis: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_small_document(
        self, pages: List[Dict[str, Any]], instructions: str, filename: str
    ) -> Dict[str, any]:
        """Analyze small documents in a single pass"""

        full_text = DocumentProcessor.format_pages_for_analysis(pages)
        return await self.analyze_document(
            full_text, instructions, "PDF", filename
        )

    async def _analyze_medium_document(
        self, pages: List[Dict[str, Any]], instructions: str, filename: str
    ) -> Dict[str, any]:
        """Analyze medium documents in batches"""

        # Process in batches of 3-5 pages
        batch_size = max(3, len(pages) // 3)

        # Create async function for processing each batch
        async def process_batch_async(
            i: int, batch_pages: List[Dict[str, Any]]
        ):
            # Determine actual page numbers for this batch
            if batch_pages:
                start_page_num = batch_pages[0].get('page', i + 1)
                end_page_num = batch_pages[-1].get(
                    'page', min(i + batch_size, len(pages))
                )
            else:
                start_page_num = i + 1
                end_page_num = min(i + batch_size, len(pages))

            batch_text = DocumentProcessor.format_pages_for_analysis(
                batch_pages
            )

            batch_instruction = (
                f"Analyze pages {start_page_num}-{end_page_num} of '{filename}' for this question: {instructions}. "
                f"If relevant information is found, provide it with page numbers. "
                f"If not relevant, say 'No relevant information found in these pages.'"
            )

            result = await self.analyze_document(
                batch_text, batch_instruction, "PDF", filename
            )

            if result["success"]:
                return {
                    "page_range": f"{start_page_num}-{end_page_num}",
                    "analysis": result["result"],
                }
            return None

        # Create all batches
        batches = [
            (i, pages[i : i + batch_size])
            for i in range(0, len(pages), batch_size)
        ]

        # Run all batch processing tasks concurrently
        batch_results_raw = await asyncio.gather(
            *[
                process_batch_async(i, batch_pages)
                for i, batch_pages in batches
            ],
            return_exceptions=True,
        )

        # Filter out None results and exceptions
        batch_results = []
        for result in batch_results_raw:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            elif result is not None:
                batch_results.append(result)

        # Synthesize results
        return await self._synthesize_batch_results(
            batch_results, instructions, filename
        )

    async def _analyze_large_document(
        self, pages: List[Dict[str, Any]], instructions: str, filename: str
    ) -> Dict[str, any]:
        """Analyze large documents relevantly by processing ALL pages"""

        logger.info(
            f"Starting relevant analysis of all {len(pages)} pages for '{filename}'"
        )

        # Process ALL pages in batches for relevant analysis
        # Use medium document approach but with larger batches for efficiency
        batch_size = max(20, len(pages) // 10)

        # Create async function for processing each large batch
        async def process_large_batch_async(
            i: int, batch_pages: List[Dict[str, Any]]
        ):
            # Determine actual page numbers for this batch
            batch_end = min(i + batch_size, len(pages))
            if batch_pages:
                start_page_num = batch_pages[0].get('page', i + 1)
                end_page_num = batch_pages[-1].get('page', batch_end)
            else:
                start_page_num = i + 1
                end_page_num = batch_end

            batch_text = DocumentProcessor.format_pages_for_analysis(
                batch_pages
            )

            batch_instruction = (
                f"You are analyzing a chunk of a larger document. This chunk covers pages {start_page_num}-{end_page_num} of '{filename}'. "
                f"The user's overall question is: {instructions}. "
                f"Provide a detailed analysis of any information in this chunk that is relevant to the user's question. "
                f"If no relevant information is found, explicitly state that this section was reviewed but contained no relevant content. "
                f"Cite page numbers for any specific findings at the end of your response."
            )

            result = await self.analyze_document(
                batch_text, batch_instruction, "PDF", filename
            )

            if result["success"]:
                return {
                    "page_range": f"{start_page_num}-{end_page_num}",
                    "analysis": result["result"],
                }
            return None

        # Create all large batches
        large_batches = [
            (i, pages[i : i + batch_size])
            for i in range(0, len(pages), batch_size)
        ]

        # Run all large batch processing tasks concurrently
        batch_results_raw = await asyncio.gather(
            *[
                process_large_batch_async(i, batch_pages)
                for i, batch_pages in large_batches
            ],
            return_exceptions=True,
        )

        # Filter out None results and exceptions
        batch_results = []
        for result in batch_results_raw:
            if isinstance(result, Exception):
                logger.error(f"Large batch processing failed: {result}")
            elif result is not None:
                batch_results.append(result)

        # Synthesize results from ALL batches
        return await self._synthesize_batch_results(
            batch_results, instructions, filename
        )

    async def _synthesize_batch_results(
        self,
        batch_results: List[Dict[str, str]],
        instructions: str,
        filename: str,
    ) -> Dict[str, any]:
        """Synthesize results from multiple batch analyses with hierarchical summarization"""

        if not batch_results:
            return {
                "success": False,
                "error": "No batch results to synthesize",
            }

        # If there are many results, create intermediate summaries first
        if len(batch_results) > 1:
            logger.info(
                f"Performing hierarchical summarization on {len(batch_results)} batch results."
            )
            intermediate_summaries = []
            intermediate_batch_size = 5

            # Create all the async tasks for intermediate summaries
            async def create_intermediate_summary(chunk):
                chunk_findings = "\n\n".join(
                    [
                        f"Analysis of pages {result['page_range']}:\n{result['analysis']}"
                        for result in chunk
                    ]
                )
                synthesis_instruction = (
                    f"Synthesize the following findings from a document analysis into a coherent intermediate summary. "
                    f"Focus only on the key points related to the user's query: {instructions}"
                )
                return await self.analyze_document(
                    chunk_findings,
                    synthesis_instruction,
                    "analysis results",
                    None,
                )

            # Create chunks and tasks
            chunks = [
                batch_results[i : i + intermediate_batch_size]
                for i in range(0, len(batch_results), intermediate_batch_size)
            ]

            # Run all intermediate summary tasks concurrently
            intermediate_results = await asyncio.gather(
                *[create_intermediate_summary(chunk) for chunk in chunks],
                return_exceptions=True,
            )

            # Process results
            for result in intermediate_results:
                if not isinstance(result, Exception) and result.get("success"):
                    intermediate_summaries.append(result["result"])

            # Combine intermediate summaries for the final synthesis
            combined_findings = "\n\n---\n\n".join(intermediate_summaries)
            synthesis_instruction = (
                f"Based on these intermediate summaries from '{filename}', provide a final concise answer to: {instructions}. "
                f"Citing page ranges where possible at the end of your response. Be sure to answer the user's question, not simply tell them where to look in the document."
            )
        else:
            # Format batch results for a single synthesis pass
            combined_findings = "\n\n".join(
                [
                    f"Analysis of pages {result['page_range']}:\n{result['analysis']}"
                    for result in batch_results
                ]
            )
            synthesis_instruction = (
                f"Based on these analyses from '{filename}', provide a concise answer to: {instructions}. "
                f"Synthesize all relevant information into a cohesive response. Be concise and to the point."
            )

        return await self.analyze_document(
            combined_findings, synthesis_instruction, "analysis results", None
        )
