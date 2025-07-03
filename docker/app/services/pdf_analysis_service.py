"""
PDF Analysis Service

This service handles intelligent analysis of PDF documents to answer user questions
by processing pages in batches and combining relevant findings.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from models.chat_config import ChatConfig
from utils.batch_processor import BatchProcessor, DocumentProcessor
from utils.config import config
from utils.executor_pool import get_shared_executor
from utils.streamlit_context import run_with_streamlit_context

logger = logging.getLogger(__name__)


class PDFAnalysisService:
    """Service for intelligent PDF document analysis and Q&A"""

    def __init__(self, config_obj: ChatConfig):
        """
        Initialize the PDF analysis service

        Args:
            config_obj: Configuration for the service
        """
        self.config = config_obj
        self.batch_size = config.file_processing.PDF_SUMMARIZATION_BATCH_SIZE  # Reuse batch size
        self.executor = get_shared_executor()

    async def analyze_pdf_for_query(self, pdf_data: Dict, user_query: str) -> str:
        """
        Analyze PDF document to answer a specific user query

        Args:
            pdf_data: PDF data from NVINGEST containing pages
            user_query: The user's question about the document

        Returns:
            Comprehensive answer based on the PDF content
        """
        try:
            pages = pdf_data.get('pages', [])
            total_pages = len(pages)
            filename = pdf_data.get('filename', 'Unknown')

            logger.info(
                f"Starting intelligent PDF analysis for query '{user_query}' on {filename} ({total_pages} pages)"
            )

            if total_pages == 0:
                return "The document appears to be empty or contains no extractable text."

            # Use DocumentProcessor to categorize and route appropriately
            doc_size = DocumentProcessor.categorize_document_size(total_pages)

            if doc_size == "small":
                return await self._analyze_document_simple(pages, user_query, filename)
            elif doc_size == "medium":
                return await self._analyze_document_batched(pages, user_query, filename)
            else:  # large
                return await self._analyze_document_intelligent(pages, user_query, filename)

        except Exception as e:
            logger.error(f"Error in PDF analysis: {e}")
            return f"I encountered an error while analyzing the document: {str(e)}"

    async def _analyze_document_simple(self, pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze small documents in a single pass"""
        try:
            # Use DocumentProcessor to format pages
            full_text = DocumentProcessor.format_pages_for_analysis(pages)

            analysis_params = {
                "task_type": "analyze",
                "text": full_text,
                "instructions": f"Based on the document '{filename}', please answer this question: {user_query}. Provide specific details and cite page numbers when referencing information.",
            }

            # Import locally to avoid circular imports
            from tools.assistant import execute_assistant_with_dict

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, run_with_streamlit_context, execute_assistant_with_dict, analysis_params
            )

            return result.result

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return f"Error analyzing document: {str(e)}"

    async def _analyze_document_batched(self, pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze medium documents using batch processing"""
        try:
            # Use BatchProcessor for efficient batch handling
            batch_processor = BatchProcessor(
                batch_size=max(3, len(pages) // 3), delay_between_batches=0.3  # 3-5 pages per batch
            )

            async def analyze_batch(batch_pages: List[Dict], start_idx: int, end_idx: int) -> Dict:
                """Analyze a single batch of pages"""
                batch_text = DocumentProcessor.format_pages_for_analysis(batch_pages)

                analysis_params = {
                    "task_type": "analyze",
                    "text": batch_text,
                    "instructions": f"Analyze pages {start_idx+1}-{end_idx} of '{filename}' for this question: {user_query}. If relevant information is found, provide it with page numbers. If not relevant, say 'No relevant information found in these pages.'",
                }

                # Import locally to avoid circular imports
                from tools.assistant import execute_assistant_with_dict

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, run_with_streamlit_context, execute_assistant_with_dict, analysis_params
                )

                return {"page_range": f"{start_idx+1}-{end_idx}", "analysis": result.result}

            # Process pages in batches
            batch_results = await batch_processor.process_in_batches(pages, analyze_batch)

            # Filter out None results (from errors)
            valid_results = [r for r in batch_results if r is not None]

            # Combine batch results into final answer
            return await self._synthesize_batch_results(valid_results, user_query, filename)

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return f"Error analyzing document: {str(e)}"

    async def _analyze_document_intelligent(self, pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze large documents using intelligent search approach"""
        try:
            # Step 1: Find relevant pages using efficient scanning
            relevant_pages = await self._find_relevant_pages(pages, user_query, filename)

            if not relevant_pages:
                return f"I searched through all {len(pages)} pages of '{filename}' but couldn't find information directly related to your question: '{user_query}'. The document may not contain relevant information, or the question might need to be rephrased."

            # Step 2: Perform detailed analysis on relevant pages
            return await self._analyze_relevant_pages(relevant_pages, user_query, filename)

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return f"Error analyzing document: {str(e)}"

    async def _find_relevant_pages(self, pages: List[Dict], user_query: str, filename: str) -> List[Dict]:
        """Find pages potentially relevant to the user query"""
        try:
            batch_processor = BatchProcessor(batch_size=5, delay_between_batches=0.2)

            async def scan_batch(batch_pages: List[Dict], start_idx: int, end_idx: int) -> List[int]:
                """Scan a batch of pages for relevance"""
                # Create summaries using DocumentProcessor
                page_summaries = []
                for page in batch_pages:
                    page_num = page.get('page', start_idx + 1)
                    page_text = page.get('text', '')[:1000]  # First 1000 chars
                    page_summaries.append(f"Page {page_num}: {page_text}...")

                batch_text = "\n\n".join(page_summaries)

                relevance_params = {
                    "task_type": "analyze",
                    "text": batch_text,
                    "instructions": f"Given this query: '{user_query}', which of these pages (if any) contain relevant information? List ONLY the page numbers that are relevant, or say 'None' if no pages are relevant.",
                }

                # Import locally to avoid circular imports
                from tools.assistant import execute_assistant_with_dict

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, run_with_streamlit_context, execute_assistant_with_dict, relevance_params
                )

                # Use DocumentProcessor to extract page numbers
                return DocumentProcessor.extract_page_numbers_from_text(
                    result.result, valid_range=(start_idx + 1, end_idx)
                )

            # Process all pages to find relevant ones
            batch_results = await batch_processor.process_in_batches(pages, scan_batch)

            # Collect all relevant page numbers
            all_relevant_nums = []
            for nums in batch_results:
                if nums:  # Skip None results
                    all_relevant_nums.extend(nums)

            # Get unique page numbers
            relevant_page_nums = list(set(all_relevant_nums))

            # Extract the full page data for relevant pages
            relevant_pages = []
            for page in pages:
                if page.get('page') in relevant_page_nums:
                    relevant_pages.append(page)

            logger.info(f"Found {len(relevant_pages)} relevant pages out of {len(pages)} total pages")
            return relevant_pages

        except Exception as e:
            logger.error(f"Error finding relevant pages: {e}")
            # Fallback: return first 10 pages
            return pages[:10]

    async def _analyze_relevant_pages(self, relevant_pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze the relevant pages found during scanning"""
        try:
            # Determine if we can analyze all at once or need batching
            if len(relevant_pages) <= 5:
                # Simple analysis for few pages
                return await self._analyze_document_simple(relevant_pages, user_query, filename)
            else:
                # Use batched analysis for many relevant pages
                return await self._analyze_document_batched(relevant_pages, user_query, filename)

        except Exception as e:
            logger.error(f"Error analyzing relevant pages: {e}")
            return f"Error during analysis: {str(e)}"

    async def _synthesize_batch_results(self, batch_results: List[Dict], user_query: str, filename: str) -> str:
        """Synthesize results from multiple batch analyses"""
        try:
            # Format batch results for synthesis
            formatted_results = []
            for result in batch_results:
                if 'page_range' in result:
                    formatted_results.append(f"Analysis of pages {result['page_range']}:\n{result['analysis']}")
                elif 'pages' in result:
                    page_list = ', '.join(map(str, result['pages']))
                    formatted_results.append(f"Pages {page_list}:\n{result['analysis']}")
                else:
                    formatted_results.append(result['analysis'])

            combined_findings = "\n\n".join(formatted_results)

            synthesis_params = {
                "task_type": "analyze",
                "text": combined_findings,
                "instructions": f"Based on these analyses from '{filename}', provide a comprehensive answer to: {user_query}. Synthesize all relevant information into a cohesive response.",
            }

            # Import locally to avoid circular imports
            from tools.assistant import execute_assistant_with_dict

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, run_with_streamlit_context, execute_assistant_with_dict, synthesis_params
            )

            return result.result

        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            # Fallback: return all individual results
            return "\n\n".join([result.get('analysis', '') for result in batch_results if result])
