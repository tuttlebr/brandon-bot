"""
PDF Analysis Service

This service handles intelligent analysis of PDF documents to answer user questions
by processing pages in batches and combining relevant findings.
"""

import asyncio
import concurrent.futures
import logging
from typing import Dict, List, Optional, Tuple

from models.chat_config import ChatConfig
from utils.config import config
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
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

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

            # For small documents (≤5 pages), process all at once
            if total_pages <= 5:
                return await self._analyze_small_document(pages, user_query, filename)

            # For medium documents (6-15 pages), process in 2-3 batches
            elif total_pages <= 15:
                return await self._analyze_medium_document(pages, user_query, filename)

            # For large documents (>15 pages), use intelligent search approach
            else:
                return await self._analyze_large_document(pages, user_query, filename)

        except Exception as e:
            logger.error(f"Error in PDF analysis: {e}")
            return f"I encountered an error while analyzing the document: {str(e)}"

    async def _analyze_small_document(self, pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze small documents (≤5 pages) all at once"""
        try:
            # Combine all pages
            full_text = "\n\n".join(
                [f"Page {page.get('page', i+1)}:\n{page.get('text', '')}" for i, page in enumerate(pages)]
            )

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
            logger.error(f"Error analyzing small document: {e}")
            return f"Error analyzing document: {str(e)}"

    async def _analyze_medium_document(self, pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze medium documents (6-15 pages) in batches"""
        try:
            batch_size = max(3, len(pages) // 3)  # 3-5 pages per batch
            batch_results = []

            # Process in batches
            for i in range(0, len(pages), batch_size):
                batch_end = min(i + batch_size, len(pages))
                batch_pages = pages[i:batch_end]

                batch_text = "\n\n".join(
                    [f"Page {page.get('page', i+j+1)}:\n{page.get('text', '')}" for j, page in enumerate(batch_pages)]
                )

                analysis_params = {
                    "task_type": "analyze",
                    "text": batch_text,
                    "instructions": f"Analyze pages {i+1}-{batch_end} of '{filename}' for this question: {user_query}. If relevant information is found, provide it with page numbers. If not relevant, say 'No relevant information found in these pages.'",
                }

                # Import locally to avoid circular imports
                from tools.assistant import execute_assistant_with_dict

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, run_with_streamlit_context, execute_assistant_with_dict, analysis_params
                )

                batch_results.append({"page_range": f"{i+1}-{batch_end}", "analysis": result.result})

                # Small delay to avoid overwhelming the LLM service
                await asyncio.sleep(0.3)

            # Combine batch results into final answer
            return await self._synthesize_batch_results(batch_results, user_query, filename)

        except Exception as e:
            logger.error(f"Error analyzing medium document: {e}")
            return f"Error analyzing document: {str(e)}"

    async def _analyze_large_document(self, pages: List[Dict], user_query: str, filename: str) -> str:
        """Analyze large documents (>15 pages) using intelligent search"""
        try:
            # Step 1: Quick scan of all pages to find potentially relevant sections
            relevant_pages = await self._find_relevant_pages(pages, user_query, filename)

            if not relevant_pages:
                return f"I searched through all {len(pages)} pages of '{filename}' but couldn't find information directly related to your question: '{user_query}'. The document may not contain relevant information, or the question might need to be rephrased."

            # Step 2: Deep analysis of relevant pages
            return await self._deep_analyze_relevant_pages(relevant_pages, user_query, filename)

        except Exception as e:
            logger.error(f"Error analyzing large document: {e}")
            return f"Error analyzing document: {str(e)}"

    async def _find_relevant_pages(self, pages: List[Dict], user_query: str, filename: str) -> List[Dict]:
        """Find pages potentially relevant to the user query"""
        try:
            relevant_pages = []
            batch_size = 5  # Scan 5 pages at a time

            for i in range(0, len(pages), batch_size):
                batch_end = min(i + batch_size, len(pages))
                batch_pages = pages[i:batch_end]

                # Create summary of each page for relevance checking
                page_summaries = []
                for j, page in enumerate(batch_pages):
                    page_text = page.get('text', '')[:1000]  # First 1000 chars
                    page_summaries.append(f"Page {page.get('page', i+j+1)}: {page_text}...")

                batch_text = "\n\n".join(page_summaries)

                relevance_params = {
                    "task_type": "analyze",
                    "text": batch_text,
                    "instructions": f"Given this query: '{user_query}', which of these pages (if any) contain relevant information? List ONLY the page numbers that are relevant, or say 'None' if no pages are relevant. Be specific about page numbers.",
                }

                # Import locally to avoid circular imports
                from tools.assistant import execute_assistant_with_dict

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, run_with_streamlit_context, execute_assistant_with_dict, relevance_params
                )

                # Parse relevant page numbers from result
                relevant_in_batch = self._extract_page_numbers(result.result, i + 1, batch_end)

                # Add full page data for relevant pages
                for page_num in relevant_in_batch:
                    for page in batch_pages:
                        if page.get('page') == page_num:
                            relevant_pages.append(page)
                            break

                await asyncio.sleep(0.2)

            logger.info(f"Found {len(relevant_pages)} relevant pages out of {len(pages)} total pages")
            return relevant_pages

        except Exception as e:
            logger.error(f"Error finding relevant pages: {e}")
            # Fallback: return first 10 pages
            return pages[:10]

    def _extract_page_numbers(self, text: str, start_page: int, end_page: int) -> List[int]:
        """Extract page numbers from LLM response"""
        import re

        if 'none' in text.lower() or 'no pages' in text.lower():
            return []

        # Find all numbers that could be page numbers
        numbers = re.findall(r'\b(\d+)\b', text)
        page_numbers = []

        for num_str in numbers:
            num = int(num_str)
            if start_page <= num <= end_page:
                page_numbers.append(num)

        return list(set(page_numbers))  # Remove duplicates

    async def _deep_analyze_relevant_pages(self, relevant_pages: List[Dict], user_query: str, filename: str) -> str:
        """Perform deep analysis on relevant pages"""
        try:
            if len(relevant_pages) <= 5:
                # Analyze all relevant pages together
                full_text = "\n\n".join(
                    [f"Page {page.get('page')}:\n{page.get('text', '')}" for page in relevant_pages]
                )

                analysis_params = {
                    "task_type": "analyze",
                    "text": full_text,
                    "instructions": f"Based on these relevant pages from '{filename}', provide a comprehensive answer to: {user_query}. Include specific details and page references.",
                }

                # Import locally to avoid circular imports
                from tools.assistant import execute_assistant_with_dict

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, run_with_streamlit_context, execute_assistant_with_dict, analysis_params
                )

                return result.result
            else:
                # Too many relevant pages, process in smaller batches
                batch_results = []
                batch_size = 3

                for i in range(0, len(relevant_pages), batch_size):
                    batch = relevant_pages[i : i + batch_size]
                    batch_text = "\n\n".join([f"Page {page.get('page')}:\n{page.get('text', '')}" for page in batch])

                    analysis_params = {
                        "task_type": "analyze",
                        "text": batch_text,
                        "instructions": f"Analyze these pages from '{filename}' for: {user_query}. Extract any relevant information with page numbers.",
                    }

                    # Import locally to avoid circular imports
                    from tools.assistant import execute_assistant_with_dict

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, run_with_streamlit_context, execute_assistant_with_dict, analysis_params
                    )

                    batch_results.append({"pages": [p.get('page') for p in batch], "analysis": result.result})

                    await asyncio.sleep(0.3)

                # Synthesize final answer
                return await self._synthesize_deep_analysis(batch_results, user_query, filename)

        except Exception as e:
            logger.error(f"Error in deep analysis: {e}")
            return f"Error during detailed analysis: {str(e)}"

    async def _synthesize_batch_results(self, batch_results: List[Dict], user_query: str, filename: str) -> str:
        """Synthesize results from multiple batches"""
        try:
            combined_findings = "\n\n".join(
                [f"Analysis of pages {result['page_range']}:\n{result['analysis']}" for result in batch_results]
            )

            synthesis_params = {
                "task_type": "analyze",
                "text": combined_findings,
                "instructions": f"Based on these analyses of different sections of '{filename}', provide a comprehensive answer to: {user_query}. Combine relevant information and provide a cohesive response.",
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
            return "\n\n".join([result['analysis'] for result in batch_results])

    async def _synthesize_deep_analysis(self, batch_results: List[Dict], user_query: str, filename: str) -> str:
        """Synthesize results from deep analysis batches"""
        try:
            combined_findings = "\n\n".join(
                [f"Pages {', '.join(map(str, result['pages']))}:\n{result['analysis']}" for result in batch_results]
            )

            synthesis_params = {
                "task_type": "analyze",
                "text": combined_findings,
                "instructions": f"Based on these detailed analyses from '{filename}', provide a final comprehensive answer to: {user_query}. Synthesize all relevant information into a cohesive response.",
            }

            # Import locally to avoid circular imports
            from tools.assistant import execute_assistant_with_dict

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, run_with_streamlit_context, execute_assistant_with_dict, synthesis_params
            )

            return result.result

        except Exception as e:
            logger.error(f"Error in final synthesis: {e}")
            # Fallback: return all individual results
            return "\n\n".join([result['analysis'] for result in batch_results])

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
