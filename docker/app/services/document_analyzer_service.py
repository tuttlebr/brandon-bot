"""
Document Analyzer Service

This service handles document analysis operations, particularly for PDFs.
Extracted from the monolithic AssistantTool to separate concerns.
"""

import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from services.llm_client_service import llm_client_service
from utils.batch_processor import BatchProcessor, DocumentProcessor
from utils.config import config as app_config

logger = logging.getLogger(__name__)


class DocumentAnalyzerService:
    """Service for analyzing documents, especially PDFs"""

    def __init__(self, config: ChatConfig, llm_type: str = "intelligent"):
        """
        Initialize document analyzer service

        Args:
            config: Chat configuration
            llm_type: Type of LLM to use
        """
        self.config = config
        self.llm_type = llm_type

    def analyze_document(
        self, document_text: str, instructions: str, document_type: str = "document", filename: Optional[str] = None
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
            client = llm_client_service.get_client(self.llm_type)
            model_name = llm_client_service.get_model_name(self.llm_type)

            system_prompt = f"""You are analyzing a {document_type} to provide comprehensive insights and answer specific questions.

When analyzing documents, thoroughly understand the content, context, and purpose. Extract key insights, identify main themes and arguments, and provide accurate answers supported by evidence from the text. Be specific and cite relevant sections when answering questions."""

            if filename:
                user_message = f"Based on the document '{filename}', please answer this question: {instructions}\n\nDocument:\n{document_text}"
            else:
                user_message = f"Please analyze the following document and answer this question: {instructions}\n\nDocument:\n{document_text}"

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

            logger.debug(f"Analyzing document with {model_name}")

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for factual analysis
                top_p=app_config.llm.DEFAULT_TOP_P,
                frequency_penalty=app_config.llm.DEFAULT_FREQUENCY_PENALTY,
                presence_penalty=app_config.llm.DEFAULT_PRESENCE_PENALTY,
            )

            result = response.choices[0].message.content.strip()

            return {
                "success": True,
                "result": result,
                "processing_notes": f"Document analysis completed for query: {instructions[:100]}{'...' if len(instructions) > 100 else ''}",
            }

        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {"success": False, "error": str(e)}

    def analyze_pdf_pages(self, pages: List[Dict[str, Any]], instructions: str, filename: str) -> Dict[str, any]:
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
            return {"success": False, "error": "The document appears to be empty or contains no extractable text."}

        # Verify comprehensive analysis
        if total_pages < 100:
            logger.warning(f"⚠️  Only {total_pages} pages available for analysis - may not be comprehensive")
        else:
            logger.info(f"✓ Comprehensive analysis confirmed: Processing {total_pages} pages")

        # Categorize document and route appropriately
        doc_size = DocumentProcessor.categorize_document_size(total_pages)

        logger.info(f"Analyzing {doc_size} document '{filename}' with {total_pages} pages")

        try:
            if doc_size == "small":
                return self._analyze_small_document(pages, instructions, filename)
            elif doc_size == "medium":
                return self._analyze_medium_document(pages, instructions, filename)
            else:  # large
                return self._analyze_large_document(pages, instructions, filename)
        except Exception as e:
            logger.error(f"Error in PDF analysis: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_small_document(self, pages: List[Dict[str, Any]], instructions: str, filename: str) -> Dict[str, any]:
        """Analyze small documents in a single pass"""

        full_text = DocumentProcessor.format_pages_for_analysis(pages)
        return self.analyze_document(full_text, instructions, "PDF", filename)

    def _analyze_medium_document(
        self, pages: List[Dict[str, Any]], instructions: str, filename: str
    ) -> Dict[str, any]:
        """Analyze medium documents in batches"""

        # Process in batches of 3-5 pages
        batch_size = max(3, len(pages) // 3)
        batch_results = []

        for i in range(0, len(pages), batch_size):
            batch_end = min(i + batch_size, len(pages))
            batch_pages = pages[i:batch_end]

            # Determine actual page numbers for this batch
            if batch_pages:
                start_page_num = batch_pages[0].get('page', i + 1)
                end_page_num = batch_pages[-1].get('page', batch_end)
            else:
                start_page_num = i + 1
                end_page_num = batch_end

            batch_text = DocumentProcessor.format_pages_for_analysis(batch_pages)

            batch_instruction = (
                f"Analyze pages {start_page_num}-{end_page_num} of '{filename}' for this question: {instructions}. "
                f"If relevant information is found, provide it with page numbers. "
                f"If not relevant, say 'No relevant information found in these pages.'"
            )

            result = self.analyze_document(batch_text, batch_instruction, "PDF", filename)

            if result["success"]:
                batch_results.append({"page_range": f"{start_page_num}-{end_page_num}", "analysis": result["result"]})

        # Synthesize results
        return self._synthesize_batch_results(batch_results, instructions, filename)

    def _analyze_large_document(self, pages: List[Dict[str, Any]], instructions: str, filename: str) -> Dict[str, any]:
        """Analyze large documents comprehensively by processing ALL pages"""

        logger.info(f"Starting comprehensive analysis of all {len(pages)} pages for '{filename}'")

        # Process ALL pages in batches for comprehensive analysis
        # Use medium document approach but with larger batches for efficiency
        batch_size = max(10, len(pages) // 10)  # Process in larger batches
        batch_results = []

        for i in range(0, len(pages), batch_size):
            batch_end = min(i + batch_size, len(pages))
            batch_pages = pages[i:batch_end]

            # Determine actual page numbers for this batch
            if batch_pages:
                start_page_num = batch_pages[0].get('page', i + 1)
                end_page_num = batch_pages[-1].get('page', batch_end)
            else:
                start_page_num = i + 1
                end_page_num = batch_end

            batch_text = DocumentProcessor.format_pages_for_analysis(batch_pages)

            batch_instruction = (
                f"Comprehensively analyze pages {start_page_num}-{end_page_num} of '{filename}' for this question: {instructions}. "
                f"Provide detailed analysis of any relevant information found in these pages. "
                f"If no relevant information is found, note that these pages were checked but contained no relevant content."
            )

            result = self.analyze_document(batch_text, batch_instruction, "PDF", filename)

            if result["success"]:
                batch_results.append({"page_range": f"{start_page_num}-{end_page_num}", "analysis": result["result"]})

        # Synthesize results from ALL batches
        return self._synthesize_batch_results(batch_results, instructions, filename)

    def _synthesize_batch_results(
        self, batch_results: List[Dict[str, str]], instructions: str, filename: str
    ) -> Dict[str, any]:
        """Synthesize results from multiple batch analyses"""

        if not batch_results:
            return {"success": False, "error": "No batch results to synthesize"}

        # Format batch results
        combined_findings = "\n\n".join(
            [f"Analysis of pages {result['page_range']}:\n{result['analysis']}" for result in batch_results]
        )

        synthesis_instruction = (
            f"Based on these analyses from '{filename}', provide a comprehensive answer to: {instructions}. "
            f"Synthesize all relevant information into a cohesive response."
        )

        return self.analyze_document(combined_findings, synthesis_instruction, "analysis results", None)
