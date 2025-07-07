"""
PDF Summary Tool

This tool generates or retrieves document summaries for uploaded PDFs.
It will create summaries on-demand if they don't already exist.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.pdf_summarization_service import PDFSummarizationService
from tools.base import BaseTool, BaseToolResponse
from utils.config import AppConfig
from utils.pdf_extractor import PDFDataExtractor

config = AppConfig()

# Configure logger
logger = logging.getLogger(__name__)


class PDFSummaryResponse(BaseToolResponse):
    """Response from PDF summary retrieval tool"""

    success: bool = Field(description="Whether the retrieval was successful")
    filename: str = Field(description="Name of the PDF file")
    summary_type: str = Field(description="Type of summary retrieved")
    document_summary: Optional[str] = Field(None, description="Overall document summary")
    page_summaries: Optional[List[Dict[str, Any]]] = Field(None, description="Page-level summaries")
    message: str = Field(description="Status message")
    direct_response: bool = Field(default=True, description="This provides a direct response to the user")


class PDFSummaryTool(BaseTool):
    """
    PDF Summary Tool

    This tool generates or retrieves summaries for PDF documents.
    It will create summaries on-demand if they don't already exist.
    """

    def __init__(self):
        super().__init__()
        self.name = "retrieve_pdf_summary"
        self.description = "ONLY use this when explicitly asked to summarize a PDF document or when the user specifically mentions 'summarize the PDF', 'summarize the document', or similar phrases. Generates comprehensive summaries of PDF documents, providing both document-level overviews and page-by-page summaries for large PDFs. DO NOT use for general questions, web searches, or when no PDF is being discussed."
        self.summarization_service = None  # Will be initialized on first use

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert the tool to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Optional filename to get summary for. If not provided, will use the most recent PDF.",
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["document", "pages", "all", "debug"],
                            "default": "document",
                            "description": "Type of summary to retrieve: 'document' for overall summary, 'pages' for page-level summaries, 'all' for both, 'debug' for troubleshooting",
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(self, params: Dict[str, Any]) -> PDFSummaryResponse:
        """Execute the tool with given parameters"""
        return self.run_with_dict(params)

    def run_with_dict(self, params: Dict[str, Any]) -> PDFSummaryResponse:
        """
        Execute summary retrieval

        Args:
            params: Dictionary containing parameters

        Returns:
            PDFSummaryResponse
        """
        filename = params.get("filename", None)
        summary_type = params.get("summary_type", "document")
        messages = params.get("messages", [])

        # Try to get PDF data from multiple sources in order of preference
        pdf_data = None
        pdf_id = None

        # 1. First check if PDF data was passed directly in params
        pdf_data_param = params.get("pdf_data", None)
        if pdf_data_param:
            logger.info("Using PDF data passed directly in parameters")
            pdf_data = pdf_data_param
            pdf_id = "direct"

        # 2. Try to extract from messages (works in thread context)
        if not pdf_data and messages:
            pdf_data = self._get_pdf_data_from_messages(messages)
            if pdf_data:
                logger.info("Extracted PDF data from system messages")
                pdf_id = "from_messages"

        # 3. Last resort: try PDFContextService (may fail in thread context)
        if not pdf_data:
            try:
                from services.pdf_context_service import PDFContextService

                config = ChatConfig.from_environment()
                pdf_context_service = PDFContextService(config)
                pdf_data = pdf_context_service.get_latest_pdf_data()
                if pdf_data:
                    logger.info("Retrieved PDF data from context service")
                    pdf_id = "latest"
            except Exception as e:
                logger.debug(f"Could not access PDF context service (expected in thread context): {e}")

        # Create pdf_documents dict for compatibility
        pdf_documents = {pdf_id: pdf_data} if pdf_data else {}

        if not pdf_documents:
            return PDFSummaryResponse(
                success=False,
                filename=filename or "Unknown",
                summary_type=summary_type,
                message="No PDF documents found. Please upload a PDF first.",
                direct_response=True,
            )

        # Find the PDF to summarize
        if not pdf_data:
            pdf_data = None
            pdf_id = None

            if filename:
                # Look for specific filename
                for pid, pdata in pdf_documents.items():
                    if pdata.get("filename", "").lower() == filename.lower():
                        pdf_data = pdata
                        pdf_id = pid
                        break
            else:
                # Get the most recent PDF
                pdf_ids = list(pdf_documents.keys())
                if pdf_ids:
                    pdf_id = pdf_ids[-1]
                    pdf_data = pdf_documents[pdf_id]

            if not pdf_data:
                return PDFSummaryResponse(
                    success=False,
                    filename=filename or "Unknown",
                    summary_type=summary_type,
                    message=f"PDF document '{filename}' not found." if filename else "No PDF documents available.",
                    direct_response=True,
                )

        actual_filename = pdf_data.get("filename", "Unknown")

        # Check if this is a batch-processed PDF
        if pdf_data.get("batch_processed", False):
            # Handle batch-processed PDF
            return self._summarize_batch_processed_pdf(pdf_data, summary_type)

        # Handle debug request
        if summary_type == "debug":
            return self.debug_pdf_processing(params)

        # Regular PDF processing continues...
        pages = pdf_data.get("pages", [])

        # Check if summarization is complete
        if not pdf_data.get("summarization_complete", False):
            total_pages = len(pdf_data.get("pages", []))

            # For large documents, generate summary on-demand using async recursive summarization
            logger.info(f"Generating on-demand summary for {actual_filename} ({total_pages} pages)")

            try:
                # Initialize summarization service if needed
                if self.summarization_service is None:
                    config = ChatConfig.from_environment()
                    self.summarization_service = PDFSummarizationService(config)

                # Log progress message (UI operations should be handled by the caller)
                logger.info(f"Generating comprehensive summary for {actual_filename} ({total_pages} pages)...")

                # Use async recursive summarization for full document processing
                # Create a new event loop for async operations
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # If no event loop exists, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Perform async recursive summarization for comprehensive results
                enhanced_pdf_data = loop.run_until_complete(
                    self.summarization_service.summarize_pdf_recursive(pdf_data)
                )

                # Update the PDF data in storage only if it came from storage
                if pdf_id != "direct":
                    from services.file_storage_service import FileStorageService

                    file_storage = FileStorageService()

                    if file_storage.update_pdf(pdf_id, enhanced_pdf_data):
                        logger.info(f"Updated PDF '{actual_filename}' with comprehensive summarization data")
                        pdf_data = enhanced_pdf_data
                    else:
                        logger.error(f"Failed to update PDF '{actual_filename}' in storage")
                else:
                    # For directly passed PDFs, just update the local copy
                    logger.info(
                        f"Updated directly passed PDF '{actual_filename}' with comprehensive summarization data"
                    )
                    pdf_data = enhanced_pdf_data

            except Exception as e:
                logger.error(f"Error generating comprehensive summary: {e}")
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message=f"❌ Error generating comprehensive summary for **{actual_filename}**: {str(e)}",
                    direct_response=True,
                )

        # Retrieve the requested summary
        document_summary = pdf_data.get("document_summary", None)
        page_summaries = pdf_data.get("page_summaries", None)

        # Format the response based on summary type
        if summary_type == "document":
            if document_summary:
                return PDFSummaryResponse(
                    success=True,
                    filename=actual_filename,
                    summary_type=summary_type,
                    document_summary=document_summary,
                    message=f"## Summary of {actual_filename}\n\n{document_summary}",
                    direct_response=True,
                )
            else:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message="Document summary not available.",
                    direct_response=True,
                )

        elif summary_type == "pages":
            if page_summaries:
                formatted_summaries = self._format_page_summaries(page_summaries)
                return PDFSummaryResponse(
                    success=True,
                    filename=actual_filename,
                    summary_type=summary_type,
                    page_summaries=page_summaries,
                    message=f"## Page Summaries for {actual_filename}\n\n{formatted_summaries}",
                    direct_response=True,
                )
            else:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message="Page summaries not available.",
                    direct_response=True,
                )

        else:  # "all"
            parts = [f"## Complete Summary of {actual_filename}\n"]

            if document_summary:
                parts.append("### Document Overview\n")
                parts.append(document_summary)
                parts.append("\n")

            if page_summaries:
                parts.append("### Detailed Page Summaries\n")
                parts.append(self._format_page_summaries(page_summaries))

            if not document_summary and not page_summaries:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message="No summaries available for this document.",
                    direct_response=True,
                )

            return PDFSummaryResponse(
                success=True,
                filename=actual_filename,
                summary_type=summary_type,
                document_summary=document_summary,
                page_summaries=page_summaries,
                message="\n".join(parts),
                direct_response=True,
            )

    def _format_page_summaries(self, page_summaries: List[Dict[str, Any]]) -> str:
        """Format page summaries for display"""
        formatted = []

        for summary in page_summaries:
            page_range = summary.get("page_range", "Unknown")
            summary_text = summary.get("summary", "No summary available")
            formatted.append(f"**Pages {page_range}:**\n{summary_text}\n")

        return "\n".join(formatted)

    def _get_pdf_data_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract PDF data from injected system messages"""
        return PDFDataExtractor.extract_from_messages(messages)

    def _summarize_batch_processed_pdf(self, pdf_data: Dict[str, Any], summary_type: str) -> PDFSummaryResponse:
        """
        Handle summarization of batch-processed PDFs using hierarchical chunking

        Args:
            pdf_data: PDF metadata with batch information
            summary_type: Type of summary requested

        Returns:
            PDFSummaryResponse
        """
        filename = pdf_data.get("filename", "Unknown")
        total_pages = pdf_data.get("total_pages", 0)
        total_batches = pdf_data.get("total_batches", 0)
        pdf_id = pdf_data.get("pdf_id")

        # Import file storage service
        from services.file_storage_service import FileStorageService

        file_storage = FileStorageService()

        # Step 1: Process each batch individually to create batch summaries
        batch_summaries = []
        pages_included = 0

        for batch_num in range(total_batches):
            batch_id = f"{pdf_id}_batch_{batch_num}"
            batch_path = file_storage.pdfs_dir / f"{batch_id}.json"

            if batch_path.exists():
                batch_data = json.loads(batch_path.read_text())
                batch_pages = batch_data.get("pages", [])

                if batch_pages:
                    # Create batch-level summary
                    batch_summary = self._summarize_batch_pages(batch_pages, batch_num, summary_type)
                    if batch_summary:
                        batch_summaries.append(batch_summary)
                        pages_included += len(batch_pages)

        if not batch_summaries:
            return PDFSummaryResponse(
                success=False,
                filename=filename,
                summary_type=summary_type,
                message="Unable to extract text from the batch-processed PDF.",
                direct_response=True,
            )

        # Step 2: If we have multiple batch summaries, combine them hierarchically
        if len(batch_summaries) > 1:
            final_summary = self._combine_batch_summaries(batch_summaries, filename, summary_type)
        else:
            final_summary = batch_summaries[0]

        # Add note about complete document coverage
        if pages_included == total_pages:
            final_summary += (
                f"\n\n**Note:** This summary covers the complete document ({pages_included} of {total_pages} pages). "
                f"The document was processed in {total_batches} batches for memory efficiency."
            )
        else:
            final_summary += (
                f"\n\n**Note:** This summary covers {pages_included} of {total_pages} total pages. "
                f"The document was processed in {total_batches} batches for memory efficiency. "
                f"For specific information from other sections, please ask about particular topics or page ranges."
            )

        formatted_summary = f"## {summary_type.title()} Summary of {filename}\n\n{final_summary}"

        return PDFSummaryResponse(
            success=True,
            filename=filename,
            summary_type=summary_type,
            summary=final_summary,
            pages_summarized=pages_included,
            total_pages=total_pages,
            message=formatted_summary,
            direct_response=True,
        )

    def _summarize_batch_pages(
        self, batch_pages: List[Dict[str, Any]], batch_num: int, summary_type: str
    ) -> Optional[str]:
        """
        Summarize a single batch of pages

        Args:
            batch_pages: List of page data for this batch
            batch_num: Batch number for context
            summary_type: Type of summary requested

        Returns:
            Summary text or None if failed
        """
        try:
            # Combine text from this batch only
            batch_text_parts = []
            for page in batch_pages:
                page_text = page.get("text", "").strip()
                if page_text:
                    page_num = page.get("page", "Unknown")
                    batch_text_parts.append(f"Page {page_num}:\n{page_text}")

            if not batch_text_parts:
                return None

            batch_text = "\n\n".join(batch_text_parts)

            # Check if this batch is too large for direct processing
            estimated_tokens = len(batch_text) // 4  # Rough token estimation
            if estimated_tokens > 100000:  # Conservative limit
                # Split batch into smaller chunks
                return self._summarize_large_batch(batch_pages, batch_num, summary_type)

            # Create batch-level summary
            if summary_type == "detailed":
                instructions = (
                    f"Create a detailed summary of this batch of pages from a larger document. "
                    f"Include key information, main topics, and important details. "
                    f"This is batch {batch_num + 1} of the document."
                )
            else:  # brief
                instructions = (
                    f"Create a concise summary of this batch of pages. "
                    f"Focus on main points and key information. "
                    f"Keep it under 200 words. This is batch {batch_num + 1} of the document."
                )

            # Use assistant tool for summarization
            summary_params = {
                "task_type": "summarize",
                "text": batch_text,
                "instructions": instructions,
            }

            from tools.assistant import execute_assistant_with_dict

            summary_result = execute_assistant_with_dict(summary_params)
            return summary_result.result if hasattr(summary_result, 'result') else str(summary_result)

        except Exception as e:
            logger.error(f"Error summarizing batch {batch_num}: {e}")
            return f"Batch {batch_num + 1} summary unavailable due to processing error."

    def _summarize_large_batch(self, batch_pages: List[Dict[str, Any]], batch_num: int, summary_type: str) -> str:
        """
        Handle summarization of large batches by splitting into smaller chunks

        Args:
            batch_pages: List of page data for this batch
            batch_num: Batch number for context
            summary_type: Type of summary requested

        Returns:
            Combined summary text
        """
        # Split batch into smaller chunks (5 pages per chunk)
        chunk_size = 5
        chunk_summaries = []

        for i in range(0, len(batch_pages), chunk_size):
            chunk_pages = batch_pages[i : i + chunk_size]

            # Create chunk-level summary
            chunk_text_parts = []
            for page in chunk_pages:
                page_text = page.get("text", "").strip()
                if page_text:
                    page_num = page.get("page", "Unknown")
                    chunk_text_parts.append(f"Page {page_num}:\n{page_text}")

            if chunk_text_parts:
                chunk_text = "\n\n".join(chunk_text_parts)

                instructions = (
                    f"Create a brief summary of these pages from batch {batch_num + 1}. "
                    f"Focus on key information and main points."
                )

                summary_params = {
                    "task_type": "summarize",
                    "text": chunk_text,
                    "instructions": instructions,
                }

                try:
                    from tools.assistant import execute_assistant_with_dict

                    summary_result = execute_assistant_with_dict(summary_params)
                    chunk_summary = summary_result.result if hasattr(summary_result, 'result') else str(summary_result)
                    chunk_summaries.append(chunk_summary)
                except Exception as e:
                    logger.error(f"Error summarizing chunk in batch {batch_num}: {e}")
                    chunk_summaries.append(f"Chunk summary unavailable due to processing error.")

        # Combine chunk summaries into batch summary
        if chunk_summaries:
            combined_chunks = "\n\n".join(chunk_summaries)

            instructions = (
                f"Combine these chunk summaries from batch {batch_num + 1} into a cohesive batch summary. "
                f"Maintain key information while eliminating redundancy."
            )

            summary_params = {
                "task_type": "summarize",
                "text": combined_chunks,
                "instructions": instructions,
            }

            try:
                from tools.assistant import execute_assistant_with_dict

                summary_result = execute_assistant_with_dict(summary_params)
                return summary_result.result if hasattr(summary_result, 'result') else str(summary_result)
            except Exception as e:
                logger.error(f"Error combining chunk summaries for batch {batch_num}: {e}")
                return f"Batch {batch_num + 1} summary unavailable due to processing error."

        return f"Batch {batch_num + 1} summary unavailable."

    def _combine_batch_summaries(self, batch_summaries: List[str], filename: str, summary_type: str) -> str:
        """
        Combine multiple batch summaries into a final document summary

        Args:
            batch_summaries: List of batch summary texts
            filename: Document filename
            summary_type: Type of summary requested

        Returns:
            Final combined summary
        """
        try:
            # Combine all batch summaries
            combined_summaries = "\n\n---\n\n".join(batch_summaries)

            # Create final summary instructions
            if summary_type == "detailed":
                instructions = (
                    f"Create a comprehensive, detailed summary of the entire document '{filename}' "
                    f"based on these batch summaries. Include all major sections, key findings, "
                    f"methodologies, and conclusions. Synthesize the information into a cohesive whole."
                )
            else:  # brief
                instructions = (
                    f"Create a concise summary of the entire document '{filename}' "
                    f"based on these batch summaries. Highlight the main purpose and key points. "
                    f"Keep it under 300 words."
                )

            # Use assistant tool for final summarization
            summary_params = {
                "task_type": "summarize",
                "text": combined_summaries,
                "instructions": instructions,
            }

            from tools.assistant import execute_assistant_with_dict

            summary_result = execute_assistant_with_dict(summary_params)
            return summary_result.result if hasattr(summary_result, 'result') else str(summary_result)

        except Exception as e:
            logger.error(f"Error combining batch summaries: {e}")
            # Fallback: return concatenated summaries
            return "\n\n---\n\n".join(batch_summaries)

    def debug_pdf_processing(self, params: Dict[str, Any]) -> PDFSummaryResponse:
        """
        Debug PDF processing issues

        Args:
            params: Dictionary containing parameters

        Returns:
            PDFSummaryResponse with debug information
        """
        messages = params.get("messages", [])
        pdf_data = self._get_pdf_data_from_messages(messages)

        if not pdf_data:
            return PDFSummaryResponse(
                success=False,
                filename="Unknown",
                summary_type="debug",
                message="No PDF document found. Please upload a PDF first.",
                direct_response=True,
            )

        filename = pdf_data.get("filename", "Unknown")
        pdf_id = pdf_data.get("pdf_id", "Unknown")

        # Import PDF context service for debugging
        from models.chat_config import ChatConfig
        from services.pdf_context_service import PDFContextService

        config = ChatConfig.from_environment()
        pdf_context_service = PDFContextService(config)

        debug_info = pdf_context_service.debug_batch_processing(pdf_id)

        return PDFSummaryResponse(
            success=True,
            filename=filename,
            summary_type="debug",
            message=f"## Debug Information for {filename}\n\n{debug_info}",
            direct_response=True,
        )


# Create global instance
pdf_summary_tool = PDFSummaryTool()


def get_pdf_summary_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    return pdf_summary_tool.to_openai_format()


def execute_pdf_summary_with_dict(params: Dict[str, Any]) -> PDFSummaryResponse:
    """Execute PDF summary retrieval with parameters as dictionary"""
    return pdf_summary_tool.run_with_dict(params)
