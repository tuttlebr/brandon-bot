"""
PDF Summary Tool

This tool generates or retrieves document summaries for uploaded PDFs.
It will create summaries on-demand if they don't already exist.
"""

import logging
from typing import Any, Dict, List, Optional

from models.chat_config import ChatConfig
from pydantic import Field
from services.pdf_summarization_service import PDFSummarizationService
from tools.base import BaseTool, BaseToolResponse

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
        self.description = "Generates comprehensive summaries of PDF documents, providing both document-level overviews and page-by-page summaries for large PDFs."
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
                            "enum": ["document", "pages", "all"],
                            "default": "document",
                            "description": "Type of summary to retrieve: 'document' for overall summary, 'pages' for page-level summaries, 'all' for both",
                        },
                    },
                    "required": [],
                },
            },
        }

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for BaseTool interface"""
        return self.to_openai_format()

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

        # Check if summarization is complete
        if not pdf_data.get("summarization_complete", False):
            total_pages = len(pdf_data.get("pages", []))

            # For small documents, inform user that summarization isn't needed
            if total_pages <= 10:
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message=f"**{actual_filename}** has only {total_pages} pages. For documents this size, I can analyze the content directly without needing a summary. Please ask specific questions about the document instead.",
                    direct_response=True,
                )

            # For large documents, generate summary on-demand
            logger.info(f"Generating on-demand summary for {actual_filename} ({total_pages} pages)")

            try:
                # Initialize summarization service if needed
                if self.summarization_service is None:
                    config = ChatConfig.from_environment()
                    self.summarization_service = PDFSummarizationService(config)

                # Log progress message (UI operations should be handled by the caller)
                logger.info(f"Generating summary for {actual_filename} ({total_pages} pages)...")

                # Perform synchronous summarization for immediate results
                enhanced_pdf_data = self.summarization_service.summarize_pdf_sync(pdf_data)

                # Update the PDF data in storage only if it came from storage
                if pdf_id != "direct":
                    from services.file_storage_service import FileStorageService

                    file_storage = FileStorageService()

                    if file_storage.update_pdf(pdf_id, enhanced_pdf_data):
                        logger.info(f"Updated PDF '{actual_filename}' with summarization data")
                        pdf_data = enhanced_pdf_data
                    else:
                        logger.error(f"Failed to update PDF '{actual_filename}' in storage")
                else:
                    # For directly passed PDFs, just update the local copy
                    logger.info(f"Updated directly passed PDF '{actual_filename}' with summarization data")
                    pdf_data = enhanced_pdf_data

            except Exception as e:
                logger.error(f"Error generating summary: {e}")
                return PDFSummaryResponse(
                    success=False,
                    filename=actual_filename,
                    summary_type=summary_type,
                    message=f"❌ Error generating summary for **{actual_filename}**: {str(e)}",
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
        if not messages:
            logger.debug("No messages provided to extract PDF data from")
            return None

        # Look for PDF content in system messages
        for message in messages:
            if message.get("role") == "system":
                content = message.get("content", "")
                if isinstance(content, str) and "## PDF Document Context" in content:
                    # This is an injected PDF context message
                    # Extract the PDF data from the content
                    try:
                        # Parse the PDF content from the system message
                        lines = content.split('\n')
                        filename = None
                        pages = []
                        in_pdf_content = False
                        current_page = None
                        current_page_text = []

                        for line in lines:
                            if "The user has uploaded a PDF document:" in line:
                                # Extract filename
                                import re

                                match = re.search(r"'([^']+)'", line)
                                if match:
                                    filename = match.group(1)
                            elif line.strip() == "---BEGIN PDF CONTENT---":
                                in_pdf_content = True
                            elif line.strip() == "---END PDF CONTENT---":
                                in_pdf_content = False
                                # Save last page if any
                                if current_page is not None and current_page_text:
                                    pages.append({"page": current_page, "text": '\n'.join(current_page_text).strip()})
                            elif in_pdf_content:
                                if line.startswith("### Page "):
                                    # Save previous page if any
                                    if current_page is not None and current_page_text:
                                        pages.append(
                                            {"page": current_page, "text": '\n'.join(current_page_text).strip()}
                                        )
                                    # Start new page
                                    try:
                                        current_page = int(line.replace("### Page ", "").strip())
                                        current_page_text = []
                                    except ValueError:
                                        pass
                                elif current_page is not None:
                                    current_page_text.append(line)

                        if pages:
                            logger.info(f"Extracted PDF data from system message: {filename} with {len(pages)} pages")
                            return {"filename": filename or "Unknown", "pages": pages}
                    except Exception as e:
                        logger.error(f"Error parsing PDF data from system message: {e}")

        logger.debug("No PDF data found in system messages")
        return None


# Create global instance
pdf_summary_tool = PDFSummaryTool()


def get_pdf_summary_tool_definition() -> Dict[str, Any]:
    """Get the OpenAI-compatible tool definition"""
    return pdf_summary_tool.to_openai_format()


def execute_pdf_summary_with_dict(params: Dict[str, Any]) -> PDFSummaryResponse:
    """Execute PDF summary retrieval with parameters as dictionary"""
    return pdf_summary_tool.run_with_dict(params)
