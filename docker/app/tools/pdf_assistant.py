"""
PDF Assistant Tool V2

Simplified tool that delegates to the new PDF services:
- PDFIngestionService for uploads
- PDFQueryServiceV2 for Q&A
- PDFSummarizerServiceV2 for summaries
"""

from typing import Any, Dict, List, Optional, Type

from models.chat_config import ChatConfig
from pydantic import Field
from services.file_storage_service import FileStorageService
from services.pdf_query_service_v2 import PDFQueryServiceV2
from services.pdf_summarizer_service_v2 import PDFSummarizerServiceV2
from services.session_state import get_active_pdf_id
from tools.base import BaseTool, BaseToolResponse, ToolController, ToolView

from utils.logging_config import get_logger

logger = get_logger(__name__)


class PDFAssistantResponse(BaseToolResponse):
    """Response from PDF assistant tool"""

    operation: str = Field(description="The operation performed")
    filename: str = Field(description="Name of the PDF file")
    result: str = Field(description="The result of the operation")
    pages_processed: Optional[List[int]] = Field(
        None, description="Page numbers that were processed"
    )
    used_vector_search: bool = Field(
        default=False, description="Whether vector search was used"
    )
    processing_notes: Optional[str] = Field(
        None, description="Additional processing notes"
    )
    direct_response: bool = Field(
        default=False, description="Show response in tool context expander"
    )


class PDFAssistantController(ToolController):
    """Controller for PDF operations"""

    def __init__(self, config_obj: ChatConfig):
        self.config = config_obj
        self.query_service = PDFQueryServiceV2(config_obj)
        self.summarizer = PDFSummarizerServiceV2(config_obj)
        self.file_storage = FileStorageService()

    def process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF operation synchronously"""
        # Run async operation in sync context
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.process_async(params))

    async def process_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process PDF operation asynchronously"""
        operation = params.get("operation", "query")
        query = params.get("query")

        # Get active PDF
        pdf_id = get_active_pdf_id()
        if not pdf_id:
            return {
                "success": False,
                "error": "No PDF document found in current session",
                "operation": operation,
                "filename": "None",
            }

        # Get PDF metadata
        pdf_data = self.file_storage.get_pdf(pdf_id)
        if not pdf_data:
            return {
                "success": False,
                "error": "PDF metadata not found",
                "operation": operation,
                "filename": "Unknown",
            }

        filename = pdf_data.get("filename", "Unknown")

        # Determine operation from query if auto
        if operation == "auto":
            operation = self._determine_operation(query)

        # Route to appropriate handler
        try:
            if operation == "summarize":
                logger.info(f"Summarizing PDF: {filename} (pdf_id: {pdf_id})")
                logger.debug(
                    "PDF data keys:"
                    f" {list(pdf_data.keys()) if pdf_data else 'None'}"
                )

                # Pass the user's query as instruction for context
                result = await self.summarizer.summarize_pdf(
                    pdf_data, user_instruction=query
                )

                logger.info(
                    "Summarization complete. Strategy:"
                    f" {result.get('strategy', 'unknown')}"
                )

                # Check if summarization failed
                if result.get("strategy") == "error":
                    return {
                        "operation": "summarize",
                        "filename": filename,
                        "success": False,
                        "error": result.get(
                            "summary", "Unknown error during summarization"
                        ),
                        "processing_notes": "Summarization failed",
                    }

                return {
                    "operation": "summarize",
                    "filename": filename,
                    "success": True,
                    "result": result["summary"],
                    "processing_notes": (
                        f"Used {result['strategy']} summarization strategy"
                    ),
                    "direct_response": True,
                }

            elif operation == "query":
                if not query:
                    return {
                        "success": False,
                        "error": "Query text is required for Q&A",
                        "operation": operation,
                        "filename": filename,
                    }

                query_result = self.query_service.query(pdf_id, query)

                # Format chunks into readable response
                if query_result["used"]:
                    chunks_text = query_result["formatted_context"]
                    result = f"\n{chunks_text}\n"
                else:
                    result = (
                        "No relevant sections found in the document for your"
                        " query."
                    )

                return {
                    "operation": "query",
                    "filename": filename,
                    "success": True,
                    "result": result,
                    "used_vector_search": True,
                    "processing_notes": f"",
                }

            elif operation == "info":
                info = f"PDF: {filename}\n"
                info += f"Pages: {pdf_data.get('total_pages', 'Unknown')}\n"
                info += (
                    f"Characters: {pdf_data.get('char_count', 'Unknown')}\n"
                )
                info += f"PDF ID: {pdf_id}"

                return {
                    "operation": "info",
                    "filename": filename,
                    "success": True,
                    "result": info,
                }

            else:
                # For now, extract and analyze operations just use query
                return await self.process_async(
                    {**params, "operation": "query"}
                )

        except Exception as e:
            logger.error(f"Error in PDF operation {operation}: {e}")
            return {
                "success": False,
                "error": str(e),
                "operation": operation,
                "filename": filename,
            }

    def _determine_operation(self, query: str) -> str:
        """Determine operation type from query"""
        if not query:
            return "info"

        query_lower = query.lower()

        # Check for summarization keywords
        if any(
            word in query_lower
            for word in ["summary", "summarize", "overview", "brief"]
        ):
            return "summarize"

        # Default to query (Q&A)
        return "query"


class PDFAssistantView(ToolView):
    """View for formatting PDF assistant responses"""

    def format_response(
        self, data: Dict[str, Any], response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format successful response"""
        return response_type(
            success=data.get("success", True),
            operation=data.get("operation", "unknown"),
            filename=data.get("filename", "Unknown"),
            result=data.get("result", ""),
            pages_processed=data.get("pages_processed"),
            used_vector_search=data.get("used_vector_search", False),
            processing_notes=data.get("processing_notes"),
            direct_response=data.get("direct_response", False),
        )

    def format_error(
        self, error: Exception, response_type: Type[BaseToolResponse]
    ) -> BaseToolResponse:
        """Format error response"""
        return response_type(
            success=False,
            error_message=str(error),
            operation="error",
            filename="Unknown",
            result=f"Error processing PDF: {str(error)}",
            direct_response=False,  # Changed to show in tool context expander
        )


class PDFAssistantTool(BaseTool):
    """
    Unified PDF Assistant Tool

    This tool handles all PDF-related operations:
    - Summarization
    - Q&A (with intelligent retrieval)
    - Page extraction
    - Document analysis
    - PDF information
    """

    def __init__(self):
        try:
            from utils.logging_config import get_logger

            logger = get_logger(__name__)
            logger.info("Creating PDFAssistantTool instance")

            super().__init__()
            self.name = "pdf_assistant"
            # Set the base description directly
            self.description = (
                "Unified tool for all PDF operations. Automatically handles"
                " summarization, Q&A, page extraction, and analysis of"
                " uploaded PDF documents. Use this tool whenever users ask"
                " about PDFs, request summaries of PDFs, or have questions"
                " about uploaded documents."
            )
            # Increase timeout for PDF operations which can take longer
            self.timeout = 256.0
            # Set execution mode to ASYNC to use process_async
            from tools.base import ExecutionMode

            self.execution_mode = ExecutionMode.ASYNC
            logger.info(
                "PDFAssistantTool initialized successfully with name:"
                f" {self.name}, timeout: {self.timeout}s, mode: ASYNC"
            )
            self._initialize_mvc()
        except Exception as e:
            logger.error(
                f"Error in PDFAssistantTool.__init__: {e}", exc_info=True
            )
            raise

    def get_dynamic_description(self) -> str:
        """Get dynamic description that indicates if a PDF is active"""
        from services.session_state import get_active_pdf_id
        from utils.pdf_upload_handler import get_active_pdf_info

        pdf_id = get_active_pdf_id()
        if pdf_id:
            pdf_info = get_active_pdf_info()
            if pdf_info:
                return (
                    "IMPORTANT: A PDF is currently uploaded and active. USE"
                    " THIS TOOL for ANY questions about PDFs, including"
                    " summarization requests. Active PDF:"
                    f" '{pdf_info['filename']}'"
                    f" ({pdf_info['total_pages']} pages). This tool handles:"
                    " summarization, Q&A, analysis, and information"
                    " extraction. When users ask to 'summarize the PDF' or"
                    " 'summarize the uploaded PDF', THIS is the tool to use."
                )

        return self.description

    def _initialize_mvc(self):
        """Initialize MVC components"""
        config_obj = ChatConfig.from_environment()
        self._controller = PDFAssistantController(config_obj)
        self._view = PDFAssistantView()

    def get_definition(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition"""
        # Get dynamic description with active PDF info
        current_description = self.get_dynamic_description()

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": current_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "summarize",
                                "query",
                                "extract",
                                "analyze",
                                "info",
                            ],
                            "default": "query",
                            "description": (
                                "Operation to perform: "
                                "'summarize' for document summary, "
                                "'query' for Q&A, "
                                "'extract' for page extraction, "
                                "'analyze' for detailed analysis, "
                                "'info' for PDF information, "
                            ),
                        },
                        "query": {
                            "type": "string",
                            "default": "Please summarize the PDF",
                            "description": (
                                "User's question or instruction. Required for"
                                " query, analyze, and auto operations. For"
                                " summarize, this can specify what aspect to"
                                " focus on."
                            ),
                        },
                        "page_numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Specific page numbers for extraction"
                                " (optional)"
                            ),
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["pages", "all"],
                            "default": "all",
                            "description": (
                                "Type of summary for summarize operation"
                            ),
                        },
                        "max_chunks": {
                            "type": "integer",
                            "default": 10,
                            "description": (
                                "Maximum chunks to retrieve for Q&A (advanced)"
                            ),
                        },
                    },
                    "required": ["operation", "query"],
                },
            },
        }

    def get_response_type(self) -> Type[BaseToolResponse]:
        """Get response type for this tool"""
        return PDFAssistantResponse


# Tool registration function
