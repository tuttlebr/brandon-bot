"""
RESTful API for the BrandonBot chatbot service

This module provides a FastAPI-based REST API that exposes the chatbot
functionality through a single /agent endpoint, accepting OpenAI-compatible
chat completion requests and returning responses in the same format.
"""

import logging
from typing import Any, Dict, List, Optional, Union

# Import the application components
from controllers.file_controller import FileController
from controllers.image_controller import ImageController
from controllers.message_controller import MessageController
from controllers.response_controller import ResponseController
from controllers.session_controller import SessionController
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models import ChatConfig
from pydantic import BaseModel, Field
from services import ChatService, ImageService, LLMService
from tools.initialize_tools import initialize_all_tools
from tools.registry import get_all_tool_definitions
from ui import ChatHistoryComponent
from utils.config import config
from utils.exceptions import ConfigurationError
from utils.startup import initialize_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    """OpenAI-compatible chat message"""

    role: str = Field(..., description="The role of the message sender")
    content: Union[str, List[Dict[str, Any]]] = Field(
        ..., description="The message content"
    )
    name: Optional[str] = Field(
        None, description="Optional name for the message sender"
    )


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""

    messages: List[ChatMessage] = Field(
        ..., description="List of conversation messages"
    )
    model: Optional[str] = Field(
        None, description="Model to use for completion"
    )
    temperature: Optional[float] = Field(
        0.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        0.95, description="Top-p sampling parameter"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens to generate"
    )
    stream: Optional[bool] = Field(
        False, description="Whether to stream the response"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="Available tools"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        "auto", description="Tool choice strategy"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice"""

    index: int = Field(0, description="Choice index")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: str = Field("stop", description="Reason for completion")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""

    id: str = Field(..., description="Response ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="Generated choices"
    )
    usage: Optional[Dict[str, Any]] = Field(
        None, description="Token usage information"
    )


class StreamingChatCompletionResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response"""

    id: str = Field(..., description="Response ID")
    object: str = Field("chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Generated choices")


class APIChatbotService:
    """API service that wraps the chatbot functionality"""

    def __init__(self):
        """Initialize the API chatbot service"""
        try:
            # Initialize app startup settings
            initialize_app()

            # Initialize configuration (API version without streamlit)
            self.config_obj = ChatConfig()

            # Initialize tools
            if len(get_all_tool_definitions()) == 0:
                logger.warning("No tools found, attempting initialization")
                initialize_all_tools()

            # Initialize services
            self.chat_service = ChatService(self.config_obj)
            self.image_service = ImageService(self.config_obj)
            self.llm_service = LLMService(self.config_obj)

            # Initialize UI components (minimal for API)
            self.chat_history_component = ChatHistoryComponent(self.config_obj)

            # Initialize controllers
            self.session_controller = SessionController(self.config_obj)
            self.message_controller = MessageController(
                self.config_obj, self.chat_service, self.session_controller
            )
            self.file_controller = FileController(
                self.config_obj,
                self.message_controller,
                self.session_controller,
            )
            self.image_controller = ImageController(
                self.config_obj,
                self.message_controller,
                self.session_controller,
            )
            self.response_controller = ResponseController(
                self.config_obj,
                self.llm_service,
                self.message_controller,
                self.session_controller,
                self.chat_history_component,
            )

            logger.info("API chatbot service initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize API chatbot service: %s", e)
            raise ConfigurationError(f"API service initialization failed: {e}")

    def _convert_messages_to_dict(
        self, messages: List[ChatMessage]
    ) -> List[Dict[str, Any]]:
        """Convert Pydantic messages to dictionary format"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
            }
            for msg in messages
        ]

    async def process_chat_completion(
        self, request: ChatCompletionRequest, session_id: Optional[str] = None
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        """
        Process a chat completion request

        Args:
            request: The chat completion request
            session_id: Optional session ID for conversation continuity

        Returns:
            Chat completion response or streaming response
        """
        try:
            # Set session ID if provided
            if session_id:
                from services.session_state import set_session_id

                set_session_id(session_id)

            # Convert messages to internal format
            messages = self._convert_messages_to_dict(request.messages)

            # Validate messages
            if not messages:
                raise HTTPException(
                    status_code=400, detail="No messages provided"
                )

            # Get the user's message (last message from user)
            user_message = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_message = content
                    elif isinstance(content, list):
                        # Handle multimodal content
                        text_parts = []
                        for part in content:
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        user_message = " ".join(text_parts)
                    break

            if not user_message:
                raise HTTPException(
                    status_code=400, detail="No user message found"
                )

            # Validate the prompt
            if not self.message_controller.validate_prompt(user_message):
                raise HTTPException(
                    status_code=400, detail="Invalid input detected"
                )

            # Add user message to chat history
            self.message_controller.safe_add_message_to_history(
                "user", user_message
            )

            # Prepare messages for processing
            prepared_messages = (
                self.message_controller.prepare_messages_for_processing(
                    messages
                )
            )

            # Determine model to use
            model_type = "llm"  # Default to standard LLM
            if request.model:
                # Map model names to types if needed
                if "fast" in request.model.lower():
                    model_type = "fast"
                elif "intelligent" in request.model.lower():
                    model_type = "intelligent"
                elif "vlm" in request.model.lower():
                    model_type = "vlm"

            # Get model name based on type
            if model_type == "fast":
                model_name = self.config_obj.fast_llm_model_name
            elif model_type == "intelligent":
                model_name = self.config_obj.intelligent_llm_model_name
            elif model_type == "vlm":
                model_name = self.config_obj.vlm_model_name
            else:
                model_name = self.config_obj.llm_model_name

            # Handle streaming vs non-streaming
            if request.stream:
                return await self._handle_streaming_response(
                    prepared_messages, model_name, model_type, request
                )
            else:
                return await self._handle_non_streaming_response(
                    prepared_messages, model_name, model_type, request
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error processing chat completion: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    async def _handle_non_streaming_response(
        self,
        prepared_messages: List[Dict[str, Any]],
        model_name: str,
        model_type: str,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion response"""

        try:
            # Generate response directly using LLM service (API-specific)
            full_response = await self._generate_response_async(
                prepared_messages, model_name, model_type
            )

            # Add the response to the session
            self.message_controller.safe_add_message_to_history(
                "assistant", full_response
            )

            # Create response
            import time
            import uuid

            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant", content=full_response
                        ),
                        finish_reason="stop",
                    )
                ],
                usage={
                    "prompt_tokens": 0,  # TODO: Implement token counting
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )

            return response

        except Exception as e:
            logger.error("Error generating non-streaming response: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}",
            )

    async def _generate_response_async(
        self,
        prepared_messages: List[Dict[str, Any]],
        model_name: str,
        model_type: str,
    ) -> str:
        """Generate response asynchronously (API-specific)"""

        try:
            # Stream the response and collect it
            full_response = ""
            async for chunk in self.llm_service.generate_streaming_response(
                prepared_messages, model_name, model_type
            ):
                full_response += chunk

            return full_response

        except Exception as e:
            logger.error("Error in async response generation: %s", e)
            return (
                "I apologize, but I encountered an error generating the "
                "response. Please try again."
            )

    async def _handle_streaming_response(
        self,
        prepared_messages: List[Dict[str, Any]],
        model_name: str,
        model_type: str,
        request: ChatCompletionRequest,
    ) -> StreamingResponse:
        """Handle streaming chat completion response"""

        async def generate_stream():
            """Generate streaming response"""
            import time
            import uuid

            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_time = int(time.time())

            # Stream the response using the LLM service
            async for chunk in self.llm_service.generate_streaming_response(
                prepared_messages, model_name, model_type
            ):
                # Create streaming response chunk
                response_chunk = StreamingChatCompletionResponse(
                    id=response_id,
                    created=created_time,
                    model=model_name,
                    choices=[
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                )

                yield f"data: {response_chunk.model_dump_json()}\n\n"

            # Send final chunk
            final_chunk = StreamingChatCompletionResponse(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            )

            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )


# Initialize the API service
api_service = APIChatbotService()

# Create FastAPI app
app = FastAPI(
    title="BrandonBot API",
    description="RESTful API for the BrandonBot chatbot service",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup"""
    try:
        # Validate environment configuration
        config.validate_environment()
        logger.info("Environment configuration validated successfully")
    except ConfigurationError as e:
        logger.error("Configuration error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected validation error: %s", e)
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BrandonBot API",
        "version": "1.0.0",
        "endpoints": {
            "/agent": "POST - Chat completion endpoint (OpenAI compatible)"
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "brandonbot-api"}


@app.post("/agent")
async def chat_completion(request: ChatCompletionRequest):
    """
    Chat completion endpoint

    Accepts OpenAI-compatible chat completion requests and returns responses
    in the same format. Supports both streaming and non-streaming responses.
    """
    return await api_service.process_chat_completion(request)


@app.post("/agent/{session_id}")
async def chat_completion_with_session(
    session_id: str, request: ChatCompletionRequest
):
    """
    Chat completion endpoint with session ID

    Accepts OpenAI-compatible chat completion requests with a specific session
    ID for conversation continuity.
    """
    return await api_service.process_chat_completion(request, session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
