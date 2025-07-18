import asyncio
import json
import logging
from typing import Any, Dict, List

import streamlit as st
import streamlit.components.v1 as components
from controllers.message_controller import MessageController
from controllers.session_controller import SessionController
from models.chat_config import ChatConfig
from services import LLMService
from ui import ChatHistoryComponent
from utils.animated_loading import get_animated_loading_html
from utils.config import config
from utils.text_processing import StreamingThinkTagFilter, strip_think_tags


class ResponseController:
    """Controller for handling LLM response generation and display"""

    def __init__(
        self,
        config_obj: ChatConfig,
        llm_service: LLMService,
        message_controller: MessageController,
        session_controller: SessionController,
        chat_history_component: ChatHistoryComponent,
    ):
        """
        Initialize the response controller

        Args:
            config_obj: Application configuration
            llm_service: LLM service for generating responses
            message_controller: Message controller for history management
            session_controller: Session controller for state management
            chat_history_component: UI component for displaying chat history
        """
        self.config_obj = config_obj
        self.llm_service = llm_service
        self.message_controller = message_controller
        self.session_controller = session_controller
        self.chat_history_component = chat_history_component

    def generate_response_with_cleanup_separation(
        self, prepared_messages: List[Dict[str, Any]]
    ):
        """
        Generate and display streaming response, returning cleanup function for later execution
        This allows the spinner to close immediately after response is displayed

        Args:
            prepared_messages: Prepared messages for API call

        Returns:
            Cleanup function to be called after spinner closes
        """
        try:
            # Generate and display response chunks
            self._generate_response_chunks(prepared_messages)

            # Return cleanup function to be called after spinner
            def cleanup():
                self._display_response()

            return cleanup

        except Exception as e:
            self._handle_response_error(e)
            return lambda: None  # Return no-op function on error

    def _generate_response_chunks(self, prepared_messages: List[Dict[str, Any]]):
        """
        Generate response chunks from LLM service using simplified streaming

        Args:
            prepared_messages: Prepared messages for API call
        """
        # Get the default model type from configuration
        from tools.tool_llm_config import TOOL_LLM_TYPES

        model_type = TOOL_LLM_TYPES["tool_selection"]
        model_name = self.session_controller.get_model_name(model_type)

        logging.info(
            f"Generating LLM response with simplified streaming (model_type: {model_type}, model: {model_name})"
        )

        # Initialize response tracking
        response_chunks = []
        full_response = ""

        # Create the chat message container for streaming
        with st.chat_message("assistant", avatar=self.config_obj.assistant_avatar):
            message_placeholder = st.empty()

            # Display loading animation inside the chat message
            with message_placeholder.container():
                components.html(
                    get_animated_loading_html(
                        icon=None,  # No icon since we're inside chat message
                        dot_size=8,
                        dot_gap=8,
                        animation_height=30,  # Smaller height for inline display
                    ),
                    height=40,  # Compact height for inline display
                )

            try:
                # Use asyncio.run for simpler async handling
                full_response = asyncio.run(
                    self._async_stream_response(
                        prepared_messages,
                        model_name,
                        model_type,
                        message_placeholder,
                        response_chunks,
                    )
                )

            except Exception as e:
                logging.error(f"Error in streaming: {e}")
                error_message = (
                    "I apologize, but I encountered an error generating the response."
                )
                message_placeholder.markdown(error_message)
                full_response = error_message

            # After streaming is complete, check for UI elements (images, etc.)
            ui_elements = self._get_ui_elements_from_llm_service()

            if ui_elements:
                # Display the synthesized response with embedded UI elements
                self._display_response_with_ui_elements(
                    full_response, ui_elements, message_placeholder
                )
            else:
                # Check for single image generation response (backward compatibility)
                image_response = self._check_for_image_generation_response()

                if image_response:
                    # Clear the text placeholder and display image instead
                    message_placeholder.empty()
                    self._display_image_generation_response(image_response)
                    self._handle_tool_context()
                    full_response = (
                        ""  # No text response to add to history for image generation
                    )
                else:
                    # Handle tool context for text response
                    self._handle_tool_context()

        # Store response for further processing
        self._full_response = full_response
        self._response_chunks = response_chunks

    async def _async_stream_response(
        self,
        prepared_messages: List[Dict[str, Any]],
        model_name: str,
        model_type: str,
        message_placeholder,
        response_chunks: List[str],
    ) -> str:
        """
        Async helper to stream response

        Args:
            prepared_messages: Messages to send
            model_name: Model to use
            model_type: Type of model (fast, llm, intelligent)
            message_placeholder: Streamlit placeholder
            response_chunks: List to collect chunks

        Returns:
            Full response text
        """
        full_response = ""
        display_text = ""
        think_filter = StreamingThinkTagFilter()
        first_chunk_received = False

        try:
            # Stream response from LLM service with consistent model type
            async for chunk in self.llm_service.generate_streaming_response(
                prepared_messages, model_name, model_type
            ):
                response_chunks.append(chunk)
                full_response += chunk

                # Process chunk through streaming filter
                filtered_chunk = think_filter.process_chunk(chunk)
                if filtered_chunk:
                    display_text += filtered_chunk

                    # Update UI with filtered response
                    # The first update will replace the loading animation
                    if display_text.strip():
                        if not first_chunk_received:
                            # Clear the animation on first real content
                            message_placeholder.empty()
                            first_chunk_received = True
                        message_placeholder.markdown(display_text)

                # Small delay for visual effect
                await asyncio.sleep(0.01)

            # Process any remaining buffered content
            final_chunk = think_filter.flush()
            if final_chunk:
                display_text += final_chunk
                message_placeholder.markdown(display_text)

        except Exception as e:
            logging.error(f"Streaming error: {e}")
            raise

        return full_response

    def _display_response(self):
        """Handle post-streaming tasks like adding to chat history and cleanup"""
        # The streaming display is now handled in _generate_response_chunks
        # This method handles post-streaming tasks

        # Get the full response that was streamed
        full_response = getattr(self, '_full_response', "")

        # Add response to chat history if there's text content
        if full_response and full_response.strip():
            self.message_controller.update_chat_history(full_response, "assistant")

        # Clear processing flag
        self.session_controller.set_processing_state(False)

    def _check_for_image_generation_response(self) -> Dict[str, Any]:
        """
        Check if the LLM service has image generation tool responses

        Returns:
            Dict containing image response data if found, empty dict otherwise
        """
        logging.debug("Checking for image generation response")

        if (
            not hasattr(self.llm_service, 'last_tool_responses')
            or not self.llm_service.last_tool_responses
        ):
            logging.debug("No tool responses found")
            return {}

        for tool_response in self.llm_service.last_tool_responses:
            if (
                tool_response.get("role") == "direct_response"
                and tool_response.get("tool_name") == "generate_image"
            ):
                logging.debug("Found image generation tool response")
                try:
                    tool_result = tool_response.get("tool_result")

                    if tool_result and hasattr(tool_result, 'success'):
                        response_data = {
                            "success": tool_result.success,
                            "image_data": getattr(tool_result, 'image_data', None),
                            "original_prompt": getattr(
                                tool_result, 'original_prompt', ''
                            ),
                            "enhanced_prompt": getattr(
                                tool_result, 'enhanced_prompt', ''
                            ),
                            "error_message": getattr(
                                tool_result, 'error_message', None
                            ),
                        }

                        if response_data["success"] and response_data["image_data"]:
                            return response_data
                        elif not response_data["success"]:
                            return response_data

                except Exception as e:
                    logging.error(
                        f"Error extracting image generation tool response: {e}"
                    )
                    continue

        return {}

    def _display_image_generation_response(self, image_response: Dict[str, Any]):
        """
        Display image generation response from tool calls

        Args:
            image_response: Dict containing image response data
        """
        logging.info("Displaying image generation response")

        try:
            success = image_response.get("success")
            image_data = image_response.get("image_data")

            if success and image_data:
                self._display_successful_image_generation(image_response)
            else:
                self._display_image_generation_error(image_response)

        except Exception as e:
            logging.error(f"Error displaying image generation response: {e}")
            error_msg = "I encountered an error while displaying the generated image."
            st.markdown(error_msg)
            self.message_controller.safe_add_message_to_history("assistant", error_msg)

    def _display_successful_image_generation(self, image_response: Dict[str, Any]):
        """
        Display successful image generation

        Args:
            image_response: Image response data
        """
        from utils.image import base64_to_pil_image

        image_data = image_response["image_data"]
        enhanced_prompt = image_response.get("enhanced_prompt", "Generated image")
        original_prompt = image_response.get("original_prompt", "")

        # Convert base64 to PIL Image for display
        generated_image = base64_to_pil_image(image_data)

        if generated_image:
            # Display the generated image
            st.image(generated_image, caption=enhanced_prompt, use_container_width=True)

            # Store image in session state
            image_id = self.session_controller.store_generated_image(
                image_data, enhanced_prompt, original_prompt
            )

            # Store the generated image in session state for easy access by other tools
            st.session_state.current_image_base64 = image_data
            st.session_state.current_image_filename = f"generated_{image_id}.png"
            st.session_state.current_image_id = image_id

            logging.info(
                f"Stored generated image in session state - image_id: {image_id}, data length: {len(image_data)}"
            )

            # Create history message
            history_message = {
                "type": "image",
                "image_id": image_id,
                # "text": f"🎨 Generated image with prompt: **{enhanced_prompt}**",
                "enhanced_prompt": enhanced_prompt,
                "original_prompt": original_prompt,
            }

            if enhanced_prompt != original_prompt and original_prompt:
                history_message[
                    "text"
                ] += f"\n\n*Enhanced from original request: \"{original_prompt}\"*"

            # Add image metadata to chat history
            self.message_controller.safe_add_message_to_history(
                "assistant", history_message
            )

            logging.debug(
                f"Successfully displayed generated image with prompt: {enhanced_prompt}"
            )
        else:
            error_msg = "Generated image but failed to display it properly."
            logging.error("Failed to convert base64 to PIL image")
            st.markdown(error_msg)
            self.message_controller.safe_add_message_to_history("assistant", error_msg)

    def _display_image_generation_error(self, image_response: Dict[str, Any]):
        """
        Display image generation error

        Args:
            image_response: Image response data with error
        """
        error_message = image_response.get("error_message", "Failed to generate image.")
        st.markdown(f"**Image Generation Error:** {error_message}")
        self.message_controller.safe_add_message_to_history(
            "assistant", f"Image generation failed: {error_message}"
        )

        # Show the enhanced prompt that was attempted
        enhanced_prompt = image_response.get("enhanced_prompt", "")
        if enhanced_prompt:
            st.markdown(f"**Attempted prompt:** {enhanced_prompt}")

    def _handle_tool_context(self):
        """Extract and display tool context from LLM responses"""
        tool_context = self._extract_tool_context_from_llm_responses()
        if tool_context:
            # Display the context expander for immediate user verification
            self.chat_history_component.display_context_expander(tool_context)
            # Store tool context in session state for chat history display
            self.session_controller.store_tool_context(tool_context)
            logging.info("Displayed and stored tool context for verification")

    def _extract_tool_context_from_llm_responses(self) -> str:
        """
        Extract context from the LLM service's last tool responses

        Returns:
            Formatted context string from tool responses, empty if none found
        """
        if (
            not hasattr(self.llm_service, 'last_tool_responses')
            or not self.llm_service.last_tool_responses
        ):
            return ""

        tool_contexts = []

        for tool_response in self.llm_service.last_tool_responses:
            # Only include regular tool responses, skip direct responses
            if tool_response.get("role") == "tool":
                try:
                    # Skip image generation responses
                    if tool_response.get("tool_name") == "generate_image":
                        continue

                    context = self._extract_context_from_tool_response(tool_response)
                    if context:
                        tool_contexts.append(context)

                except Exception as e:
                    logging.error(f"Error extracting tool context: {e}")
                    continue

        if tool_contexts:
            combined_context = config.tool_context.CONTEXT_SEPARATOR.join(tool_contexts)
            logging.info(f"Extracted tool context with {len(tool_contexts)} entries")
            return combined_context

        return ""

    def _extract_context_from_tool_response(self, tool_response: Dict[str, Any]) -> str:
        """
        Extract context from a single tool response

        Args:
            tool_response: Single tool response to extract context from

        Returns:
            Formatted context string or empty string
        """
        # Since we now only process regular tool responses, not direct responses
        tool_content = tool_response.get("content", "")

        if isinstance(tool_content, str):
            try:
                tool_data = json.loads(tool_content)
                if isinstance(tool_data, dict):
                    return self._format_tool_data_context(tool_data)
            except json.JSONDecodeError:
                # If not JSON, treat as plain text
                if tool_content.strip():
                    return f"**Tool Response:**\n{tool_content.strip()}"

        return ""

    def _format_tool_data_context(self, tool_data: Dict[str, Any]) -> str:
        """
        Format tool data into readable context

        Args:
            tool_data: Parsed tool response data

        Returns:
            Formatted context string
        """
        # Skip image generation data using centralized keywords
        if any(key in tool_data for key in config.image_generation.DETECTION_KEYWORDS):
            return ""

        # Check for formatted_results first (preferred format)
        if "formatted_results" in tool_data:
            formatted_results = tool_data["formatted_results"]
            if formatted_results and formatted_results.strip():
                return f"**Tool Response Data:**\n\n{formatted_results}"

        # Handle different tool response formats
        if "results" in tool_data:
            results = tool_data["results"]
            if isinstance(results, list) and results:
                return f"**Tool found {len(results)} results**"
            elif isinstance(results, str) and results.strip():
                return f"**Tool Response:**\n\n{results}"

        # Handle weather tool response format
        if "location" in tool_data and "current" in tool_data:
            location = tool_data.get("location", "Unknown")
            current = tool_data.get("current", {})
            temp = current.get("temperature", "N/A")
            return f"**Weather data for {location}** (Current: {temp}°F)"

        # Handle PDF content retrieval
        if (
            "filename" in tool_data
            and "content" in tool_data
            and isinstance(tool_data.get("content"), list)
        ):
            return self._format_pdf_content_context(tool_data)

        # Handle web extraction results
        if "url" in tool_data and "content" in tool_data and "success" in tool_data:
            url = tool_data.get("url", "Unknown URL")
            content = tool_data.get("content", "")
            success = tool_data.get("success", False)

            if success and content.strip():
                # Show extracted content with URL context
                content_preview = content[: config.tool_context.PREVIEW_TEXT_LENGTH]
                if len(content) > config.tool_context.PREVIEW_TEXT_LENGTH:
                    content_preview += config.tool_context.CONTEXT_TRUNCATION_SUFFIX

                return f"**Web Content Extracted from:** {url}\n\n{content_preview}"
            elif not success:
                error_msg = tool_data.get("error_message", "Extraction failed")
                return f"**Web Extraction Failed:** {url}\n\nError: {error_msg}"
            else:
                return f"**Web Extraction:** {url} (No content extracted)"

        # Generic fallback
        summary_parts = []
        if "query" in tool_data:
            summary_parts.append(f"Query: {tool_data['query']}")
        if "total_results" in tool_data:
            summary_parts.append(f"Results: {tool_data['total_results']}")
        if summary_parts:
            return f"**Tool Response:** {', '.join(summary_parts)}"

        return ""

    def _format_pdf_content_context(self, tool_data: Dict[str, Any]) -> str:
        """
        Format PDF content retrieval context

        Args:
            tool_data: PDF tool response data

        Returns:
            Formatted PDF context string
        """
        filename = tool_data.get("filename", "Unknown PDF")
        content = tool_data.get("content", [])
        pages_requested = tool_data.get("pages_requested", [])

        if content:
            context_parts = [f"**PDF Content Retrieved from:** {filename}"]
            if pages_requested:
                context_parts.append(
                    f"**Pages:** {', '.join(map(str, pages_requested))}"
                )

            # Show retrieved content (truncated for readability) using config
            max_pages = config.tool_context.MAX_PAGES_IN_CONTEXT
            preview_length = config.tool_context.PREVIEW_TEXT_LENGTH
            truncation_suffix = config.tool_context.CONTEXT_TRUNCATION_SUFFIX

            for page_content in content[:max_pages]:
                page_num = page_content.get("page_number", 1)
                page_text = page_content.get("text", "")
                if page_text:
                    # Truncate long text but show meaningful preview
                    preview_text = page_text[:preview_length] + (
                        truncation_suffix if len(page_text) > preview_length else ""
                    )
                    context_parts.append(f"**Page {page_num}:**\n{preview_text}")

            if len(content) > max_pages:
                context_parts.append(f"... and {len(content) - max_pages} more pages")

            return "\n\n".join(context_parts)
        else:
            return (
                f"**PDF Query Result:** {tool_data.get('message', 'No content found')}"
            )

    def _get_ui_elements_from_llm_service(self) -> List[Dict[str, Any]]:
        """
        Get UI elements (images, etc.) from the LLM service

        Returns:
            List of UI element dictionaries
        """
        if hasattr(self.llm_service, 'last_ui_elements'):
            return self.llm_service.last_ui_elements
        return []

    def _display_response_with_ui_elements(
        self, full_response: str, ui_elements: List[Dict[str, Any]], message_placeholder
    ):
        """
        Display the synthesized response with embedded UI elements

        Args:
            full_response: The synthesized text response from LLM
            ui_elements: List of UI elements to display
            message_placeholder: Streamlit placeholder for the message
        """
        from utils.image import base64_to_pil_image

        # First, display the full text response
        display_response = strip_think_tags(full_response)
        message_placeholder.markdown(display_response)

        # Then display each UI element
        for element in ui_elements:
            if element["type"] == "image":
                image_data = element["data"]
                if image_data["success"] and image_data["image_data"]:
                    # Convert base64 to PIL Image
                    generated_image = base64_to_pil_image(image_data["image_data"])
                    if generated_image:
                        # Display the image below the text
                        st.image(
                            generated_image,
                            caption=image_data["enhanced_prompt"],
                            use_container_width=True,
                        )

                        # Store image in session state
                        image_id = self.session_controller.store_generated_image(
                            image_data["image_data"],
                            image_data["enhanced_prompt"],
                            image_data["original_prompt"],
                        )

                        # Store the generated image in session state for easy access by other tools
                        st.session_state.current_image_base64 = image_data["image_data"]
                        st.session_state.current_image_filename = (
                            f"generated_{image_id}.png"
                        )
                        st.session_state.current_image_id = image_id

                        logging.info(
                            f"Stored generated image in session state - image_id: {image_id}, data length: {len(image_data['image_data'])}"
                        )

                        # Create history message for the image
                        history_message = {
                            "type": "image",
                            "image_id": image_id,
                            # "text": f"Generated image with prompt: {image_data['enhanced_prompt']}",
                            "enhanced_prompt": image_data["enhanced_prompt"],
                            "original_prompt": image_data["original_prompt"],
                        }

                        # Add image to chat history
                        self.message_controller.safe_add_message_to_history(
                            "assistant", history_message
                        )

        # Handle tool context
        self._handle_tool_context()

    def _handle_response_error(self, error: Exception):
        """
        Handle errors during response generation

        Args:
            error: The exception that occurred
        """
        error_msg = f"Error generating response: ```json\n{error}\n```"
        logging.error(error_msg)
        self.message_controller.update_chat_history(
            "I apologize, but I encountered an error while generating a response.",
            "assistant",
        )
        self.session_controller.set_processing_state(False)
        st.rerun()
