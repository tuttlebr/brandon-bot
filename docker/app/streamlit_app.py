import asyncio
import logging
import random
from typing import Any, Dict, Generator, List

import streamlit as st
from models import ChatConfig
from services import ChatService, ImageService, LLMService
from ui import ChatHistoryComponent, apply_custom_styles, get_typing_indicator_html
from utils.image import pil_image_to_base64
from utils.system_prompt import SYSTEM_PROMPT, TOOL_PROMPT, greeting_prompt


class StreamlitChatApp:
    """Main Streamlit chat application that orchestrates UI and services"""

    def __init__(self):
        """Initialize the Streamlit chat application"""
        # Initialize configuration
        self.config = ChatConfig.from_environment()

        # Apply custom styling and setup UI
        apply_custom_styles()
        st.markdown(
            f'<h1 style="color: #76b900; text-align: center;">{greeting_prompt()}</h1>', unsafe_allow_html=True,
        )

        # Initialize services
        self.chat_service = ChatService(self.config)
        self.image_service = ImageService(self.config)
        self.llm_service = LLMService(self.config)

        # Initialize UI components
        self.chat_history_component = ChatHistoryComponent(self.config)

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state with default values"""
        if not hasattr(st.session_state, "initialized"):
            st.session_state.initialized = True
            st.session_state.fast_llm_model_name = self.config.fast_llm_model_name
            st.session_state.llm_model_name = self.config.llm_model_name
            st.session_state.intelligent_llm_model_name = self.config.intelligent_llm_model_name
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.current_page = 0

            # Initialize image storage
            if 'generated_images' not in st.session_state:
                st.session_state.generated_images = {}

        # Clean up old image data periodically
        self._cleanup_old_images()

    def _cleanup_old_images(self, max_images: int = 50):
        """
        Clean up old image data from session state to prevent memory issues

        Args:
            max_images: Maximum number of images to keep in session state
        """
        if hasattr(st.session_state, 'generated_images') and st.session_state.generated_images:
            current_count = len(st.session_state.generated_images)

            if current_count > max_images:
                # Sort by image_id (which includes timestamp) and keep the most recent ones
                sorted_images = sorted(st.session_state.generated_images.items(), key=lambda x: x[0], reverse=True)

                # Keep only the most recent max_images
                st.session_state.generated_images = dict(sorted_images[:max_images])

                removed_count = current_count - max_images
                logging.info(
                    f"Cleaned up {removed_count} old images from session state. "
                    f"Kept {max_images} most recent images."
                )

    def display_chat_history(self):
        """Display the chat history using the chat history component"""
        self.chat_history_component.display_chat_history(st.session_state.messages)

    def display_tool_context_expander(self, context: str):
        """
        Display tool context in an expandable section for user verification

        Args:
            context: Tool response context to display
        """
        if context:
            self.chat_history_component.display_context_expander(context)

    def _clean_chat_history_of_tool_calls(self):
        """
        Clean existing chat history to remove any messages containing tool call instructions
        """
        if not hasattr(st.session_state, 'messages') or not st.session_state.messages:
            return

        original_count = len(st.session_state.messages)
        cleaned_messages = []

        for message in st.session_state.messages:
            content = message.get("content", "")

            # Keep system messages and non-string content as-is
            if message.get("role") == "system" or not isinstance(content, str):
                cleaned_messages.append(message)
                continue

            # Check if message contains tool call instructions
            if self._contains_tool_call_instructions(content):
                logging.warning(
                    f"Removing {message.get('role', 'unknown')} message with tool call instructions from chat history"
                )
                continue

            # Keep clean messages
            cleaned_messages.append(message)

        if len(cleaned_messages) != original_count:
            st.session_state.messages = cleaned_messages
            logging.info(f"Cleaned chat history: {original_count} -> {len(cleaned_messages)} messages")

    def process_prompt(self, prompt: str):
        """
        Process user prompt and generate response

        Args:
            prompt: The user's text prompt
        """
        # Clean the prompt
        prompt = prompt.strip()

        # Clear any previous tool context when processing a new user message
        if hasattr(st.session_state, 'last_tool_context'):
            st.session_state.last_tool_context = None

        # Clear previous tool responses from LLM service
        if hasattr(self.llm_service, 'last_tool_responses'):
            self.llm_service.last_tool_responses = []

        # Validate that the prompt doesn't contain tool call instructions (shouldn't happen from user input)
        if self._contains_tool_call_instructions(prompt):
            logging.error("User prompt contains tool call instructions - this should not happen")
            st.error("Invalid input detected. Please try again with a different message.")
            st.session_state.processing = False
            return

        # Clean existing chat history of any tool call instructions
        self._clean_chat_history_of_tool_calls()

        # Clean previous chat history from context
        st.session_state.messages = self.chat_service.clean_chat_history_context(st.session_state.messages)

        # Display user message immediately in the UI
        with st.chat_message("user", avatar=self.config.user_avatar):
            st.markdown(prompt[:2048] + "..." if len(prompt) > 2048 else prompt)

        # Add user message to chat history using safe method
        self._safe_add_message_to_history("user", prompt)

        try:
            prepared_messages = self.chat_service.prepare_messages_for_api(st.session_state.messages)

            # Generate and display response
            self._generate_and_display_response(prepared_messages)

        except Exception as e:
            logging.error(f"Error processing prompt: {e}")
            st.error("An error occurred while processing your request. Please try again.")
            st.session_state.processing = False

    def _extract_tool_context_from_llm_responses(self) -> str:
        """
        Extract context from the LLM service's last tool responses

        Returns:
            Formatted context string from tool responses, empty if none found
        """
        if not hasattr(self.llm_service, 'last_tool_responses') or not self.llm_service.last_tool_responses:
            logging.debug("No tool responses found in LLM service")
            return ""

        logging.debug(f"Found {len(self.llm_service.last_tool_responses)} tool responses to process")

        tool_contexts = []

        for tool_response in self.llm_service.last_tool_responses:
            logging.debug(f"Processing tool response: {tool_response}")
            # Handle both regular tool responses and direct responses (like image generation)
            if tool_response.get("role") in ["tool", "direct_response"]:
                try:
                    # Skip image generation responses - they are handled separately
                    # Check multiple ways image generation might be identified
                    tool_name = tool_response.get("tool_name")
                    if tool_name == "generate_image":
                        logging.debug(
                            f"Skipping image generation response (tool_name={tool_name}) in tool context extraction"
                        )
                        continue

                    # Additional check: if this is a direct response with image-related content, skip it
                    if tool_response.get("role") == "direct_response":
                        tool_content_raw = tool_response.get("content", "")
                        # Check if content contains image generation indicators
                        if isinstance(tool_content_raw, str) and any(
                            keyword in tool_content_raw.lower()
                            for keyword in [
                                "image_data",
                                "enhanced_prompt",
                                "original_prompt",
                                "cfg_scale",
                                "dimensions",
                            ]
                        ):
                            logging.debug("Skipping likely image generation response based on content analysis")
                            continue

                    # For direct responses, extract content from the tool result
                    if tool_response.get("role") == "direct_response":
                        tool_result = tool_response.get("tool_result")
                        if tool_result and hasattr(tool_result, 'result'):
                            tool_content = tool_result.result
                        else:
                            tool_content = tool_response.get("content", "")
                    else:
                        tool_content = tool_response.get("content", "")

                    logging.debug(f"Tool content type: {type(tool_content)}, length: {len(str(tool_content))}")
                    if isinstance(tool_content, str):
                        # Try to parse as JSON to extract formatted results
                        import json

                        try:
                            tool_data = json.loads(tool_content)
                            if isinstance(tool_data, dict):
                                # Additional safety check: skip if this looks like image generation data
                                if any(
                                    key in tool_data
                                    for key in [
                                        "image_data",
                                        "enhanced_prompt",
                                        "original_prompt",
                                        "cfg_scale",
                                        "dimensions",
                                    ]
                                ):
                                    logging.debug("Skipping tool response that contains image generation data")
                                    continue

                                # Check for formatted_results first (preferred format)
                                if "formatted_results" in tool_data:
                                    formatted_results = tool_data["formatted_results"]
                                    if formatted_results and formatted_results.strip():
                                        tool_contexts.append(f"**Tool Response Data:**\n{formatted_results}")
                                        logging.info(f"Found formatted_results from tool response")

                                # Check for other common tool response formats
                                elif "results" in tool_data:
                                    results = tool_data["results"]
                                    if isinstance(results, list) and results:
                                        tool_contexts.append(f"**Tool found {len(results)} results**")
                                        logging.info(f"Found {len(results)} results from tool response")
                                    elif isinstance(results, str) and results.strip():
                                        tool_contexts.append(f"**Tool Response:**\n{results}")
                                        logging.info(f"Found string results from tool response")

                                # Handle weather tool response format
                                elif "location" in tool_data and "current" in tool_data:
                                    location = tool_data.get("location", "Unknown")
                                    current = tool_data.get("current", {})
                                    temp = current.get("temperature", "N/A")
                                    tool_contexts.append(f"**Weather data for {location}** (Current: {temp}°F)")
                                    logging.info(f"Found weather tool response for {location}")

                                # Generic fallback for other structured data
                                else:
                                    # Try to create a summary of the tool response
                                    summary_parts = []
                                    if "query" in tool_data:
                                        summary_parts.append(f"Query: {tool_data['query']}")
                                    if "total_results" in tool_data:
                                        summary_parts.append(f"Results: {tool_data['total_results']}")
                                    if summary_parts:
                                        tool_contexts.append(f"**Tool Response:** {', '.join(summary_parts)}")

                        except json.JSONDecodeError:
                            # If not JSON, treat as plain text but format it nicely
                            if tool_content.strip():
                                tool_contexts.append(f"**Tool Response:**\n{tool_content.strip()}")
                                logging.info(f"Found plain text tool response")
                except Exception as e:
                    logging.error(f"Error extracting tool context: {e}")
                    continue

        # Combine all tool contexts with proper formatting
        if tool_contexts:
            combined_context = "\n\n---\n\n".join(tool_contexts)
            logging.info(
                f"Extracted tool context with {len(tool_contexts)} entries, total length: {len(combined_context)}"
            )
            return combined_context

        logging.debug("No tool context found in LLM responses")
        return ""

    def _extract_tool_context_from_messages(self, messages: List[Dict[str, Any]] = None) -> str:
        """
        Extract context from tool responses in the message history

        Args:
            messages: Optional list of messages to search, defaults to session state messages

        Returns:
            Formatted context string from tool responses, empty if none found
        """
        # Use provided messages or fall back to session state
        search_messages = messages if messages is not None else st.session_state.messages

        tool_contexts = []

        # Look for recent tool messages
        for message in reversed(search_messages):
            if message.get("role") == "tool":
                try:
                    # Parse tool response content
                    tool_content = message.get("content", "")
                    if isinstance(tool_content, str):
                        # Try to parse as JSON to extract formatted results
                        import json

                        try:
                            tool_data = json.loads(tool_content)
                            if isinstance(tool_data, dict):
                                # Additional safety check: skip if this looks like image generation data
                                if any(
                                    key in tool_data
                                    for key in [
                                        "image_data",
                                        "enhanced_prompt",
                                        "original_prompt",
                                        "cfg_scale",
                                        "dimensions",
                                    ]
                                ):
                                    logging.debug("Skipping tool response that contains image generation data")
                                    continue

                                # Check for formatted_results first (preferred format)
                                if "formatted_results" in tool_data:
                                    formatted_results = tool_data["formatted_results"]
                                    if formatted_results and formatted_results.strip():
                                        tool_contexts.append(f"**Tool Response Data:**\n{formatted_results}")
                                        logging.info(f"Found formatted_results from tool response")

                                # Check for other common tool response formats
                                elif "results" in tool_data:
                                    results = tool_data["results"]
                                    if isinstance(results, list) and results:
                                        tool_contexts.append(f"**Tool found {len(results)} results**")
                                        logging.info(f"Found {len(results)} results from tool response")
                                    elif isinstance(results, str) and results.strip():
                                        tool_contexts.append(f"**Tool Response:**\n{results}")
                                        logging.info(f"Found string results from tool response")

                                # Handle weather tool response format
                                elif "location" in tool_data and "current" in tool_data:
                                    location = tool_data.get("location", "Unknown")
                                    current = tool_data.get("current", {})
                                    temp = current.get("temperature", "N/A")
                                    tool_contexts.append(f"**Weather data for {location}** (Current: {temp}°F)")
                                    logging.info(f"Found weather tool response for {location}")

                                # Generic fallback for other structured data
                                else:
                                    # Try to create a summary of the tool response
                                    summary_parts = []
                                    if "query" in tool_data:
                                        summary_parts.append(f"Query: {tool_data['query']}")
                                    if "total_results" in tool_data:
                                        summary_parts.append(f"Results: {tool_data['total_results']}")
                                    if summary_parts:
                                        tool_contexts.append(f"**Tool Response:** {', '.join(summary_parts)}")

                        except json.JSONDecodeError:
                            # If not JSON, treat as plain text but format it nicely
                            if tool_content.strip():
                                tool_contexts.append(f"**Tool Response:**\n{tool_content.strip()}")
                                logging.info(f"Found plain text tool response")
                except Exception as e:
                    logging.error(f"Error extracting tool context: {e}")
                    continue

        # Combine all tool contexts with proper formatting
        if tool_contexts:
            combined_context = "\n\n---\n\n".join(tool_contexts)
            logging.info(
                f"Extracted tool context with {len(tool_contexts)} entries, total length: {len(combined_context)}"
            )
            return combined_context

        logging.debug("No tool context found in messages")
        return ""

    def _run_async_streaming_response(self, prepared_messages: list, model: str) -> Generator[str, None, str]:
        """
        Wrapper to run async streaming response in a synchronous context

        Args:
            prepared_messages: Prepared messages for API call
            model: Model name to use

        Yields:
            Filtered content chunks

        Returns:
            Complete filtered response
        """
        try:

            async def collect_and_yield():
                """Collect all chunks from async generator"""
                chunks = []
                async_gen = self.llm_service.generate_streaming_response(prepared_messages, model)
                async for chunk in async_gen:
                    chunks.append(chunk)
                return chunks

            # Run the async function
            chunks = asyncio.run(collect_and_yield())

            # Yield all chunks for streaming display
            full_response = ""
            for chunk in chunks:
                full_response += chunk
                yield chunk

            return full_response

        except Exception as e:
            error_msg = f"Error in async streaming wrapper: {e}"
            logging.error(error_msg)
            yield "I apologize, but I encountered an error while generating a response."
            return "Error occurred during response generation"

    def _run_async_streaming_response_with_image_check(
        self, prepared_messages: list, model: str
    ) -> Generator[str, None, str]:
        """
        Wrapper to run async streaming response that checks for image generation first

        Args:
            prepared_messages: Prepared messages for API call
            model: Model name to use

        Yields:
            Filtered content chunks (empty if image generation detected)

        Returns:
            Complete filtered response
        """
        try:

            async def collect_and_yield():
                """Collect all chunks from async generator"""
                chunks = []
                async_gen = self.llm_service.generate_streaming_response(prepared_messages, model)
                async for chunk in async_gen:
                    chunks.append(chunk)
                return chunks

            # Run the async function to populate tool responses
            chunks = asyncio.run(collect_and_yield())

            # Check if this is an image generation response
            image_response = self._check_for_image_generation_response()
            if image_response:
                # If it's image generation, don't yield any text chunks
                logging.info("Detected image generation response, suppressing text output")
                return ""

            # For non-image responses, yield all chunks for streaming display
            full_response = ""
            for chunk in chunks:
                full_response += chunk
                yield chunk

            return full_response

        except Exception as e:
            error_msg = f"Error in async streaming wrapper with image check: {e}"
            logging.error(error_msg)
            # Check if it's an image generation error before yielding error message
            image_response = self._check_for_image_generation_response()
            if not image_response:
                yield "I apologize, but I encountered an error while generating a response."
            return "Error occurred during response generation"

    def _generate_and_display_response(self, prepared_messages: list):
        """
        Generate and display streaming response from LLM

        Args:
            prepared_messages: Prepared messages for API call
        """
        try:
            # Show spinner during the slow LLM API call
            random_icon = ["🤖", "🧠", "🤔", "🤓", "⚡"]
            with st.spinner(f"{random_icon[random.randint(0, len(random_icon) - 1)]}"):
                # FIRST: Generate the LLM response to populate tool responses (without streaming yet)
                logging.info("Main flow: Running LLM service to populate tool responses")

                async def collect_responses():
                    """Collect responses to populate tool responses"""
                    chunks = []
                    async_gen = self.llm_service.generate_streaming_response(
                        prepared_messages, st.session_state["fast_llm_model_name"]
                    )
                    async for chunk in async_gen:
                        chunks.append(chunk)
                    return chunks

                # Run the async function to populate tool responses
                response_chunks = asyncio.run(collect_responses())
                logging.info(f"Main flow: LLM service completed, got {len(response_chunks)} chunks")

            # Create a chat message container for the assistant response (no spinner needed here)
            with st.chat_message("assistant", avatar=self.config.assistant_avatar):
                # NOW check if we have image generation tool responses
                image_response = self._check_for_image_generation_response()
                logging.debug(f"Main flow: image_response = {image_response}")

                # Handle image generation response if present
                if image_response:
                    logging.info("Main flow: Calling _display_image_generation_response")
                    self._display_image_generation_response(image_response, full_response="")
                    full_response = ""  # No text response to add to history for image generation
                    logging.info("Main flow: Finished _display_image_generation_response")

                # Display text response only if no image generation is present
                text_response = "".join(response_chunks)
                if text_response.strip() and not image_response:
                    logging.info("Main flow: Displaying text response (no image generation)")
                    st.markdown(text_response)
                    full_response = text_response
                elif text_response.strip() and image_response:
                    logging.info("Main flow: Suppressing text response due to image generation")
                    # Check if the text response contains non-image content that should be displayed
                    if not self._is_image_generation_json(text_response):
                        logging.info("Main flow: Text response contains non-image content, displaying it")
                        st.markdown(text_response)

                # ALWAYS extract and display tool context from LLM service tool responses
                # This ensures other tools' results are shown even when image generation occurs
                tool_context = self._extract_tool_context_from_llm_responses()
                if tool_context:
                    # Display the context expander for immediate user verification
                    self.display_tool_context_expander(tool_context)
                    # Store tool context in session state for chat history display
                    st.session_state.last_tool_context = tool_context
                    logging.info("Displayed and stored tool context for verification")
                else:
                    logging.debug("No tool context found to display")

            # Add the complete response to chat history (only if there's a text response)
            if full_response and full_response.strip():
                self._update_chat_history(full_response, "assistant")

            # Clear processing flag - no need to rerun since we've already displayed the response
            st.session_state.processing = False

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            logging.error(error_msg)
            self._update_chat_history(
                "I apologize, but I encountered an error while generating a response.", "assistant"
            )
            # Clear processing flag even on error
            st.session_state.processing = False
            st.rerun()

    def _is_image_generation_json(self, text: str) -> bool:
        """
        Check if the text response is image generation JSON that should be suppressed

        Args:
            text: The text response to check

        Returns:
            True if this appears to be image generation JSON, False otherwise
        """
        if not text or not isinstance(text, str):
            return False

        text_lower = text.strip().lower()

        # Check if it looks like JSON
        if not (text_lower.startswith('{') and text_lower.endswith('}')):
            return False

        # Check if it contains image generation indicators
        image_keywords = [
            "enhanced_prompt",
            "original_prompt",
            "cfg_scale",
            "dimensions",
            "successfully generated",
            "image with cfg_scale",
        ]

        return any(keyword in text_lower for keyword in image_keywords)

    def _check_for_image_generation_response(self) -> Dict[str, Any]:
        """
        Check if the LLM service has image generation tool responses

        Returns:
            Dict containing image response data if found, empty dict otherwise
        """
        logging.debug("_check_for_image_generation_response: Starting check")

        if not hasattr(self.llm_service, 'last_tool_responses') or not self.llm_service.last_tool_responses:
            logging.debug("_check_for_image_generation_response: No tool responses found")
            return {}

        logging.debug(
            f"_check_for_image_generation_response: Found {len(self.llm_service.last_tool_responses)} tool responses"
        )

        for tool_response in self.llm_service.last_tool_responses:
            logging.debug(
                f"_check_for_image_generation_response: Checking tool response: role={tool_response.get('role')}, tool_name={tool_response.get('tool_name')}"
            )

            if tool_response.get("role") == "direct_response" and tool_response.get("tool_name") == "generate_image":
                logging.debug("_check_for_image_generation_response: Found image generation tool response")
                try:
                    # Get the full tool result object directly (no need to parse JSON)
                    tool_result = tool_response.get("tool_result")
                    logging.debug(
                        f"_check_for_image_generation_response: tool_result = {type(tool_result)}, has success attr: {hasattr(tool_result, 'success') if tool_result else False}"
                    )

                    if tool_result and hasattr(tool_result, 'success'):
                        # Convert the tool result to a dict for easier handling
                        response_data = {
                            "success": tool_result.success,
                            "image_data": getattr(tool_result, 'image_data', None),
                            "original_prompt": getattr(tool_result, 'original_prompt', ''),
                            "enhanced_prompt": getattr(tool_result, 'enhanced_prompt', ''),
                            "error_message": getattr(tool_result, 'error_message', None),
                        }

                        logging.debug(
                            f"_check_for_image_generation_response: response_data success={response_data['success']}, has_image_data={bool(response_data['image_data'])}"
                        )

                        if response_data["success"] and response_data["image_data"]:
                            logging.debug(
                                "_check_for_image_generation_response: Returning successful image generation response"
                            )
                            return response_data
                        elif not response_data["success"]:
                            logging.debug(
                                "_check_for_image_generation_response: Returning failed image generation response"
                            )
                            return response_data

                except Exception as e:
                    logging.error(f"Error extracting image generation tool response: {e}")
                    continue

        logging.debug("_check_for_image_generation_response: No image generation response found, returning empty dict")
        return {}

    def _display_image_generation_response(self, image_response: Dict[str, Any], full_response: str = ""):
        """
        Display image generation response from tool calls

        Args:
            image_response: Dict containing image response data
            full_response: Any additional text response
        """
        logging.info(f"Displaying image generation response...")
        try:
            # Debug the response structure
            success = image_response.get("success")
            image_data = image_response.get("image_data")
            logging.info(f"Response success: {success}, has image_data: {bool(image_data)}")

            if success and image_data:
                # Successful image generation
                from utils.image import base64_to_pil_image

                image_data = image_response["image_data"]
                enhanced_prompt = image_response.get("enhanced_prompt", "Generated image")
                original_prompt = image_response.get("original_prompt", "")

                # Convert base64 to PIL Image for display (st.image needs PIL Image, not base64 string)
                generated_image = base64_to_pil_image(image_data)

                if generated_image:
                    # Display the generated image
                    st.image(generated_image, caption=f"{enhanced_prompt}", use_container_width=True)

                    # Display information about prompt enhancement
                    # if enhanced_prompt != original_prompt and original_prompt:
                    # st.markdown(f"**Enhanced from:** {original_prompt}")
                    # st.markdown(f"{enhanced_prompt}")

                    # Initialize session state for storing image data if not exists
                    if 'generated_images' not in st.session_state:
                        st.session_state.generated_images = {}

                    # Generate a unique ID for this image
                    import time

                    image_id = f"img_{int(time.time() * 1000)}"

                    # Store image data in session state for visual persistence
                    st.session_state.generated_images[image_id] = {
                        'image_data': image_data,
                        'enhanced_prompt': enhanced_prompt,
                        'original_prompt': original_prompt,
                    }

                    # Store lightweight image metadata in chat history (NOT the base64 data)
                    history_message = {
                        "type": "image",
                        "image_id": image_id,
                        "text": f"🎨 Generated image with prompt: **{enhanced_prompt}**",
                        "enhanced_prompt": enhanced_prompt,
                        "original_prompt": original_prompt,
                    }

                    if enhanced_prompt != original_prompt and original_prompt:
                        history_message["text"] += f"\n\n*Enhanced from original request: \"{original_prompt}\"*"

                    # Add image metadata to chat history (no base64 data)
                    self._safe_add_message_to_history("assistant", history_message)

                    logging.debug(f"Successfully displayed generated image with enhanced prompt: {enhanced_prompt}")
                else:
                    # Error converting image data
                    error_msg = "Generated image but failed to display it properly."
                    logging.error(
                        f"Failed to convert base64 to PIL image: {len(image_data) if image_data else 0} bytes"
                    )
                    st.markdown(error_msg)
                    self._safe_add_message_to_history("assistant", error_msg)
            else:
                # Debug why the condition failed
                logging.error(
                    f"Image generation condition failed - success: {success}, image_data length: {len(image_data) if image_data else 0}"
                )
                if not success:
                    # Failed image generation
                    error_message = image_response.get("error_message", "Failed to generate image.")
                    st.markdown(f"**Image Generation Error:** {error_message}")
                    self._safe_add_message_to_history("assistant", f"Image generation failed: {error_message}")

                    # Show the enhanced prompt that was attempted
                    enhanced_prompt = image_response.get("enhanced_prompt", "")
                    if enhanced_prompt:
                        st.markdown(f"**Attempted prompt:** {enhanced_prompt}")
                else:
                    # success=True but no image_data
                    logging.error("Image generation succeeded but no image data received")
                    st.markdown("**Image Generation Error:** Image was generated but no data received.")
                    self._safe_add_message_to_history("assistant", "Image generation failed: No image data received")

            # Display any additional text response
            if full_response and full_response.strip():
                st.markdown(full_response)

        except Exception as e:
            logging.error(f"Error displaying image generation response: {e}")
            error_msg = "I encountered an error while displaying the generated image."
            st.markdown(error_msg)
            self._safe_add_message_to_history("assistant", error_msg)

        # Clear processing flag
        st.session_state.processing = False

    def _contains_tool_call_instructions(self, content: str) -> bool:
        """
        Check if content contains custom tool call instructions

        Args:
            content: The content to check

        Returns:
            True if content contains tool call instructions, False otherwise
        """
        if not isinstance(content, str):
            return False

        # Check for various custom tool call patterns
        import re

        patterns = [
            r'<TOOLCALL-\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\[.*?\]</TOOLCALL>',
            r'<TOOLCALL"\[.*?\]</TOOLCALL>',
            r'<TOOLCALL\s*-\s*\[.*?\]</TOOLCALL>',
            r'<toolcall-\[.*?\]</toolcall>',
            r'<TOOLCALL.?\[.*?\]</TOOLCALL>',
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.DOTALL | re.IGNORECASE):
                return True

        return False

    def _safe_add_message_to_history(self, role: str, content: Any):
        """
        Safely add a message to chat history with validation

        Args:
            role: The role of the message sender
            content: The content of the message (can be string, dict for images, etc.)
        """
        # Handle different content types
        if isinstance(content, str):
            # String content - validate it's not empty and doesn't contain tool calls
            if not content or not content.strip():
                logging.warning(f"Attempted to add empty {role} message to chat history, skipping")
                return

            # Check for tool call instructions
            if self._contains_tool_call_instructions(content):
                logging.warning(
                    f"Attempted to add {role} message with tool call instructions to chat history, skipping"
                )
                return

            content = content.strip()
        elif isinstance(content, dict):
            # Dict content (like image messages) - ensure it has meaningful data
            if not content:
                logging.warning(f"Attempted to add empty dict {role} message to chat history, skipping")
                return
        else:
            # Other content types - ensure they're truthy
            if not content:
                logging.warning(f"Attempted to add empty {role} message to chat history, skipping")
                return

        # Add the validated message to history
        st.session_state.messages.append({"role": role, "content": content})
        logging.debug(f"Added {role} message to chat history")

    def _update_chat_history(self, text: str, role: str):
        """
        Update chat history with new response

        Args:
            text: The response text
            role: The role of the message sender
        """
        # Clean up message format before saving
        st.session_state.messages = self.chat_service.drop_verbose_messages_context(st.session_state.messages)

        # Use the safe method to add the message
        self._safe_add_message_to_history(role, text)

    def run(self):
        """Run the main application"""
        # Display chat history (will include any new messages)
        self.display_chat_history()

        # Check if we're currently processing a message to prevent concurrent processing
        if st.session_state.get("processing", False):
            st.info("Processing your message, please wait...")
            return

        # Handle user input - let Streamlit's natural refresh cycle with smooth transitions handle updates
        if prompt := st.chat_input("Hello, how are you?"):
            # Set processing flag to prevent concurrent requests
            st.session_state.processing = True
            self.process_prompt(prompt)


def main():
    """Main function to run the Streamlit app"""
    # Create and run the application
    app = StreamlitChatApp()
    app.run()


if __name__ == "__main__":
    main()
