from typing import Any, Dict, List, Union
from utils.text_processing import strip_think_tags


class ChatMessage:
    """Class to handle chat message operations"""

    def __init__(
        self, role: str, content: Union[str, List[Dict[str, Any]], Dict[str, Any]]
    ):
        """
        Initialize a chat message

        Args:
            role: The role of the message sender (system, user, assistant)
            content: The content of the message, which can be text or structured content
        """
        self.role = role
        self.content = content

    def get_display_content(self) -> str:
        """
        Extract displayable content from message

        Returns:
            The text content that should be displayed in the UI
        """
        # Extract the raw content first
        raw_content = ""
        if isinstance(self.content, list):
            raw_content = self.content[0].get("text", "")
        elif isinstance(self.content, dict):
            # Handle image messages with metadata
            raw_content = self.content.get("text", "")
        else:
            raw_content = self.content

        # Strip think tags before returning
        return strip_think_tags(raw_content)

    def is_image_message(self) -> bool:
        """
        Check if this message contains an image

        Returns:
            True if the message contains an image, False otherwise
        """
        return isinstance(self.content, dict) and self.content.get("type") == "image"

    def get_image_data(self) -> tuple[str, str, str]:
        """
        Get image data and metadata from the message

        Returns:
            Tuple of (image_id, enhanced_prompt, original_prompt)
        """
        if self.is_image_message():
            return (
                self.content.get("image_id", ""),
                self.content.get("enhanced_prompt", ""),
                self.content.get("original_prompt", ""),
            )
        return "", "", ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary format for API calls

        Returns:
            Message in dictionary format
        """
        return {"role": self.role, "content": self.content}
