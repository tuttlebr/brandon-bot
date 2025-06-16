from typing import Any, Dict, List, Union


class ChatMessage:
    """Class to handle chat message operations"""

    def __init__(self, role: str, content: Union[str, List[Dict[str, Any]], Dict[str, Any]]):
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
        if isinstance(self.content, list):
            return self.content[0].get("text", "")
        elif isinstance(self.content, dict):
            # Handle image messages
            if self.content.get("type") == "image":
                return self.content.get("text", "")
            return self.content.get("text", "")
        return self.content

    def is_image_message(self) -> bool:
        """
        Check if this message contains an image

        Returns:
            True if the message contains an image, False otherwise
        """
        return isinstance(self.content, dict) and self.content.get("type") == "image"

    def get_image_data(self) -> tuple[str, str]:
        """
        Get image data and caption from the message

        Returns:
            Tuple of (base64_image_data, caption)
        """
        if self.is_image_message():
            return (self.content.get("image_data", ""), self.content.get("image_caption", ""))
        return "", ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary format for API calls

        Returns:
            Message in dictionary format
        """
        return {"role": self.role, "content": self.content}
