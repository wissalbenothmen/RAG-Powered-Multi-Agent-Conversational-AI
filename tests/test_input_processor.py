"""
Overview:
This module contains unit tests for the input_processor module in a RAG-powered AI system.
It verifies the process_input function's behavior with text and image inputs.

Technical Details:
- Framework: Uses unittest with mocking to test process_input.
- Scope: Tests text-only and image-with-text processing scenarios.
- Mocking: Patches GenerativeModel to simulate responses.
"""
import unittest
from unittest.mock import patch, MagicMock

from agents.input_processor import process_input


class TestInputProcessor(unittest.TestCase):
    """Unit tests for the input_processor module's process_input function."""

    @patch("agents.input_processor.GenerativeModel")
    def test_process_input_text(self, mock_model: MagicMock) -> None:
        """Test process_input with a text-only input."""
        result = process_input("Hello")
        self.assertEqual(result, "Hello")

    @patch("agents.input_processor.GenerativeModel")
    def test_process_input_with_image(self, mock_model: MagicMock) -> None:
        """Test process_input with text and image input."""
        mock_model.return_value.generate_content.return_value.text = "Image response"
        mock_image = MagicMock()
        mock_image.read.return_value = b"image_data"

        result = process_input("Describe this", image=mock_image)
        self.assertEqual(result, "Image response")
        mock_model.return_value.generate_content.assert_called_once()

if __name__ == "__main__":
    unittest.main()

#end-of-file
