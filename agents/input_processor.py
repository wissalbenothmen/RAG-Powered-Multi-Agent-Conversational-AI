"""
Overview:
This module provides input processing for a RAG-powered AI system using Google's Gemini model.
It handles user text input and optional image data, generating responses via the GenerativeModel.

Technical Details:
- Model: Uses 'gemini-1.5-pro' from google.generativeai for content generation.
- Input: Accepts text and optional image binary data, processed together if both are provided.
- Error Handling: Logs errors and re-raises exceptions to maintain system stability.
- Logging: Utilizes Python's logging module for debugging and error tracking.
"""

import logging
from google.generativeai import GenerativeModel

# Initialize logger for this module
logger = logging.getLogger(__name__)


def process_input(user_input, image=None):
    """Process user input with an optional image using the Gemini model.

    Args:
        user_input: The text input from the user (string).
        image: Optional image file object (default is None).

    Returns:
        The processed response text (string).

    Raises:
        Exception: If processing fails, logged and re-raised.
    """
    try:
        # Initialize the generative model
        model = GenerativeModel("gemini-1.5-pro")
        logger.debug("Processing input: %s...", user_input[:50])

        # Handle input with image if provided
        if image:
            image_content = image.read()
            logger.info("Processing input with image")
            response = model.generate_content(
                contents=[user_input, image_content]
            )
            logger.debug("Image response: %s...", response.text[:50])
            return response.text

        # Handle text-only input
        logger.debug("Returning plain user input (no image)")
        return user_input

    except Exception as e:
        logger.error(
            "Error processing input: %s",
            str(e),
            exc_info=True
        )
        raise

#end-of-file
