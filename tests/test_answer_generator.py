"""
Overview:
This module contains unit tests for the answer_generator module in a RAG-powered AI system.
It verifies the search_web function's behavior with specific queries.

Technical Details:
- Framework: Uses unittest to test search_web functionality.
- Scope: Tests RAG-specific query responses against expected outputs.
- Imports: Relies on agents.answer_generator for the function under test.
"""
import unittest

from agents.answer_generator import search_web


class TestAnswerGeneratorSimple(unittest.TestCase):
    """Unit tests for the answer_generator module's search_web function."""

    def test_search_web_rag(self) -> None:
        """Test search_web with a RAG query."""
        result = search_web("What is RAG?")
        expected = (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances "
            "language models by combining retrieval of relevant documents with "
            "text generation. For more details, see: "
            "https://arxiv.org/abs/2005.11401 (original RAG paper by Lewis et al.)."
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

#end-of-file
