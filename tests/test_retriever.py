"""
Overview:
This module contains unit tests for the retriever module in a RAG-powered AI system.
It verifies validate_api_key and retrieve_information functions.

Technical Details:
- Framework: Uses unittest with mocking to test retriever functionality.
- Scope: Tests API key validation and document retrieval scenarios.
- Mocking: Patches os.getenv and ChatGoogleGenerativeAI for controlled testing.
"""
import unittest
from unittest.mock import patch, MagicMock

from agents.retriever import retrieve_information, validate_api_key


class TestRetriever(unittest.TestCase):
    """Unit tests for the retriever module's functions."""

    @patch("os.getenv")
    def test_validate_api_key_success(self, mock_getenv: MagicMock) -> None:
        """Test validate_api_key with a valid API key."""
        mock_getenv.return_value = "AIza123"
        api_key = validate_api_key()
        self.assertEqual(api_key, "AIza123")

    @patch("os.getenv")
    def test_validate_api_key_missing(self, mock_getenv: MagicMock) -> None:
        """Test validate_api_key with missing API key."""
        mock_getenv.return_value = None
        with self.assertRaises(ValueError) as context:
            validate_api_key()
        self.assertEqual(
            str(context.exception),
            "GOOGLE_API_KEY environment variable is not set"
        )

    @patch("os.getenv")
    def test_validate_api_key_invalid(self, mock_getenv: MagicMock) -> None:
        """Test validate_api_key with an invalid API key."""
        mock_getenv.return_value = "InvalidKey"
        with self.assertRaises(ValueError) as context:
            validate_api_key()
        self.assertEqual(
            str(context.exception),
            "GOOGLE_API_KEY appears to be invalid (should start with 'AIza')"
        )

    @patch("agents.retriever.ChatGoogleGenerativeAI")
    @patch("os.getenv")
    def test_retrieve_information_success(
        self,
        mock_getenv: MagicMock,
        mock_llm: MagicMock
    ) -> None:
        """Test retrieve_information with successful retrieval."""
        mock_getenv.return_value = "AIza123"
        mock_vector_db = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_db.as_retriever.return_value = mock_retriever
        mock_vector_db.search.return_value = [
            {"source": "doc1.pdf", "chunk_text": "RAG...", "similarity_score": 0.9}
        ]
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.invoke.return_value = "RAG explanation"

        results = retrieve_information("What is RAG?", mock_vector_db, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "doc1.pdf")
        mock_vector_db.search.assert_called_once_with("What is RAG?", top_k=1)

    @patch("agents.retriever.ChatGoogleGenerativeAI")
    @patch("os.getenv")
    def test_retrieve_information_chain_failure(
        self,
        mock_getenv: MagicMock,
        mock_llm: MagicMock
    ) -> None:
        """Test retrieve_information with chain failure fallback."""
        mock_getenv.return_value = "AIza123"
        mock_vector_db = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_db.as_retriever.return_value = mock_retriever
        mock_vector_db.search.return_value = [
            {"source": "doc1.pdf", "chunk_text": "RAG...", "similarity_score": 0.9}
        ]
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.invoke.side_effect = Exception("Chain error")

        results = retrieve_information("What is RAG?", mock_vector_db, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "doc1.pdf")
        mock_vector_db.search.assert_called_once_with("What is RAG?", top_k=1)


if __name__ == "__main__":
    unittest.main()

#end-of-file
