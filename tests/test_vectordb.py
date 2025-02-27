"""
Overview:
This module contains unit tests for the vectordb module in a RAG-powered AI system.
It verifies VectorDB's indexing and search functionality with mocked dependencies.

Technical Details:
- Framework: Uses unittest with mocking to test VectorDB methods.
- Scope: Tests index_documents and search with temporary directories.
- Mocking: Patches FAISS, PyMuPDFLoader, and DatabaseManager for isolation.
"""
import unittest
from unittest.mock import patch, MagicMock
import logging
import os
import shutil
from pathlib import Path

from rag.vectordb import VectorDB


class TestVectorDB(unittest.TestCase):
    """Unit tests for the vectordb module's VectorDB class."""

    def setUp(self) -> None:
        """Set up test environment with temporary directories."""
        logging.disable(logging.CRITICAL)
        self.data_dir = "tests/data"
        self.index_dir = "tests/index"
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up test environment by removing temporary directories."""
        logging.disable(logging.NOTSET)
        for dir_path in [self.data_dir, self.index_dir]:
            if Path(dir_path).exists():
                shutil.rmtree(dir_path)

    @patch("rag.vectordb.FAISS")
    @patch("rag.vectordb.PyMuPDFLoader")
    @patch("rag.vectordb.DatabaseManager")
    def test_index_documents(
        self,
        mock_db_manager: MagicMock,
        mock_loader: MagicMock,
        mock_faiss: MagicMock
    ) -> None:
        """Test index_documents with mocked dependencies."""
        pdf_path = Path("tests/data/test.pdf")
        mock_db_manager.return_value.get_unprocessed_files.return_value = [pdf_path]
        # Mock Document object for splitting
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content"
        mock_loader.return_value.load.return_value = [mock_doc]
        with patch("rag.vectordb.RecursiveCharacterTextSplitter") as mock_splitter:
            mock_splitter.return_value.split_documents.return_value = [mock_doc]
            mock_faiss.from_documents.return_value = MagicMock()

            # Initialize VectorDB and test indexing
            vector_db = VectorDB(data_dir="tests/data", index_dir="tests/index")
            vector_db.vectorstore = None
            vector_db.index_documents()

            # Verify mock calls
            mock_loader.assert_called_once_with(str(pdf_path))
            mock_faiss.from_documents.assert_called_once()

    @patch("rag.vectordb.FAISS")
    def test_search(self, mock_faiss: MagicMock) -> None:
        """Test search with mocked vectorstore."""
        # Mock vectorstore behavior
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (MagicMock(page_content="Test", metadata={"source": "doc1.pdf"}), 0.1)
        ]
        # Initialize VectorDB with mocked vectorstore
        vector_db = VectorDB(data_dir="tests/data", index_dir="tests/index")
        vector_db.vectorstore = mock_vectorstore

        results = vector_db.search("What is RAG?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "doc1.pdf")


if __name__ == "__main__":
    unittest.main()

#end-of-file
