"""
Overview:
This module contains unit tests for the process_docs module in a RAG-powered AI system.
It verifies the process_documents function's behavior with mocked VectorDB.

Technical Details:
- Framework: Uses unittest with mocking to test process_documents.
- Scope: Tests document indexing and retrieval with sample queries.
- Mocking: Patches VectorDB to simulate indexing and search responses.
"""
import unittest
from unittest.mock import patch

from rag.process_docs import process_documents


class TestProcessDocs(unittest.TestCase):
    """Unit tests for the process_docs module's process_documents function."""

    @patch("rag.process_docs.VectorDB")
    def test_process_documents(self, mock_vector_db: unittest.mock.MagicMock) -> None:
        """Test process_documents with mocked VectorDB operations."""
        mock_db_instance = mock_vector_db.return_value
        mock_db_instance.index_documents.return_value = None
        mock_db_instance.db_manager.get_processed_files.return_value = ["doc1.pdf"]
        mock_db_instance.db_manager.get_summary.return_value = {
            "total_documents": 1,
            "processed_documents": 1,
            "total_chunks": 10,
            "last_processed": "2025-02-25",
        }
        mock_db_instance.search.side_effect = [
            [{"source": "doc1.pdf", "chunk_text": "RAG is...", "similarity_score": 0.9}],
            [],
            [{"source": "doc1.pdf", "chunk_text": "Difference...", "similarity_score": 0.8}],
        ]

        process_documents(data_dir="tests/data", top_k=1)

        mock_db_instance.index_documents.assert_called_once()
        self.assertEqual(mock_db_instance.search.call_count, 3)


if __name__ == "__main__":
    unittest.main()

#end-of-file
