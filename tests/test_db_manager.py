"""
Overview:
This module contains unit tests for the db_manager module in a RAG-powered AI system.
It verifies DatabaseManager's database operations with mocked SQLite connections.

Technical Details:
- Framework: Uses unittest with mocking to test DatabaseManager methods.
- Scope: Tests initialization, document addition, file retrieval, and summary.
- Mocking: Patches SQLite connections and file system operations.
"""
import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from pathlib import Path

from database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Unit tests for the db_manager module's DatabaseManager class."""

    def setUp(self) -> None:
        """Set up test environment with temporary directories."""
        self.db_path = "test_data/documents.db"
        Path("test_data").mkdir(exist_ok=True)

    def tearDown(self) -> None:
        """Clean up test environment by removing temporary files."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

    @patch("sqlite3.connect")
    def test_init(self, mock_sqlite: MagicMock) -> None:
        """Test DatabaseManager initialization."""
        mock_conn = MagicMock()
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        db = DatabaseManager(db_path=self.db_path)
        mock_conn.execute.assert_called_once_with(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                processed_timestamp TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )

    @patch("sqlite3.connect")
    def test_add_document(self, mock_sqlite: MagicMock) -> None:
        """Test adding a document to the database."""
        mock_conn = MagicMock()
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        db = DatabaseManager(db_path=self.db_path)
        mock_conn.execute.reset_mock()  # Clear init calls
        with patch("database.db_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2025-02-25T00:00:00"
            result = db.add_document("test.pdf", 5)
        self.assertTrue(result)
        mock_conn.execute.assert_called_once_with(
            """
            INSERT OR REPLACE INTO documents (
                filename, processed_timestamp, chunk_count, status
            ) VALUES (?, ?, ?, ?)
            """,
            ("test.pdf", "2025-02-25T00:00:00", 5, "processed"),
        )

    @patch("sqlite3.connect")
    def test_get_processed_files(self, mock_sqlite: MagicMock) -> None:
        """Test retrieving processed files from the database."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("test.pdf",)]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        db = DatabaseManager(db_path=self.db_path)
        files = db.get_processed_files()
        self.assertEqual(files, ["test.pdf"])

    @patch("sqlite3.connect")
    def test_get_unprocessed_files(self, mock_sqlite: MagicMock) -> None:
        """Test retrieving unprocessed files."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("processed.pdf",)]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        db = DatabaseManager(db_path=self.db_path)
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [
                Path("processed.pdf"),
                Path("unprocessed.pdf")
            ]
            unprocessed = db.get_unprocessed_files(Path("test_data"))
            self.assertEqual(
                [p.name for p in unprocessed],
                ["unprocessed.pdf"]
            )

    @patch("sqlite3.connect")
    def test_get_summary(self, mock_sqlite: MagicMock) -> None:
        """Test generating a summary of database stats."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(2,), (1,), (10,), ("2025-02-25",)]
        mock_conn.execute.return_value = mock_cursor
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        db = DatabaseManager(db_path=self.db_path)
        summary = db.get_summary()
        self.assertEqual(summary["total_documents"], 2)
        self.assertEqual(summary["processed_documents"], 1)
        self.assertEqual(summary["total_chunks"], 10)
        self.assertEqual(summary["last_processed"], "2025-02-25")


if __name__ == "__main__":
    unittest.main()

#end-of-file
