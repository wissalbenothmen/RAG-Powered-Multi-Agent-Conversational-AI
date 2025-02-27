"""
Overview:
This module contains unit tests for the RAG-Powered-Multi-Agent-Conversational-AI Flask app.
It tests routes, database operations, and API endpoints using mocking.

Technical Details:
- Framework: Uses unittest with Flask test client and mock objects.
- Database: Tests with a temporary SQLite database for chat history.
- Scope: Covers index, history, chat, and database management functions.
- Mocking: Simulates external dependencies like API keys and database connections.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from flask import Flask, session
from app import (
    app,
    init_chat_db,
    save_chat_session_async,
    get_chat_history,
    delete_chat_session,
    rename_chat_session
)


class TestApp(unittest.TestCase):
    """Unit tests for the RAG-Powered-Multi-Agent-Conversational-AI Flask app."""

    def setUp(self) -> None:
        """Set up Flask test client and context with a temporary database."""
        self.app = app.test_client()
        self.app.testing = True
        app.config["SECRET_KEY"] = "test-secret-key"

        # Use a temporary directory and database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chat_db_path = os.path.join(self.temp_dir.name, "chat_history_test.db")
        app.chat_db_path = self.chat_db_path  # Override global chat_db_path

        with patch(
            "os.getenv",
            side_effect=lambda key, default=None: "fake-api-key" if key == "GOOGLE_API_KEY" else default
        ):
            self.ctx = app.app_context()
            self.ctx.push()
        init_chat_db()  # Initialize temp database

    def tearDown(self) -> None:
        """Tear down app context and clean up temporary directory."""
        self.ctx.pop()
        self.temp_dir.cleanup()  # Removes directory and contents

    def test_index(self) -> None:
        """Test the index route renders correctly."""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    @patch("app.get_chat_history")
    def test_history(self, mock_get_chat_history: MagicMock) -> None:
        """Test history route with mocked chat history."""
        mock_get_chat_history.return_value = [
            {"session_id": "1", "title": "Test", "timestamp": "2025-02-25T12:00:00", "conversation": []}
        ]
        response = self.app.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<h1 class="history-title">', response.data)

    @patch("app.process_input")
    @patch("app.vector_db")
    @patch("app.generate_answer")
    @patch("app.track_sources")
    @patch("app.search_arxiv")
    def test_chat_post(
        self,
        mock_search_arxiv: MagicMock,
        mock_track_sources: MagicMock,
        mock_generate_answer: MagicMock,
        mock_vector_db: MagicMock,
        mock_process_input: MagicMock
    ) -> None:
        """Test chat POST endpoint with a sample query."""
        mock_process_input.return_value = "What is RAG?"
        mock_vector_db.search.return_value = [
            {"source": "doc1.pdf", "chunk_text": "RAG is...", "similarity_score": 0.9}
        ]
        mock_generate_answer.return_value = "RAG is Retrieval-Augmented Generation."
        mock_track_sources.return_value = [{"source": "doc1.pdf"}]
        mock_search_arxiv.return_value = [{"title": "RAG Paper", "id": "1234"}]

        response = self.app.post("/chat", json={"user_input": "What is RAG?"})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["answer"], "RAG is Retrieval-Augmented Generation.")

    @patch("app.sqlite3.connect")
    def test_init_chat_db(self, mock_sqlite: MagicMock) -> None:
        """Test chat database initialization."""
        mock_conn = MagicMock()
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        init_chat_db()
        mock_conn.execute.assert_any_call(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                conversation TEXT NOT NULL
            )
            """
        )
        mock_conn.execute.assert_any_call(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_sessions(timestamp)"
        )

    @patch("app.sqlite3.connect")
    def test_save_chat_session_async(self, mock_sqlite: MagicMock) -> None:
        """Test async chat session saving."""
        mock_conn = MagicMock()
        mock_sqlite.return_value.__enter__.return_value = mock_conn

        with patch("app.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            conversation = [{"type": "user", "content": "Hi"}]
            convo_json = json.dumps(conversation)

            with patch("app.datetime") as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = "2025-02-25T12:00:00"
                save_chat_session_async("session1", "Test Title", conversation)

                mock_thread.assert_called_once_with(target=unittest.mock.ANY)
                mock_thread_instance.start.assert_called_once()

                target_func = mock_thread.call_args[1]["target"]
                target_func()

                mock_conn.execute.assert_called_once_with(
                    """
                    INSERT OR REPLACE INTO chat_sessions (
                        session_id, title, timestamp, conversation)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("session1", "Test Title", "2025-02-25T12:00:00", convo_json),
                )

    @patch("app.sqlite3.connect")
    def test_get_chat_history(self, mock_sqlite: MagicMock) -> None:
        """Test retrieving chat history from the database."""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("1", "Test", "2025-02-25T12:00:00", '[{"type": "user", "content": "Hi"}]')
        ]
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        history = get_chat_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["session_id"], "1")

    @patch("app.sqlite3.connect")
    def test_delete_chat_session(self, mock_sqlite: MagicMock) -> None:
        """Test deleting a chat session."""
        mock_conn = MagicMock()
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        delete_chat_session("session1")
        mock_conn.execute.assert_called_once_with(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            ("session1",)
        )

    @patch("app.sqlite3.connect")
    def test_rename_chat_session(self, mock_sqlite: MagicMock) -> None:
        """Test renaming a chat session."""
        mock_conn = MagicMock()
        mock_sqlite.return_value.__enter__.return_value = mock_conn
        rename_chat_session("session1", "New Title")
        mock_conn.execute.assert_called_once_with(
            "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
            ("New Title", "session1")
        )


if __name__ == "__main__":
    unittest.main()

#end-of-file
