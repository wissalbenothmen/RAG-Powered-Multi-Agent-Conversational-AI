"""
Overview:
This module defines the DatabaseManager class to track document processing 
in a RAG-powered AI system.
It uses PostgreSQL to store metadata about processed documents 
from a Google Cloud Storage bucket.

Technical Details:
- Database: PostgreSQL with a 'documents' table for filename,
 timestamp, chunk count, and status.
- Connection: Managed via psycopg2 with environment variables for Cloud SQL.
- GCS Integration: Tracks unprocessed files from a specified bucket.
- Error Handling: Returns defaults (e.g., [], {}) on failure to ensure stability.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import dotenv
import psycopg2
import google.cloud.storage as storage

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Class to manage document metadata in a PostgreSQL database."""

    def __init__(self, db_path: str = "data/documents.db"):
        """Initialize the database manager.

        Args:
            db_path (str): Ignored for Cloud SQL; kept for compatibility.
        """
        # db_path is unused but retained for potential local DB support
        self._initialize_database()

    def get_connection(self) -> psycopg2.extensions.connection:
        """Create and return a PostgreSQL database connection."""
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),  # Public IP: 34.38.195.191
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME"),
            port=5432,  # Default PostgreSQL port
            cursor_factory=psycopg2.extras.DictCursor,
        )

    def _initialize_database(self) -> None:
        """Initialize the documents table in PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) UNIQUE NOT NULL,
                        processed_timestamp TIMESTAMP NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending'
                    )
                    """
                )
            conn.commit()
        logger.info("Documents database initialized with PostgreSQL")

    def add_document(self, filename: str, chunk_count: int) -> bool:
        """Add or update a document's metadata in the database.

        Args:
            filename (str): Name of the document file.
            chunk_count (int): Number of chunks in the document.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO documents (
                            filename, processed_timestamp, chunk_count, status
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (filename) DO UPDATE SET
                            processed_timestamp = EXCLUDED.processed_timestamp,
                            chunk_count = EXCLUDED.chunk_count,
                            status = EXCLUDED.status
                        """,
                        (filename, datetime.now().isoformat(), chunk_count, "processed"),
                    )
                conn.commit()
            logger.info(
                "Added/Updated document %s with %d chunks",
                filename,
                chunk_count
            )
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error adding document %s: %s", filename, str(e), exc_info=True)
            return False

    def get_processed_files(self) -> List[str]:
        """Retrieve filenames of processed documents.

        Returns:
            List[str]: List of processed filenames.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT filename FROM documents WHERE status = 'processed'"
                    )
                    return [row["filename"] for row in cursor.fetchall()]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error retrieving processed files: %s", str(e), exc_info=True)
            return []

    def get_unprocessed_files(self, data_dir: str) -> List[Path]:
        """Retrieve unprocessed files from a GCS bucket.

        Args:
            data_dir (str): Ignored; kept for compatibility (uses GCS bucket).

        Returns:
            List[Path]: List of unprocessed file paths.
        """
        try:
            processed_files = set(self.get_processed_files())
            # Use GCS bucket instead of local data_dir (kept for compatibility)
            bucket_name = os.getenv("DOCUMENTS_BUCKET", "rag-multiagent-documents")
            bucket = storage.Client().bucket(bucket_name)
            all_files = {
                blob.name for blob in bucket.list_blobs()
                if blob.name.endswith(".pdf")
            }
            unprocessed = all_files - processed_files
            return [Path(f) for f in unprocessed]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error finding unprocessed files: %s", str(e), exc_info=True)
            return []

    def get_summary(self) -> Dict:
        """Generate a summary of document processing stats.

        Returns:
            Dict: Summary with total, processed, chunks, and last processed time.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) AS total FROM documents")
                    total = cursor.fetchone()["total"]
                    cursor.execute(
                        "SELECT COUNT(*) AS processed FROM documents "
                        "WHERE status = 'processed'"
                    )
                    processed = cursor.fetchone()["processed"]
                    cursor.execute("SELECT SUM(chunk_count) AS chunks FROM documents")
                    chunks = cursor.fetchone()["chunks"] or 0
                    cursor.execute(
                        "SELECT MAX(processed_timestamp) AS latest FROM documents"
                    )
                    latest = cursor.fetchone()["latest"]
                return {
                    "total_documents": total,
                    "processed_documents": processed,
                    "total_chunks": chunks,
                    "last_processed": latest.isoformat() if latest else "N/A",
                }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error generating summary: %s", str(e), exc_info=True)
            return {}

#end-of-file
