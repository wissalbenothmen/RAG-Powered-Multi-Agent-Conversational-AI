"""
Overview:
This module defines the FeedbackCollector class for managing user feedback in a RAG-powered AI system.
It uses PostgreSQL to store and retrieve feedback data, ensuring persistence and statistical analysis.
The class handles feedback collection, retrieval, and stats computation, integrating with the system via
environment variables for database configuration. It employs psycopg2 for database interactions and
includes error handling to maintain system stability.

Technical Details:
- Database: PostgreSQL with a 'feedback' table storing fields like id, timestamp, question, etc.
- Connection: Managed via psycopg2 with DictCursor for dictionary-based row access.
- Methods: Includes initialization, feedback collection, retrieval (all/by ID), and stats calculation.
- Error Handling: Exceptions are logged and either re-raised or returned as safe defaults to prevent crashes.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

import psycopg2

# Initialize logger for this module
logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Class to manage collection and retrieval of user feedback."""

    def __init__(self):
        """Initialize the FeedbackCollector and ensure the feedback table exists."""
        self._ensure_feedback_table_exists()

    def get_connection(self):
        """Create and return a PostgreSQL database connection."""
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME"),
            cursor_factory=psycopg2.extras.DictCursor,
        )

    def _ensure_feedback_table_exists(self) -> None:
        """Ensure the feedback table exists in the database."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS feedback (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP NOT NULL,
                            question TEXT NOT NULL,
                            satisfaction_score INTEGER NOT NULL,
                            domain VARCHAR(100) NOT NULL,
                            user_feedback TEXT,
                            response_time FLOAT,
                            response_type VARCHAR(50),
                            sources_count INTEGER,
                            accuracy_score FLOAT,
                            article_category VARCHAR(100)
                        )
                        """
                    )
                conn.commit()
            logger.info("Feedback table initialized with PostgreSQL")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error initializing feedback table: %s",
                str(e),
                exc_info=True
            )
            raise

    def collect_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and store feedback data, returning it with metadata.

        Args:
            feedback_data (Dict[str, Any]): Feedback details from the user.

        Returns:
            Dict[str, Any]: Stored feedback with added ID and timestamp.
        """
        required_fields = ["question", "satisfaction_score", "domain"]
        for field in required_fields:
            if field not in feedback_data:
                raise ValueError(f"Missing required field: {field}")

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO feedback (
                            timestamp, question, satisfaction_score, domain,
                            user_feedback, response_time, response_type,
                            sources_count, accuracy_score, article_category
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            datetime.now().isoformat(),
                            feedback_data["question"],
                            feedback_data["satisfaction_score"],
                            feedback_data["domain"],
                            feedback_data.get("user_feedback", None),
                            feedback_data.get("response_time", None),
                            feedback_data.get("response_type", None),
                            feedback_data.get("sources_count", None),
                            feedback_data.get("accuracy_score", None),
                            feedback_data.get("article_category", None),
                        ),
                    )
                    new_id = cursor.fetchone()["id"]
                conn.commit()
            feedback_with_metadata = {
                "id": new_id,
                "timestamp": datetime.now().isoformat(),
                **feedback_data
            }
            logger.info("Feedback collected: ID %s", new_id)
            return feedback_with_metadata
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error collecting feedback: %s",
                str(e),
                exc_info=True
            )
            raise

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Retrieve all feedback entries from the database.

        Returns:
            List[Dict[str, Any]]: List of all feedback records.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM feedback")
                    return [
                        {
                            k: v.isoformat() if isinstance(v, datetime) else v
                            for k, v in row.items()
                        }
                        for row in cursor.fetchall()
                    ]
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error reading feedback: %s",
                str(e),
                exc_info=True
            )
            return []

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Calculate and return statistics about stored feedback.

        Returns:
            Dict[str, Any]: Stats including total, average score, and more.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) AS total, "
                        "AVG(satisfaction_score) AS avg FROM feedback"
                    )
                    stats = cursor.fetchone()
                    total = stats["total"]
                    avg_score = stats["avg"] or 0
                    cursor.execute(
                        "SELECT domain, COUNT(*) AS count "
                        "FROM feedback GROUP BY domain"
                    )
                    domain_stats = {
                        row["domain"]: row["count"]
                        for row in cursor.fetchall()
                    }
                    cursor.execute(
                        "SELECT * FROM feedback "
                        "ORDER BY timestamp DESC LIMIT 1"
                    )
                    latest = cursor.fetchone()
                    latest = (
                        {
                            k: v.isoformat() if isinstance(v, datetime) else v
                            for k, v in latest.items()
                        }
                        if latest
                        else None
                    )
                return {
                    "total_feedback": total,
                    "average_satisfaction": round(avg_score, 2),
                    "feedback_by_domain": domain_stats,
                    "latest_feedback": latest,
                }
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error calculating stats: %s",
                str(e),
                exc_info=True
            )
            return {
                "total_feedback": 0,
                "average_satisfaction": 0,
                "feedback_by_domain": {},
                "latest_feedback": None
            }

    def get_feedback_by_id(self, feedback_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific feedback entry by its ID.

        Args:
            feedback_id (int): The ID of the feedback to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Feedback record if found, else None.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT * FROM feedback WHERE id = %s",
                        (feedback_id,)
                    )
                    row = cursor.fetchone()
                    return (
                        {
                            k: v.isoformat() if isinstance(v, datetime) else v
                            for k, v in row.items()
                        }
                        if row
                        else None
                    )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error getting feedback ID: %s",
                str(e),
                exc_info=True
            )
            return None

#end-of-file
