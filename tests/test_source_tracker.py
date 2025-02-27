"""
Overview:
This module contains unit tests for the source_tracker module in a RAG-powered AI system.
It verifies the track_sources function's deduplication and scoring logic.

Technical Details:
- Framework: Uses unittest to test track_sources functionality.
- Scope: Tests source deduplication based on highest similarity scores.
- Dependencies: Relies on agents.source_tracker for track_sources function.
"""
import unittest

from agents.source_tracker import track_sources


class TestSourceTracker(unittest.TestCase):
    """Unit tests for the source_tracker module's track_sources function."""

    def test_track_sources(self) -> None:
        """Test track_sources deduplication with multiple sources."""
        sources = [
            {"source": "doc1.pdf", "chunk_text": "Text", "similarity_score": 0.9},
            {"source": "doc1.pdf", "chunk_text": "Text2", "similarity_score": 0.95},
            {"source": "doc2.pdf", "chunk_text": "Text3", "similarity_score": 0.8},
        ]
        tracked = track_sources(sources)
        self.assertEqual(len(tracked), 2)
        self.assertEqual(tracked[0]["source"], "doc1.pdf")
        self.assertEqual(tracked[0]["similarity_score"], 0.95)


if __name__ == "__main__":
    unittest.main()

#end-of-file