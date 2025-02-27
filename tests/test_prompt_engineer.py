"""
Overview:
This module contains unit tests for the prompt_engineer module in a RAG-powered AI system.
It verifies the PromptEngineer class's query analysis and prompt crafting.

Technical Details:
- Framework: Uses unittest to test PromptEngineer methods.
- Scope: Tests analyze_query and craft_prompt with various query types.
- Dependencies: Relies on agents.prompt_engineer for PromptEngineer class.
"""
import unittest

from agents.prompt_engineer import PromptEngineer


class TestPromptEngineer(unittest.TestCase):
    """Unit tests for the prompt_engineer module's PromptEngineer class."""

    def setUp(self) -> None:
        """Set up PromptEngineer instance for tests."""
        self.engineer = PromptEngineer()

    def test_analyze_query_default(self) -> None:
        """Test analyze_query with a default query type."""
        sources = [{"source": "doc1.pdf", "chunk_text": "Text"}]
        strategy = self.engineer.analyze_query("What is RAG?", sources)
        self.assertEqual(strategy, "default")

    def test_analyze_query_detailed(self) -> None:
        """Test analyze_query with a detailed query type."""
        sources = [{"source": "doc1.pdf", "chunk_text": "Text"}]
        strategy = self.engineer.analyze_query("Explain how RAG works", sources)
        self.assertEqual(strategy, "detailed")

    def test_analyze_query_creative(self) -> None:
        """Test analyze_query with a creative query type."""
        sources = [{"source": "doc1.pdf", "chunk_text": "Text"}]
        strategy = self.engineer.analyze_query("Imagine a story about RAG", sources)
        self.assertEqual(strategy, "creative")

    def test_craft_prompt(self) -> None:
        """Test craft_prompt returns a template with question placeholder."""
        sources = [{"source": "doc1.pdf", "chunk_text": "RAG is...", "similarity_score": 0.9}]
        prompt = self.engineer.craft_prompt("What is RAG?", sources)
        # Check for placeholder in template
        self.assertIn("Question: {question}", prompt.template)


if __name__ == "__main__":
    unittest.main()

#end-of-file
