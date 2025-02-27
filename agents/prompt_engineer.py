"""
Overview:
This module defines the PromptEngineer class for creating dynamic prompts in a RAG-powered AI system.
It selects and crafts prompt templates based on query analysis for varied response styles.

Technical Details:
- Templates: Uses langchain_core's PromptTemplate with three styles: default, detailed, creative.
- Analysis: Determines prompt style by inspecting query keywords and source availability.
- Logging: Tracks prompt selection and errors using Python's logging module.
- Error Handling: Returns a default template on failure to ensure system stability.
"""

import logging
from typing import List, Dict
from langchain_core.prompts import PromptTemplate

# Initialize logger for this module
logger = logging.getLogger(__name__)


class PromptEngineer:
    """Class to manage and craft prompt templates for query responses."""

    def __init__(self):
        """Initialize PromptEngineer with a set of predefined prompt templates."""
        self.prompt_templates = {
            "default": PromptTemplate(
                input_variables=["question", "context"],
                template="""\
                Question: {question}

                Context:
                {context}

                Instructions:
                Provide a clear, concise, and accurate answer based on the context.
                Use markdown formatting with sections and bullet points where appropriate.
                Cite sources as '[Source: {source}]'.
                If the context is insufficient, acknowledge limitations and suggest rephrasing.

                Answer:
                """,
            ),
            "detailed": PromptTemplate(
                input_variables=["question", "context"],
                template="""\
                Question: {question}

                Context:
                {context}

                Instructions:
                Provide a detailed, comprehensive answer that:
                - Starts with a direct response
                - Explains the reasoning step-by-step
                - Incorporates all relevant details from the context
                - Uses markdown with headings, bullet points, and tables if needed
                - Cites sources as '[Source: {source}]'
                - Highlights uncertainties or gaps in the context

                Answer:
                """,
            ),
            "creative": PromptTemplate(
                input_variables=["question", "context"],
                template="""\
                Question: {question}

                Context:
                {context}

                Instructions:
                Provide a creative and engaging answer that:
                - Starts with a catchy hook
                - Weaves context into a narrative or analogy
                - Uses markdown for readability
                - Cites sources as '[Source: {source}]'
                - Maintains accuracy while adding flair

                Answer:
                """,
            ),
        }

    def analyze_query(self, query: str, sources: List[Dict]) -> str:
        """Analyze the query to select an appropriate prompt strategy.

        Args:
            query (str): The user's query.
            sources (List[Dict]): List of retrieved source data.

        Returns:
            str: The selected prompt strategy key (default, detailed, creative).
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ["explain", "how", "why", "detail"]):
            logger.debug("Selected 'detailed' prompt strategy")
            return "detailed"
        if any(word in query_lower for word in ["imagine", "story", "example"]):
            logger.debug("Selected 'creative' prompt strategy")
            return "creative"
        if not sources or len(sources) < 2:
            logger.warning("Limited sources detected; using 'default' prompt")
            return "default"
        logger.debug("Selected 'default' prompt strategy")
        return "default"

    def craft_prompt(self, query: str, sources: List[Dict]) -> PromptTemplate:
        """Craft a prompt template based on query and sources.

        Args:
            query (str): The user's query.
            sources (List[Dict]): List of retrieved source data.

        Returns:
            PromptTemplate: The crafted prompt for generating a response.
        """
        try:
            context = "\n\n".join(
                [
                    f"Source: {s['source']}\n"
                    f"Content: {s['chunk_text']}\n"
                    f"(Relevance: {s['similarity_score']:.2%})"
                    for s in sources
                ]
            )
            strategy = self.analyze_query(query, sources)
            prompt = self.prompt_templates.get(
                strategy,
                self.prompt_templates["default"]
            )
            logger.info("Crafted prompt with strategy: %s", strategy)
            return prompt
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error crafting prompt: %s",
                str(e),
                exc_info=True
            )
            return self.prompt_templates["default"]

#end-of-file
