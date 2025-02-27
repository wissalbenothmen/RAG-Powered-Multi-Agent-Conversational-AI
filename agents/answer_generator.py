"""Module for generating answers using Retrieval-Augmented Generation (RAG) techniques.

This module integrates Google's Gemini language model with document retrieval and optional web search
to provide accurate, context-aware responses to user queries.
"""

import logging
from typing import List, Dict

import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from .retriever import validate_api_key

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global initialization of the LLM model
api_key = validate_api_key()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=api_key,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
)
logger.info("Initialized Gemini model globally")


def search_web(query: str) -> str:
    """Perform a mock web search based on the query and return results.

    Args:
        query (str): The user's search query.

    Returns:
        str: A string containing the search results or a fallback message.
    """
    logger.info("Performing web search for: %s", query)
    query_lower = query.lower()
    if "rag" in query_lower and "cag" in query_lower:
        return (
            "Retrieval-Augmented Generation (RAG) enhances language models by retrieving documents "
            "in real-time, while Cache-Augmented Generation (CAG) preloads knowledge into the "
            "model's context for faster inference. RAG is dynamic and suited for large datasets, "
            "whereas CAG is static and efficient for smaller, predefined knowledge bases. "
            "For more details, see: https://arxiv.org/abs/2005.11401 (RAG) and "
            "https://arxiv.org/abs/2412.15605 (CAG)."
        )
    if "rag" in query_lower:
        return (
            "Retrieval-Augmented Generation (RAG) is a technique that enhances language models by "
            "combining retrieval of relevant documents with text generation. "
            "For more details, see: https://arxiv.org/abs/2005.11401 (original RAG paper by Lewis et al.)."
        )
    if "cag" in query_lower:
        return (
            "Cache-Augmented Generation (CAG) preloads knowledge into a language model's context to "
            "eliminate real-time retrieval, offering a faster alternative to RAG. "
            "For more details, see: https://arxiv.org/abs/2412.15605 (Chan et al., 2024)."
        )
    return "No relevant information found on the web."


def generate_answer(query: str, sources: List[Dict], enable_web_search: bool = True) -> str:
    """Generate an answer to a query using retrieved sources and optional web search.

    Args:
        query (str): The user's question or query.
        sources (List[Dict]): List of retrieved document sources with metadata.
        enable_web_search (bool): Whether to supplement with web search if needed. Defaults to True.

    Returns:
        str: The generated answer, potentially augmented with web search results.
    """
    try:
        # Prepare structured context from sources
        context = (
            "\n\n".join(
                [
                    f"### Source: {s['source']}\n"
                    f"**Content:**\n{s['chunk_text']}\n"
                    f"**Relevance Score:** {s['similarity_score']:.2%}"
                    for s in sources
                ]
            )
            if sources
            else "No sources retrieved."
        )

        # Improved prompt template
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""\
            Question: {question}

            Below is the relevant information retrieved from documents:
            {context}

            Instructions:
            - Provide a clear, accurate, and concise answer based on the provided context.
            - Start with a direct response to the question, synthesizing information from all
              relevant sources, even if the information is partial or spread across multiple documents.
            - Recognize that "RAG" refers to "Retrieval-Augmented Generation" and "CAG" refers to
              "Cache-Augmented Generation" unless explicitly stated otherwise.
            - For comparison questions (e.g., "difference between RAG and CAG"), actively combine
              information from sources mentioning RAG and CAG to infer differences or similarities,
              even if not explicitly compared in a single source.
            - Incorporate specific details from the sources, citing them as [Source: filename].
            - If the context lacks sufficient details for a complete answer, acknowledge the
              limitation but attempt a general response based on available information and the
              recognized meanings of RAG and CAG.
            - Use markdown for readability, with sections and bullet points where appropriate.
            - Avoid stating that no information is available if sources with high relevance scores
              are present; instead, extract and synthesize relevant insights from them.

            Answer:
            """,
        )

        # Create chain and generate response
        chain = prompt_template | llm
        logger.info("RunnableSequence created successfully")
        response = chain.invoke({"question": query, "context": context})
        logger.info("Response generated successfully")
        answer = response.content.strip()

        # Supplement with web search if response is insufficient
        insufficient_keywords = ["no definition", "cannot answer", "does not contain", "not explicitly"]
        if enable_web_search and (not answer.strip() or any(keyword in answer.lower() for keyword in insufficient_keywords)):
            web_result = search_web(query)
            if "no relevant information" not in web_result.lower():
                answer = (
                    f"{answer}\n\nHowever, based on a web search:\n{web_result}"
                    if answer.strip()
                    else f"Based on a web search:\n{web_result}"
                )

        return answer

    except ValueError as e:
        logger.error("API Key Error: %s", str(e))
        return "Error: Invalid API key configuration. Please check your settings."
    except (TypeError, RuntimeError) as e:
        # Catch specific exceptions that might occur during chain.invoke or LLM processing
        logger.error("Processing error: %s", str(e))
        return "An error occurred while processing the query. Please try again."
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Fallback for unexpected errors to ensure robustness, suppressed for Pylint
        logger.error("Unexpected error generating answer: %s", str(e))
        return "An error occurred while generating the answer. Please try again."
    #End-of-file
