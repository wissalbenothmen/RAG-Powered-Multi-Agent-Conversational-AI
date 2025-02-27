"""
Overview:
This module provides retrieval functionality for a RAG-powered AI system using LangChain and Gemini.
It validates API keys and retrieves relevant documents from a vector database based on user queries.

Technical Details:
- Dependencies: LangChain for retrieval and prompt chaining, Google Gemini for LLM integration.
- Retrieval: Uses a vector database and falls back to direct search if the chain fails.
- Environment: Loads API key via dotenv from GOOGLE_API_KEY.
- Error Handling: Returns empty list on failure to ensure system stability.
"""

import logging
import os
from typing import List, Dict

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_api_key() -> str:
    """Validate that the Google API key is set and properly formatted.

    Returns:
        str: The validated API key.

    Raises:
        ValueError: If the API key is missing or invalid.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    if not api_key.startswith("AIza"):
        raise ValueError(
            "GOOGLE_API_KEY seems invalid (should start with 'AIza')"
        )
    return api_key


def retrieve_information(query: str, vector_db, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant documents for a query using RAG.

    Args:
        query (str): The user's query.
        vector_db: The pre-initialized VectorDB instance.
        top_k (int): Number of documents to return (default is 3).

    Returns:
        List[Dict]: List of relevant documents with metadata.
    """
    try:
        # Validate API key
        api_key = validate_api_key()

        # Configure retriever from vector database
        retriever = vector_db.as_retriever(search_kwargs={"k": top_k})

        # Initialize Gemini model for processing
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
        )

        # Define prompt template
        template = """Answer the question based only on the context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        # Build retrieval chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Attempt retrieval with chain, fallback to direct search
        try:
            chain.invoke(query)  # Execute chain for side effects
            results = vector_db.search(query, top_k=top_k)
            return results
        except Exception as chain_error:
            logger.warning(
                "Chain retrieval failed: %s, using direct search",
                chain_error
            )
            return vector_db.search(query, top_k=top_k)

    except ValueError as e:
        logger.error("API Key Error: %s", str(e))
        return []
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error in retrieve_information: %s", str(e))
        return []

#end-of-file
