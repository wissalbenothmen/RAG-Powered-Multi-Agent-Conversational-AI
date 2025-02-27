"""
Overview:
This module processes PDF documents in a GCS bucket for a RAG-powered AI system.
It leverages VectorDB to index and test document retrieval.

Technical Details:
- Input: GCS bucket URL with PDFs.
- Processing: Skips processed files unless forced, indexes via VectorDB.
- Testing: Runs sample queries to verify retrieval.
- Logging: Tracks progress and results.
"""
import logging
from pathlib import Path

from .vectordb import VectorDB

# Initialize logger for this module
logger = logging.getLogger(__name__)


def process_documents(
    data_dir: str = "gs://rag-multiagent-documents/",
    top_k: int = 10,
    force_replace_index: bool = False,
    force_reindex_all: bool = False
) -> None:
    """Process documents in a GCS bucket, skipping processed ones unless forced.

    Args:
        data_dir (str): GCS bucket URL with PDFs (default: "gs://rag-multiagent-documents/").
        top_k (int): Number of top results for test queries (default: 10).
        force_replace_index (bool): Replace existing FAISS index if True.
        force_reindex_all (bool): Reprocess all documents if True.
    """
    try:
        # Initialize VectorDB with GCS bucket
        db = VectorDB(data_dir=data_dir)
        logger.info("Starting document processing from GCS bucket")

        # Index documents, respecting force flags
        db.index_documents(
            force_replace_index=force_replace_index,
            force_reindex_all=force_reindex_all
        )

        # Log processed files
        processed_files = db.db_manager.get_processed_files()
        logger.info("Processed files: %s", processed_files)
        logger.info("Total files processed: %d", len(processed_files))

        # Test retrieval with sample queries
        test_queries = [
            "What is RAG?",
            "What is CAG?",
            "What is the difference between RAG and CAG?"
        ]

        for test_query in test_queries:
            logger.debug("Testing search with query: %s", test_query)
            results = db.search(test_query, top_k=top_k)

            if results:
                logger.info("\nTest search results for '%s':", test_query)
                for i, result in enumerate(results, 1):
                    logger.info("Result %d:", i)
                    logger.info("Source: %s", result['source'])
                    logger.info("Similarity: %.2f%%", result['similarity_score'] * 100)
                    text = result["chunk_text"][:200].replace("\n", " ")
                    logger.info("Text snippet: %s...", text)
            else:
                logger.warning("\nNo results found for test query: %s", test_query)

        # Log processing summary
        summary = db.db_manager.get_summary()
        logger.info("\nProcessing Summary:")
        logger.info("Total Documents: %d", summary['total_documents'])
        logger.info("Processed Documents: %d", summary['processed_documents'])
        logger.info("Total Chunks: %d", summary['total_chunks'])
        logger.info("Last Processed: %s", summary['last_processed'])

    except Exception as e:
        logger.error("Error processing documents: %s", str(e), exc_info=True)
        raise

#end-of-file
