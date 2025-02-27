"""
Overview:
This module provides source tracking for a RAG-powered AI system.
It processes a list of document sources to identify unique entries and assign focus areas.

Technical Details:
- Input: List of source dictionaries with metadata (e.g., source path, similarity score).
- Logic: Deduplicates sources by path, keeping the highest similarity score.
- Focus Areas: Derived from filenames if not provided, with fallback to 'General AI'.
- Output: List of unique source dictionaries with enriched metadata.
"""

import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


def track_sources(sources) -> list:
    """Track and deduplicate document sources, assigning focus areas.

    Args:
        sources: List of dictionaries containing source metadata.

    Returns:
        list: Deduplicated list of source dictionaries with focus areas.
    """
    unique_sources = {}

    for source in sources:
        # Extract source path from metadata
        source_path = source["source"]

        # Set default focus area or use existing one
        focus_area = source.get("focus_area", "Unknown")
        if not focus_area or focus_area == "Unknown":
            # Derive focus area from filename if not specified
            filename = source_path.split("\\")[-1].lower()
            if "rag" in filename and "cache" in filename:
                focus_area = "Cache-Augmented Generation"
            elif "rag" in filename:
                focus_area = "Retrieval-Augmented Generation"
            elif "enterprise" in filename:
                focus_area = "Enterprise Applications"
            else:
                focus_area = "General AI"

        # Update unique sources with highest similarity score
        if (source_path not in unique_sources or
                source["similarity_score"] > unique_sources[source_path]["similarity_score"]):
            unique_sources[source_path] = {
                "source": source_path,
                "focus_area": focus_area,
                "similarity_score": source["similarity_score"],
                "similarity_type": source.get("similarity_type", "cosine"),
            }

    # Convert dictionary to list of tracked sources
    tracked = list(unique_sources.values())
    logger.debug("Tracked %d unique sources", len(tracked))
    return tracked

#end-of-file
