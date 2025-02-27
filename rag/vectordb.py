"""
Overview:
This module defines the VectorDB class for managing a vector database 
in a RAG-powered AI system.
It handles document indexing and retrieval using FAISS and HuggingFace embeddings.

Technical Details:
- Storage: Indexes PDFs from a GCS bucket, stores FAISS index in another bucket.
- Embedding: Uses HuggingFace model 'all-MiniLM-L6-v2' for vectorization.
- Database: Tracks processed files via DatabaseManager.
- Resilience: Implements retries for GCS operations.
"""
import logging
import os
import io
import tempfile
import time
import shutil
from typing import List, Dict
from pathlib import Path
from zipfile import ZipFile

import dotenv
from tqdm import tqdm
import google.cloud.storage as storage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from database.db_manager import DatabaseManager

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorDB:
    """Class to manage vector indexing and retrieval for RAG documents."""

    def __init__(
        self,
        data_dir: str = "gs://rag-multiagent-documents/",
        index_dir: str = "gs://rag-multiagent-index/",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize VectorDB with GCS and embedding settings.

        Args:
            data_dir (str): GCS bucket for documents.
            index_dir (str): GCS bucket for FAISS index.
            chunk_size (int): Size of document chunks.
            chunk_overlap (int): Overlap between chunks.
            model_name (str): HuggingFace embedding model name.
        """
        # Instance attributes for document and index management
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.bucket_name = os.getenv("DOCUMENTS_BUCKET", "rag-multiagent-documents")
        self.index_bucket_name = os.getenv("INDEX_BUCKET", "rag-multiagent-index")
        self.storage_client = storage.Client()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_manager = DatabaseManager()
        # Load embeddings with CPU fallback
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Loaded embedding model: %s", model_name)
        except Exception as e:
            logger.error("Failed to load embedding model: %s", str(e), exc_info=True)
            raise
        self.index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index").rstrip("/")
        # Check and load existing index from GCS
        if self._index_exists_in_gcs():
            self._load_index_from_gcs()
        else:
            self.vectorstore = None
            logger.info("No existing index found in GCS")

    def _index_exists_in_gcs(self) -> bool:
        """Check if FAISS index exists in GCS bucket."""
        bucket = self.storage_client.bucket(self.index_bucket_name)
        blob = storage.Blob(self.index_path, bucket)
        return blob.exists()

    def _load_index_from_gcs(self) -> None:
        """Load FAISS index from GCS into memory."""
        bucket = self.storage_client.bucket(self.index_bucket_name)
        blob = bucket.blob(self.index_path)
        temp_buffer = io.BytesIO()
        blob.download_to_file(temp_buffer)
        temp_buffer.seek(0)
        temp_dir = tempfile.mkdtemp()
        try:
            with ZipFile(temp_buffer, "r") as zipf:
                zipf.extractall(temp_dir)
            self.vectorstore = FAISS.load_local(
                temp_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded FAISS index from GCS")
        except Exception as e:
            logger.error("Failed to load FAISS index from GCS: %s", str(e), exc_info=True)
            self.vectorstore = None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temporary directory: %s", temp_dir)
            temp_buffer.close()

    def _save_index_to_gcs(self, force_replace: bool = False) -> None:
        """Save FAISS index to GCS with retry logic.

        Args:
            force_replace (bool): Overwrite existing index if True.
        """
        if not self.vectorstore:
            logger.error("No vectorstore available to save")
            return

        bucket = self.storage_client.bucket(self.index_bucket_name)
        blob = bucket.blob(self.index_path)

        if blob.exists() and not force_replace:
            logger.info(
                "FAISS index exists at %s in %s. Use force_replace=True to overwrite.",
                self.index_path,
                self.index_bucket_name
            )
            return

        save_dir = tempfile.mkdtemp()
        save_path = os.path.join(save_dir, "faiss_index")
        try:
            self.vectorstore.save_local(save_path)
            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, "w") as zipf:
                for root, _, files in os.walk(save_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, save_path)
                        zipf.write(file_path, arcname)
            zip_buffer.seek(0)

            max_attempts = 5
            delay = 2
            for attempt in range(max_attempts):
                try:
                    blob.upload_from_file(zip_buffer, rewind=True, timeout=300)
                    logger.info(
                        "Saved FAISS index to GCS bucket %s at %s",
                        self.index_bucket_name,
                        self.index_path
                    )
                    break
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "Attempt %d failed: %s. Retrying in %d seconds...",
                            attempt + 1,
                            str(e),
                            delay
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(
                            "Failed to upload index after %d attempts: %s",
                            max_attempts,
                            str(e),
                            exc_info=True
                        )
                        raise
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)
            logger.debug("Cleaned up temporary directory: %s", save_dir)
            if "zip_buffer" in locals():
                zip_buffer.close()

    def _load_and_split_document(self, blob_name: str) -> List:
        """Load and split a document from GCS, ensuring original filename in metadata.

        Args:
            blob_name (str): Name of the blob in GCS bucket.

        Returns:
            List: List of document chunks.
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(blob_name)
                temp_buffer = io.BytesIO()
                blob.download_to_file(temp_buffer)
                temp_buffer.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(temp_buffer.read())
                    temp_file_path = temp_file.name

                try:
                    loader = PyMuPDFLoader(temp_file_path)
                    documents = loader.load()
                    for doc in documents:
                        # Keep only original blob_name in metadata
                        doc.metadata = {"source": blob_name}
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len,
                        add_start_index=True,
                        separators=["\n\n", "\n", ". ", " ", ""],
                    )
                    chunks = text_splitter.split_documents(documents)
                    logger.debug("Split %s into %d chunks", blob_name, len(chunks))
                    return chunks
                finally:
                    os.unlink(temp_file_path)
                    logger.debug("Deleted temporary file: %s", temp_file_path)
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Attempt %d failed for %s: %s. Retrying...",
                        attempt + 1,
                        blob_name,
                        str(e)
                    )
                    time.sleep(2)
                else:
                    logger.error(
                        "Failed to process %s after %d attempts: %s",
                        blob_name,
                        max_attempts,
                        str(e),
                        exc_info=True
                    )
                    return []
            finally:
                temp_buffer.close()
        return []

    def index_documents(self, force_replace_index: bool = False, force_reindex_all: bool = False) -> None:
        """Index documents from GCS bucket.

        Args:
            force_replace_index (bool): Replace existing index if True.
            force_reindex_all (bool): Reprocess all files if True.
        """
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = [blob for blob in bucket.list_blobs() if blob.name.endswith(".pdf")]

        if not blobs:
            logger.warning("No PDF files found in the documents bucket")
            return

        if force_reindex_all:
            unprocessed_blobs = [blob.name for blob in blobs]
            logger.info("Forcing reindex of all %d PDF files", len(unprocessed_blobs))
        else:
            unprocessed_blobs = [
                blob.name for blob in blobs
                if blob.name not in self.db_manager.get_processed_files()
            ]
            logger.info("Found %d unprocessed PDF files", len(unprocessed_blobs))

        if not unprocessed_blobs:
            logger.info("No unprocessed PDF files to index")
            if self.vectorstore is None:
                logger.warning("No existing index and no new files to process")
            return

        all_chunks = []
        for blob_name in tqdm(unprocessed_blobs, desc="Processing PDFs"):
            chunks = self._load_and_split_document(blob_name)
            if chunks:
                all_chunks.extend(chunks)
                self.db_manager.add_document(blob_name, len(chunks))
                logger.info("Processed %s with %d chunks", blob_name, len(chunks))
            else:
                logger.warning("No chunks extracted from %s", blob_name)

        if all_chunks:
            if self.vectorstore is None or force_replace_index:
                self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
                logger.info("Created new FAISS index")
            else:
                self.vectorstore.add_documents(all_chunks)
                logger.info("Added chunks to existing FAISS index")
            self._save_index_to_gcs(force_replace=force_replace_index)
        else:
            logger.info("No new chunks; index unchanged")

    def search(self, query: str, top_k: int = 30) -> List[Dict]:
        """Search the vector database for relevant documents.

        Args:
            query (str): Search query string.
            top_k (int): Number of top results to return (default: 30).

        Returns:
            List[Dict]: List of result dictionaries with source, text, and score.
        """
        try:
            if self.vectorstore is None:
                logger.error("No index available. Creating from unprocessed docs...")
                self.index_documents(force_replace_index=False, force_reindex_all=False)
                if self.vectorstore is None:
                    logger.error("Failed to create index. No results available.")
                    return []
            results = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
            seen_sources = {}
            for doc, score in results:
                source = doc.metadata.get("source", "unknown")
                similarity = float(1 - (score / 2))
                if (source not in seen_sources or
                        similarity > seen_sources[source]["similarity_score"]):
                    seen_sources[source] = {
                        "source": source,
                        "chunk_text": doc.page_content,
                        "similarity_score": similarity,
                        "metadata": doc.metadata,
                    }
            formatted_results = sorted(
                seen_sources.values(),
                key=lambda x: x["similarity_score"],
                reverse=True
            )[:top_k]
            logger.debug(
                "Search results for '%s...': %s",
                query[:50],
                [r['source'] for r in formatted_results]
            )
            return formatted_results
        except Exception as e:
            logger.error("Error during search: %s", str(e), exc_info=True)
            return []

    def as_retriever(self, search_kwargs=None):
        """Return the vectorstore as a LangChain retriever.

        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 30}).

        Returns:
            Retriever object for LangChain compatibility.
        """
        if self.vectorstore is None:
            logger.error("No vectorstore available. Attempting to create index...")
            self.index_documents(force_replace_index=False, force_reindex_all=False)
            if self.vectorstore is None:
                raise ValueError("No index available")
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 30}
        )

#end-of-file
