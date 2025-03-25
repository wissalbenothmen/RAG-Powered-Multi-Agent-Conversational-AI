from pathlib import Path


class Config:
    FLASK_DEBUG = True
    GOOGLE_API_KEY = "api_key"
    FAISS_INDEX_PATH = str(Path("rag/index/faiss_index"))
    VECTOR_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2 model
