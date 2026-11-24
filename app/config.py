import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Corporate RAG Chatbot"
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", os.path.join(base_dir, "db", "chroma_db"))
    
    # Models
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Ingestion
    chunk_size: int = 800
    chunk_overlap: int = 120
    
    # Generation
    max_context_chars: int = 15000 
    
    # GUARDRAILS
    # Cross-Encoder scores are logits. For ms-marco, > 0 is usually relevant.
    # We set a stricter positive threshold to ensure high quality.
    threshold_factual: float = 0.45
    
    # Set to False to disable the heavy vision processing
    enable_charts: bool = True
    
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()