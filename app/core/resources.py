import logging
import torch
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings as ChromaSettings
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class ResourceManager:
    """
    Singleton Manager: Ensures we only load heavy models ONCE.
    Used by both Text RAG and Chart RAG to save memory.
    """
    _embed_model = None
    _rerank_model = None
    _chroma_client = None

    @staticmethod
    def get_device():
        """Auto-detect best available accelerator."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps" # For Mac M1/M2/M3
        return "cpu"

    @classmethod
    def initialize_all(cls):
        """
        Call this on App Startup to warm up models.
        """
        logger.info("🚀 Warming up AI Resources...")
        cls.ensure_nltk()
        cls.get_embedding_model()
        cls.get_rerank_model()
        cls.get_chroma_client()
        logger.info("✅ All Resources Loaded & Ready.")

    @classmethod
    def ensure_nltk(cls):
        """Checks and downloads required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("📥 Downloading NLTK 'punkt_tab'...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('punkt', quiet=True)

    @classmethod
    def get_embedding_model(cls):
        """
        Returns the shared Bi-Encoder. 
        Chart Ingestor will use this same instance.
        """
        if cls._embed_model is None:
            device = cls.get_device()
            logger.info(f"🧠 Loading Embedding Model: {settings.embed_model} on [{device}]")
            cls._embed_model = SentenceTransformer(settings.embed_model, device=device)
        return cls._embed_model

    @classmethod
    def get_rerank_model(cls):
        if cls._rerank_model is None:
            device = cls.get_device()
            logger.info(f"⚖️ Loading Rerank Model: {settings.rerank_model} on [{device}]")
            cls._rerank_model = CrossEncoder(settings.rerank_model, device=device)
        return cls._rerank_model

    @classmethod
    def get_chroma_client(cls):
        if cls._chroma_client is None:
            logger.info(f"💾 Connecting to ChromaDB at {settings.chroma_db_path}")
            cls._chroma_client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return cls._chroma_client