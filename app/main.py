import logging
import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import get_settings
from .routes import documents, chat
from .core.resources import ResourceManager
from .services.llm import init_client, close_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy libraries
logging.getLogger("rapidocr").setLevel(logging.ERROR)

settings = get_settings()

def clean_temp_folder():
    """Wipes temp folder on startup to prevent storage leaks."""
    temp_dir = "temp_uploads"
    if os.path.exists(temp_dir):
        logger.info(f"🧹 Cleaning temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 STARTUP: System Init...")
    clean_temp_folder()
    
    # Singleton Model Warmup & NLTK Check
    # This single call now handles NLTK, Embeddings, Reranker, and ChromaDB
    ResourceManager.initialize_all()
    
    # Network Init (Groq Async Client)
    await init_client()
    
    yield
    
    logger.info("🛑 SHUTDOWN: Cleanup...")
    await close_client()

app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["Docs"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/health")
def health(): return {"status": "healthy"}