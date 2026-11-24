import asyncio
import numpy as np 
from typing import List, Dict
from .embedding import store
from app.core.resources import ResourceManager
from ..config import get_settings

settings = get_settings()

def sigmoid(x):
    """
    Squashes raw Cross-Encoder logits (unbounded) into a 0-1 probability range.
    CRITICAL for consistent thresholding (0.45 vs 0.15).
    """
    return 1 / (1 + np.exp(-x))

async def semantic_rerank(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    """
    Stage 2: High Precision Re-ranking
    Uses the Cross-Encoder to deeply analyze query-document relevance.
    """
    if not candidates: return []
    
    # Prepare pairs for the Cross-Encoder: [[Query, Text], [Query, Text], ...]
    pairs = [[query, doc["text"]] for doc in candidates]
    reranker = ResourceManager.get_rerank_model()
    
    # Predict scores (blocking CPU operation run in a separate thread)
    # These return raw logits (e.g., -4.5 to +8.2)
    raw_scores = await asyncio.to_thread(reranker.predict, pairs)
    
    # Apply Sigmoid to convert logits to probabilities (0.0 to 1.0)
    scores = sigmoid(np.array(raw_scores))
    
    for i, doc in enumerate(candidates):
        doc["rerank_score"] = float(scores[i])
        
    # Sort by the new high-precision score
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # Return only the best top_k
    return candidates[:top_k]

async def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    The Industrial Two-Stage Pipeline:
    1. Fast Vector Search (High Recall)
    2. Precise Cross-Encoder Reranking (High Precision)
    """
    
    # 1. Stage 1: Fast Filtering (Bi-Encoder)
    # We fetch 20x the requested amount to ensure we catch relevant chunks
    # even if there are typos (e.g., "captheorem") or vague terms.
    initial_k = top_k * 20 
    
    results = await store.search(query, top_k=initial_k)
    if not results: return []
    
    # 2. Stage 2: Precision Re-ranking (Cross-Encoder)
    # This filters the 20x down to the best top_k using deep semantic analysis.
    return await semantic_rerank(query, results, top_k)