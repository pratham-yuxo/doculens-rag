import logging
import asyncio
from typing import List, Dict
from app.core.resources import ResourceManager
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingStore:
    def __init__(self):
        # USE SINGLETON: Do not create a new client; get the shared one.
        # This prevents loading the model multiple times.
        self.client = ResourceManager.get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name="pdf_rag_prod",
            metadata={"hnsw:space": "cosine"}
        )

    def _sanitize(self, meta: Dict) -> Dict:
        """Prevent DB crash on None types."""
        return {k: str(v) if isinstance(v, (list, dict, type(None))) else v for k, v in meta.items()}

    async def add_chunks(self, chunks: List[Dict]):
        if not chunks: return
        # Async offload to prevent blocking main thread
        await asyncio.to_thread(self._add_sync, chunks)

    def _add_sync(self, chunks: List[Dict]):
        # USE SINGLETON MODEL
        model = ResourceManager.get_embedding_model()
        
        texts = [c["text"] for c in chunks]
        # Generate embeddings using the shared model
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        ids, docs, metas = [], [], []
        for i, c in enumerate(chunks):
            # Create a unique ID: filename_chunkID
            ids.append(f"{c['filename']}_{c['chunk_id']}")
            docs.append(c["text"])
            
            # Prepare metadata
            m = c.get("metadata", {})
            m.update({
                "filename": c["filename"], 
                "page_no": c["page_no"], 
                "chunk_id": c["chunk_id"]
            })
            metas.append(self._sanitize(m))

        self.collection.upsert(
            ids=ids, 
            documents=docs, 
            embeddings=embeddings.tolist(), 
            metadatas=metas
        )

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        return await asyncio.to_thread(self._search_sync, query, top_k)
    
    def _search_sync(self, query: str, top_k: int):
        # USE SINGLETON MODEL
        model = ResourceManager.get_embedding_model()
        
        q_emb = model.encode(query, normalize_embeddings=True).tolist()
        
        res = self.collection.query(
            query_embeddings=[q_emb], 
            n_results=top_k, 
            include=["documents", "metadatas", "distances"]
        )
        
        out = []
        if not res["ids"] or not res["ids"][0]: return out
        
        for i in range(len(res["ids"][0])):
            out.append({
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                # Convert distance to similarity score (1 - distance)
                "score": 1.0 - res["distances"][0][i] 
            })
        return out

    async def clear(self):
        """Asynchronously clears the DB."""
        await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self):
        """Synchronously clears/resets the collection."""
        try:
            self.client.delete_collection("pdf_rag_prod")
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(
            name="pdf_rag_prod",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("🧹 ChromaDB collection cleared.")

    def get_all_documents(self) -> List[Dict]:
        """Summary of uploaded docs for the UI."""
        try:
            results = self.collection.get(include=["metadatas"])
            metas = results.get("metadatas", []) or []
            
            docs = {}
            for m in metas:
                if not m: continue
                fname = m.get("filename", "unknown")
                if fname not in docs:
                    docs[fname] = {"filename": fname, "pages": set(), "chunk_count": 0}
                
                docs[fname]["chunk_count"] += 1
                if "page_no" in m:
                    docs[fname]["pages"].add(m["page_no"])
            
            output = []
            for d in docs.values():
                output.append({
                    "filename": d["filename"],
                    "chunks": d["chunk_count"],
                    "pages": len(d["pages"])
                })
            return output
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

store = EmbeddingStore()