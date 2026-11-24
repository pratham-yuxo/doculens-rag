import logging
import nltk
import numpy as np
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.resources import ResourceManager
from ..config import get_settings

# Docling Imports
from docling.datamodel.document import DocItemLabel

logger = logging.getLogger(__name__)
settings = get_settings()

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

def semantic_split(text: str, threshold_percentile: int = 90) -> List[str]:
    """
    Splits text into semantically coherent chunks using the embedding model.
    """
    if not text.strip(): return []
    
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2: return [text]
        
    model = ResourceManager.get_embedding_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    
    distances = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1])
        distances.append(1 - sim)
        
    if not distances: return [text]
        
    breakpoint_dist = np.percentile(distances, threshold_percentile)
    chunks = []
    current_chunk = [sentences[0]]
    
    for i, dist in enumerate(distances):
        if dist > breakpoint_dist:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def build_chunks_from_docling(docling_doc, filename: str) -> List[Dict]:
    """
    Extracts text PAGE-BY-PAGE from Docling.
    PRESERVES PAGE NUMBERS accurately for RAG citations.
    """
    final_chunks = []
    chunk_counter = 0
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Iterate pages to maintain strict Page Number metadata
    for page_no, page in docling_doc.pages.items():
        
        page_text_items = []
        
        # Iterate items in the document to find ones on this page
        for item, level in docling_doc.iterate_items():
            
            # 1. Filter: Item must be on the current page
            # prov is a list of locations (e.g. for split tables). We take the first.
            if not item.prov or item.prov[0].page_no != page_no:
                continue

            # 2. Filter: Skip images/formulas in Text Stream (Handled by Chart Ingestor)
            if item.label in [DocItemLabel.PICTURE, DocItemLabel.FORMULA]:
                continue
            
            # 3. Content: Prefer Markdown (preserves Tables/Lists) > Text
            if hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown(doc=docling_doc)
            elif hasattr(item, "text"):
                text = item.text
            else:
                continue
                
            if text and text.strip():
                page_text_items.append(text)

        # Combine page content
        full_page_text = "\n\n".join(page_text_items)
        if not full_page_text.strip():
            continue

        # 4. Hybrid Chunking (Semantic + Recursive)
        try:
            semantic_groups = semantic_split(full_page_text)
        except Exception:
            semantic_groups = [full_page_text]
            
        for group in semantic_groups:
            chunks = splitter.split_text(group)
            for txt in chunks:
                final_chunks.append({
                    "filename": filename,
                    "page_no": page_no, # ✅ Exact Page Match
                    "chunk_id": chunk_counter,
                    "text": txt,
                    "metadata": {
                        "source": filename, 
                        "page": str(page_no)
                    }
                })
                chunk_counter += 1
                
    return final_chunks