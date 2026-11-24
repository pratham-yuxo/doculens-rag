import logging
import re
from typing import List, Dict, Any, Set, Union
from app.core.resources import ResourceManager

# Setup detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_figure_ids(text: str) -> List[str]:
    """
    Extracts explicit figure IDs from user queries.
    Matches: Figure 5.5, Fig 5.5, Image 5.5, Chart 5.5, Graph 5.5
    """
    # Expanded regex to catch "Graph" and different separators (dot, space, hyphen)
    pattern = r'(?:Figure|Fig\.?|Chart|Image|Plate|Table|Graph)\s*[:\s-]*(\d+(?:[\.-]\d+)*[a-zA-Z]?)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(matches))

async def retrieve_charts(query: str, top_k: int = 5) -> List[Dict]:
    """
    Tri-Layer Retriever: Deterministic ID -> Context Bridge -> Semantic Visual.
    """
    logger.info(f"🚀 [Retriever] Starting retrieval for query: '{query}'")
    
    client = ResourceManager.get_chroma_client()
    chart_collection = None
    text_collection = None

    try:
        available_colls = [c.name for c in client.list_collections()]
        
        # 1. Load Chart Collection
        if "pdf_rag_charts" in available_colls:
            chart_collection = client.get_collection("pdf_rag_charts")
        
        # 2. Smart Text Collection Loading (Auto-detects Prod vs Dev)
        text_coll_name = "pdf_rag"
        if "pdf_rag_prod" in available_colls:
            text_coll_name = "pdf_rag_prod"
        
        if text_coll_name in available_colls:
            text_collection = client.get_collection(text_coll_name)

        if not chart_collection or chart_collection.count() == 0:
            logger.warning("⚠️ Chart collection is empty or missing.")
            return []
            
    except Exception as e:
        logger.error(f"❌ Database Init Error: {e}")
        return []

    model = ResourceManager.get_embedding_model()
    final_results = []
    seen_ids = set()
    found_fig_ids = set() # Audit trail for missing images
    
    # --- LAYER 1: DETERMINISTIC LOOKUP (O(1) Accuracy) ---
    target_ids = extract_figure_ids(query)
    
    if target_ids and chart_collection:
        logger.info(f"   🎯 [Layer 1] User requested IDs: {target_ids}")
        try:
            # Query DB specifically for these IDs
            direct_hits = chart_collection.get(
                where={"figure_id": {"$in": target_ids}},
                include=["documents", "metadatas"]
            )
            
            if direct_hits['ids']:
                count = len(direct_hits['ids'])
                logger.info(f"   ✅ DIRECT HIT: Found {count} charts.")
                
                for i in range(count):
                    chunk_id = direct_hits['ids'][i]
                    meta = direct_hits['metadatas'][i]
                    
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        final_results.append({
                            "text": direct_hits['documents'][i],
                            "metadata": meta,
                            "rerank_score": 1.0, # Perfect Score
                            "id": chunk_id,
                            "source_type": "deterministic_id_match"
                        })
                        fid = meta.get('figure_id')
                        if fid: found_fig_ids.add(fid)

        except Exception as e:
            logger.warning(f"Layer 1 (Direct ID) failed: {e}")

    # --- LAYER 2: THE CONTEXT BRIDGE (Source/Page Mapping) ---
    # Used if Layer 1 missed something, or if the query implies a specific location
    source_page_map: Dict[str, Set[Union[int, str]]] = {}
    
    if text_collection:
        try:
            # A. Broad Semantic Search (Find text discussing the query)
            text_results = text_collection.query(
                query_texts=[query], n_results=3, include=["documents", "metadatas"]
            )
            
            found_docs = text_results['documents'][0] if text_results['documents'] else []
            found_metas = text_results['metadatas'][0] if text_results['metadatas'] else []

            # B. Pinpoint Text Search (If we have IDs but missed them in Layer 1, find their mention in text)
            if target_ids:
                pinpoint_queries = [f"Figure {ref}" for ref in target_ids]
                pinpoint_results = text_collection.query(
                    query_texts=pinpoint_queries, n_results=2, include=["documents", "metadatas"]
                )
                if pinpoint_results['documents']:
                    for i in range(len(pinpoint_results['documents'])):
                        found_docs.extend(pinpoint_results['documents'][i])
                        found_metas.extend(pinpoint_results['metadatas'][i])

            # C. Map Text -> Pages
            # We look at the found text. If it mentions our target ID, we grab that page.
            for i, doc in enumerate(found_docs):
                doc_refs = extract_figure_ids(doc)
                
                # Logic: If this text chunk mentions a target figure, OR if we have no targets (general query)
                is_relevant = any(ref in target_ids for ref in doc_refs) or (not target_ids)
                
                if is_relevant:
                    meta = found_metas[i]
                    src = meta.get('source') or meta.get('file_name') or meta.get('filename')
                    pg = meta.get('page_no')
                    
                    if src and pg:
                        if src not in source_page_map: source_page_map[src] = set()
                        source_page_map[src].add(pg)

        except Exception as e:
            logger.warning(f"Layer 2 (Bridge Logic) warning: {e}")

    # Execute Bridge Fetch
    if source_page_map:
        logger.info(f"   2️⃣  [Layer 2] Bridge Map: {source_page_map}")
        for source, pages in source_page_map.items():
            
            # Safe split for Chroma types (Int vs Str) to prevent crashes
            int_pages = []
            for p in pages:
                try:
                    if isinstance(p, int): int_pages.append(p)
                    elif isinstance(p, str) and p.isdigit(): int_pages.append(int(p))
                except: pass
            
            if int_pages:
                try:
                    # Fetch ALL charts on these specific pages
                    res = chart_collection.get(
                        where={"page_no": {"$in": int_pages}},
                        include=["documents", "metadatas"]
                    )
                    
                    if res['ids']:
                        for i in range(len(res['ids'])):
                            chunk_id = res['ids'][i]
                            meta = res['metadatas'][i]
                            
                            # Soft Source Filter (Python-side)
                            # Checks if the chart's source loosely matches the text's source
                            c_src = meta.get('source') or meta.get('filename') or ""
                            if any(target in c_src or c_src in target for target in source_page_map.keys()):
                                if chunk_id not in seen_ids:
                                    seen_ids.add(chunk_id)
                                    final_results.append({
                                        "text": res['documents'][i],
                                        "metadata": meta,
                                        "rerank_score": 0.95, # High Confidence
                                        "id": chunk_id,
                                        "source_type": "context_bridge"
                                    })
                                    fid = meta.get('figure_id')
                                    if fid: found_fig_ids.add(fid)

                except Exception as e:
                    logger.warning(f"Layer 2 (Bridge Fetch) failed: {e}")

    # --- LAYER 3: SEMANTIC FALLBACK (Visual Description Search) ---
    # Run only if we haven't filled our quota or if no specific IDs were found
    if len(final_results) < top_k:
        logger.info("   3️⃣  [Layer 3] Running semantic visual search...")
        try:
            q_emb = model.encode(query, normalize_embeddings=True).tolist()
            vec_results = chart_collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if vec_results['ids'] and vec_results['ids'][0]:
                for i in range(len(vec_results['ids'][0])):
                    chunk_id = vec_results['ids'][0][i]
                    meta = vec_results['metadatas'][0][i]
                    
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        dist = vec_results['distances'][0][i]
                        # Score: 1.0 - Distance (Max 0.9 to stay below Layer 1/2)
                        score = min(0.9, 1.0 - dist)
                        
                        final_results.append({
                            "text": vec_results['documents'][0][i],
                            "metadata": meta,
                            "rerank_score": score,
                            "id": chunk_id,
                            "source_type": "semantic_visual"
                        })
                        fid = meta.get('figure_id')
                        if fid: found_fig_ids.add(fid)
        except Exception as e:
            logger.error(f"Layer 3 failed: {e}")

    # --- AUDIT: MISSED IMAGES ---
    if target_ids:
        missed = [tid for tid in target_ids if tid not in found_fig_ids]
        if missed:
            logger.warning(f"⚠️ [MISSED] Requested IDs not found in DB: {missed}")
            logger.info("   -> Reason: ID might be missing from metadata or OCR failed.")
        else:
            logger.info("✅ All requested figures found.")

    # Finish
    final_results.sort(key=lambda x: x['rerank_score'], reverse=True)
    logger.info(f"🏁 Retrieval complete. Returning {len(final_results)} charts.")
    
    out = []
    # Return top_k + 2 to allow context matches to shine
    for r in final_results[:top_k+2]:
        out.append({
            "text": r['text'],
            "metadata": r['metadata'],
            "rerank_score": round(r['rerank_score'], 4)
        })
    return out