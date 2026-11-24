import uuid
import logging
import base64
import io
import time
import re
import hashlib
import requests
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from app.core.resources import ResourceManager
from app.config import get_settings

from docling.datamodel.document import DocItemLabel

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
settings = get_settings()

# --- MODEL CONFIGURATION ---
MODEL_PRIMARY = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_BACKUP = "meta-llama/llama-4-maverick-17b-128e-instruct"

def sanitize_image(pil_image) -> str:
    """Converts PIL Image to clean RGB JPEG base64 string."""
    try:
        buffer = io.BytesIO()
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize to 1024x1024 to save tokens and speed up
        if pil_image.width > 1024 or pil_image.height > 1024:
            pil_image.thumbnail((1024, 1024))
            
        pil_image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Image sanitization failed: {e}")
        return ""

def call_groq_vision_with_retry(model_id: str, b64_img: str, known_id: str = None) -> str:
    """Makes API call with EXPONENTIAL BACKOFF."""
    
    # --- PROMPT OPTIMIZATION ---
    # If ID is known from layout, skip the extraction step to save tokens/latency.
    if known_id:
        system_instruction = (
            f"You are a financial chart analyst. We have identified this chart as '{known_id}' from the document layout.\n"
            "TASK: Provide a dense, detailed summary of the chart content, data trends, X/Y axes, and key labels.\n"
            "Do NOT repeat the ID. Just describe the visual data."
        )
    else:
        system_instruction = (
            "You are a financial chart analyst. Extract metadata and describe this chart.\n"
            "1. EXTRACT LABEL: Look for 'Figure X.X', 'Fig X', 'Chart X' inside the image.\n"
            "   - If found, start response with 'ID: Figure X.X'.\n"
            "   - If NOT found, start response with 'ID: None'.\n"
            "2. DESCRIBE: Provide a dense summary of the data, trends, and axes.\n"
        )

    payload = {
        "model": model_id,
        "messages": [{
            "role": "user", 
            "content": [
                {"type": "text", "text": system_instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }],
        "max_tokens": 500
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json=payload,
                timeout=45
            )
            
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            
            elif resp.status_code == 429:
                wait_time = (attempt + 1) * 5 
                logger.warning(f"⏳ Rate Limit (429) for {model_id}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue 
            
            else:
                return f"ERROR_{resp.status_code}: {resp.text}"
                
        except Exception as e:
            logger.error(f"Connection Error: {e}")
            return "ERROR_CONN"
            
    return "ERROR_RATE_LIMIT_EXHAUSTED"

def analyze_chart_with_groq(pil_image, known_id=None) -> str:
    if not settings.groq_api_key: return ""
    b64_img = sanitize_image(pil_image)
    if not b64_img: return ""
    
    # Try Priority 1
    result = call_groq_vision_with_retry(MODEL_PRIMARY, b64_img, known_id)
    if "ERROR" not in result: return result
        
    logger.warning(f"⚠️ Primary model failed. Switching to backup...")
    
    # Try Priority 2
    result_backup = call_groq_vision_with_retry(MODEL_BACKUP, b64_img, known_id)
    if "ERROR" not in result_backup: return result_backup
        
    logger.error(f"❌ All vision models failed.")
    return ""

def extract_figure_id_from_llm(text: str) -> Optional[str]:
    """Parses LLM output for ID."""
    if not text: return None
    strict_pattern = r'(?:\*\*|__)?ID(?:\*\*|__)?[:\s-]*\s*(?:Figure|Fig\.?|Chart|Plate|Graph)\s*[:\s-]*(\d+(?:[\.-]\d+)*[a-zA-Z]?)'
    match = re.search(strict_pattern, text, re.IGNORECASE)
    if match: return match.group(1)
    
    # Relaxed
    intro_text = text[:150]
    relaxed_pattern = r'(?:Figure|Fig\.?|Chart|Plate|Graph)\s*[:\s-]*(\d+(?:[\.-]\d+)*[a-zA-Z]?)'
    match_relaxed = re.search(relaxed_pattern, intro_text, re.IGNORECASE)
    if match_relaxed: return match_relaxed.group(1)
    return None

def find_caption_in_neighbors(items: List[Any], current_idx: int) -> Optional[str]:
    """
    OPTIMAL ALGORITHM: Scans neighbors for text patterns like "Figure 5.5".
    Priority: [Next, Prev, Next+1, Prev-1]
    Filters: Ignores Headers/Footers and Long Paragraphs.
    """
    caption_pattern = r'^(?:Figure|Fig\.?|Chart|Plate|Graph)\s*(\d+(?:[\.-]\d+)*[a-zA-Z]?)'
    
    # Check Next item first (most common for captions), then Prev
    priority_offsets = [1, -1, 2, -2]
    
    for offset in priority_offsets:
        idx = current_idx + offset
        
        # Boundary Check
        if idx < 0 or idx >= len(items):
            continue
            
        neighbor = items[idx]
        
        # 1. Label Filter: Ignore Page Headers/Footers to prevent "1.1" hallucinations
        if hasattr(neighbor, 'label'):
            if neighbor.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                continue
        
        # 2. Text Check
        if hasattr(neighbor, 'text') and neighbor.text:
            text = neighbor.text.strip()
            
            # 3. Length Heuristic: Captions are usually short. 
            # If text > 250 chars, it's likely a paragraph referencing the figure, not the caption itself.
            if len(text) > 250:
                continue

            match = re.search(caption_pattern, text, re.IGNORECASE)
            if match:
                fid = match.group(1)
                
                # 4. Section Header Sanity Check
                if hasattr(neighbor, 'label') and neighbor.label == DocItemLabel.SECTION_HEADER:
                     # Only accept if it strictly says "Figure 1.1", not just "1.1 Introduction"
                     if not re.match(r'^(?:Figure|Fig|Chart|Graph)', text, re.IGNORECASE):
                         continue
                
                return fid
    return None

def process_charts_from_docling(docling_doc, filename: str):
    """
    Ingests charts using Hybrid Approach: Layout Analysis (Docling) + Vision (Groq).
    """
    if not settings.enable_charts: 
        logger.info("🚫 Chart processing disabled in settings.")
        return

    logger.info(f"🎨 [Chart Ingestion] Starting processing for: {filename}")
    
    client = ResourceManager.get_chroma_client()
    collection = client.get_or_create_collection("pdf_rag_charts")
    model = ResourceManager.get_embedding_model() 

    ids, docs, metas, embeddings = [], [], [], []
    count = 0
    skipped_count = 0
    
    # Optimization: Convert iterator to list for neighbor access
    all_items = [item for item, level in docling_doc.iterate_items()]
    
    for i, item in enumerate(all_items):
        
        if hasattr(item, 'label') and item.label == DocItemLabel.PICTURE:
            
            # --- STRICT PAGE LOGIC ---
            if item.prov and len(item.prov) > 0:
                page_no = item.prov[0].page_no
            else:
                page_no = None # Sentinel

            if not isinstance(page_no, int):
                skipped_count += 1
                continue
            
            try:
                img_obj = item.get_image(docling_doc)
                if not img_obj: continue

                # Filter tiny icons
                if img_obj.width < 100 or img_obj.height < 100: 
                    logger.debug(f"   -> Skipped (Page {page_no}): Image too tiny.")
                    skipped_count += 1
                    continue

                # --- STEP 1: DETERMINISTIC CAPTION HUNTING ---
                detected_id = find_caption_in_neighbors(all_items, i)
                
                if detected_id:
                    logger.info(f"   🎯 [Layout] Found ID '{detected_id}' from caption text!")
                else:
                    logger.info(f"   🔍 [Layout] No caption found. Falling back to Vision LLM...")

                # --- STEP 2: VISION ANALYSIS ---
                time.sleep(1.0) # Reduced throttle slightly
                desc = analyze_chart_with_groq(img_obj, known_id=detected_id)
                
                if not desc or "ERROR" in desc:
                    skipped_count += 1
                    continue
                if "Not a chart" in desc or "icon" in desc.lower() and len(desc) < 50: 
                    skipped_count += 1
                    continue
                
                # Determine Final ID
                final_id = detected_id
                if not final_id:
                    # Only parse LLM output if we didn't find it in layout
                    final_id = extract_figure_id_from_llm(desc)
                    if final_id: logger.info(f"   🧠 [Vision] Extracted ID '{final_id}' from image pixels.")

                chunk_text = f"Figure/Chart Analysis (Page {page_no}): {desc}"
                
                # --- DETERMINISTIC ID GENERATION ---
                # Fixes duplicate ingestion pollution. 
                # Hash based on filename + page + index so re-running is safe.
                unique_str = f"{filename}_p{page_no}_idx{i}"
                chart_uuid = hashlib.md5(unique_str.encode()).hexdigest()
                chart_id = f"{filename}_chart_{chart_uuid}"
                
                ids.append(chart_id)
                docs.append(chunk_text)
                
                metas.append({
                    "filename": filename,
                    "source": filename,
                    "page_no": page_no,
                    "is_chart": True,
                    "visual_summary": desc,
                    "figure_id": final_id if final_id else "unknown" 
                })
                
                embeddings.append(model.encode(chunk_text, normalize_embeddings=True).tolist())
                count += 1
                
            except Exception as e:
                logger.error(f"   -> ERROR processing image on Page {page_no}: {e}")
                skipped_count += 1

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        logger.info(f"✅ [Success] Indexed {count} charts for {filename}")
    else:
        logger.warning(f"⚠️ [Info] No valid charts found for {filename}.")