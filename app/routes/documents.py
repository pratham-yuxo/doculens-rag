import logging
import os
import tempfile
import asyncio
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.models.schemas import UploadResponse, DocumentInfo
from app.services.embedding import store
from app.config import get_settings

# --- DOCLING IMPORTS ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    AcceleratorOptions, 
    AcceleratorDevice
)
from docling.datamodel.base_models import InputFormat

# --- LOGIC IMPORTS ---
from app.services.ingestion import build_chunks_from_docling
try:
    from app.charts.ingestor import process_charts_from_docling
except ImportError:
    process_charts_from_docling = None

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

def run_docling_pipeline(file_path: str, filename: str):
    """
    Unified Background Task:
    1. Parse PDF with Docling (High Quality).
    2. Extract Markdown Text -> Vector DB.
    3. Extract Images -> Vision Analysis -> Vector DB.
    """
    try:
        logger.info(f"🚀 [Docling] Parsing {filename}...")
        
        # --- PERFORMANCE OPTIMIZATION ---
        pipeline_options = PdfPipelineOptions()
        
        # 1. OCR Settings (Fixes RapidOCR spam & Latency)
        # Disable OCR for digital PDFs to speed up processing by 10x
        pipeline_options.do_ocr = False 
        
        # 2. Table Structure (Keep this True for table accuracy)
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST 
        
        # 3. Image Extraction (Required for Charts)
        # DISABLE generate_page_images to prevent std::bad_alloc (OOM)
        pipeline_options.generate_page_images = True 
        # Temporarily disable picture images to fix crash
        pipeline_options.generate_picture_images = True 
        
        # 4. Accelerator (Low Memory Mode)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, 
            device=AcceleratorDevice.CUDA
        )

        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        })
        
        logger.info(f"⏳ [Docling] Starting conversion for {filename}...")
        
        # 1. Run Docling (Heavy Lifting)
        doc = converter.convert(file_path).document
        
        # 2. Text Ingestion (Page-Aware Chunking)
        chunks = build_chunks_from_docling(doc, filename)
        
        # Add to Vector Store (Async wrapper for background thread)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(store.add_chunks(chunks))
        finally:
            loop.close()
            
        logger.info(f"✅ [Text] Indexed {len(chunks)} chunks from Docling.")
        
        # 3. Chart Ingestion (Using Docling's detected figures)
        if settings.enable_charts and process_charts_from_docling:
            process_charts_from_docling(doc, filename)
            
    except Exception as e:
        logger.error(f"❌ [Docling] Failed {filename}: {e}", exc_info=True)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
@router.delete("/clear")
async def clear_documents():
    """
    Deletes the entire ChromaDB collection, removing all documents, chunks, and embeddings.
    """
    await store.clear()
    return {"status": "ok", "message": "All documents have been cleared from the database."}

@router.get("/list", response_model=List[DocumentInfo])
def list_documents():
    """
    Returns a list of all ingested documents in the vector DB.
    """
    return store.get_all_documents()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")
    
    try:
        # Write to temp file (Docling needs a path)
        safe_name = f"upload_{file.filename}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Trigger Unified Background Task
        background_tasks.add_task(run_docling_pipeline, tmp_path, file.filename)
        
        return UploadResponse(
            filename=file.filename,
            status="success",
            message="Processing started. Docling is parsing text and charts in the background."
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))