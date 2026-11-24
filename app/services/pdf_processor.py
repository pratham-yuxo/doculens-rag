import logging
from typing import List, Dict
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)

def extract_pdf_content(file_path: str) -> List[Dict]:
    """
    Uses Docling to parse PDF into structured Markdown.
    Returns a list of chunks (not just pages) because Docling handles structure.
    """
    try:
        logger.info(f"🚀 [Docling] Parsing {file_path.split('/')[-1]}...")
        
        # Configure Docling for PDF
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        # Convert
        result = converter.convert(file_path)
        doc = result.document
        
        # We return the Docling document object directly to be chunked later
        # OR we can return a dummy list if we want to keep the signature compatible
        # But better to return the doc object and let ingestion.py handle it.
        # However, to keep compatibility with existing 'pages' logic in routes,
        # we might need to adapt.
        
        # Let's return the doc object wrapped in a list to satisfy the type hint for now,
        # but the real work happens in ingestion.py
        return [doc] 

    except Exception as e:
        logger.error(f"❌ [Docling] Failed {file_path.split('/')[-1]}: {e}")
        raise e

